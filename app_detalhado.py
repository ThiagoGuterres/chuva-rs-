import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium.plugins import HeatMap, Fullscreen, MiniMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error

st.set_page_config(page_title="Chuva no RS — mapa detalhado", page_icon="🌧️", layout="wide")

# ----------------------------------------------------------------- configuração
CIDADES = {
    "Porto Alegre": (-30.03, -51.23),
    "Caxias do Sul": (-29.17, -51.18),
    "Pelotas": (-31.77, -52.34),
    "Santa Maria": (-29.69, -53.81),
    "Passo Fundo": (-28.26, -52.41),
    "Uruguaiana": (-29.76, -57.09),
    "Santa Rosa": (-27.87, -54.48),
    "Bagé": (-31.33, -54.10),
    "Erechim": (-27.63, -52.27),
    "Santa Cruz do Sul": (-29.72, -52.43),
    "Lajeado": (-29.47, -51.96),
    "Vacaria": (-28.51, -50.93),
    "Rio Grande": (-32.03, -52.10),
    "São Borja": (-28.66, -56.00),
}
LIMITES_RS = dict(lat_min=-34.5, lat_max=-26.5, lon_min=-58.5, lon_max=-48.5)
INICIO = "20150101"
URL_MALHA_RS = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/43?formato=application/vnd.geo+json&qualidade=maxima"

RENOMEAR = {
    "PRECTOTCORR": "chuva", "T2M": "temp", "T2M_MAX": "tmax", "T2M_MIN": "tmin",
    "RH2M": "umid", "WS2M": "vento", "PS": "pressao",
}
BASE = ["chuva", "temp", "tmax", "tmin", "umid", "vento", "pressao"]


def dentro_do_rs(lat, lon):
    L = LIMITES_RS
    return L["lat_min"] <= lat <= L["lat_max"] and L["lon_min"] <= lon <= L["lon_max"]


# ----------------------------------------------------------------- dados
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def baixar_contorno_rs():
    try:
        r = requests.get(URL_MALHA_RS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def baixar_power(lat, lon):
    fim = (pd.Timestamp.today() - pd.Timedelta(days=3)).strftime("%Y%m%d")
    r = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params=dict(
            parameters=",".join(RENOMEAR), community="AG",
            latitude=lat, longitude=lon, start=INICIO, end=fim, format="JSON",
        ),
        timeout=180,
    )
    r.raise_for_status()
    dados = r.json()["properties"]["parameter"]
    df = pd.DataFrame(dados).rename(columns=RENOMEAR)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"
    df = df.replace(-999, np.nan)
    return df.loc[:df["chuva"].last_valid_index()]


def criar_features(df):
    df = df.sort_index().copy()
    for c in BASE:
        for l in (1, 2, 3):
            df[f"{c}_l{l}"] = df[c].shift(l)
    df["chuva_7d"] = df["chuva"].rolling(7).sum()
    df["chuva_30d"] = df["chuva"].rolling(30).sum()
    df["dif_pressao"] = df["pressao"].diff()
    df["dif_temp"] = df["temp"].diff()
    doy = df.index.dayofyear
    df["sen_ano"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_ano"] = np.cos(2 * np.pi * doy / 365.25)
    df["alvo_mm"] = df["chuva"].shift(-1)
    df["alvo_chuva"] = (df["alvo_mm"] >= 1).astype(float).where(df["alvo_mm"].notna())
    return df


def novo_clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)


def novo_reg():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)


@st.cache_resource(
    ttl=12 * 3600,
    show_spinner="Baixando dados da NASA e treinando o modelo (na primeira vez leva alguns minutos)...",
)
def preparar():
    partes = []
    for nome, (lat, lon) in CIDADES.items():
        f = criar_features(baixar_power(lat, lon))
        f["lat"], f["lon"], f["cidade"] = lat, lon, nome
        partes.append(f)
    feats = pd.concat(partes)
    cols = [c for c in feats.columns if c not in ("alvo_mm", "alvo_chuva", "cidade")]

    com_alvo = feats.dropna(subset=["alvo_mm"])
    treino = com_alvo[com_alvo.index < "2024-01-01"]
    teste = com_alvo[com_alvo.index >= "2024-01-01"]

    clf_av = novo_clf().fit(treino[cols], treino["alvo_chuva"])
    reg_av = novo_reg().fit(treino[cols], treino["alvo_mm"])
    prob = clf_av.predict_proba(teste[cols])[:, 1]
    mm = np.clip(reg_av.predict(teste[cols]), 0, None)
    metricas = {
        "auc": roc_auc_score(teste["alvo_chuva"], prob),
        "f1": f1_score(teste["alvo_chuva"], prob >= 0.5),
        "f1_ing": f1_score(teste["alvo_chuva"], (teste["chuva"] >= 1).astype(int)),
        "mae": mean_absolute_error(teste["alvo_mm"], mm),
        "mae_ing": mean_absolute_error(teste["alvo_mm"], teste["chuva"]),
    }

    clf = novo_clf().fit(com_alvo[cols], com_alvo["alvo_chuva"])
    reg = novo_reg().fit(com_alvo[cols], com_alvo["alvo_mm"])

    ultimo = feats.groupby("cidade").tail(1).copy()
    ultimo["prob"] = clf.predict_proba(ultimo[cols])[:, 1]
    ultimo["mm"] = np.clip(reg.predict(ultimo[cols]), 0, None)
    return dict(clf=clf, reg=reg, cols=cols, metricas=metricas, ultimo=ultimo)


# ----------------------------------------------------------------- mapa
def cor(p):
    return "#2e7d32" if p < 0.3 else ("#f9a825" if p < 0.6 else "#c62828")


def montar_mapa(ultimo, contorno, ponto=None):
    m = folium.Map(
        location=[-29.7, -53.5], zoom_start=7, min_zoom=6, tiles="cartodbpositron",
        max_bounds=True, min_lat=LIMITES_RS["lat_min"], max_lat=LIMITES_RS["lat_max"],
        min_lon=LIMITES_RS["lon_min"], max_lon=LIMITES_RS["lon_max"],
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("cartodbdark_matter", name="Escuro").add_to(m)

    if contorno:
        folium.GeoJson(
            contorno, name="Contorno do RS",
            style_function=lambda _: {"color": "#1565c0", "weight": 2.5, "fillOpacity": 0},
        ).add_to(m)

    camada_calor = folium.FeatureGroup(name="Mapa de calor (risco)")
    HeatMap(
        [[r["lat"], r["lon"], max(r["prob"], 0.05)] for _, r in ultimo.iterrows()],
        radius=55, blur=35, max_zoom=7,
    ).add_to(camada_calor)
    camada_calor.add_to(m)

    camada_cidades = folium.FeatureGroup(name="Cidades")
    for _, r in ultimo.iterrows():
        popup = folium.Popup(
            f"<b>{r['cidade']}</b><br>"
            f"Chance de chuva amanhã: <b>{r['prob'] * 100:.0f}%</b><br>"
            f"Estimativa: {r['mm']:.1f} mm<br>"
            f"Temp. atual: {r['temp']:.1f} °C<br>"
            f"Umidade: {r['umid']:.0f}%<br>"
            f"Vento: {r['vento']:.1f} m/s<br>"
            f"Pressão: {r['pressao']:.1f} kPa",
            max_width=250,
        )
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=11, color="#222", weight=1,
            fill=True, fill_color=cor(r["prob"]), fill_opacity=0.9,
            tooltip=f"{r['cidade']}: {r['prob'] * 100:.0f}%", popup=popup,
        ).add_to(camada_cidades)
    camada_cidades.add_to(m)

    if ponto:
        lat, lon, p, mm = ponto
        folium.Marker(
            [lat, lon], icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
            popup=f"Ponto pesquisado<br>{p * 100:.0f}% de chance, ~{mm:.1f} mm",
        ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def grafico_serie(df):
    d = df.tail(365)
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("Chuva (mm/dia)", "Temperatura (°C)", "Umidade relativa (%)",
                        "Vento (m/s)", "Pressão (kPa)"),
    )
    fig.add_bar(x=d.index, y=d["chuva"], row=1, col=1)
    fig.add_scatter(x=d.index, y=d["temp"], row=2, col=1, mode="lines")
    fig.add_scatter(x=d.index, y=d["umid"], row=3, col=1, mode="lines")
    fig.add_scatter(x=d.index, y=d["vento"], row=4, col=1, mode="lines")
    fig.add_scatter(x=d.index, y=d["pressao"], row=5, col=1, mode="lines")
    fig.update_layout(height=850, showlegend=False, margin=dict(t=40))
    return fig


def grafico_clima(df):
    mensal = df["chuva"].resample("MS").sum()
    clim = mensal.groupby(mensal.index.month).mean()
    fig = go.Figure(go.Bar(x=list(clim.index), y=clim.values))
    fig.update_layout(xaxis_title="mês", yaxis_title="mm/mês (média 2015 até hoje)", height=400)
    return fig


# ----------------------------------------------------------------- app
st.title("🌧️ Chuva no Rio Grande do Sul — mapa detalhado")
st.caption(
    "Dados diários da NASA POWER + machine learning. Contorno real do RS (IBGE), "
    "mapa de calor de risco e camadas de mapa (clique no ícone de camadas no canto do mapa)."
)

M = preparar()
contorno = baixar_contorno_rs()

st.sidebar.header("Pesquisa")
opcoes = ["(clicar no mapa)"] + list(CIDADES)
cidade_sel = st.sidebar.selectbox("Cidade", opcoes)

if "ponto" not in st.session_state:
    st.session_state.ponto = CIDADES["Porto Alegre"]
    st.session_state.ultimo_clique = None

if cidade_sel != "(clicar no mapa)":
    st.session_state.ponto = CIDADES[cidade_sel]

st.sidebar.info(
    "A NASA POWER não é em tempo real: os dados chegam com alguns dias de atraso. "
    "A previsão vale para o dia seguinte ao último dia com dados."
)
if contorno is None:
    st.sidebar.warning("Não consegui buscar o contorno oficial do RS (API do IBGE); o mapa segue sem ele.")

lat, lon = st.session_state.ponto
try:
    with st.spinner("Buscando dados do ponto na NASA..."):
        df = baixar_power(lat, lon)
except Exception as e:
    st.error(f"Não consegui buscar os dados da NASA para este ponto: {e}")
    st.stop()

f = criar_features(df)
ult = f.tail(1)
prob = float(M["clf"].predict_proba(ult[M["cols"]])[0, 1])
mm = float(max(M["reg"].predict(ult[M["cols"]])[0], 0))

col_mapa, col_info = st.columns([3, 2])

with col_mapa:
    mapa = st_folium(
        montar_mapa(M["ultimo"], contorno, (lat, lon, prob, mm)),
        height=560, use_container_width=True, returned_objects=["last_clicked"],
    )

with col_info:
    st.subheader(f"Ponto: {lat:.2f}, {lon:.2f}")
    st.metric("Chance de chuva no dia seguinte", f"{prob * 100:.0f}%")
    st.metric("Chuva estimada", f"{mm:.1f} mm")
    st.caption(f"Último dia com dados da NASA: {df.index.max():%d/%m/%Y}")
    st.caption("Clique em qualquer ponto do mapa (dentro do RS) para pesquisar esse local.")

clique = mapa.get("last_clicked") if mapa else None
if clique and clique != st.session_state.ultimo_clique:
    st.session_state.ultimo_clique = clique
    if dentro_do_rs(clique["lat"], clique["lng"]):
        st.session_state.ponto = (clique["lat"], clique["lng"])
        st.rerun()
    else:
        st.warning("Esse ponto está fora do Rio Grande do Sul.")

aba1, aba2, aba3 = st.tabs(["Últimos 12 meses", "Chuva média por mês", "Sobre o modelo"])
with aba1:
    st.plotly_chart(grafico_serie(df))
with aba2:
    st.plotly_chart(grafico_clima(df))
with aba3:
    mt = M["metricas"]
    st.write("Teste com dados de 2024 em diante (o modelo foi treinado só com dados até 2023):")
    st.table(pd.DataFrame(
        {
            "Modelo": [f"{mt['f1']:.3f}", f"{mt['mae']:.2f} mm"],
            "Ingênuo (repete o dia de hoje)": [f"{mt['f1_ing']:.3f}", f"{mt['mae_ing']:.2f} mm"],
        },
        index=["F1 (chove amanhã?)", "Erro médio (mm)"],
    ))
    st.write(f"ROC-AUC do modelo: {mt['auc']:.3f}")
    st.caption(
        "Os dados da NASA POWER têm resolução de cerca de 50 km e o modelo foi treinado com "
        f"{len(CIDADES)} cidades, então pontos muito longe delas tendem a ser menos confiáveis."
    )
