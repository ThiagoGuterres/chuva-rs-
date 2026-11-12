# 🌧️ Previsão de Chuva no Rio Grande do Sul

Projeto de machine learning que estima a **chance de chuva e o volume (mm) para o dia seguinte** em qualquer ponto do Rio Grande do Sul, usando dados meteorológicos da **NASA POWER**. Inclui um notebook para o Google Colab e um aplicativo web em Streamlit com mapa e gráficos.

## O que o projeto faz

- Baixa dados diários da API **NASA POWER** (2015 até o dia mais recente disponível) para 14 cidades de referência do RS.
- Usa as variáveis: precipitação, temperatura (média, máxima e mínima), umidade relativa, vento a 2 m e pressão de superfície.
- Treina dois modelos com `scikit-learn` (`HistGradientBoosting`):
  - **Classificação:** vai chover amanhã? (chuva ≥ 1 mm)
  - **Regressão:** quantos mm devem chover amanhã?
- Avalia o modelo com divisão temporal (treino até 2023, teste de 2024 em diante) e compara com um modelo ingênuo ("amanhã repete hoje").
- Mostra os resultados em mapa e gráficos interativos.

## Aplicativo web (Streamlit)

Duas versões do app:

| Arquivo | Descrição |
|---|---|
| `app.py` | Mapa estático (matplotlib), leve e sem dependência de tiles externos. |
| `app_detalhado.py` | Mapa interativo (folium) com contorno oficial do RS (IBGE), mapa de calor de risco, camadas, minimapa, tela cheia e **clique no mapa** para pesquisar qualquer ponto do estado. |

Ambos mostram: chance de chuva, volume estimado, gráficos dos últimos 12 meses (chuva, temperatura, umidade, vento e pressão), chuva média por mês e as métricas do modelo.

## Notebook (Google Colab)

`chuva_rs_nasa.ipynb` traz o passo a passo completo: download dos dados, criação das variáveis, treino, avaliação, gráficos e uma interface de pesquisa por cidade ou coordenada.

## Como rodar localmente

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>
pip install -r requirements.txt
streamlit run app_detalhado.py   # ou: streamlit run app.py
```

Na primeira execução o app baixa os dados da NASA e treina os modelos (alguns minutos). Depois, o resultado fica em cache.

## Publicação

O app pode ser publicado gratuitamente no [Streamlit Community Cloud](https://share.streamlit.io): conecte este repositório e aponte para `app_detalhado.py` (ou `app.py`).

## Estrutura

```
├── app.py                  # app Streamlit com mapa estático
├── app_detalhado.py        # app Streamlit com mapa interativo
├── chuva_rs_nasa.ipynb     # notebook do Colab
├── requirements.txt        # dependências
└── README.md
```

## Limitações

- Os dados da NASA POWER **não são em tempo real**: chegam com alguns dias de atraso. A previsão vale para o dia seguinte ao último dia disponível, não para "amanhã" de fato.
- A resolução dos dados é de cerca de 50 km, então não capta chuvas muito localizadas.
- O modelo é treinado com 14 cidades; pontos muito distantes delas tendem a ser menos confiáveis.
- Prever chuva diária é um problema difícil. Consulte as métricas na aba "Sobre o modelo" do app.

## Tecnologias

Python · pandas · NumPy · scikit-learn · Plotly · matplotlib · folium · Streamlit

## Fontes de dados

- [NASA POWER](https://power.larc.nasa.gov/): dados meteorológicos diários.
- [IBGE](https://servicodados.ibge.gov.br/api/docs/malhas): contorno do estado do Rio Grande do Sul.