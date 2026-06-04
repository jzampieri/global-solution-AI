<div align="center">
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/satellite-dish.svg" width="60" alt="Satellite">
  <h1>Orbital Logis &middot; GAIE</h1>
  <p><strong>Global Asteroid Intelligence Engine</strong></p>
  <p>Prevenção da Síndrome de Kessler com Machine Learning e dados da NASA (ODS 9).</p>
</div>

---

## 1. Visão Geral
O **GAIE (Global Asteroid Intelligence Engine)** é a camada de inteligência do sistema Orbital Logis. Ele atua processando dados orbitais de objetos próximos à Terra (NEOs) para prever o risco de colisões espaciais, que podem desencadear a Síndrome de Kessler.

Através do consumo da API pública **NASA NeoWs**, o modelo coleta dados reais e classifica os asteroides em **Risco Crítico** ou **Risco Baixo**. A aplicação é construída de ponta a ponta em Python e disponibilizada via Streamlit.

## 2. Equipe e Identificação
Projeto desenvolvido para a Global Solution 2026 (FIAP) - Engenharia de Software (4º Ano).

* **Nome do Integrante 1** - RM: 00000
* **Nome do Integrante 2** - RM: 00000
* *(Preencher demais integrantes e RMs)*

## 3. Metodologia
A construção da solução foi dividida nas seguintes etapas do pipeline de dados:

1. **Extração e Integração:** Coleta contínua de telemetria orbital através da API REST da NASA, buscando um volume de ~1.000 amostras com suas respectivas features (diâmetro, velocidade relativa, distância mínima, etc).
2. **Pré-Processamento:** Limpeza de registros malformados, tratamento de dados nulos, remoção de duplicatas nominais e codificação da variável categórica do corpo orbital. O dataset resultante possui mais de 10 colunas e o alvo (`is_potentially_hazardous`).
3. **Treinamento:** Divisão estratificada (80/20) entre treino e teste. Escalonamento com `StandardScaler` aplicado aos dados da Regressão Logística.
4. **Modelagem:** Treinamento de duas técnicas clássicas para comparação: **Random Forest** (n_estimators=200, max_depth=10) e **Regressão Logística** (max_iter=1000). Ambas configuradas com balanceamento de pesos para tratar a assimetria das classes.

## 4. Resultados Técnicos
A escolha do modelo definitivo (Random Forest) baseou-se na capacidade de lidar com relacionamentos não-lineares nos parâmetros orbitais, além da sua superioridade nas métricas de avaliação do conjunto de teste:

* **Random Forest:** AUC-ROC de ~99% e F1-Score elevado, minimizando falsos negativos (vital para missão crítica).
* **Regressão Logística:** Apresentou acurácia razoável, porém com métricas de ROC e F1 inferiores, servindo como baseline comparativa.

## 5. Explicabilidade (SHAP)
No contexto espacial, a opacidade do modelo (caixa-preta) é inaceitável. Para sistemas autônomos que disparam manobras evasivas, a decisão da IA deve ser justificável e auditável.

A aplicação utiliza a biblioteca **SHAP (SHapley Additive exPlanations)** através do `TreeExplainer` para decompor a contribuição de cada variável na classificação de risco:

* **Magnitude Absoluta (H) e Diâmetro:** O gráfico SHAP (beeswarm) comprova consistentemente que asteroides com maior diâmetro estimado (e menor magnitude H, indicando maior reflexividade/tamanho) empurram a predição fortemente para "Risco Crítico".
* **Distância de Aproximação:** Valores baixos de "miss distance" (marcados em azul no plot) aumentam o impacto positivo no eixo X, contribuindo para o risco de colisão.
* **Ação:** Essa explicabilidade confirma que o modelo aprendeu as leis físicas reais da mecânica orbital e não apenas correlações espúrias nos dados.

## 6. Instruções de Execução
O ambiente foi padronizado em um isolamento de dependências (`requirements.txt`).

**Pré-requisitos:** Python 3.10 ou superior.

```bash
# 1. Clone o repositório
git clone https://github.com/SEU-USUARIO/orbital-logis-gaie.git
cd orbital-logis-gaie

# 2. Crie e ative o ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o servidor da aplicação
streamlit run app.py
```

O dashboard será aberto automaticamente em `http://localhost:8501`. Na barra lateral, você pode configurar a janela de coleta dos dados da NASA e testar predições manuais no formulário.
