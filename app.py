# ORBITAL LOGIS | GAIE - Global Asteroid Intelligence Engine
# Global Solution 2026 - FIAP
# Prevenção de Colisões Espaciais (Síndrome de Kessler) | ODS 9
# Dependências: ver requirements.txt
# Execução: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ORBITAL LOGIS | GAIE",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Fundo branco */
  .stApp {
    background-color: #ffffff;
    color: #111111;
  }

  [data-testid="stAppViewContainer"],
  [data-testid="stMain"] {
    background-color: #ffffff;
  }

  [data-testid="stHeader"] {
    background-color: #ffffff;
    border-bottom: 1px solid #e8e8e8;
  }

  /* Texto escuro apenas em containers de conteudo (nao em botoes) */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
  .stApp li {
    color: #111111;
  }

  .stMarkdown p,
  .stMarkdown span,
  .stMarkdown li,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] span {
    color: #111111 !important;
  }

  /* Caption */
  [data-testid="stCaptionContainer"] p {
    color: #888888 !important;
    font-size: 0.8rem !important;
  }

  /* Labels de inputs */
  label,
  .stTextInput label,
  .stNumberInput label,
  .stSlider label,
  .stSelectbox label,
  .stCheckbox label,
  .stRadio label {
    color: #333333 !important;
    font-weight: 500 !important;
  }

  /* Campos de texto */
  input, textarea {
    color: #111111 !important;
    background-color: #ffffff !important;
  }

  /* Botoes: forca texto branco diretamente no elemento button */
  button[kind="primary"],
  button[kind="primary"] *,
  button[kind="primary"]:hover,
  button[kind="primary"]:hover *,
  button[kind="primary"]:focus,
  button[kind="primary"]:focus * {
    color: #ffffff !important;
    background-color: #111111 !important;
  }

  /* Qualquer button dentro dos wrappers do Streamlit */
  .stButton button,
  .stFormSubmitButton button,
  .stButton button *,
  .stFormSubmitButton button * {
    color: #ffffff !important;
  }

  /* Expander — fundo claro e texto legivel */
  [data-testid="stExpander"] {
    background-color: #f7f7f7;
    border: 1px solid #e8e8e8;
    border-radius: 4px;
  }

  [data-testid="stExpander"] summary {
    background-color: #f7f7f7;
    color: #111111 !important;
  }

  /* Slider valores */
  [data-testid="stSlider"] span {
    color: #333333 !important;
  }

  /* st.info / st.warning / st.error / st.success */
  [data-testid="stAlert"] p,
  [data-testid="stAlert"] span,
  [data-testid="stNotification"] p,
  div[role="alert"] p {
    color: #111111 !important;
  }

  /* Expander header */
  details summary span,
  [data-testid="stExpander"] summary span {
    color: #111111 !important;
    font-weight: 500 !important;
  }

  /* Spinner */
  [data-testid="stSpinner"] p {
    color: #555555 !important;
  }

  /* Botao de form submit */
  .stFormSubmitButton > button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
  }

  .stFormSubmitButton > button:hover {
    background-color: #333333 !important;
  }

  /* Hero header — minimalista */
  .hero-header {
    border-bottom: 2px solid #111111;
    padding: 2.5rem 0 1.8rem 0;
    margin-bottom: 2.5rem;
  }

  .hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 0.6rem;
  }

  .hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #111111;
    letter-spacing: -1px;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
  }

  .hero-subtitle {
    font-size: 1rem;
    color: #555555;
    font-weight: 400;
    line-height: 1.6;
    max-width: 680px;
  }

  .ods-badge {
    display: inline-block;
    border: 1px solid #cccccc;
    color: #444444;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    margin-top: 1.2rem;
  }

  /* Section divider */
  .section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888888;
    border-top: 1px solid #e8e8e8;
    padding-top: 1.8rem;
    margin: 2rem 0 1.2rem 0;
  }

  /* Metric cards */
  .metric-card {
    background: #f7f7f7;
    border: 1px solid #e8e8e8;
    border-radius: 4px;
    padding: 1.4rem 1.2rem;
    text-align: left;
    transition: border-color 0.15s;
  }

  .metric-card:hover {
    border-color: #aaaaaa;
  }

  .metric-label {
    font-size: 0.7rem;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
    font-weight: 600;
  }

  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Roboto Mono', monospace;
    color: #111111;
    line-height: 1;
  }

  .metric-value.good { color: #1a7a4a; }
  .metric-value.mid  { color: #111111; }
  .metric-value.warn { color: #b85c00; }

  /* Info box */
  .info-box {
    background: #f7f7f7;
    border-left: 3px solid #111111;
    padding: 1rem 1.4rem;
    color: #444444;
    font-size: 0.9rem;
    line-height: 1.7;
    margin: 1rem 0;
  }

  /* Alert boxes */
  .alert-critical {
    background: #fff5f5;
    border: 1px solid #f0c0c0;
    border-left: 3px solid #cc2222;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    color: #cc2222;
    font-weight: 600;
    font-size: 1rem;
    text-align: center;
  }

  .alert-safe {
    background: #f4fbf7;
    border: 1px solid #b0dcc0;
    border-left: 3px solid #1a7a4a;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    color: #1a7a4a;
    font-weight: 600;
    font-size: 1rem;
    text-align: center;
  }

  /* Plot container */
  .plot-container {
    background: #f7f7f7;
    border: 1px solid #e8e8e8;
    border-radius: 4px;
    padding: 1.5rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #f7f7f7 !important;
    border-right: 1px solid #e8e8e8;
  }

  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #111111 !important;
    font-weight: 600;
  }

  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #333333 !important;
  }

  [data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #111111 !important;
  }

  /* Dataframe */
  .stDataFrame {
    border: 1px solid #e8e8e8;
    border-radius: 4px;
  }

  /* Buttons */
  .stButton > button[kind="primary"] {
    background-color: #111111;
    color: #ffffff;
    border: none;
    border-radius: 3px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: background-color 0.15s;
  }

  .stButton > button[kind="primary"]:hover {
    background-color: #333333;
  }

  /* Divider */
  hr { border-color: #e8e8e8; }

  /* Botão retrátil da sidebar */
  [data-testid="stIconMaterial"] {
    color: #111111 !important;
  }

  /* Table */
  table { width: 100% !important; }
  th { background: #f0f0f0 !important; color: #111111 !important; font-weight: 600 !important; }
  td { color: #333333 !important; }
</style>
""", unsafe_allow_html=True)

NASA_API_KEY = "DEMO_KEY"
NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"

FEATURES = [
    "estimated_diameter_min_km",
    "estimated_diameter_max_km",
    "relative_velocity_kmh",
    "miss_distance_km",
    "absolute_magnitude_h",
    "orbiting_body_encoded",
]

TARGET = "is_potentially_hazardous"

def fetch_neo_data(api_key: str, days_back: int = 60) -> pd.DataFrame:
    """
    Busca dados de asteroides (NEO) na NASA API.
    """
    records = []
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)
    current = start_date

    while current < end_date and len(records) < 1200:
        chunk_end = min(current + timedelta(days=7), end_date)
        params = {
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date":   chunk_end.strftime("%Y-%m-%d"),
            "api_key":    api_key,
        }
        try:
            resp = requests.get(NASA_NEO_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.warning(f"Erro na requisição ({current.date()} → {chunk_end.date()}): {e}")
            current = chunk_end
            continue

        for date_str, neos in data.get("near_earth_objects", {}).items():
            for neo in neos:
                try:
                    diam = neo["estimated_diameter"]["kilometers"]
                    approach = neo["close_approach_data"][0]
                    record = {
                        "name":                        neo.get("name", ""),
                        "estimated_diameter_min_km":   float(diam["estimated_diameter_min"]),
                        "estimated_diameter_max_km":   float(diam["estimated_diameter_max"]),
                        "relative_velocity_kmh":       float(approach["relative_velocity"]["kilometers_per_hour"]),
                        "miss_distance_km":            float(approach["miss_distance"]["kilometers"]),
                        "absolute_magnitude_h":        float(neo.get("absolute_magnitude_h", np.nan)),
                        "orbiting_body":               approach.get("orbiting_body", "Earth"),
                        "is_potentially_hazardous":    int(neo.get("is_potentially_hazardous_asteroid", False)),
                        "close_approach_date":         approach.get("close_approach_date", ""),
                    }
                    records.append(record)
                except (KeyError, IndexError, ValueError):
                    continue  
        current = chunk_end

    df = pd.DataFrame(records)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e prepara o dataset.
    """
    df["orbiting_body_encoded"] = df["orbiting_body"].astype("category").cat.codes

    df = df.dropna(subset=FEATURES + [TARGET])

    df = df.drop_duplicates(subset=["name"])

    cols = ["name", "close_approach_date"] + FEATURES + [TARGET]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].reset_index(drop=True)

def train_models(df: pd.DataFrame):
    """
    Treina tres classificadores com validação cruzada 5-fold.
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_cv    = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Logistic Regression (pipeline com scaler para CV correta)
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=42, solver="lbfgs",
    )
    lr.fit(X_train_scaled, y_train)
    lr_pred  = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pipe  = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"
    ))])
    lr_cv    = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred  = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_cv    = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "Random Forest": {
            "accuracy":    round(accuracy_score(y_test, rf_pred), 4),
            "f1":          round(f1_score(y_test, rf_pred, zero_division=0), 4),
            "auc_roc":     round(roc_auc_score(y_test, rf_proba), 4),
            "cv_auc_mean": round(rf_cv.mean(), 4),
            "cv_auc_std":  round(rf_cv.std(), 4),
        },
        "Logistic Regression": {
            "accuracy":    round(accuracy_score(y_test, lr_pred), 4),
            "f1":          round(f1_score(y_test, lr_pred, zero_division=0), 4),
            "auc_roc":     round(roc_auc_score(y_test, lr_proba), 4),
            "cv_auc_mean": round(lr_cv.mean(), 4),
            "cv_auc_std":  round(lr_cv.std(), 4),
        },
        "Gradient Boosting": {
            "accuracy":    round(accuracy_score(y_test, gb_pred), 4),
            "f1":          round(f1_score(y_test, gb_pred, zero_division=0), 4),
            "auc_roc":     round(roc_auc_score(y_test, gb_proba), 4),
            "cv_auc_mean": round(gb_cv.mean(), 4),
            "cv_auc_std":  round(gb_cv.std(), 4),
        },
    }

    splits = {
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "predictions": {
            "Random Forest":       (rf_pred, rf_proba),
            "Logistic Regression": (lr_pred, lr_proba),
            "Gradient Boosting":   (gb_pred, gb_proba),
        },
    }

    return rf, lr, gb, scaler, splits, metrics

def generate_shap_plot(model, X_train: np.ndarray, feature_names: list) -> plt.Figure:
    """
    Gera SHAP Summary Plot.
    """
    plt.close("all")

    explainer   = shap.TreeExplainer(model)
    sample_size = min(300, len(X_train))
    X_sample    = shap.sample(X_train, sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, 'values'):
        sv = shap_values.values[..., 1] if shap_values.values.ndim == 3 else shap_values.values
    else:
        sv = shap_values

    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 5),
        color_bar=True,
    )

    fig = plt.gcf()
    ax  = plt.gca()

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for text in fig.findobj(plt.Text):
        text.set_color("#111111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    ax.tick_params(colors="#555555")

    ax.set_title(
        "SHAP - Impacto das Features na Classificacao de Risco",
        color="#111111",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    return fig

def generate_visual_plots(splits: dict, metrics: dict) -> dict:
    """
    Gera curvas ROC comparativas e matrizes de confusão para todos os modelos.
    """
    plt.close("all")
    y_test      = splits["y_test"]
    predictions = splits["predictions"]
    model_names = list(predictions.keys())
    colors      = ["#111111", "#e07b00", "#1a7a4a"]

    # --- Curvas ROC ---
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    fig_roc.patch.set_facecolor("#ffffff")
    ax_roc.set_facecolor("#f7f7f7")

    for (model_name, (pred, proba)), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = metrics[model_name]["auc_roc"]
        ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})", color=color, lw=2)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Aleatório (AUC=0.500)")
    ax_roc.set_xlabel("Taxa de Falsos Positivos", color="#333333")
    ax_roc.set_ylabel("Taxa de Verdadeiros Positivos", color="#333333")
    ax_roc.set_title("Curvas ROC — Comparativo de Modelos", color="#111111", fontweight="bold", fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.tick_params(colors="#555555")
    for spine in ax_roc.spines.values():
        spine.set_edgecolor("#dddddd")
    plt.tight_layout()

    # --- Matrizes de Confusão ---
    fig_cm, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig_cm.patch.set_facecolor("#ffffff")

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        pred, _ = predictions[model_name]
        cm = confusion_matrix(y_test, pred)
        ax = axes[i]
        ax.set_facecolor("#ffffff")
        ax.imshow(cm, interpolation="nearest", cmap="Greys")
        ax.set_title(model_name, color="#111111", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predito", color="#333333", fontsize=9)
        ax.set_ylabel("Real", color="#333333", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Baixo Risco", "Risco Crítico"], color="#333333", fontsize=8)
        ax.set_yticklabels(["Baixo Risco", "Risco Crítico"], color="#333333", fontsize=8)
        ax.tick_params(colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#dddddd")
        thresh = cm.max() / 2.0
        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[row, col] > thresh else "#111111",
                )

    fig_cm.suptitle("Matrizes de Confusão — Conjunto de Teste", color="#111111", fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()

    return {"roc": fig_roc, "cm": fig_cm}


def predict_single(
    rf_model,
    scaler,
    diameter_min: float,
    diameter_max: float,
    velocity_kmh: float,
    miss_distance_km: float,
    magnitude: float,
    orbiting_encoded: int,
) -> tuple[int, float]:
    """
    Predicao single (RF).
    """
    X = np.array([[
        diameter_min,
        diameter_max,
        velocity_kmh,
        miss_distance_km,
        magnitude,
        orbiting_encoded,
    ]])
    pred  = rf_model.predict(X)[0]
    proba = rf_model.predict_proba(X)[0][1]
    return int(pred), float(proba)

def main():

    st.markdown("""
    <div class="hero-header">
      <p class="hero-eyebrow">Global Solution 2026 &mdash; FIAP</p>
      <p class="hero-title">Orbital Logis &middot; GAIE</p>
      <p class="hero-subtitle">
        Global Asteroid Intelligence Engine &mdash; Prevenção da Síndrome de Kessler
        com Machine Learning e dados reais da NASA.
      </p>
      <span class="ods-badge">ODS 9 &middot; Indústria, Inovação e Infraestrutura</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>Síndrome de Kessler:</strong> A proliferação de detritos orbitais pode tornar certas
      órbitas inutilizáveis em cascata, ameaçando satélites de comunicação, GPS e monitoramento climático
      que sustentam a infraestrutura global (ODS 9). O GAIE classifica objetos próximos à Terra em
      <strong>Risco Crítico</strong> ou <strong>Risco Baixo</strong> usando dados reais da API NeoWs da NASA,
      com explicabilidade via SHAP para justificar cada decisão do modelo.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Configurações")
        api_key = st.text_input(
            "Chave API NASA",
            value=NASA_API_KEY,
            type="password",
            help="Obtenha sua chave gratuita em https://api.nasa.gov/",
        )
        days_back = st.slider(
            "Janela de coleta (dias passados)",
            min_value=14,
            max_value=60,
            value=60,
            step=7,
            help="Cada semana retorna ~50 a 100 NEOs. 60 dias equivale a aproximadamente 1.000 registros.",
        )
        run_button = st.button("Executar Análise", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("## Predição em Tempo Real")
        st.caption("Insira os parâmetros do objeto e veja o risco estimado pelo modelo.")

        with st.form("prediction_form"):
            vel  = st.number_input("Velocidade relativa (km/h)",    min_value=0.0, value=50_000.0,  step=1000.0,   format="%.0f")
            dist = st.number_input("Distância de aproximação (km)", min_value=0.0, value=500_000.0, step=10_000.0, format="%.0f")
            dmin = st.number_input("Diâmetro mínimo estimado (km)", min_value=0.0, value=0.1,       step=0.01,     format="%.3f")
            dmax = st.number_input("Diâmetro máximo estimado (km)", min_value=0.0, value=0.3,       step=0.01,     format="%.3f")
            mag  = st.number_input("Magnitude absoluta (H)",        min_value=0.0, value=22.0,      step=0.5,      format="%.1f")
            submitted = st.form_submit_button("Prever Risco", use_container_width=True)
        st.markdown(
            "<style>button{color:#ffffff!important}button *{color:#ffffff!important}</style>",
            unsafe_allow_html=True,
        )

    if run_button:
        st.session_state.pop("df", None)
        st.session_state.pop("models_ready", None)
        st.session_state.pop("shap_fig", None)
        st.session_state.pop("visual_figs", None)
        with st.spinner("Buscando dados de asteroides na API NeoWs da NASA..."):
            raw_df = fetch_neo_data(api_key=api_key, days_back=days_back)
        if raw_df.empty:
            st.error("Nenhum dado retornado pela API. Verifique sua chave ou conexão.")
            st.stop()
        clean_df = preprocess_data(raw_df)
        st.session_state["df"] = clean_df

    if "df" not in st.session_state:
        st.info("⚙️ Configure sua chave da API NASA na barra lateral e clique em **Executar Análise** para iniciar.")
        st.stop()

    df = st.session_state["df"]

    st.markdown('<p class="section-header">Dataset &mdash; NASA Near Earth Objects</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    hazardous_count = int(df[TARGET].sum())
    safe_count      = len(df) - hazardous_count

    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Total de NEOs</div>
          <div class="metric-value mid">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Features</div>
          <div class="metric-value mid">{len(FEATURES)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Risco Cr&iacute;tico</div>
          <div class="metric-value warn">{hazardous_count:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Risco Baixo</div>
          <div class="metric-value good">{safe_count:,}</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Visualizar amostra do dataset", expanded=False):
        st.dataframe(
            df.head(30).style.format({
                "estimated_diameter_min_km": "{:.4f}",
                "estimated_diameter_max_km": "{:.4f}",
                "relative_velocity_kmh":     "{:,.0f}",
                "miss_distance_km":          "{:,.0f}",
                "absolute_magnitude_h":      "{:.2f}",
            }),
            use_container_width=True,
        )

    if "models_ready" not in st.session_state:
        if len(df) < 50:
            st.error(f"Dataset insuficiente ({len(df)} linhas). Aumente a janela de coleta ou verifique a API.")
            st.stop()

        with st.spinner("Treinando 3 modelos com validação cruzada 5-fold..."):
            rf, lr, gb, scaler, splits, metrics = train_models(df)

        st.session_state.update({
            "models_ready": True,
            "rf":           rf,
            "lr":           lr,
            "gb":           gb,
            "scaler":       scaler,
            "splits":       splits,
            "metrics":      metrics,
        })

    rf      = st.session_state["rf"]
    lr      = st.session_state["lr"]
    gb      = st.session_state["gb"]
    scaler  = st.session_state["scaler"]
    splits  = st.session_state["splits"]
    metrics = st.session_state["metrics"]

    st.markdown('<p class="section-header">Comparativo de Modelos</p>', unsafe_allow_html=True)

    col_rf, col_lr, col_gb = st.columns(3)

    for col, model_name in zip([col_rf, col_lr, col_gb], ["Random Forest", "Logistic Regression", "Gradient Boosting"]):
        m = metrics[model_name]
        color_acc = "good" if m["accuracy"] >= 0.85 else "mid"
        color_f1  = "good" if m["f1"]       >= 0.75 else "warn"
        color_auc = "good" if m["auc_roc"]  >= 0.85 else "mid"

        with col:
            st.markdown(f"##### {model_name}")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Acurácia</div>
                  <div class="metric-value {color_acc}">{m['accuracy']:.2%}</div>
                </div>""", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">F1-Score</div>
                  <div class="metric-value {color_f1}">{m['f1']:.2%}</div>
                </div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">AUC-ROC</div>
                  <div class="metric-value {color_auc}">{m['auc_roc']:.2%}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("")
    metrics_rows = []
    for model_name, m in metrics.items():
        metrics_rows.append({
            "Modelo": model_name,
            "Acurácia": m["accuracy"],
            "F1-Score": m["f1"],
            "AUC-ROC (teste)": m["auc_roc"],
            "AUC-ROC CV (média)": m["cv_auc_mean"],
            "CV ± std": m["cv_auc_std"],
        })
    metrics_df = pd.DataFrame(metrics_rows)

    best_model_name = metrics_df.loc[metrics_df["AUC-ROC (teste)"].idxmax(), "Modelo"]

    def highlight_best(row):
        return ["background-color: rgba(0,212,160,0.12); font-weight:700;" if row["Modelo"] == best_model_name else "" for _ in row]

    st.dataframe(
        metrics_df.style
            .apply(highlight_best, axis=1)
            .format({
                "Acurácia": "{:.2%}",
                "F1-Score": "{:.2%}",
                "AUC-ROC (teste)": "{:.2%}",
                "AUC-ROC CV (média)": "{:.2%}",
                "CV ± std": "{:.4f}",
            }),
        use_container_width=True,
        hide_index=True,
    )
    st.info(f"**Melhor modelo:** {best_model_name} — AUC-ROC teste: {metrics[best_model_name]['auc_roc']:.2%} | CV 5-fold: {metrics[best_model_name]['cv_auc_mean']:.2%} ± {metrics[best_model_name]['cv_auc_std']:.4f}")

    st.markdown('<p class="section-header">Avaliação Visual dos Modelos</p>', unsafe_allow_html=True)

    if "visual_figs" not in st.session_state:
        with st.spinner("Gerando curvas ROC e matrizes de confusão..."):
            visual_figs = generate_visual_plots(splits, metrics)
            st.session_state["visual_figs"] = visual_figs

    visual_figs = st.session_state["visual_figs"]

    tab_roc, tab_cm = st.tabs(["Curvas ROC", "Matrizes de Confusão"])

    with tab_roc:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(visual_figs["roc"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(
            "Cada curva mostra o trade-off entre Verdadeiros Positivos e Falsos Positivos. "
            "Quanto mais próxima do canto superior esquerdo, melhor o modelo."
        )

    with tab_cm:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(visual_figs["cm"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(
            "Diagonal principal = acertos. Fora da diagonal = erros. "
            "Falsos Negativos (linha 1, col 0) são críticos: objeto perigoso classificado como seguro."
        )

    st.markdown('<p class="section-header">SHAP &mdash; Explicabilidade do Melhor Modelo</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      O <strong>SHAP (SHapley Additive exPlanations)</strong> decompõe cada predição do modelo,
      mostrando o quanto cada feature contribuiu para classificar um objeto como <em>Risco Crítico</em>.
      Em sistemas de missão crítica como o GAIE, a explicabilidade não é opcional &mdash; é um requisito de segurança.
    </div>
    """, unsafe_allow_html=True)

    if "shap_fig" not in st.session_state:
        with st.spinner("Calculando valores SHAP..."):
            shap_fig = generate_shap_plot(rf, splits["X_train"], FEATURES)
            st.session_state["shap_fig"] = shap_fig

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.pyplot(st.session_state["shap_fig"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Azul = valor baixo da feature   |   Vermelho = valor alto da feature   |   "
        "Eixo X = impacto na predicao de Risco Critico"
    )

    st.markdown('<p class="section-header">Predicao em Tempo Real</p>', unsafe_allow_html=True)

    if dmax < dmin:
        st.warning("Diametro maximo deve ser maior ou igual ao diametro minimo.")
    else:
        pred_class, pred_proba = predict_single(
            rf_model=rf,
            scaler=scaler,
            diameter_min=dmin,
            diameter_max=dmax,
            velocity_kmh=vel,
            miss_distance_km=dist,
            magnitude=mag,
            orbiting_encoded=0,
        )

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if pred_class == 1:
                st.markdown(f"""
                <div class="alert-critical">
                  RISCO CR&Iacute;TICO<br/>
                  <span style="font-size:2.5rem; font-family:'Roboto Mono',monospace;">{pred_proba:.1%}</span><br/>
                  <span style="font-size:0.85rem; font-weight:400;">probabilidade de amea&ccedil;a</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-safe">
                  RISCO BAIXO<br/>
                  <span style="font-size:2.5rem; font-family:'Roboto Mono',monospace;">{1-pred_proba:.1%}</span><br/>
                  <span style="font-size:0.85rem; font-weight:400;">probabilidade de seguran&ccedil;a</span>
                </div>""", unsafe_allow_html=True)

        with col_res2:
            st.markdown("**Parametros utilizados:**")
            params_df = pd.DataFrame({
                "Feature": [
                    "Velocidade relativa",
                    "Distancia de aproximacao",
                    "Diametro minimo",
                    "Diametro maximo",
                    "Magnitude absoluta",
                ],
                "Valor": [
                    f"{vel:,.0f} km/h",
                    f"{dist:,.0f} km",
                    f"{dmin:.3f} km",
                    f"{dmax:.3f} km",
                    f"{mag:.1f}",
                ],
            })
            st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#aaaaaa; font-size:0.78rem; margin-top:1rem; letter-spacing:0.3px;">
      <strong style="color:#888888;">ORBITAL LOGIS &middot; GAIE</strong> &mdash; Global Solution 2026 &middot; FIAP<br/>
      Dados: <a href="https://api.nasa.gov/" style="color:#666666; text-decoration:underline;">NASA NeoWs API</a>
      &nbsp;&middot;&nbsp; Modelos: Random Forest &amp; Logistic Regression &amp; Gradient Boosting
      &nbsp;&middot;&nbsp; CV 5-fold &nbsp;&middot;&nbsp; Explicabilidade: SHAP &nbsp;&middot;&nbsp; ODS 9
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
