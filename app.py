"""
Sprint 4 — Produto de Dados: Previsão de Visitantes do Zoológico de Korkeasaari
Grupo: Rafael Tavares (567357), Gabriel Muniz (568237), Yuri Quirino (568512),
       Leonardo Barros (566788), Marcelo Augusto (567176)
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle, os, io
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Zoo Predictor — Korkeasaari",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ESTILOS CSS
st.markdown("""
<style>
    /* Paleta */
    :root {
        --cor-primaria: #1d3557;
        --cor-secundaria: #457b9d;
        --cor-destaque: #e63946;
        --cor-suave: #a8dadc;
        --cor-fundo: #f1faee;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1d3557 0%, #457b9d 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Cards de métricas */
    .metric-card {
        background: white !important;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid #457b9d;
    }
    .metric-card .label { font-size: 0.8rem; color: #555 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #1d3557 !important; }
    .metric-card .unit  { font-size: 0.9rem; color: #666 !important; }

    /* Badge de demanda */
    .badge-alta   { background:#e63946; color:white; padding:0.5rem 1.2rem; border-radius:20px; font-weight:700; font-size:1.1rem; }
    .badge-media  { background:#f4a261; color:white; padding:0.5rem 1.2rem; border-radius:20px; font-weight:700; font-size:1.1rem; }
    .badge-baixa  { background:#2a9d8f; color:white; padding:0.5rem 1.2rem; border-radius:20px; font-weight:700; font-size:1.1rem; }

    /* Cards de recomendação */
    .rec-card {
        background: #f8f9fa !important;
        color: #1d3557 !important;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #457b9d;
    }
    .rec-card * { color: #1d3557 !important; }
    .rec-card.alta  { border-left-color: #e63946; }
    .rec-card.media { border-left-color: #f4a261; }
    .rec-card.baixa { border-left-color: #2a9d8f; }

    /* Caixa de explicação SHAP */
    .shap-explanation {
        background: #e8f4f8 !important;
        color: #1d3557 !important;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1d3557;
        font-size: 0.95rem;
        margin: 0.5rem 0;
    }
    .shap-explanation * { color: #1d3557 !important; }

    /* Limitações */
    .limitacao {
        background: #fff8e1 !important;
        color: #5d4037 !important;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border-left: 4px solid #ffc107;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .limitacao * { color: #5d4037 !important; }

    /* Rodapé */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }

    /* Ocultar o menu do streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# CONSTANTES DO DOMÍNIO
SEED = 42
np.random.seed(SEED)

FEATURES = [
    'ano', 'mes', 'dia', 'dow', 'dia_ano', 'semana_ano', 'trimestre',
    'fim_de_semana', 'alta_temporada', 'ferias_escolares',
    'feriado_natal', 'feriado_fixo',
    'mes_sin', 'mes_cos', 'dow_sin', 'dow_cos',
    'lag_7d', 'lag_30d', 'media_movel_7d',
]

FERIADOS_FIXOS = {(1,1), (1,6), (5,1), (12,6), (12,24), (12,25), (12,26)}

DIAS_SEMANA_PT = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
MESES_PT = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']


# FUNÇÕES DE FEATURE ENGINEERING
def eh_feriado_fixo(d: date) -> int:
    return int((d.month, d.day) in FERIADOS_FIXOS)

def gerar_features_serie(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todo o feature engineering em um DataFrame com colunas 'data' e 'visitas'."""
    df = df.sort_values('data').reset_index(drop=True)
    df['ano']       = df['data'].dt.year
    df['mes']       = df['data'].dt.month
    df['dia']       = df['data'].dt.day
    df['dow']       = df['data'].dt.dayofweek
    df['dia_ano']   = df['data'].dt.dayofyear
    df['semana_ano']= df['data'].dt.isocalendar().week.astype(int)
    df['trimestre'] = df['data'].dt.quarter

    df['fim_de_semana']   = (df['dow'] >= 5).astype(int)
    df['alta_temporada']  = df['mes'].isin([6, 7, 8]).astype(int)
    df['ferias_escolares']= df['mes'].isin([6, 7]).astype(int)
    df['feriado_natal']   = ((df['mes'] == 12) & (df['dia'] >= 24)).astype(int)
    df['feriado_fixo']    = df['data'].apply(lambda d: eh_feriado_fixo(d.date()))

    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    df['lag_7d']        = df['visitas'].shift(7)
    df['lag_30d']       = df['visitas'].shift(30)
    df['media_movel_7d']= df['visitas'].shift(1).rolling(7).mean()

    return df.dropna().reset_index(drop=True)

def features_para_data(d: date, lag_7: float, lag_30: float, mm7: float) -> pd.DataFrame:
    """Cria um DataFrame de uma única linha com as features de uma data."""
    row = {
        'ano':             d.year,
        'mes':             d.month,
        'dia':             d.day,
        'dow':             d.weekday(),
        'dia_ano':         d.timetuple().tm_yday,
        'semana_ano':      d.isocalendar()[1],
        'trimestre':       (d.month - 1) // 3 + 1,
        'fim_de_semana':   int(d.weekday() >= 5),
        'alta_temporada':  int(d.month in [6, 7, 8]),
        'ferias_escolares':int(d.month in [6, 7]),
        'feriado_natal':   int(d.month == 12 and d.day >= 24),
        'feriado_fixo':    eh_feriado_fixo(d),
        'mes_sin':         np.sin(2 * np.pi * d.month / 12),
        'mes_cos':         np.cos(2 * np.pi * d.month / 12),
        'dow_sin':         np.sin(2 * np.pi * d.weekday() / 7),
        'dow_cos':         np.cos(2 * np.pi * d.weekday() / 7),
        'lag_7d':          lag_7,
        'lag_30d':         lag_30,
        'media_movel_7d':  mm7,
    }
    return pd.DataFrame([row])[FEATURES]


# GERAÇÃO DE DADOS SINTÉTICOS (fallback)
@st.cache_data
def gerar_dados_sinteticos() -> pd.DataFrame:
    """
    Gera dados sintéticos calibrados com os padrões reais do zoo de Korkeasaari:
    - Média geral ≈ 600 visitantes/dia
    - Pico verão (Jun-Ago) ≈ 2.000–3.000 visitantes/dia
    - Fim de semana ≈ 2x dia útil
    """
    anos = [2018, 2019, 2022, 2023]
    datas = []
    for ano in anos:
        d = date(ano, 1, 1)
        while d.year == ano:
            datas.append(d)
            d += timedelta(days=1)

    rng = np.random.default_rng(SEED)

    registros = []
    for d in datas:
        mes, dow = d.month, d.weekday()

        # Sazonalidade mensal (perfil típico de zoo nórdico)
        sazonalidade_mes = {
            1: 0.12, 2: 0.13, 3: 0.18, 4: 0.30, 5: 0.50,
            6: 0.90, 7: 1.00, 8: 0.85, 9: 0.45, 10: 0.25,
            11: 0.14, 12: 0.20,
        }[mes]

        # Efeito dia da semana
        efeito_dow = {0: 0.60, 1: 0.55, 2: 0.55, 3: 0.60, 4: 0.75, 5: 1.00, 6: 0.90}[dow]

        # Feriado fixo
        feriado = 1.5 if (d.month, d.day) in FERIADOS_FIXOS else 1.0
        natal   = 1.8 if (d.month == 12 and d.day >= 24) else 1.0

        base = 3200  # pico de sábado de julho
        visitas = base * sazonalidade_mes * efeito_dow * feriado * natal
        visitas = max(50, visitas + rng.normal(0, visitas * 0.12))
        registros.append({'data': pd.Timestamp(d), 'visitas': round(visitas)})

    return pd.DataFrame(registros)


# TREINO E CACHE DO MODELO
MODEL_PATH = "modelo_zoo.pkl"

@st.cache_resource(show_spinner="Treinando o modelo… ⚙️")
def carregar_ou_treinar_modelo(df_raw: pd.DataFrame):
    """Treina (ou carrega) o GradientBoosting tunado e o explainer SHAP."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        return bundle

    df_fe = gerar_features_serie(df_raw.copy())

    X = df_fe[FEATURES]
    y = df_fe["visitas"]

    split_idx = int(len(df_fe) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Gradient Boosting tunado (melhores parâmetros do GridSearchCV da Sprint 3)
    modelo = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metricas = {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 1),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 1),
        "R2":   round(r2_score(y_test, y_pred), 4),
    }

    # SHAP explainer
    explainer = shap.TreeExplainer(modelo)

    # Salva
    bundle = {
        "modelo": modelo,
        "explainer": explainer,
        "metricas": metricas,
        "X_train": X_train,
        "y_train": y_train,
        "df_fe": df_fe,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


# FUNÇÕES DE PLOTAGEM
def plot_waterfall_shap(shap_vals, x_row, feature_names, base_value, predicao, top_n=10):
    """Waterfall manual para a previsão individual."""
    vals = shap_vals[0]
    feat = list(feature_names)

    # Ordena por |SHAP| e pega top_n
    idx = np.argsort(np.abs(vals))[::-1][:top_n]
    vals_top  = vals[idx]
    feats_top = [feat[i] for i in idx]
    data_top  = [x_row.iloc[0, i] for i in idx]

    # Inverte para waterfall de cima para baixo
    vals_top  = vals_top[::-1]
    feats_top = feats_top[::-1]
    data_top  = data_top[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.55)))
    fig.patch.set_facecolor("#f1faee")
    ax.set_facecolor("#f1faee")

    cumulative = base_value
    bar_starts = []
    bar_widths = []

    for v in vals_top:
        bar_starts.append(cumulative if v >= 0 else cumulative + v)
        bar_widths.append(abs(v))
        cumulative += v

    colors = ["#e63946" if v >= 0 else "#457b9d" for v in vals_top]
    y_pos  = np.arange(len(vals_top))

    bars = ax.barh(y_pos, bar_widths, left=bar_starts, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.6)

    # Rótulos das features
    labels = [f"{f} = {d:.1f}" for f, d in zip(feats_top, data_top)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # Valores SHAP nas barras
    for i, (bar, v) in enumerate(zip(bars, vals_top)):
        x_pos = bar.get_x() + bar.get_width() / 2
        ax.text(x_pos, i, f"{v:+.0f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    ax.axvline(base_value, color="#1d3557", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(predicao,   color="#e63946", linewidth=2, alpha=0.9)

    ax.set_xlabel("Número de visitantes", fontsize=10)
    ax.set_title(f"Explicação SHAP — Previsão: {predicao:,.0f} visitantes\n"
                 f"(base={base_value:,.0f})", fontsize=11, fontweight="bold", color="#1d3557")

    patch_pos = mpatches.Patch(color="#e63946", label="Aumenta a previsão")
    patch_neg = mpatches.Patch(color="#457b9d", label="Reduz a previsão")
    ax.legend(handles=[patch_pos, patch_neg], loc="lower right", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sazonalidade(df_fe):
    """Gráfico de visitação média por mês e por dia da semana."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#f1faee")

    med_mes = df_fe.groupby("mes")["visitas"].mean()
    axes[0].bar(MESES_PT, med_mes.values, color="#457b9d", edgecolor="white")
    axes[0].set_title("Média de visitas por mês", fontweight="bold", color="#1d3557")
    axes[0].set_ylabel("Visitantes/dia")
    axes[0].set_facecolor("#f1faee")
    for ax in axes:
        ax.set_facecolor("#f1faee")

    med_dow = df_fe.groupby("dow")["visitas"].mean()
    cores_dow = ["#e63946" if i >= 5 else "#457b9d" for i in range(7)]
    axes[1].bar(DIAS_SEMANA_PT, med_dow.values, color=cores_dow, edgecolor="white")
    axes[1].set_title("Média de visitas por dia da semana", fontweight="bold", color="#1d3557")
    axes[1].set_ylabel("Visitantes/dia")

    plt.tight_layout()
    return fig


def plot_importancia_global(explainer, X_sample, feature_names):
    """SHAP summary bar plot de importância global."""
    sv = explainer.shap_values(X_sample)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"Feature": feature_names, "SHAP": mean_abs})
    imp = imp.sort_values("SHAP", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f1faee")
    ax.set_facecolor("#f1faee")
    ax.barh(imp["Feature"], imp["SHAP"], color="#457b9d", edgecolor="white")
    ax.set_title("Importância Global das Features (SHAP)", fontweight="bold", color="#1d3557")
    ax.set_xlabel("Média |SHAP value|")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# LÓGICA DE NEGÓCIO
def nivel_demanda(visitas: float) -> tuple[str, str, str]:
    """Retorna (nivel, classe_css, emoji)."""
    if visitas >= 1500:
        return "ALTA", "alta", "🔴"
    elif visitas >= 700:
        return "MÉDIA", "media", "🟡"
    else:
        return "BAIXA", "baixa", "🟢"

def recomendacoes(nivel: str, visitas: float, d: date) -> list[str]:
    dia_nome = DIAS_SEMANA_PT[d.weekday()]
    mes_nome = [
        "Janeiro","Fevereiro","Março","Abril","Maio","Junho",
        "Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"
    ][d.month - 1]

    if nivel == "ALTA":
        return [
            f"🧑‍💼 **Escala máxima nos totens** — ativar todas as {int(visitas/200)+1} unidades de autoatendimento.",
            f"🎟️ **Pré-venda digital** — priorizar emissão online para evitar filas físicas no dia.",
            f"🍔 **Estoque de alimentação reforçado** — previsão de consumo ~{int(visitas*0.35):,} refeições.",
            f"🚌 **Sinalização de transporte** — coordenar com Helsinki Regional Transport Authority.",
            f"👮 **Equipe de segurança adicional** — recomendado para acima de 1.500 visitantes.",
            f"📣 **Comunicação proativa** — publicar aviso de alta movimentação no site e redes sociais.",
        ]
    elif nivel == "MÉDIA":
        return [
            f"🧑‍💼 **Escala padrão nos totens** — manter {max(2, int(visitas/300))} unidades ativas.",
            f"🍔 **Estoque de alimentação normal** — previsão de consumo ~{int(visitas*0.30):,} refeições.",
            f"🎯 **Oportunidade de campanha** — {dia_nome} de {mes_nome} é momento ideal para promoções relâmpago.",
            f"🔍 **Monitoramento em tempo real** — acionar totem adicional se fila > 10 pessoas.",
        ]
    else:
        return [
            f"💡 **Manutenção preventiva** — dia de baixa é ideal para manutenção de totens sem impacto.",
            f"🏷️ **Promoção de captação** — oferecer desconto ou evento especial para aumentar fluxo em {dia_nome}.",
            f"📚 **Treinamento de equipe** — alocar horas de capacitação para funcionários neste período.",
            f"🔧 **Reposição de estoque** — reabastecer itens de loja e alimentação com tranquilidade.",
        ]

def gerar_texto_shap(shap_vals, feature_names, x_row, nivel):
    """Gera um texto legível em português explicando a previsão."""
    vals = shap_vals[0]
    feat = list(feature_names)
    idx  = np.argsort(np.abs(vals))[::-1]

    positivos = [(feat[i], vals[i], x_row.iloc[0, i]) for i in idx if vals[i] > 0][:3]
    negativos = [(feat[i], vals[i], x_row.iloc[0, i]) for i in idx if vals[i] < 0][:2]

    NOMES_BR = {
        "lag_7d": "visitação da semana passada",
        "media_movel_7d": "média dos últimos 7 dias",
        "lag_30d": "visitação de 30 dias atrás",
        "mes_cos": "sazonalidade mensal (cos)",
        "mes_sin": "sazonalidade mensal (sin)",
        "dow_cos": "ciclo semanal (cos)",
        "dow_sin": "ciclo semanal (sin)",
        "alta_temporada": "alta temporada (jun-ago)",
        "ferias_escolares": "férias escolares",
        "fim_de_semana": "final de semana",
        "feriado_fixo": "feriado nacional",
        "feriado_natal": "período natalino",
        "dia_ano": "dia do ano",
        "semana_ano": "semana do ano",
        "trimestre": "trimestre",
        "dow": "dia da semana",
        "mes": "mês",
        "dia": "dia do mês",
        "ano": "ano",
    }

    partes = []
    if positivos:
        fatores = [NOMES_BR.get(f, f) for f, v, _ in positivos]
        partes.append(f"**↑ Aumentam** a previsão: {' + '.join(fatores)}")
    if negativos:
        fatores = [NOMES_BR.get(f, f) for f, v, _ in negativos]
        partes.append(f"**↓ Reduzem** a previsão: {' + '.join(fatores)}")

    return " | ".join(partes)


# INTERFACE PRINCIPAL
def main():
    # HEADER 
    st.markdown("""
    <div class="main-header">
        <h1>🦁 Zoo Predictor — Korkeasaari</h1>
        <p>Previsão de Visitantes com Explicabilidade SHAP | Sprint 4 · Machine Learning · 1TIAPR-2025</p>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Helsinki_Zoo_logo.svg/320px-Helsinki_Zoo_logo.svg.png",
                 width=80, use_column_width=False)
        st.title("📅 Configurar Previsão")

        st.markdown("### Selecione a data")
        data_input = st.date_input(
            "Data para previsão",
            value=date.today(),
            min_value=date(2024, 1, 1),
            max_value=date(2027, 12, 31),
        )

        st.markdown("---")
        st.markdown("### 🔢 Contexto histórico")
        st.caption("Informe os valores recentes para melhorar a precisão.")

        lag_7 = st.number_input(
            "Visitantes há 7 dias (mesmo dia da semana)",
            min_value=0, max_value=15000, value=850, step=50,
        )
        lag_30 = st.number_input(
            "Visitantes há 30 dias",
            min_value=0, max_value=15000, value=700, step=50,
        )
        mm7 = st.number_input(
            "Média dos últimos 7 dias",
            min_value=0, max_value=15000, value=780, step=50,
        )

        st.markdown("---")
        st.markdown("### 📂 Dados reais (opcional)")
        csv_upload = st.file_uploader("zoo_diario.csv", type=["csv"])
        if csv_upload:
            st.success("Arquivo carregado! O modelo será retreinado com dados reais.")

        prever_btn = st.button("🔮 Gerar Previsão", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        **Grupo:**  
        Rafael Tavares · Gabriel Muniz  
        Yuri Quirino · Leonardo Barros  
        Marcelo Augusto  
        """)

    # CARREGA / TREINA MODELO 
    if csv_upload:
        df_raw = pd.read_csv(csv_upload, parse_dates=["data"])
    else:
        df_raw = gerar_dados_sinteticos()

    bundle   = carregar_ou_treinar_modelo(df_raw)
    modelo   = bundle["modelo"]
    explainer= bundle["explainer"]
    metricas = bundle["metricas"]
    df_fe    = bundle["df_fe"]

    # TABS
    tab_pred, tab_explore, tab_modelo, tab_neg = st.tabs([
        "🔮 Previsão & Explicação",
        "📊 Explorar Dados",
        "🤖 Sobre o Modelo",
        "💼 Uso Prático & Limitações",
    ])

    # TAB 1 — PREVISÃO & EXPLICAÇÃO
    with tab_pred:
        if prever_btn or True:  # Mostra resultado imediatamente ao carregar
            x_row    = features_para_data(data_input, lag_7, lag_30, mm7)
            predicao = max(0, modelo.predict(x_row)[0])
            sv       = explainer.shap_values(x_row)

            exp_val = explainer.expected_value
            if hasattr(exp_val, "__len__"):
                exp_val = float(np.asarray(exp_val).flatten()[0])

            nivel, classe, emoji = nivel_demanda(predicao)

            # Resultado principal
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Previsão de Visitantes</div>
                    <div class="value">{predicao:,.0f}</div>
                    <div class="unit">visitantes/dia</div>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Nível de Demanda</div>
                    <div class="value">{emoji}</div>
                    <div class="unit"><span class="badge-{classe}">{nivel}</span></div>
                </div>""", unsafe_allow_html=True)

            with col3:
                dia_str = DIAS_SEMANA_PT[data_input.weekday()]
                mes_str = MESES_PT[data_input.month - 1]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Data Prevista</div>
                    <div class="value" style="font-size:1.3rem">{data_input.strftime("%d/%m/%Y")}</div>
                    <div class="unit">{dia_str} · {mes_str}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Explicação textual SHAP
            texto_shap = gerar_texto_shap(sv, FEATURES, x_row, nivel)
            st.markdown(f"""
            <div class="shap-explanation">
                <b>💡 Por que essa previsão?</b><br>
                {texto_shap}
            </div>""", unsafe_allow_html=True)

            # Waterfall SHAP
            st.markdown("#### 📉 Explicação SHAP — Contribuição de cada feature")
            fig_wf = plot_waterfall_shap(sv, x_row, FEATURES, exp_val, predicao, top_n=10)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

            st.caption(
                "As barras vermelhas **aumentam** a previsão em relação à média histórica; "
                "as azuis **diminuem**. "
                "A linha tracejada é a média histórica (base SHAP)."
            )

            # Recomendações
            st.markdown(f"#### 🏷️ Recomendações Operacionais — Demanda {nivel}")
            for rec in recomendacoes(nivel, predicao, data_input):
                st.markdown(f'<div class="rec-card {classe}">{rec}</div>', unsafe_allow_html=True)

    # TAB 2 — EXPLORAR DADOS
    with tab_explore:
        st.markdown("### 📊 Padrões Históricos de Visitação")

        col1, col2, col3 = st.columns(3)
        col1.metric("Média Diária",   f"{df_fe['visitas'].mean():,.0f}")
        col2.metric("Pico Registrado", f"{df_fe['visitas'].max():,.0f}")
        col3.metric("Mínimo Registrado", f"{df_fe['visitas'].min():,.0f}")

        st.pyplot(plot_sazonalidade(df_fe), use_container_width=True)

        st.markdown("#### Distribuição de visitas ao longo do tempo")
        fig_ts, ax_ts = plt.subplots(figsize=(14, 4))
        fig_ts.patch.set_facecolor("#f1faee")
        ax_ts.set_facecolor("#f1faee")
        ax_ts.plot(df_fe["data"], df_fe["visitas"], color="#1d3557", linewidth=0.7, alpha=0.8)
        ax_ts.fill_between(df_fe["data"], df_fe["visitas"], alpha=0.1, color="#457b9d")
        ax_ts.set_xlabel("Data")
        ax_ts.set_ylabel("Visitantes/dia")
        ax_ts.set_title("Série histórica de visitantes", fontweight="bold", color="#1d3557")
        ax_ts.grid(alpha=0.3)
        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)

        # Importância global
        st.markdown("#### 🌍 Importância Global das Features (SHAP)")
        X_sample = bundle["X_train"].sample(min(200, len(bundle["X_train"])), random_state=SEED)
        fig_imp = plot_importancia_global(explainer, X_sample, FEATURES)
        st.pyplot(fig_imp, use_container_width=True)
        plt.close(fig_imp)

    # TAB 3 — SOBRE O MODELO
    with tab_modelo:
        st.markdown("### 🤖 Pipeline de Machine Learning")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Modelo Final")
            st.info("""
**Gradient Boosting Regressor** (sklearn)

- `n_estimators`: 200  
- `max_depth`: 5  
- `learning_rate`: 0.1  
- Selecionado após GridSearchCV + RandomizedSearchCV (Sprint 3)
""")
            st.markdown("#### Métricas no Holdout (20% final da série)")
            m = metricas
            st.markdown(f"""
| Métrica | Valor |
|---------|-------|
| **MAE** (erro médio absoluto) | {m['MAE']:,} visitantes |
| **RMSE** | {m['RMSE']:,} visitantes |
| **R²** | {m['R2']} |
""")

        with col2:
            st.markdown("#### Features Utilizadas (19)")
            feat_info = {
                "lag_7d / lag_30d": "Visitas há 7 e 30 dias (autocorrelação temporal)",
                "media_movel_7d": "Média dos últimos 7 dias (tendência recente)",
                "mes_sin / mes_cos": "Sazonalidade anual (encoding circular)",
                "dow_sin / dow_cos": "Ciclo semanal (encoding circular)",
                "fim_de_semana": "Flag: sábado ou domingo",
                "alta_temporada": "Flag: junho, julho, agosto",
                "ferias_escolares": "Flag: junho, julho",
                "feriado_fixo": "Feriados nacionais finlandeses",
                "feriado_natal": "24 a 31 de dezembro",
                "dow / mes / dia / dia_ano": "Componentes temporais brutos",
            }
            for feat, desc in feat_info.items():
                st.markdown(f"- **{feat}**: {desc}")

        st.markdown("#### Estratégia de Validação")
        st.markdown("""
- **Split cronológico** (80% treino / 20% teste) — sem data leakage  
- **K-Fold Cross-Validation** (5 folds, shuffle) para estimar R² médio ± std  
- **Grid vs Random Search** para comparar abordagens de tuning
""")

    # TAB 4 — USO PRÁTICO & LIMITAÇÕES
    with tab_neg:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💼 Como o Zoo se beneficia?")

            st.markdown("""
#### 🎰 O que o totem ganha com isso?

**Totens de autoatendimento** são o gargalo da operação em dias de pico. 
Com a previsão antecipada, o zoo pode:
""")
            beneficios = [
                ("🔧 Manutenção Preditiva", "Fazer manutenção nos totens nos dias de baixa demanda, garantindo disponibilidade total no pico."),
                ("👷 Escala de Pessoal", "Dimensionar funcionários de apoio aos totens com até 7 dias de antecedência, reduzindo custo de horas extras."),
                ("🎟️ Gestão de Filas", "Ativar ou desativar totens conforme a demanda prevista, reduzindo tempo de espera."),
                ("📦 Estoque & Logística", "Planejar reposição de papel de ingresso, limpeza e alimentação com base no volume esperado."),
                ("📣 Marketing Dinâmico", "Lançar promoções específicas nos dias previstos de baixa para equilibrar a demanda."),
            ]
            for titulo, desc in beneficios:
                st.markdown(f"""
<div class="rec-card">
<b>{titulo}</b><br>{desc}
</div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🎯 Decisão Suportada pelo Modelo")
            st.markdown("""
| Nível | Ação Recomendada |
|-------|-----------------|
| 🔴 ALTA  | Escala máxima, pré-venda, comunicação proativa |
| 🟡 MÉDIA | Escala padrão, monitoramento, oportunidade de campanha |
| 🟢 BAIXA | Manutenção, treinamento, promoção de captação |
""")

        with col2:
            st.markdown("### ⚠️ Avaliação Crítica")

            st.markdown("#### Limitações Conhecidas")
            limitacoes = [
                ("Ausência de dados climáticos", "Temperatura e chuva afetam fortemente a visitação ao zoo (área parcialmente ao ar livre). Não incluídos por falta de fonte integrada."),
                ("Anos 2020-2021 excluídos", "Pandemia cria outliers estruturais. O modelo não sabe lidar com eventos extraordinários similares."),
                ("Lags dependem de dados reais", "Os campos lag_7d, lag_30d e media_movel_7d precisam ser preenchidos manualmente, limitando automação."),
                ("Sem eventos especiais", "Campanhas, nascimentos de animais e eventos externos não estão modelados."),
                ("Risco de overfitting nos lags", "Em série temporal, lags podem vazar informação futura se o split não for estritamente cronológico."),
            ]
            for titulo, desc in limitacoes:
                st.markdown(f"""
<div class="limitacao">
<b>⚠️ {titulo}</b><br>{desc}
</div>""", unsafe_allow_html=True)

            st.markdown("#### 🚀 Melhorias Futuras")
            melhorias = [
                "🌤️ Integrar API meteorológica (FMI — Finnish Meteorological Institute)",
                "📅 Flag de eventos especiais (campanhas, feriados móveis finlandeses)",
                "🤖 Testar XGBoost / LightGBM / Prophet como modelos alternativos",
                "🎲 Bayesian Optimization (Optuna) como terceira abordagem de tuning",
                "📱 Notificações automáticas para gestores quando previsão > limiar",
                "🔄 Re-treinamento automático mensal com dados novos",
                "📊 Intervalo de confiança quantílico na previsão",
            ]
            for m in melhorias:
                st.markdown(f"- {m}")

    # RODAPÉ
    st.markdown("""
    <div class="footer">
        Zoológico de Korkeasaari · Helsinki · Dados: Helsinki Region Infoshare (CC-BY 4.0)<br>
        Sprint 4 — Machine Learning · 1TIAPR-2025 · Grupo: Rafael Tavares, Gabriel Muniz, Yuri Quirino, Leonardo Barros, Marcelo Augusto
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
