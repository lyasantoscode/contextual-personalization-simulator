import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# Configuração da página

st.set_page_config(
    page_title="Simulador Q-Learning Somático",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Estilo visual

st.markdown(
    """
    <style>
        :root {
            --bg-sidebar: #14213d;
            --bg-sidebar-2: #0f172a;
            --border: #d7dee8;
            --border-soft: #e8eef5;
            --text-main: #1f2933;
            --text-muted: #64748b;
            --text-sidebar: #f8fafc;
            --text-sidebar-muted: #b9c3d5;
            --accent: #1683d8;
            --accent-dark: #0f66aa;
            --accent-soft: #e7f2fb;
            --sidebar-accent: #42a5f5;
        }

        header[data-testid="stHeader"] {
            background: rgba(246, 246, 243, 0.96);
            height: 3.15rem;
            border-bottom: 1px solid #e1e1dc;
            z-index: 999999;
        }

        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"] {
            visibility: visible;
        }

        button[kind="header"] {
            z-index: 1000000;
        }

        #MainMenu, footer { visibility: hidden; }

        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(66, 165, 245, 0.12) 0, rgba(66, 165, 245, 0.00) 34%),
                radial-gradient(circle at 92% 12%, rgba(20, 33, 61, 0.08) 0, rgba(20, 33, 61, 0.00) 32%),
                linear-gradient(180deg, #f7fbff 0%, #f3f7fb 46%, #eef3f7 100%);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 3.75rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-sidebar) 0%, var(--bg-sidebar-2) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 4px 0 18px rgba(15, 23, 42, 0.20);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 3.75rem;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 1.6rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: var(--text-sidebar) !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #ffffff !important;
            border-radius: 10px !important;
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] div,
        section[data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #111827 !important;
            -webkit-text-fill-color: #111827 !important;
            opacity: 1 !important;
        }

        .sidebar-brand {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 15px;
            padding: 0.95rem 0.9rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.18);
        }

        .sidebar-brand-top {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 0.65rem;
        }

        .sidebar-logo {
            width: 38px;
            height: 38px;
            border-radius: 11px;
            background: linear-gradient(135deg, #42a5f5 0%, #7dd3fc 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff !important;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.03em;
            box-shadow: 0 6px 15px rgba(66, 165, 245, 0.32);
        }

        .sidebar-title { font-size: 1rem; font-weight: 850; color: #ffffff !important; line-height: 1.18; }
        .sidebar-subtitle { font-size: 0.75rem; font-weight: 600; color: var(--text-sidebar-muted) !important; }
        .sidebar-description { color: var(--text-sidebar-muted) !important; font-size: 0.78rem; line-height: 1.5; margin-top: 0.2rem; }

        .side-section {
            margin-top: 1.05rem;
            margin-bottom: 0.55rem;
            padding-top: 0.8rem;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
        }

        .side-section-title {
            color: #ffffff !important;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .side-section-title::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: var(--sidebar-accent);
            display: inline-block;
        }

        .side-hint {
            background: rgba(66, 165, 245, 0.13);
            border: 1px solid rgba(66, 165, 245, 0.20);
            color: #d8ecff !important;
            border-radius: 11px;
            padding: 0.7rem 0.75rem;
            font-size: 0.76rem;
            line-height: 1.45;
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .sidebar-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.12);
            margin-top: 1.1rem;
            padding-top: 0.8rem;
            color: var(--text-sidebar-muted) !important;
            font-size: 0.72rem;
            line-height: 1.4;
        }

        h1, h2, h3, h4 { color: var(--text-main) !important; }
        p, div, label, span { color: var(--text-main); }

        .topbar {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--border);
            border-radius: 12px;
            min-height: 62px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
            backdrop-filter: blur(8px);
        }

        .brand-area { display: flex; align-items: center; gap: 12px; }
        .brand-mark {
            width: 36px;
            height: 36px;
            border-radius: 7px;
            background: linear-gradient(135deg, #1683d8 0%, #58a8df 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff !important;
            font-weight: 800;
            font-size: 13px;
        }
        .brand-title { font-size: 1.05rem; font-weight: 800; color: var(--text-main); white-space: nowrap; }
        .nav-area { display: flex; gap: 22px; font-size: 0.92rem; align-items: center; }
        .nav-area span { color: var(--text-muted); font-weight: 600; }
        .nav-active { color: var(--text-main) !important; font-weight: 800 !important; }

        .page-title { margin-top: 0; margin-bottom: 16px; }
        .page-title h1 { font-size: 1.72rem !important; margin-bottom: 0.35rem !important; font-weight: 850 !important; letter-spacing: -0.02em; }
        .page-title p { margin-top: 0; color: #526171 !important; font-size: 0.96rem; line-height: 1.55; max-width: 980px; }

        .metric-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 16px 14px 16px;
            min-height: 118px;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.07);
        }
        .metric-label { color: var(--text-muted); font-size: 0.80rem; font-weight: 750; margin-bottom: 10px; }
        .metric-value { color: var(--text-main); font-size: 1.78rem; font-weight: 850; line-height: 1.08; overflow-wrap: anywhere; }
        .metric-detail { color: var(--text-muted); font-size: 0.79rem; margin-top: 8px; line-height: 1.35; }
        .metric-blue { background: linear-gradient(180deg, #1689e3 0%, #2379ba 100%); border: 1px solid #156fb6; box-shadow: 0 2px 5px rgba(22, 131, 216, 0.22); }
        .metric-blue .metric-label, .metric-blue .metric-value, .metric-blue .metric-detail { color: #ffffff !important; }

        .info-strip {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid var(--border);
            border-radius: 10px !important;
            padding: 10px 14px;
            margin-top: 12px;
            margin-bottom: 12px;
            color: var(--text-muted);
            font-size: 0.88rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045) !important;
        }
        .info-strip b { color: var(--text-main); }

        .panel-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-soft); padding-bottom: 8px; margin-bottom: 10px; }
        .panel-title { font-size: 0.95rem; font-weight: 850; color: var(--text-main); }
        .panel-tag { background: var(--accent-soft); color: var(--accent-dark); border: 1px solid #c3dff4; border-radius: 999px; padding: 3px 9px; font-size: 0.72rem; font-weight: 800; }
        .small-muted { color: var(--text-muted); font-size: 0.84rem; line-height: 1.5; }

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
    padding: 0.64rem 1rem !important;
    box-shadow: 0 2px 5px rgba(22, 131, 216, 0.20);
}

.stButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
    color: #ffffff !important;
}

.stButton > button:hover p,
.stDownloadButton > button:hover p {
    color: #ffffff !important;
}

.stButton > button:focus,
.stDownloadButton > button:focus,
.stButton > button:active,
.stDownloadButton > button:active {
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
    color: #ffffff !important;
}

.stButton > button:focus p,
.stDownloadButton > button:focus p,
.stButton > button:active p,
.stDownloadButton > button:active p {
    color: #ffffff !important;
}

/* Corrige a caixa branca do upload CSV no menu lateral */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px dashed rgba(255, 255, 255, 0.25) !important;
    border-radius: 10px !important;
    padding: 0.55rem !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px dashed rgba(255, 255, 255, 0.25) !important;
    border-radius: 9px !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] div,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #e5edf7 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 7px !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
    color: #ffffff !important;
}

/* Corrige o card interno branco do arquivo já carregado */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px dashed rgba(255, 255, 255, 0.28) !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(15, 31, 55, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 8px !important;
    color: #e5edf7 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    color: #e5edf7 !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg {
    color: #e5edf7 !important;
    fill: #e5edf7 !important;
}

/* Reforço para versões do Streamlit que deixam o card como div interna */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div[data-testid*="File"] {
    background-color: rgba(15, 31, 55, 0.95) !important;
    color: #e5edf7 !important;
    border-radius: 8px !important;
}

        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255, 255, 255, 0.92) !important; border-color: var(--border) !important; border-radius: 12px !important; box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055) !important; }
        .footer { text-align: center; color: #64748b !important; font-size: 0.82rem; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)


# Tipos e configurações

State = Tuple[str, str, str, str, str]
Feature = str
Action = str

# Estado: feature, luminosidade, local, período e atividade.
features = ["brilho", "volume"]
luminosidades = ["escuro", "moderado", "claro"]
locais = ["residencia", "trabalho", "transporte", "area_publica"]
periodos = ["dia", "noite"]
atividades = ["parado", "deslocamento", "reuniao"]
acoes = ["baixo", "medio", "alto"]

estados: List[State] = [
    (feature, lum, loc, per, ati)
    for feature in features
    for lum in luminosidades
    for loc in locais
    for per in periodos
    for ati in atividades
]

acao_idx = {acao: i for i, acao in enumerate(acoes)}


# Normalização e CSV

def remover_acentos(valor: str) -> str:
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in texto if not unicodedata.combining(ch))


def normalizar_token(valor: Any) -> str:
    return remover_acentos(valor).replace(" ", "_").replace("-", "_")


MAPA_COLUNAS = {
    "step": "passo",
    "id": "id_origem",
    "timestamp": "timestamp",
    "datahora": "timestamp",
    "data_hora": "timestamp",
    "cenario": "cenario",
    "scenario": "cenario",
    "contexto": "cenario",
    "feature": "feature",
    "funcao": "feature",
    "configuracao": "feature",
    "tipo_ajuste": "feature",
    "variavel": "feature",
    "luminosity": "luminosidade",
    "luminosidade": "luminosidade",
    "luz": "lux",
    "lux": "lux",
    "luz_ambiente": "lux",
    "local_bright": "luminosidade",
    "location": "local",
    "local": "local",
    "localizacao": "local",
    "localizacao_contextual": "local",
    "period": "periodo",
    "periodo": "periodo",
    "day_shift": "periodo",
    "turno": "periodo",
    "weekday": "dia_semana",
    "week_day": "dia_semana",
    "dia_semana": "dia_semana",
    "preferencia": "preferencia_generica",
    "preferencia_usuario": "preferencia_generica",
    "target": "preferencia_generica",
    "user_preference": "preferencia_generica",
    "user_brightness_preference": "preferencia_brilho",
    "brightness_preference": "preferencia_brilho",
    "preferencia_brilho": "preferencia_brilho",
    "preferencia_brightness": "preferencia_brilho",
    "preferencia_volume": "preferencia_volume",
    "volume_preference": "preferencia_volume",
    "user_volume_preference": "preferencia_volume",
    "bright": "brilho_observado",
    "brightness": "brilho_observado",
    "brilho": "brilho_observado",
    "brilho_ajustado": "brilho_observado",
    "brilho_ajustado_pct": "brilho_pct",
    "acao": "acao_observada",
    "action": "acao_observada",
    "activity": "atividade",
    "atividade": "atividade",
    "user_activity": "atividade",
    "detected_activity": "atividade",
    "schedule": "agenda",
    "meeting": "agenda",
    "reuniao": "agenda",
    "battery_level": "bateria",
    "bateria": "bateria",
    "battery_charging": "carregando",
    "charging": "carregando",
    "screen": "tela",
    "screen_on": "tela",
    "headset": "fone",
    "bluetooth": "bluetooth",
    "ring_mode": "modo_toque",
    "volume": "volume_observado",
}


SINONIMOS_FEATURE = {
    "brilho": "brilho", "brightness": "brilho", "tela": "brilho", "screen_brightness": "brilho", "luz_tela": "brilho", "0": "brilho",
    "volume": "volume", "audio": "volume", "som": "volume", "media_volume": "volume", "1": "volume",
}

SINONIMOS_LUMINOSIDADE = {
    "escuro": "escuro", "dark": "escuro", "baixo_lux": "escuro", "baixa_luz": "escuro", "0": "escuro",
    "moderado": "moderado", "medio": "moderado", "media": "moderado", "moderada": "moderado", "medium": "moderado", "1": "moderado",
    "claro": "claro", "alta_luz": "claro", "bright": "claro", "alto_lux": "claro", "2": "claro",
}

SINONIMOS_LOCAL = {
    "residencia": "residencia", "residencial": "residencia", "casa": "residencia", "home": "residencia",
    "quarto": "residencia", "apartamento": "residencia", "domicilio": "residencia", "d4": "residencia", "0": "residencia",
    "trabalho": "trabalho", "work": "trabalho", "escritorio": "trabalho", "office": "trabalho", "empresa": "trabalho", "1": "trabalho",
    "transporte": "transporte", "vehicle": "transporte", "veiculo": "transporte", "on_vehicle": "transporte",
    "on_foot": "transporte", "deslocamento": "transporte", "caminhando": "transporte", "walking": "transporte", "2": "transporte",
    "area_publica": "area_publica", "public_area": "area_publica", "ambiente_externo": "area_publica",
    "externo": "area_publica", "outdoor": "area_publica", "rua": "area_publica", "shopping": "area_publica",
    "supermercado": "area_publica", "campus": "area_publica", "praca": "area_publica", "porto": "area_publica", "3": "area_publica",
}

SINONIMOS_PERIODO = {
    "dia": "dia", "day": "dia", "manha": "dia", "tarde": "dia", "morning": "dia", "afternoon": "dia", "0": "dia",
    "noite": "noite", "night": "noite", "madrugada": "noite", "evening": "noite", "dawn": "noite", "1": "noite",
}

SINONIMOS_ATIVIDADE = {
    "parado": "parado", "still": "parado", "stopped": "parado", "idle": "parado", "sentado": "parado",
    "unknown": "parado", "desconhecido": "parado", "3": "parado", "4": "parado",
    "deslocamento": "deslocamento", "walking": "deslocamento", "andando": "deslocamento", "caminhando": "deslocamento",
    "running": "deslocamento", "correndo": "deslocamento", "vehicle": "deslocamento", "veiculo": "deslocamento",
    "in_vehicle": "deslocamento", "on_vehicle": "deslocamento", "on_foot": "deslocamento",
    "bicicleta": "deslocamento", "bicycle": "deslocamento", "on_bicycle": "deslocamento",
    "transporte": "deslocamento", "0": "deslocamento", "1": "deslocamento", "2": "deslocamento", "7": "deslocamento", "8": "deslocamento",
    "reuniao": "reuniao", "meeting": "reuniao", "event": "reuniao", "evento": "reuniao", "agenda": "reuniao", "schedule": "reuniao",
}

SINONIMOS_ACAO = {
    "baixo": "baixo", "baixa": "baixo", "low": "baixo", "diminuir": "baixo", "reduzir": "baixo", "0": "baixo",
    "medio": "medio", "media": "medio", "médio": "medio", "medium": "medio", "manter": "medio", "1": "medio",
    "alto": "alto", "alta": "alto", "high": "alto", "aumentar": "alto", "2": "alto",
}

def categorizar_lux(valor: Any) -> Optional[str]:
    try:
        lux = float(str(valor).replace(",", "."))
    except Exception:
        return None
    if lux <= 20:
        return "escuro"
    if lux <= 300:
        return "moderado"
    return "claro"


def normalizar_valor(valor: Any, mapa: Dict[str, str]) -> Optional[str]:
    if pd.isna(valor):
        return None
    chave = normalizar_token(valor)
    return mapa.get(chave)


def normalizar_colunas_csv(df: pd.DataFrame) -> pd.DataFrame:
    renomear = {}
    usados = set()
    for coluna in df.columns:
        chave = normalizar_token(coluna)
        destino = MAPA_COLUNAS.get(chave, chave)
        if destino in usados:
            destino = f"{destino}_extra"
        usados.add(destino)
        renomear[coluna] = destino
    return df.rename(columns=renomear)


def extrair_colunas_extras(linha: pd.Series) -> Dict[str, Any]:
    extras = {}
    colunas = [
        "id_origem", "timestamp", "lux", "dia_semana", "bateria", "carregando",
        "tela", "fone", "bluetooth", "agenda", "brilho_observado",
        "brilho_pct", "acao_observada", "volume_observado", "modo_toque"
    ]
    for coluna in colunas:
        if coluna in linha.index and not pd.isna(linha[coluna]):
            extras[coluna] = linha[coluna]
    return extras


@dataclass
class Config:
    n_execucoes: int
    n_episodios: int
    episodios_treino_passivo: int
    passos_por_episodio: int
    alpha: float
    gamma: float
    epsilon_inicial: float
    epsilon_min: float
    epsilon_decay: float
    beta_somatico: float
    d_memoria: float
    recompensa_aceitacao: float
    recompensa_correcao: float
    recompensa_reversao: float
    sinal_aceitacao: float
    sinal_correcao: float
    sinal_reversao: float
    seed: int
    perfil: str
    janela_estabilizacao: int
    paciencia_estabilizacao: int
    limiar_estabilizacao: float


# Funções visuais

def metric_card(label: str, value: str, detail: str = "", blue: bool = False):
    css_class = "metric-card metric-blue" if blue else "metric-card"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def panel_header(title: str, tag: Optional[str] = None):
    tag_html = f'<span class="panel-tag">{tag}</span>' if tag else ""
    st.markdown(
        f"""
        <div class="panel-header">
            <div class="panel-title">{title}</div>
            {tag_html}
        </div>
        """,
        unsafe_allow_html=True
    )


# Topo da interface

st.markdown(
    """
    <div class="topbar">
        <div class="brand-area">
            <div class="brand-mark">QLS</div>
            <div class="brand-title">Ambiente Experimental</div>
        </div>
        <div class="nav-area">
            <span class="nav-active">Simulação</span>
            <span>Cenários</span>
            <span>Métricas</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="page-title">
        <h1>Simulador de Q-Learning Somático</h1>
        <p>
            Simulador experimental para comparação entre Q-Learning tradicional e Q-Learning Somático
            em cenários controlados de personalização contextual de brilho e volume em dispositivos móveis.
            O estado contextual combina feature, luminosidade, tipo de ambiente, período e atividade.
            O experimento considera treinamento passivo, teste ativo, resposta probabilística do usuário sintético,
            aceitação inferida, recompensa, estabilização e efeito da memória somática sobre a escolha de ações.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-top">
            <div class="sidebar-logo">QLS</div>
            <div>
                <div class="sidebar-title">Q-Learning Somático</div>
                <div class="sidebar-subtitle">Ambiente de simulação</div>
            </div>
        </div>
        <div class="sidebar-description">
            Configure o experimento, selecione o perfil sintético e controle os cenários simulados.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="side-section"><div class="side-section-title">Execução</div></div>', unsafe_allow_html=True)
n_execucoes = st.sidebar.slider("Número de execuções", 1, 50, 30)
n_episodios = st.sidebar.slider("Número de episódios", 50, 1000, 300, step=50)
episodios_treino_passivo = st.sidebar.slider(
    "Episódios de treinamento passivo",
    0,
    max(0, n_episodios - 10),
    min(50, max(0, n_episodios - 10)),
    step=10
)
passos_por_episodio = st.sidebar.slider("Passos por episódio", 10, 200, 20, step=10)

st.sidebar.markdown('<div class="side-section"><div class="side-section-title">Aprendizagem</div></div>', unsafe_allow_html=True)
alpha = st.sidebar.slider("Alpha", 0.01, 1.0, 0.25, step=0.01)
gamma = st.sidebar.slider("Gamma", 0.0, 1.0, 0.90, step=0.01)
epsilon_inicial = st.sidebar.slider("Epsilon inicial", 0.0, 1.0, 0.30, step=0.01)
epsilon_min = st.sidebar.slider("Epsilon mínimo", 0.0, 0.5, 0.03, step=0.01)
epsilon_decay = st.sidebar.slider("Decaimento do epsilon", 0.900, 1.000, 0.985, step=0.001)

st.sidebar.markdown('<div class="side-section"><div class="side-section-title">Camada somática</div></div>', unsafe_allow_html=True)
beta_somatico = st.sidebar.slider("Beta somático", 0.0, 1.0, 0.55, step=0.01)
d_memoria = st.sidebar.slider("Atualização da memória d", 0.0, 1.0, 0.35, step=0.01)

with st.sidebar.expander("Sinais e recompensas", expanded=False):
    recompensa_aceitacao = st.slider("Recompensa: aceitação", -1.0, 1.0, 1.0, step=0.1)
    recompensa_correcao = st.slider("Recompensa: correção moderada", -1.0, 1.0, -0.4, step=0.1)
    recompensa_reversao = st.slider("Recompensa: reversão forte", -1.0, 1.0, -1.0, step=0.1)
    sinal_aceitacao = st.slider("S: aceitação", -1.0, 1.0, 1.0, step=0.1)
    sinal_correcao = st.slider("S: correção moderada", -1.0, 1.0, -0.6, step=0.1)
    sinal_reversao = st.slider("S: reversão forte", -1.0, 1.0, -1.0, step=0.1)

st.sidebar.markdown('<div class="side-section"><div class="side-section-title">Cenários</div></div>', unsafe_allow_html=True)
seed = int(st.sidebar.number_input("ID experimental", value=42, step=1))
perfil = st.sidebar.selectbox(
    "Perfil sintético de usuário",
    [
        "Sensível à luminosidade",
        "Sensível ao período",
        "Sensível ao local",
        "Misto"
    ],
    index=3
)

modo_execucao = st.sidebar.radio(
    "Modo de execução",
    [
        "Gerar cenários automaticamente",
        "Carregar cenários por CSV"
    ]
)

arquivo_cenarios = None
if modo_execucao == "Carregar cenários por CSV":
    arquivo_cenarios = st.sidebar.file_uploader("Arquivo CSV de cenários", type=["csv"])

modelo_csv = """passo,cenario,feature,luminosidade,local,periodo,atividade,lux,preferencia_brilho,preferencia_volume
1,residencia_escuro_noite,brilho,escuro,residencia,noite,parado,8,baixo,
2,residencia_escuro_noite,brilho,escuro,residencia,noite,parado,12,baixo,
3,trabalho_moderado_dia,brilho,moderado,trabalho,dia,parado,180,medio,
4,transporte_claro_dia,brilho,claro,transporte,dia,deslocamento,1200,alto,
5,area_publica_claro_noite,brilho,claro,area_publica,noite,deslocamento,450,medio,
6,reuniao_trabalho_volume,volume,moderado,trabalho,dia,reuniao,220,,baixo
7,residencia_noite_volume,volume,moderado,residencia,noite,parado,95,,baixo
8,transporte_dia_volume,volume,moderado,transporte,dia,deslocamento,300,,medio
9,area_publica_volume,volume,claro,area_publica,dia,deslocamento,900,,baixo
10,trabalho_dia_volume,volume,moderado,trabalho,dia,parado,180,,baixo
"""

st.sidebar.download_button(
    label="Baixar modelo CSV",
    data=modelo_csv.encode("utf-8"),
    file_name="modelo_cenarios.csv",
    mime="text/csv"
)

st.sidebar.markdown('<div class="side-section"><div class="side-section-title">Métricas científicas</div></div>', unsafe_allow_html=True)
janela_estabilizacao = st.sidebar.slider("Janela da média móvel", 5, 50, 10, step=5)
paciencia_estabilizacao = st.sidebar.slider("Persistência para estabilizar", 5, 50, 20, step=5)
limiar_estabilizacao = st.sidebar.slider("Limiar de correção", 0.01, 0.30, 0.05, step=0.01)
executar_sensibilidade = st.sidebar.checkbox("Executar análise de beta", value=False)
betas_texto = st.sidebar.text_input("Betas para análise", value="0.25,0.40,0.55,0.70")

st.sidebar.markdown(
    """
    <div class="side-hint">
        Use 30 ou mais rodadas independentes para reduzir variação amostral. O modo CSV permite reproduzir uma sequência externa de cenários.
    </div>
    <div class="sidebar-footer">
        Simulador QLS<br>
        Simulação comparativa e análise experimental.
    </div>
    """,
    unsafe_allow_html=True
)

cfg = Config(
    n_execucoes=n_execucoes,
    n_episodios=n_episodios,
    episodios_treino_passivo=episodios_treino_passivo,
    passos_por_episodio=passos_por_episodio,
    alpha=alpha,
    gamma=gamma,
    epsilon_inicial=epsilon_inicial,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    beta_somatico=beta_somatico,
    d_memoria=d_memoria,
    recompensa_aceitacao=recompensa_aceitacao,
    recompensa_correcao=recompensa_correcao,
    recompensa_reversao=recompensa_reversao,
    sinal_aceitacao=sinal_aceitacao,
    sinal_correcao=sinal_correcao,
    sinal_reversao=sinal_reversao,
    seed=seed,
    perfil=perfil,
    janela_estabilizacao=janela_estabilizacao,
    paciencia_estabilizacao=paciencia_estabilizacao,
    limiar_estabilizacao=limiar_estabilizacao,
)

# Estado da sessão

if "resultados_simulacao" not in st.session_state:
    st.session_state.resultados_simulacao = None

if "config_assinatura" not in st.session_state:
    st.session_state.config_assinatura = None


def assinatura_configuracao(cfg_local: Config, modo_execucao_local: str) -> tuple:
    """Retorna uma assinatura da configuração atual."""
    return (
        cfg_local.n_execucoes,
        cfg_local.n_episodios,
        cfg_local.episodios_treino_passivo,
        cfg_local.passos_por_episodio,
        cfg_local.alpha,
        cfg_local.gamma,
        cfg_local.epsilon_inicial,
        cfg_local.epsilon_min,
        cfg_local.epsilon_decay,
        cfg_local.beta_somatico,
        cfg_local.d_memoria,
        cfg_local.recompensa_aceitacao,
        cfg_local.recompensa_correcao,
        cfg_local.recompensa_reversao,
        cfg_local.sinal_aceitacao,
        cfg_local.sinal_correcao,
        cfg_local.sinal_reversao,
        cfg_local.seed,
        cfg_local.perfil,
        cfg_local.janela_estabilizacao,
        cfg_local.paciencia_estabilizacao,
        cfg_local.limiar_estabilizacao,
        modo_execucao_local,
    )
# Resumo inicial do experimento

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    metric_card("Execuções", f"{cfg.n_execucoes}", "Rodadas independentes", blue=True)
with col_b:
    metric_card("Episódios", f"{cfg.n_episodios}", f"Treino passivo: {cfg.episodios_treino_passivo}", blue=True)
with col_c:
    metric_card("Passos", f"{cfg.passos_por_episodio}", "Interações por episódio", blue=True)
with col_d:
    modo_curto = "Automático" if modo_execucao == "Gerar cenários automaticamente" else "CSV"
    metric_card("Modo", modo_curto, "Geração de contextos", blue=True)

st.markdown(
    f"""
    <div class="info-strip">
        <b>Perfil:</b> {cfg.perfil} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>ID experimental:</b> {cfg.seed} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>α:</b> {cfg.alpha:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>γ:</b> {cfg.gamma:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>ε inicial:</b> {cfg.epsilon_inicial:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>β somático:</b> {cfg.beta_somatico:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>d:</b> {cfg.d_memoria:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Treino passivo:</b> {cfg.episodios_treino_passivo} ep. &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Features:</b> brilho e volume &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Estados:</b> {len(estados)}
    </div>
    """,
    unsafe_allow_html=True
)


# Modelo do usuário sintético

def carregar_e_validar_csv(arquivo) -> Optional[pd.DataFrame]:
    """Carrega e normaliza um arquivo CSV de cenários."""
    try:
        df = pd.read_csv(arquivo)
    except Exception as exc:
        st.error(f"Erro ao ler CSV: {exc}")
        return None

    if df.empty:
        st.error("Erro no CSV: o arquivo está vazio.")
        return None

    df = normalizar_colunas_csv(df)

    colunas_minimas = ["passo", "cenario", "local", "periodo"]
    for coluna in colunas_minimas:
        if coluna not in df.columns:
            st.error(f"Erro no CSV: coluna obrigatória ausente: {coluna}")
            return None

    if "feature" not in df.columns:
        df["feature"] = "brilho"

    if "atividade" not in df.columns:
        if "agenda" in df.columns:
            df["atividade"] = df["agenda"].apply(
                lambda v: "reuniao" if str(v).strip().lower() not in ["0", "nao", "não", "false", "none", "nan", ""]
                else "parado"
            )
        else:
            df["atividade"] = "parado"

    if "luminosidade" not in df.columns:
        if "lux" in df.columns:
            df["luminosidade"] = df["lux"].apply(categorizar_lux)
        else:
            # Em volume, a luminosidade não é fator principal, mas é mantida no estado para compatibilidade.
            df["luminosidade"] = "moderado"

    df["feature"] = df["feature"].apply(lambda v: normalizar_valor(v, SINONIMOS_FEATURE))
    df["luminosidade"] = df["luminosidade"].apply(lambda v: normalizar_valor(v, SINONIMOS_LUMINOSIDADE))
    df["local"] = df["local"].apply(lambda v: normalizar_valor(v, SINONIMOS_LOCAL))
    df["periodo"] = df["periodo"].apply(lambda v: normalizar_valor(v, SINONIMOS_PERIODO))
    df["atividade"] = df["atividade"].apply(lambda v: normalizar_valor(v, SINONIMOS_ATIVIDADE))

    preferencias = []
    for _, linha in df.iterrows():
        feature = linha.get("feature")
        valor = None

        if "preferencia_generica" in df.columns and not pd.isna(linha.get("preferencia_generica")):
            valor = linha.get("preferencia_generica")
        elif feature == "volume" and "preferencia_volume" in df.columns and not pd.isna(linha.get("preferencia_volume")):
            valor = linha.get("preferencia_volume")
        elif feature == "brilho" and "preferencia_brilho" in df.columns and not pd.isna(linha.get("preferencia_brilho")):
            valor = linha.get("preferencia_brilho")
        elif "preferencia_brilho" in df.columns and not pd.isna(linha.get("preferencia_brilho")):
            valor = linha.get("preferencia_brilho")

        if valor is None or pd.isna(valor):
            if feature in features:
                estado_tmp = (feature, linha["luminosidade"], linha["local"], linha["periodo"], linha["atividade"])
                preferencias.append(preferencia_usuario(estado_tmp, cfg.perfil))
            else:
                preferencias.append(None)
        else:
            preferencias.append(normalizar_valor(valor, SINONIMOS_ACAO))

    df["preferencia_usuario"] = preferencias

    erros = []
    for coluna in ["feature", "luminosidade", "local", "periodo", "atividade", "preferencia_usuario"]:
        invalidos = df[df[coluna].isna()].index.tolist()
        if invalidos:
            erros.append(f"{coluna}: linhas {[i + 2 for i in invalidos[:8]]}")

    if erros:
        st.error("Erro no CSV: valores não reconhecidos após normalização. " + " | ".join(erros))
        return None

    try:
        df["passo"] = pd.to_numeric(df["passo"], errors="raise").astype(int)
    except Exception:
        st.error("Erro no CSV: a coluna passo deve ser numérica.")
        return None

    df["cenario"] = df["cenario"].astype(str).str.strip().replace("", "cenario")
    df = df.sort_values("passo").reset_index(drop=True)
    return df


def preferencia_brilho(estado: State, perfil_usuario: str) -> Action:
    _, luminosidade, local, periodo, atividade = estado

    if atividade == "reuniao":
        if luminosidade == "escuro":
            return "baixo"
        return "medio"

    if perfil_usuario == "Sensível à luminosidade":
        if luminosidade == "escuro":
            return "baixo"
        if luminosidade == "moderado":
            return "medio"
        return "alto"

    if perfil_usuario == "Sensível ao período":
        if periodo == "noite":
            if luminosidade == "claro" and local in ["transporte", "area_publica"]:
                return "medio"
            return "baixo"
        if luminosidade == "claro":
            return "alto"
        if luminosidade == "moderado":
            return "medio"
        return "baixo"

    if perfil_usuario == "Sensível ao local":
        if local == "residencia":
            return "baixo" if periodo == "noite" else "medio"
        if local == "trabalho":
            return "medio"
        if local == "transporte":
            return "alto" if luminosidade == "claro" else "medio"
        if local == "area_publica":
            return "alto" if luminosidade == "claro" and periodo == "dia" else "medio"

    if luminosidade == "escuro":
        return "baixo"
    if local == "transporte" and luminosidade == "claro":
        return "alto"
    if local == "area_publica" and luminosidade == "claro" and periodo == "dia":
        return "alto"
    if local == "area_publica" and periodo == "noite":
        return "medio"
    if luminosidade == "claro" and periodo == "noite":
        return "medio"
    if luminosidade == "moderado" and periodo == "noite":
        return "baixo"
    if luminosidade == "claro" and local == "residencia" and periodo == "dia":
        return "alto"
    return "medio"


def preferencia_volume(estado: State, perfil_usuario: str) -> Action:
    """Define a preferência sintética de volume."""
    _, luminosidade, local, periodo, atividade = estado

    if atividade == "reuniao":
        return "baixo"

    if local == "trabalho":
        return "baixo" if periodo == "dia" else "medio"

    if local == "area_publica":
        return "baixo"

    if local == "transporte":
        return "medio"

    if local == "residencia" and periodo == "noite":
        return "baixo"

    if local == "residencia" and periodo == "dia":
        return "medio"

    return "medio"


def preferencia_usuario(estado: State, perfil_usuario: str) -> Action:
    feature = estado[0]
    if feature == "volume":
        return preferencia_volume(estado, perfil_usuario)
    return preferencia_brilho(estado, perfil_usuario)


def normalizar_probabilidades(p_aceitar: float, p_corrigir: float, p_reverter: float) -> Tuple[float, float, float]:
    """Normaliza uma distribuição de três probabilidades."""
    valores = np.array([p_aceitar, p_corrigir, p_reverter], dtype=float)
    valores = np.clip(valores, 0.0, None)
    soma = float(valores.sum())
    if soma <= 0:
        return (1.0, 0.0, 0.0)
    valores = valores / soma
    return float(valores[0]), float(valores[1]), float(valores[2])


def tornar_resposta_mais_severa(probabilidades: Tuple[float, float, float], intensidade: float) -> Tuple[float, float, float]:
    """Ajusta a distribuição para contextos mais sensíveis."""
    p_aceitar, p_corrigir, p_reverter = probabilidades
    intensidade = float(np.clip(intensidade, 0.0, 0.35))

    deslocamento = min(p_aceitar * intensidade, p_aceitar)
    p_aceitar -= deslocamento
    p_corrigir += deslocamento * 0.60
    p_reverter += deslocamento * 0.40

    return normalizar_probabilidades(p_aceitar, p_corrigir, p_reverter)


def tornar_resposta_mais_tolerante(probabilidades: Tuple[float, float, float], intensidade: float) -> Tuple[float, float, float]:
    """Ajusta a distribuição para contextos menos sensíveis."""
    p_aceitar, p_corrigir, p_reverter = probabilidades
    intensidade = float(np.clip(intensidade, 0.0, 0.25))

    deslocamento_reversao = min(p_reverter * intensidade, p_reverter)
    deslocamento_correcao = min(p_corrigir * intensidade * 0.50, p_corrigir)

    p_reverter -= deslocamento_reversao
    p_corrigir -= deslocamento_correcao
    p_aceitar += deslocamento_reversao + deslocamento_correcao

    return normalizar_probabilidades(p_aceitar, p_corrigir, p_reverter)


def distribuicao_resposta(
    feature: Feature,
    distancia: int,
    perfil_usuario: str,
    luminosidade: Optional[str] = None,
    local: Optional[str] = None,
    periodo: Optional[str] = None,
    atividade: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Calcula as probabilidades de aceitação, correção e reversão."""
    distancia = int(distancia)

    # Distribuição base por feature e distância do erro.
    if feature == "volume":
        tabela = {
            0: (0.93, 0.07, 0.00),
            1: (0.15, 0.75, 0.10),
            2: (0.01, 0.19, 0.80),
        }
    else:
        tabela = {
            0: (0.95, 0.05, 0.00),
            1: (0.20, 0.70, 0.10),
            2: (0.02, 0.28, 0.70),
        }

    probabilidades = tabela[distancia]

    # Mesmo ações adequadas podem ter pequenas correções.
    if distancia == 0:
        return probabilidades

    # Ajustes por perfil sintético.
    if perfil_usuario == "Sensível à luminosidade":
        if feature == "brilho":
            intensidade = 0.18
            if luminosidade in ["escuro", "claro"]:
                intensidade += 0.07
            probabilidades = tornar_resposta_mais_severa(probabilidades, intensidade)
        elif feature == "volume":
            probabilidades = tornar_resposta_mais_tolerante(probabilidades, 0.04)

    elif perfil_usuario == "Sensível ao período":
        if periodo == "noite":
            intensidade = 0.14
            if feature == "brilho":
                intensidade += 0.06
            probabilidades = tornar_resposta_mais_severa(probabilidades, intensidade)
        elif periodo == "dia" and feature == "brilho":
            probabilidades = tornar_resposta_mais_tolerante(probabilidades, 0.03)

    elif perfil_usuario == "Sensível ao local":
        if local in ["trabalho", "area_publica", "transporte"] or atividade == "reuniao":
            intensidade = 0.15
            if feature == "volume" and (atividade == "reuniao" or local in ["trabalho", "area_publica"]):
                intensidade += 0.08
            probabilidades = tornar_resposta_mais_severa(probabilidades, intensidade)
        elif local == "residencia":
            probabilidades = tornar_resposta_mais_tolerante(probabilidades, 0.05)

    else:  # Perfil Misto
        intensidade = 0.06
        if feature == "brilho" and luminosidade in ["escuro", "claro"]:
            intensidade += 0.04
        if feature == "volume" and (atividade == "reuniao" or local in ["trabalho", "area_publica"]):
            intensidade += 0.05
        if periodo == "noite" and feature == "brilho":
            intensidade += 0.03
        probabilidades = tornar_resposta_mais_severa(probabilidades, intensidade)

    return probabilidades

def montar_resposta(rotulo: str, distancia: int, cfg_local: Config, p_aceitar: float, p_corrigir: float, p_reverter: float) -> Dict[str, float]:
    if rotulo == "aceitar":
        return {
            "aceitou": 1,
            "corrigiu": 0,
            "reverteu": 0,
            "recompensa": cfg_local.recompensa_aceitacao,
            "sinal_somatico": cfg_local.sinal_aceitacao,
            "distancia": distancia,
            "resposta": "aceitacao",
            "p_aceitar": p_aceitar,
            "p_corrigir": p_corrigir,
            "p_reverter": p_reverter,
        }

    if rotulo == "corrigir":
        return {
            "aceitou": 0,
            "corrigiu": 1,
            "reverteu": 0,
            "recompensa": cfg_local.recompensa_correcao,
            "sinal_somatico": cfg_local.sinal_correcao,
            "distancia": distancia,
            "resposta": "correcao_moderada",
            "p_aceitar": p_aceitar,
            "p_corrigir": p_corrigir,
            "p_reverter": p_reverter,
        }

    return {
        "aceitou": 0,
        "corrigiu": 1,
        "reverteu": 1,
        "recompensa": cfg_local.recompensa_reversao,
        "sinal_somatico": cfg_local.sinal_reversao,
        "distancia": distancia,
        "resposta": "reversao_forte",
        "p_aceitar": p_aceitar,
        "p_corrigir": p_corrigir,
        "p_reverter": p_reverter,
    }


def resposta_positiva_observada(acao: Action, cfg_local: Config) -> Dict[str, float]:
    """Resposta usada durante o treinamento passivo."""
    return {
        "aceitou": 1,
        "corrigiu": 0,
        "reverteu": 0,
        "recompensa": cfg_local.recompensa_aceitacao,
        "sinal_somatico": cfg_local.sinal_aceitacao,
        "distancia": 0,
        "resposta": "observacao_passiva",
        "p_aceitar": 1.0,
        "p_corrigir": 0.0,
        "p_reverter": 0.0,
    }


def resposta_usuario_csv(estado: State, acao: Action, preferencia: Action, cfg_local: Config, probabilistica: bool = True) -> Dict[str, float]:
    feature, luminosidade, local, periodo, atividade = estado
    distancia = abs(acao_idx[acao] - acao_idx[preferencia])

    p_aceitar, p_corrigir, p_reverter = distribuicao_resposta(
        feature=feature,
        distancia=distancia,
        perfil_usuario=cfg_local.perfil,
        luminosidade=luminosidade,
        local=local,
        periodo=periodo,
        atividade=atividade,
    )

    if probabilistica:
        rotulo = random.choices(
            ["aceitar", "corrigir", "reverter"],
            weights=[p_aceitar, p_corrigir, p_reverter],
            k=1
        )[0]
    else:
        rotulo = "aceitar" if distancia == 0 else "corrigir" if distancia == 1 else "reverter"

    return montar_resposta(rotulo, distancia, cfg_local, p_aceitar, p_corrigir, p_reverter)


def resposta_usuario(estado: State, acao: Action, cfg_local: Config) -> Dict[str, float]:
    pref = preferencia_usuario(estado, cfg_local.perfil)
    return resposta_usuario_csv(estado, acao, pref, cfg_local, probabilistica=True)


# Agentes

class QLearning:
    def __init__(self, cfg_local: Config):
        self.cfg = cfg_local
        self.q: Dict[State, Dict[Action, float]] = {
            estado: {acao: 0.0 for acao in acoes}
            for estado in estados
        }
        self.epsilon = cfg_local.epsilon_inicial

    def escolher_acao(self, estado: State) -> Action:
        if random.random() < self.epsilon:
            return random.choice(acoes)
        valores = self.q[estado]
        return max(valores, key=valores.get)

    def atualizar(self, estado: State, acao: Action, recompensa: float, proximo_estado: State):
        q_atual = self.q[estado][acao]
        melhor_proximo = max(self.q[proximo_estado].values())
        novo_q = q_atual + self.cfg.alpha * (
            recompensa + self.cfg.gamma * melhor_proximo - q_atual
        )
        self.q[estado][acao] = novo_q

    def reduzir_exploracao(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


class QLearningSomatico(QLearning):
    def __init__(self, cfg_local: Config, beta_override: Optional[float] = None):
        super().__init__(cfg_local)
        self.beta = cfg_local.beta_somatico if beta_override is None else beta_override
        self.ms: Dict[State, Dict[Action, float]] = {
            estado: {acao: 0.0 for acao in acoes}
            for estado in estados
        }

    def normalizar_q(self, estado: State) -> Dict[Action, float]:
        valores = np.array([self.q[estado][acao] for acao in acoes], dtype=float)
        minimo = valores.min()
        maximo = valores.max()
        if np.isclose(maximo, minimo):
            return {acao: 0.0 for acao in acoes}
        normalizados = 2 * ((valores - minimo) / (maximo - minimo)) - 1
        return {acao: float(normalizados[i]) for i, acao in enumerate(acoes)}

    def escolher_acao(self, estado: State) -> Action:
        if random.random() < self.epsilon:
            return random.choice(acoes)
        q_normalizado = self.normalizar_q(estado)
        utilidades = {
            acao: (1 - self.beta) * q_normalizado[acao] + self.beta * self.ms[estado][acao]
            for acao in acoes
        }
        return max(utilidades, key=utilidades.get)

    def atualizar_memoria_somatica(self, estado: State, acao: Action, sinal_somatico: float):
        valor_antigo = self.ms[estado][acao]
        novo_valor = (1 - self.cfg.d_memoria) * valor_antigo + self.cfg.d_memoria * sinal_somatico
        self.ms[estado][acao] = float(np.clip(novo_valor, -1.0, 1.0))


# Simulação

def extrair_memoria_somatica(agente, execucao: int, beta_valor: Optional[float] = None) -> pd.DataFrame:
    memoria = []
    if isinstance(agente, QLearningSomatico):
        for estado, acoes_ms in agente.ms.items():
            for acao, valor in acoes_ms.items():
                memoria.append({
                    "execucao": execucao,
                    "beta": agente.beta if beta_valor is None else beta_valor,
                    "feature": estado[0],
                    "luminosidade": estado[1],
                    "local": estado[2],
                    "periodo": estado[3],
                    "atividade": estado[4],
                    "acao": acao,
                    "MS(s,a)": valor,
                })
    return pd.DataFrame(memoria)


def atualizar_agente_com_observacao(agente, estado: State, acao: Action, resposta: Dict[str, float], proximo_estado: State):
    agente.atualizar(estado, acao, resposta["recompensa"], proximo_estado)
    if isinstance(agente, QLearningSomatico):
        agente.atualizar_memoria_somatica(estado, acao, resposta["sinal_somatico"])


def rodar_simulacao(tipo_agente, nome_modelo: str, execucao: int, cfg_local: Config, beta_override: Optional[float] = None):
    if tipo_agente is QLearningSomatico:
        agente = tipo_agente(cfg_local, beta_override=beta_override)
    else:
        agente = tipo_agente(cfg_local)

    resultados = []
    resultados_contexto = []
    episodios_treino = min(cfg_local.episodios_treino_passivo, max(cfg_local.n_episodios - 1, 0))

    for episodio in range(cfg_local.n_episodios):
        fase = "treino_passivo" if episodio < episodios_treino else "teste_ativo"
        episodio_ativo = episodio - episodios_treino

        aceitos = correcoes = reversoes = 0
        recompensa_total = 0.0
        acoes_executadas = 0
        estado = random.choice(estados)

        for passo in range(cfg_local.passos_por_episodio):
            preferencia = preferencia_usuario(estado, cfg_local.perfil)

            if fase == "treino_passivo":
                acao = preferencia
                resposta = resposta_positiva_observada(acao, cfg_local)
            else:
                acao = agente.escolher_acao(estado)
                resposta = resposta_usuario_csv(estado, acao, preferencia, cfg_local, probabilistica=True)

            proximo_estado = random.choice(estados)
            atualizar_agente_com_observacao(agente, estado, acao, resposta, proximo_estado)

            if fase == "teste_ativo":
                aceitos += int(resposta["aceitou"])
                correcoes += int(resposta["corrigiu"])
                reversoes += int(resposta["reverteu"])
                recompensa_total += float(resposta["recompensa"])
                acoes_executadas += 1

                resultados_contexto.append({
                    "modelo": nome_modelo,
                    "execucao": execucao,
                    "episodio": episodio_ativo,
                    "episodio_global": episodio,
                    "fase": fase,
                    "perfil_usuario": cfg_local.perfil,
                    "passo": passo,
                    "cenario": "automatico",
                    "feature": estado[0],
                    "luminosidade": estado[1],
                    "local": estado[2],
                    "periodo": estado[3],
                    "atividade": estado[4],
                    "acao": acao,
                    "preferencia": preferencia,
                    "resposta": resposta["resposta"],
                    "aceitou": int(resposta["aceitou"]),
                    "corrigiu": int(resposta["corrigiu"]),
                    "reverteu": int(resposta["reverteu"]),
                    "recompensa": float(resposta["recompensa"]),
                    "sinal_somatico": float(resposta["sinal_somatico"]),
                    "distancia": int(resposta["distancia"]),
                    "p_aceitar": float(resposta["p_aceitar"]),
                    "p_corrigir": float(resposta["p_corrigir"]),
                    "p_reverter": float(resposta["p_reverter"]),
                })

            estado = proximo_estado

        agente.reduzir_exploracao()

        if fase == "teste_ativo":
            resultados.append({
                "modelo": nome_modelo,
                "execucao": execucao,
                "episodio": episodio_ativo,
                "episodio_global": episodio,
                "fase": fase,
                "correcoes": correcoes,
                "aceitos": aceitos,
                "reversoes": reversoes,
                "recompensa_total": recompensa_total,
                "recompensa_media": recompensa_total / max(acoes_executadas, 1),
                "taxa_aceitacao": aceitos / max(acoes_executadas, 1),
                "taxa_correcao": correcoes / max(acoes_executadas, 1),
            })

    return pd.DataFrame(resultados), pd.DataFrame(resultados_contexto), extrair_memoria_somatica(agente, execucao, beta_override)


def rodar_simulacao_csv(tipo_agente, nome_modelo: str, execucao: int, cfg_local: Config, df_cenarios: pd.DataFrame, beta_override: Optional[float] = None):
    if tipo_agente is QLearningSomatico:
        agente = tipo_agente(cfg_local, beta_override=beta_override)
    else:
        agente = tipo_agente(cfg_local)

    resultados = []
    resultados_contexto = []
    total_linhas = len(df_cenarios)
    episodios_treino = min(cfg_local.episodios_treino_passivo, max(cfg_local.n_episodios - 1, 0))

    for episodio in range(cfg_local.n_episodios):
        fase = "treino_passivo" if episodio < episodios_treino else "teste_ativo"
        episodio_ativo = episodio - episodios_treino

        aceitos = correcoes = reversoes = 0
        recompensa_total = 0.0
        acoes_executadas = 0

        for passo in range(cfg_local.passos_por_episodio):
            indice = (episodio * cfg_local.passos_por_episodio + passo) % total_linhas
            linha = df_cenarios.iloc[indice]
            estado = (linha["feature"], linha["luminosidade"], linha["local"], linha["periodo"], linha["atividade"])
            preferencia = linha["preferencia_usuario"]

            if fase == "treino_passivo":
                acao = preferencia
                resposta = resposta_positiva_observada(acao, cfg_local)
            else:
                acao = agente.escolher_acao(estado)
                resposta = resposta_usuario_csv(estado, acao, preferencia, cfg_local, probabilistica=True)

            proximo_indice = (indice + 1) % total_linhas
            proxima_linha = df_cenarios.iloc[proximo_indice]
            proximo_estado = (
                proxima_linha["feature"],
                proxima_linha["luminosidade"],
                proxima_linha["local"],
                proxima_linha["periodo"],
                proxima_linha["atividade"]
            )

            atualizar_agente_com_observacao(agente, estado, acao, resposta, proximo_estado)

            if fase == "teste_ativo":
                aceitos += int(resposta["aceitou"])
                correcoes += int(resposta["corrigiu"])
                reversoes += int(resposta["reverteu"])
                recompensa_total += float(resposta["recompensa"])
                acoes_executadas += 1

                registro_contexto = {
                    "modelo": nome_modelo,
                    "execucao": execucao,
                    "episodio": episodio_ativo,
                    "episodio_global": episodio,
                    "fase": fase,
                    "perfil_usuario": cfg_local.perfil,
                    "passo": passo,
                    "passo_csv": linha["passo"],
                    "cenario": linha["cenario"],
                    "feature": estado[0],
                    "luminosidade": estado[1],
                    "local": estado[2],
                    "periodo": estado[3],
                    "atividade": estado[4],
                    "acao": acao,
                    "preferencia": preferencia,
                    "resposta": resposta["resposta"],
                    "aceitou": int(resposta["aceitou"]),
                    "corrigiu": int(resposta["corrigiu"]),
                    "reverteu": int(resposta["reverteu"]),
                    "recompensa": float(resposta["recompensa"]),
                    "sinal_somatico": float(resposta["sinal_somatico"]),
                    "distancia": int(resposta["distancia"]),
                    "p_aceitar": float(resposta["p_aceitar"]),
                    "p_corrigir": float(resposta["p_corrigir"]),
                    "p_reverter": float(resposta["p_reverter"]),
                }
                registro_contexto.update(extrair_colunas_extras(linha))
                resultados_contexto.append(registro_contexto)

        agente.reduzir_exploracao()

        if fase == "teste_ativo":
            resultados.append({
                "modelo": nome_modelo,
                "execucao": execucao,
                "episodio": episodio_ativo,
                "episodio_global": episodio,
                "fase": fase,
                "correcoes": correcoes,
                "aceitos": aceitos,
                "reversoes": reversoes,
                "recompensa_total": recompensa_total,
                "recompensa_media": recompensa_total / max(acoes_executadas, 1),
                "taxa_aceitacao": aceitos / max(acoes_executadas, 1),
                "taxa_correcao": correcoes / max(acoes_executadas, 1),
            })

    return pd.DataFrame(resultados), pd.DataFrame(resultados_contexto), extrair_memoria_somatica(agente, execucao, beta_override)


def executar_experimento(cfg_local: Config, df_cenarios: Optional[pd.DataFrame] = None):
    todos_resultados = []
    todos_contextos = []
    memorias = []

    for execucao in range(cfg_local.n_execucoes):
        random.seed(cfg_local.seed + execucao)
        np.random.seed(cfg_local.seed + execucao)

        if df_cenarios is None:
            resultado_ql, contexto_ql, _ = rodar_simulacao(QLearning, "Q-Learning", execucao, cfg_local)
        else:
            resultado_ql, contexto_ql, _ = rodar_simulacao_csv(QLearning, "Q-Learning", execucao, cfg_local, df_cenarios)

        random.seed(cfg_local.seed + execucao)
        np.random.seed(cfg_local.seed + execucao)

        if df_cenarios is None:
            resultado_qls, contexto_qls, memoria = rodar_simulacao(QLearningSomatico, "Q-Learning Somático", execucao, cfg_local)
        else:
            resultado_qls, contexto_qls, memoria = rodar_simulacao_csv(QLearningSomatico, "Q-Learning Somático", execucao, cfg_local, df_cenarios)

        todos_resultados.extend([resultado_ql, resultado_qls])
        todos_contextos.extend([contexto_ql, contexto_qls])
        memorias.append(memoria)

    return (
        pd.concat(todos_resultados, ignore_index=True),
        pd.concat(todos_contextos, ignore_index=True),
        pd.concat(memorias, ignore_index=True) if memorias else pd.DataFrame(),
    )


# Métricas

def resumo_por_modelo(resultados: pd.DataFrame) -> pd.DataFrame:
    run_metrics = resultados.groupby(["modelo", "execucao"]).agg(
        correcoes=("correcoes", "sum"),
        reversoes=("reversoes", "sum"),
        aceitos=("aceitos", "sum"),
        recompensa_total=("recompensa_total", "sum"),
        recompensa_media=("recompensa_media", "mean"),
        taxa_aceitacao=("taxa_aceitacao", "mean"),
        taxa_correcao=("taxa_correcao", "mean"),
    ).reset_index()

    resumo = run_metrics.groupby("modelo").agg(
        correcoes_media=("correcoes", "mean"),
        correcoes_dp=("correcoes", "std"),
        reversoes_media=("reversoes", "mean"),
        reversoes_dp=("reversoes", "std"),
        recompensa_media=("recompensa_media", "mean"),
        recompensa_dp=("recompensa_media", "std"),
        taxa_aceitacao_media=("taxa_aceitacao", "mean"),
        taxa_aceitacao_dp=("taxa_aceitacao", "std"),
        taxa_correcao_media=("taxa_correcao", "mean"),
        taxa_correcao_dp=("taxa_correcao", "std"),
    ).reset_index()

    return resumo, run_metrics


def calcular_ganhos(resumo: pd.DataFrame) -> pd.DataFrame:
    if set(resumo["modelo"]) != {"Q-Learning", "Q-Learning Somático"}:
        return pd.DataFrame()

    ql = resumo[resumo["modelo"] == "Q-Learning"].iloc[0]
    qls = resumo[resumo["modelo"] == "Q-Learning Somático"].iloc[0]

    def reducao(valor_ql, valor_qls):
        return 100 * (valor_ql - valor_qls) / valor_ql if valor_ql != 0 else np.nan

    def aumento(valor_ql, valor_qls):
        return 100 * (valor_qls - valor_ql) / abs(valor_ql) if valor_ql != 0 else np.nan

    return pd.DataFrame([
        {"Métrica": "Correções", "Q-Learning": ql["correcoes_media"], "Q-Learning Somático": qls["correcoes_media"], "Diferença (%)": reducao(ql["correcoes_media"], qls["correcoes_media"])},
        {"Métrica": "Reversões", "Q-Learning": ql["reversoes_media"], "Q-Learning Somático": qls["reversoes_media"], "Diferença (%)": reducao(ql["reversoes_media"], qls["reversoes_media"])},
        {"Métrica": "Recompensa média", "Q-Learning": ql["recompensa_media"], "Q-Learning Somático": qls["recompensa_media"], "Diferença (%)": aumento(ql["recompensa_media"], qls["recompensa_media"])},
        {"Métrica": "Aceitação inferida", "Q-Learning": ql["taxa_aceitacao_media"], "Q-Learning Somático": qls["taxa_aceitacao_media"], "Diferença (%)": aumento(ql["taxa_aceitacao_media"], qls["taxa_aceitacao_media"])},
    ])


def calcular_wilcoxon(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calcula o teste de Wilcoxon pareado por execução."""
    if not SCIPY_AVAILABLE:
        return pd.DataFrame([{
            "Métrica": "SciPy indisponível",
            "W": np.nan,
            "p-valor": np.nan,
            "significativo_0.05": False,
            "interpretação": "teste não executado"
        }])

    ql = run_metrics[run_metrics["modelo"] == "Q-Learning"].sort_values("execucao").reset_index(drop=True)
    qls = run_metrics[run_metrics["modelo"] == "Q-Learning Somático"].sort_values("execucao").reset_index(drop=True)

    metricas = [
        ("correcoes", "Correções"),
        ("reversoes", "Reversões"),
        ("recompensa_media", "Recompensa média"),
        ("taxa_aceitacao", "Aceitação inferida"),
        ("taxa_correcao", "Taxa de correção"),
    ]

    linhas = []
    for coluna, nome in metricas:
        diferencas = ql[coluna].to_numpy(dtype=float) - qls[coluna].to_numpy(dtype=float)

        if np.allclose(diferencas, 0.0):
            stat, p_value = np.nan, np.nan
            interpretacao = "sem diferença observada"
            significativo = False
        else:
            try:
                stat, p_value = wilcoxon(
                    ql[coluna],
                    qls[coluna],
                    zero_method="wilcox",
                    alternative="two-sided"
                )
                significativo = bool(p_value < 0.05) if not pd.isna(p_value) else False
                interpretacao = "diferença significativa" if significativo else "não significativo"
            except ValueError:
                stat, p_value = np.nan, np.nan
                significativo = False
                interpretacao = "teste não aplicável"

        linhas.append({
            "Métrica": nome,
            "W": stat,
            "p-valor": p_value,
            "significativo_0.05": significativo,
            "interpretação": interpretacao,
        })

    return pd.DataFrame(linhas)


def formatar_wilcoxon_exibicao(testes: pd.DataFrame) -> pd.DataFrame:
    """Formata a tabela do Wilcoxon para exibição."""
    df = testes.copy()

    def fmt_num(valor: Any, casas: int = 6) -> str:
        if pd.isna(valor):
            return "n/a"
        return f"{float(valor):.{casas}f}"

    df["W"] = df["W"].apply(lambda v: fmt_num(v, 4))
    df["p-valor"] = df["p-valor"].apply(lambda v: fmt_num(v, 6))
    df["significativo_0.05"] = df["significativo_0.05"].map({True: "sim", False: "não"})
    return df


def episodio_estabilizacao(dados_modelo: pd.DataFrame, cfg_local: Config) -> float:
    dados = dados_modelo.sort_values("episodio")
    serie = dados["taxa_correcao"].rolling(cfg_local.janela_estabilizacao, min_periods=1).mean().values
    episodios = dados["episodio"].values

    limite = len(serie) - cfg_local.paciencia_estabilizacao + 1
    for i in range(0, max(limite, 0)):
        trecho = serie[i:i + cfg_local.paciencia_estabilizacao]
        if len(trecho) == cfg_local.paciencia_estabilizacao and np.all(trecho <= cfg_local.limiar_estabilizacao):
            return float(episodios[i])
    return np.nan


def calcular_estabilizacao(resultados: pd.DataFrame, cfg_local: Config) -> pd.DataFrame:
    ep_modelo = resultados.groupby(["modelo", "episodio"]).agg(
        taxa_correcao=("taxa_correcao", "mean")
    ).reset_index()

    linhas = []
    for modelo in ep_modelo["modelo"].unique():
        ep = episodio_estabilizacao(ep_modelo[ep_modelo["modelo"] == modelo], cfg_local)
        linhas.append({"modelo": modelo, "episodio_estabilizacao": ep})

    tabela = pd.DataFrame(linhas)
    tabela["ganho_vs_ql_%"] = np.nan
    tabela["observacao"] = "critério não atingido"

    ql_linha = tabela[tabela["modelo"] == "Q-Learning"]
    qls_linha = tabela[tabela["modelo"] == "Q-Learning Somático"]

    if not ql_linha.empty and not qls_linha.empty:
        ql_ep = ql_linha["episodio_estabilizacao"].iloc[0]
        qls_ep = qls_linha["episodio_estabilizacao"].iloc[0]

        if not pd.isna(ql_ep):
            tabela.loc[tabela["modelo"] == "Q-Learning", "observacao"] = "critério atingido"

        if not pd.isna(qls_ep):
            tabela.loc[tabela["modelo"] == "Q-Learning Somático", "observacao"] = "critério atingido"

        if not pd.isna(ql_ep) and not pd.isna(qls_ep) and ql_ep != 0:
            ganho = 100 * (ql_ep - qls_ep) / ql_ep
            tabela.loc[tabela["modelo"] == "Q-Learning", "ganho_vs_ql_%"] = 0.0
            tabela.loc[tabela["modelo"] == "Q-Learning Somático", "ganho_vs_ql_%"] = ganho
        elif pd.isna(ql_ep) and not pd.isna(qls_ep):
            tabela.loc[tabela["modelo"] == "Q-Learning Somático", "observacao"] = "critério atingido; baseline não estabilizou"

    return tabela


def formatar_estabilizacao_exibicao(estabilizacao: pd.DataFrame) -> pd.DataFrame:
    """Formata a tabela de estabilização para exibição."""
    df = estabilizacao.copy()

    def fmt_ep(valor: Any) -> str:
        if pd.isna(valor):
            return "não estabilizou"
        return f"episódio {int(valor)}"

    def fmt_ganho(valor: Any) -> str:
        if pd.isna(valor):
            return "n/a"
        return f"{float(valor):.2f}%"

    df["Estabilização"] = df["episodio_estabilizacao"].apply(fmt_ep)
    df["Ganho relativo"] = df["ganho_vs_ql_%"].apply(fmt_ganho)
    df = df.rename(columns={"modelo": "Modelo", "observacao": "Observação"})
    return df[["Modelo", "Estabilização", "Ganho relativo", "Observação"]]


def comparar_contextos(resultados_contexto: pd.DataFrame) -> pd.DataFrame:
    agrupado = resultados_contexto.groupby(["modelo", "feature", "cenario", "luminosidade", "local", "periodo", "atividade"]).agg(
        correcoes=("corrigiu", "sum"),
        reversoes=("reverteu", "sum"),
        aceitacao=("aceitou", "mean"),
        recompensa_media=("recompensa", "mean"),
    ).reset_index()

    ql = agrupado[agrupado["modelo"] == "Q-Learning"].copy()
    qls = agrupado[agrupado["modelo"] == "Q-Learning Somático"].copy()

    comp = ql.merge(
        qls,
        on=["feature", "cenario", "luminosidade", "local", "periodo", "atividade"],
        suffixes=("_QL", "_QLS")
    )

    comp["reducao_correcoes_%"] = np.where(
        comp["correcoes_QL"] != 0,
        100 * (comp["correcoes_QL"] - comp["correcoes_QLS"]) / comp["correcoes_QL"],
        np.nan
    )
    comp["ganho_aceitacao_pp"] = 100 * (comp["aceitacao_QLS"] - comp["aceitacao_QL"])

    return comp[[
        "feature", "cenario", "luminosidade", "local", "periodo", "atividade",
        "correcoes_QL", "correcoes_QLS", "reducao_correcoes_%",
        "aceitacao_QL", "aceitacao_QLS", "ganho_aceitacao_pp",
        "reversoes_QL", "reversoes_QLS",
        "recompensa_media_QL", "recompensa_media_QLS"
    ]].sort_values("reducao_correcoes_%", ascending=False)


def comparar_features(resultados_contexto: pd.DataFrame) -> pd.DataFrame:
    agrupado = resultados_contexto.groupby(["modelo", "feature", "execucao"]).agg(
        correcoes=("corrigiu", "sum"),
        reversoes=("reverteu", "sum"),
        aceitacao=("aceitou", "mean"),
        recompensa_media=("recompensa", "mean"),
        taxa_correcao=("corrigiu", "mean"),
    ).reset_index()

    resumo = agrupado.groupby(["modelo", "feature"]).agg(
        correcoes_media=("correcoes", "mean"),
        correcoes_dp=("correcoes", "std"),
        reversoes_media=("reversoes", "mean"),
        reversoes_dp=("reversoes", "std"),
        aceitacao_media=("aceitacao", "mean"),
        aceitacao_dp=("aceitacao", "std"),
        recompensa_media=("recompensa_media", "mean"),
        recompensa_dp=("recompensa_media", "std"),
        taxa_correcao_media=("taxa_correcao", "mean"),
        taxa_correcao_dp=("taxa_correcao", "std"),
    ).reset_index()

    return resumo


# Gráficos

def criar_grafico_linha(df: pd.DataFrame, coluna_y: str, titulo: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.8, 3.35))
    cores = {"Q-Learning": "#555555", "Q-Learning Somático": "#1683d8"}

    for modelo in df["modelo"].unique():
        dados_modelo = df[df["modelo"] == modelo]
        ax.plot(dados_modelo["episodio"], dados_modelo[coluna_y], linewidth=2.0, label=modelo, color=cores.get(modelo))

    ax.set_xlabel("Episódio", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(titulo, fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(True, linestyle="-", linewidth=0.45, alpha=0.28)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#555555", labelsize=8)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.tight_layout()
    return fig


def criar_grafico_barras(resumo: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 3.35))
    modelos = resumo["modelo"].tolist()
    valores = resumo["correcoes_media"].tolist()
    cores = ["#555555" if m == "Q-Learning" else "#1683d8" for m in modelos]
    ax.bar(modelos, valores, color=cores, width=0.55)
    ax.set_title("Correções médias por modelo", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_ylabel("Correções médias por execução", fontsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.45, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#555555", labelsize=8)
    fig.tight_layout()
    return fig


def criar_grafico_beta(beta_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 3.35))
    ax.plot(beta_df["beta"], beta_df["correcoes_media"], marker="o", linewidth=2.0, color="#1683d8")
    ax.set_title("Análise de sensibilidade do beta somático", fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlabel("Beta somático")
    ax.set_ylabel("Correções médias")
    ax.grid(True, linestyle="-", linewidth=0.45, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# Exportação

def formatar_resumo_comparativo(resumo: pd.DataFrame) -> pd.DataFrame:
    df = resumo.copy()
    df["Correções (média ± dp)"] = df.apply(lambda r: f"{r['correcoes_media']:.2f} ± {r['correcoes_dp']:.2f}", axis=1)
    df["Reversões (média ± dp)"] = df.apply(lambda r: f"{r['reversoes_media']:.2f} ± {r['reversoes_dp']:.2f}", axis=1)
    df["Aceitação (%) (média ± dp)"] = df.apply(lambda r: f"{100*r['taxa_aceitacao_media']:.2f} ± {100*r['taxa_aceitacao_dp']:.2f}", axis=1)
    df["Recompensa (média ± dp)"] = df.apply(lambda r: f"{r['recompensa_media']:.4f} ± {r['recompensa_dp']:.4f}", axis=1)
    return df[["modelo", "Correções (média ± dp)", "Reversões (média ± dp)", "Aceitação (%) (média ± dp)", "Recompensa (média ± dp)"]].rename(columns={"modelo": "Modelo"})


def df_to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, escape=False)


def formatar_ganhos_latex(ganhos: pd.DataFrame) -> pd.DataFrame:
    df = ganhos.copy()
    if df.empty:
        return df
    df["Q-Learning"] = df["Q-Learning"].apply(lambda v: f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}")
    df["Q-Learning Somático"] = df["Q-Learning Somático"].apply(lambda v: f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}")
    df["Diferença (%)"] = df["Diferença (%)"].apply(lambda v: f"{v:.2f}")
    return df


def formatar_features_latex(features_comp: pd.DataFrame) -> pd.DataFrame:
    df = features_comp.copy()
    if df.empty:
        return df
    out = pd.DataFrame({
        "Modelo": df["modelo"],
        "Feature": df["feature"],
        "Correções (média ± dp)": df.apply(lambda r: f"{r['correcoes_media']:.2f} ± {r['correcoes_dp']:.2f}", axis=1),
        "Reversões (média ± dp)": df.apply(lambda r: f"{r['reversoes_media']:.2f} ± {r['reversoes_dp']:.2f}", axis=1),
        "Aceitação (%) (média ± dp)": df.apply(lambda r: f"{100*r['aceitacao_media']:.2f} ± {100*r['aceitacao_dp']:.2f}", axis=1),
        "Recompensa (média ± dp)": df.apply(lambda r: f"{r['recompensa_media']:.4f} ± {r['recompensa_dp']:.4f}", axis=1),
        "Taxa correção (%)": df.apply(lambda r: f"{100*r['taxa_correcao_media']:.2f} ± {100*r['taxa_correcao_dp']:.2f}", axis=1),
    })
    return out


def formatar_contextos_latex(contexto_comp: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = contexto_comp.copy().head(top_n)
    if df.empty:
        return df
    return pd.DataFrame({
        "Feature": df["feature"],
        "Luminosidade": df["luminosidade"],
        "Local": df["local"],
        "Período": df["periodo"],
        "Atividade": df["atividade"],
        "Correções QL": df["correcoes_QL"].astype(int),
        "Correções QLS": df["correcoes_QLS"].astype(int),
        "Redução (%)": df["reducao_correcoes_%"].apply(lambda v: f"{v:.2f}"),
        "Aceitação QL (%)": df["aceitacao_QL"].apply(lambda v: f"{100*v:.2f}"),
        "Aceitação QLS (%)": df["aceitacao_QLS"].apply(lambda v: f"{100*v:.2f}"),
        "Ganho aceitação (p.p.)": df["ganho_aceitacao_pp"].apply(lambda v: f"{v:.2f}"),
    })


def formatar_contextos_piores_latex(contexto_comp: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = contexto_comp.copy().sort_values("reducao_correcoes_%", ascending=True).head(top_n)
    if df.empty:
        return df
    return pd.DataFrame({
        "Feature": df["feature"],
        "Luminosidade": df["luminosidade"],
        "Local": df["local"],
        "Período": df["periodo"],
        "Atividade": df["atividade"],
        "Correções QL": df["correcoes_QL"].astype(int),
        "Correções QLS": df["correcoes_QLS"].astype(int),
        "Redução (%)": df["reducao_correcoes_%"].apply(lambda v: f"{v:.2f}"),
        "Aceitação QL (%)": df["aceitacao_QL"].apply(lambda v: f"{100*v:.2f}"),
        "Aceitação QLS (%)": df["aceitacao_QLS"].apply(lambda v: f"{100*v:.2f}"),
    })


def formatar_memoria_latex(memoria_somatica: pd.DataFrame, tipo: str = "positivos", top_n: int = 10) -> pd.DataFrame:
    df = memoria_somatica.copy()
    if df.empty:
        return df
    if tipo == "negativos":
        df = df.sort_values("MS(s,a)", ascending=True).head(top_n)
    else:
        df = df.sort_values("MS(s,a)", ascending=False).head(top_n)
    df = df[["feature", "luminosidade", "local", "periodo", "atividade", "acao", "MS(s,a)"]].copy()
    df["MS(s,a)"] = df["MS(s,a)"].apply(lambda v: f"{v:.4f}")
    return df.rename(columns={
        "feature": "Feature",
        "luminosidade": "Luminosidade",
        "local": "Local",
        "periodo": "Período",
        "atividade": "Atividade",
        "acao": "Ação",
    })

def agregar_memoria_somatica(memoria_somatica: pd.DataFrame) -> pd.DataFrame:
    """Agrega a memória somática por par estado-ação."""
    if memoria_somatica.empty:
        return pd.DataFrame()

    colunas_estado_acao = [
        "feature",
        "luminosidade",
        "local",
        "periodo",
        "atividade",
        "acao",
    ]

    memoria_agregada = memoria_somatica.groupby(colunas_estado_acao).agg(
        ms_media=("MS(s,a)", "mean"),
        ms_dp=("MS(s,a)", "std"),
        ms_min=("MS(s,a)", "min"),
        ms_max=("MS(s,a)", "max"),
        n_execucoes=("MS(s,a)", "count"),
    ).reset_index()

    memoria_agregada["ms_dp"] = memoria_agregada["ms_dp"].fillna(0.0)
    memoria_agregada["abs_ms_media"] = memoria_agregada["ms_media"].abs()

    return memoria_agregada


def formatar_memoria_agregada_latex(
    memoria_agregada: pd.DataFrame,
    tipo: str = "positivos",
    top_n: int = 10
) -> pd.DataFrame:
    """Formata a memória somática agregada para LaTeX."""
    if memoria_agregada.empty:
        return pd.DataFrame()

    df = memoria_agregada.copy()

    if tipo == "negativos":
        df = df.sort_values("ms_media", ascending=True).head(top_n)
    elif tipo == "intensos":
        df = df.sort_values("abs_ms_media", ascending=False).head(top_n)
    else:
        df = df.sort_values("ms_media", ascending=False).head(top_n)

    out = pd.DataFrame({
        "Feature": df["feature"],
        "Luminosidade": df["luminosidade"],
        "Local": df["local"],
        "Período": df["periodo"],
        "Atividade": df["atividade"],
        "Ação": df["acao"],
        "$\\overline{MS}(s,a)$": df["ms_media"].apply(lambda v: f"{v:.4f}"),
        "DP": df["ms_dp"].apply(lambda v: f"{v:.4f}"),
        "n": df["n_execucoes"].astype(int),
    })

    return out


def montar_latex_tabela(nome: str, df: pd.DataFrame, caption: str, label: str) -> str:
    corpo = df_to_latex(df)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{corpo}\n"
        "\\end{table}\n"
    )


def montar_latex_todas_tabelas(resumo_comparativo: pd.DataFrame, ganhos: pd.DataFrame, testes: pd.DataFrame,
                               estabilizacao: pd.DataFrame, features_comp: pd.DataFrame,
                               contexto_comp: pd.DataFrame, memoria_somatica: pd.DataFrame) -> str:
    blocos = []
    blocos.append(montar_latex_tabela("resumo", resumo_comparativo,
        "Resumo comparativo entre Q-Learning e Q-Learning Somático.", "tab:resumo_comparativo"))
    blocos.append(montar_latex_tabela("ganhos", formatar_ganhos_latex(ganhos),
        "Diferenças relativas entre Q-Learning e Q-Learning Somático.", "tab:ganhos_relativos"))
    blocos.append(montar_latex_tabela("wilcoxon", formatar_wilcoxon_exibicao(testes),
        "Teste de Wilcoxon pareado por métrica.", "tab:wilcoxon"))
    blocos.append(montar_latex_tabela("estabilizacao", formatar_estabilizacao_exibicao(estabilizacao),
        "Episódio de estabilização segundo o critério de média móvel.", "tab:estabilizacao"))
    blocos.append(montar_latex_tabela("features", formatar_features_latex(features_comp),
        "Resumo dos resultados por feature avaliada.", "tab:resultados_feature"))
    blocos.append(montar_latex_tabela("contextos", formatar_contextos_latex(contexto_comp, top_n=10),
        "Dez contextos com maior redução de correções no Q-Learning Somático.", "tab:top_contextos"))
    blocos.append(montar_latex_tabela("mem_pos", formatar_memoria_latex(memoria_somatica, "positivos", top_n=10),
        "Principais marcadores somáticos positivos aprendidos.", "tab:memoria_positiva"))
    blocos.append(montar_latex_tabela("mem_neg", formatar_memoria_latex(memoria_somatica, "negativos", top_n=10),
        "Principais marcadores somáticos negativos aprendidos.", "tab:memoria_negativa"))
    return "\n\n".join(blocos)


# Área de execução

with st.container(border=True):
    panel_header("Execução do experimento", "controle")
    exec_col1, exec_col2 = st.columns([4, 1.25])
    with exec_col1:
        st.markdown(
            """
            <div class="small-muted">
                Execute os dois modelos sob a mesma configuração experimental. O painel calcula métricas agregadas,
                média ± desvio padrão, Wilcoxon pareado, estabilização, análise por contexto, memória somática final
                e arquivos exportáveis para reprodutibilidade.
            </div>
            """,
            unsafe_allow_html=True
        )
    with exec_col2:
        executar = st.button("Executar simulação")


# Execução e resultados


if executar:
    with st.spinner("Executando simulação..."):
        if modo_execucao == "Carregar cenários por CSV":
            if arquivo_cenarios is None:
                st.error("Envie um arquivo CSV para executar este modo.")
                st.stop()
            df_cenarios = carregar_e_validar_csv(arquivo_cenarios)
            if df_cenarios is None:
                st.stop()
        else:
            df_cenarios = None

        resultados, resultados_contexto, memoria_somatica = executar_experimento(cfg, df_cenarios)

        resumo, run_metrics = resumo_por_modelo(resultados)
        ganhos = calcular_ganhos(resumo)
        testes = calcular_wilcoxon(run_metrics)
        estabilizacao = calcular_estabilizacao(resultados, cfg)
        contexto_comp = comparar_contextos(resultados_contexto)
        features_comp = comparar_features(resultados_contexto)
        resumo_comparativo = formatar_resumo_comparativo(resumo)

        st.session_state.resultados_simulacao = {
            "resultados": resultados,
            "resultados_contexto": resultados_contexto,
            "memoria_somatica": memoria_somatica,
            "resumo": resumo,
            "run_metrics": run_metrics,
            "ganhos": ganhos,
            "testes": testes,
            "estabilizacao": estabilizacao,
            "contexto_comp": contexto_comp,
            "features_comp": features_comp,
            "resumo_comparativo": resumo_comparativo,
            "df_cenarios": df_cenarios,
            "cfg": cfg,
            "modo_execucao": modo_execucao,
        }

        st.session_state.config_assinatura = assinatura_configuracao(cfg, modo_execucao)

    st.success("Simulação concluída.")


dados_salvos = st.session_state.resultados_simulacao

if dados_salvos is not None:
    resultados = dados_salvos["resultados"]
    resultados_contexto = dados_salvos["resultados_contexto"]
    memoria_somatica = dados_salvos["memoria_somatica"]
    resumo = dados_salvos["resumo"]
    run_metrics = dados_salvos["run_metrics"]
    ganhos = dados_salvos["ganhos"]
    testes = dados_salvos["testes"]
    estabilizacao = dados_salvos["estabilizacao"]
    contexto_comp = dados_salvos["contexto_comp"]
    features_comp = dados_salvos["features_comp"]
    resumo_comparativo = dados_salvos["resumo_comparativo"]
    df_cenarios = dados_salvos["df_cenarios"]

    assinatura_atual = assinatura_configuracao(cfg, modo_execucao)
    if st.session_state.config_assinatura != assinatura_atual:
        st.warning(
            "Os resultados exibidos pertencem à última simulação executada. "
            "Você alterou algum parâmetro; clique em 'Executar simulação' para atualizar os resultados."
        )

    ql_corr = resumo[resumo["modelo"] == "Q-Learning"]["correcoes_media"].iloc[0]
    qls_corr = resumo[resumo["modelo"] == "Q-Learning Somático"]["correcoes_media"].iloc[0]
    reducao = ((ql_corr - qls_corr) / ql_corr) * 100 if ql_corr != 0 else 0

    ql_acc = resumo[resumo["modelo"] == "Q-Learning"]["taxa_aceitacao_media"].iloc[0]
    qls_acc = resumo[resumo["modelo"] == "Q-Learning Somático"]["taxa_aceitacao_media"].iloc[0]

    st.markdown("### Indicadores principais")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Correções QL", f"{ql_corr:.2f}", "Média por execução")
    with k2:
        metric_card("Correções QLS", f"{qls_corr:.2f}", "Média por execução")
    with k3:
        metric_card("Redução", f"{reducao:.2f}%", "Correções evitadas")
    with k4:
        metric_card("Aceitação", f"{100*qls_acc:.2f}%", f"QL: {100*ql_acc:.2f}%")

    st.markdown("### Painel de análise")
    graficos_ep = resultados.groupby(["modelo", "episodio"]).agg(
        correcoes=("correcoes", "mean"),
        taxa_correcao=("taxa_correcao", "mean"),
        taxa_aceitacao=("taxa_aceitacao", "mean"),
        recompensa_media=("recompensa_media", "mean"),
    ).reset_index()

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        with st.container(border=True):
            panel_header("Correções por episódio", "linha temporal")
            st.pyplot(criar_grafico_linha(graficos_ep, "correcoes", "Correções ao longo dos episódios", "Correções"), use_container_width=True)
    with row1_col2:
        with st.container(border=True):
            panel_header("Taxa de aceitação", "linha temporal")
            st.pyplot(criar_grafico_linha(graficos_ep, "taxa_aceitacao", "Aceitação ao longo dos episódios", "Aceitação"), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True):
            panel_header("Recompensa média", "linha temporal")
            st.pyplot(criar_grafico_linha(graficos_ep, "recompensa_media", "Recompensa média ao longo dos episódios", "Recompensa"), use_container_width=True)
    with row2_col2:
        with st.container(border=True):
            panel_header("Comparação agregada", "resumo")
            st.pyplot(criar_grafico_barras(resumo), use_container_width=True)

    st.markdown("### Resultados tabulares")
    tab_resumo, tab_ganhos, tab_estatistica, tab_feature, tab_contexto, tab_memoria, tab_cenarios, tab_beta, tab_download = st.tabs(
        [
            "Resumo",
            "Ganhos",
            "Estatística",
            "Features",
            "Contextos",
            "Memória somática",
            "Cenários carregados",
            "Sensibilidade beta",
            "Exportação"
        ]
    )

    with tab_resumo:
        with st.container(border=True):
            panel_header("Resumo comparativo", "média ± dp")
            st.dataframe(resumo_comparativo, use_container_width=True)
        with st.container(border=True):
            panel_header("Resumo numérico completo", "métricas")
            st.dataframe(resumo.round(4), use_container_width=True)

    with tab_ganhos:
        with st.container(border=True):
            panel_header("Diferenças relativas", "QLS vs QL")
            st.dataframe(ganhos.round(4), use_container_width=True)

    with tab_estatistica:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            with st.container(border=True):
                panel_header("Teste de Wilcoxon pareado", "p-valor")
                st.dataframe(formatar_wilcoxon_exibicao(testes), use_container_width=True)
        with col_s2:
            with st.container(border=True):
                panel_header("Episódio de estabilização", "média móvel")
                st.dataframe(formatar_estabilizacao_exibicao(estabilizacao), use_container_width=True)
                st.caption(
                    f"Critério: média móvel da taxa de correção ≤ {cfg.limiar_estabilizacao:.2f} "
                    f"por {cfg.paciencia_estabilizacao} episódios consecutivos."
                )

    with tab_feature:
        with st.container(border=True):
            panel_header("Resumo por feature", "brilho vs volume")
            view_feature = features_comp.copy()
            if not view_feature.empty:
                view_feature["aceitacao_media"] = 100 * view_feature["aceitacao_media"]
                view_feature["aceitacao_dp"] = 100 * view_feature["aceitacao_dp"]
                view_feature["taxa_correcao_media"] = 100 * view_feature["taxa_correcao_media"]
                view_feature["taxa_correcao_dp"] = 100 * view_feature["taxa_correcao_dp"]
                st.dataframe(view_feature.round(4), use_container_width=True)
            else:
                st.info("Nenhum resultado por feature disponível.")

    with tab_contexto:
        with st.container(border=True):
            panel_header("Comparação por contexto", "QL vs QLS")
            view = contexto_comp.copy()
            view["aceitacao_QL"] = (100 * view["aceitacao_QL"]).round(2)
            view["aceitacao_QLS"] = (100 * view["aceitacao_QLS"]).round(2)
            view["reducao_correcoes_%"] = view["reducao_correcoes_%"].round(2)
            view["ganho_aceitacao_pp"] = view["ganho_aceitacao_pp"].round(2)
            view["recompensa_media_QL"] = view["recompensa_media_QL"].round(4)
            view["recompensa_media_QLS"] = view["recompensa_media_QLS"].round(4)
            st.dataframe(view, use_container_width=True)

    with tab_memoria:
        with st.container(border=True):
            panel_header("Memória somática final", "MS(s,a)")
            if not memoria_somatica.empty:
                memoria_view = memoria_somatica.copy()
                memoria_view["MS(s,a)"] = memoria_view["MS(s,a)"].round(4)

                memoria_agregada = agregar_memoria_somatica(memoria_somatica)
                memoria_agregada_view = memoria_agregada.copy()
                memoria_agregada_view["ms_media"] = memoria_agregada_view["ms_media"].round(4)
                memoria_agregada_view["ms_dp"] = memoria_agregada_view["ms_dp"].round(4)
                memoria_agregada_view["ms_min"] = memoria_agregada_view["ms_min"].round(4)
                memoria_agregada_view["ms_max"] = memoria_agregada_view["ms_max"].round(4)

                mem_ag_pos, mem_ag_neg, mem_ag_int, mem_pos, mem_neg, mem_all = st.tabs([
                    "Agregada +",
                    "Agregada -",
                    "Agregada intensa",
                    "Individuais +",
                    "Individuais -",
                    "Todos"
                ])

                with mem_ag_pos:
                    st.dataframe(
                        memoria_agregada_view.sort_values("ms_media", ascending=False).head(60),
                        use_container_width=True
                    )

                with mem_ag_neg:
                    st.dataframe(
                        memoria_agregada_view.sort_values("ms_media", ascending=True).head(60),
                        use_container_width=True
                    )

                with mem_ag_int:
                    st.dataframe(
                        memoria_agregada_view.sort_values("abs_ms_media", ascending=False).head(60),
                        use_container_width=True
                    )

                with mem_pos:
                    st.dataframe(
                        memoria_view.sort_values("MS(s,a)", ascending=False).head(60),
                        use_container_width=True
                    )

                with mem_neg:
                    st.dataframe(
                        memoria_view.sort_values("MS(s,a)", ascending=True).head(60),
                        use_container_width=True
                    )

                with mem_all:
                    st.dataframe(memoria_view, use_container_width=True)

            else:
                st.info("Nenhum dado de memória somática disponível.")

    with tab_cenarios:
        with st.container(border=True):
            panel_header("Cenários carregados", "CSV")
            if df_cenarios is not None:
                st.write(f"Total de linhas no arquivo: {len(df_cenarios)}")
                st.dataframe(df_cenarios.head(100), use_container_width=True)
            else:
                st.write("O experimento foi executado com geração automática de cenários.")

    beta_resultados = pd.DataFrame()
    with tab_beta:
        with st.container(border=True):
            panel_header("Análise de sensibilidade do beta", "opcional")
            if executar_sensibilidade:
                try:
                    beta_values = [float(x.strip()) for x in betas_texto.split(",") if x.strip()]
                    linhas_beta = []
                    progress = st.progress(0)
                    for idx, beta_val in enumerate(beta_values):
                        cfg_beta = Config(**{**cfg.__dict__, "beta_somatico": beta_val})
                        res_b, _, _ = executar_experimento(cfg_beta, df_cenarios)
                        resumo_b, _ = resumo_por_modelo(res_b)
                        qls_b = resumo_b[resumo_b["modelo"] == "Q-Learning Somático"].iloc[0]
                        linhas_beta.append({
                            "beta": beta_val,
                            "correcoes_media": qls_b["correcoes_media"],
                            "reversoes_media": qls_b["reversoes_media"],
                            "taxa_aceitacao_media": qls_b["taxa_aceitacao_media"],
                            "recompensa_media": qls_b["recompensa_media"],
                        })
                        progress.progress((idx + 1) / len(beta_values))
                    beta_resultados = pd.DataFrame(linhas_beta).sort_values("beta")
                    st.pyplot(criar_grafico_beta(beta_resultados), use_container_width=True)
                    view_beta = beta_resultados.copy()
                    view_beta["taxa_aceitacao_media"] = 100 * view_beta["taxa_aceitacao_media"]
                    st.dataframe(view_beta.round(4), use_container_width=True)
                except Exception as exc:
                    st.error(f"Erro na análise de beta: {exc}")
            else:
                st.write("Ative a opção 'Executar análise de beta' na barra lateral para comparar diferentes pesos somáticos.")

    with tab_download:
        with st.container(border=True):
            panel_header("Exportação dos resultados", "CSV / LaTeX")
            st.caption("Os CSVs completos preservam todos os registros. Os arquivos LaTeX abaixo são versões resumidas e formatadas para uso documental.")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.download_button("CSV: episódios", resultados.to_csv(index=False).encode("utf-8"), "resultados_episodios.csv", "text/csv")
            with d2:
                st.download_button("CSV: contexto completo", resultados_contexto.to_csv(index=False).encode("utf-8"), "resultados_contexto.csv", "text/csv")
            with d3:
                st.download_button("CSV: resumo", resumo.to_csv(index=False).encode("utf-8"), "resumo_estatistico.csv", "text/csv")
            with d4:
                st.download_button("CSV: memória somática", memoria_somatica.to_csv(index=False).encode("utf-8"), "memoria_somatica.csv", "text/csv")

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.download_button("CSV: ganhos", ganhos.to_csv(index=False).encode("utf-8"), "ganhos.csv", "text/csv")
            with e2:
                st.download_button("CSV: Wilcoxon", testes.to_csv(index=False).encode("utf-8"), "wilcoxon.csv", "text/csv")
            with e3:
                st.download_button("CSV: estabilização", estabilizacao.to_csv(index=False).encode("utf-8"), "estabilizacao.csv", "text/csv")
            with e4:
                st.download_button("CSV: feature", features_comp.to_csv(index=False).encode("utf-8"), "resumo_por_feature.csv", "text/csv")

            st.markdown("#### Tabelas LaTeX")
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.download_button("LaTeX: resumo", df_to_latex(resumo_comparativo).encode("utf-8"), "tabela_resumo_latex.tex", "text/plain")
            with l2:
                st.download_button("LaTeX: ganhos", df_to_latex(formatar_ganhos_latex(ganhos)).encode("utf-8"), "tabela_ganhos_latex.tex", "text/plain")
            with l3:
                st.download_button("LaTeX: Wilcoxon", df_to_latex(formatar_wilcoxon_exibicao(testes)).encode("utf-8"), "tabela_wilcoxon_latex.tex", "text/plain")
            with l4:
                st.download_button("LaTeX: estabilização", df_to_latex(formatar_estabilizacao_exibicao(estabilizacao)).encode("utf-8"), "tabela_estabilizacao_latex.tex", "text/plain")

            l5, l6, l7, l8 = st.columns(4)
            with l5:
                st.download_button("LaTeX: features", df_to_latex(formatar_features_latex(features_comp)).encode("utf-8"), "tabela_features_latex.tex", "text/plain")
            with l6:
                st.download_button("LaTeX: top contextos", df_to_latex(formatar_contextos_latex(contexto_comp, top_n=10)).encode("utf-8"), "tabela_top_contextos_latex.tex", "text/plain")
            with l7:
                st.download_button("LaTeX: memória +", df_to_latex(formatar_memoria_latex(memoria_somatica, "positivos", top_n=10)).encode("utf-8"), "tabela_memoria_positiva_latex.tex", "text/plain")
            with l8:
                st.download_button("LaTeX: memória -", df_to_latex(formatar_memoria_latex(memoria_somatica, "negativos", top_n=10)).encode("utf-8"), "tabela_memoria_negativa_latex.tex", "text/plain")

            memoria_agregada = agregar_memoria_somatica(memoria_somatica)

            st.markdown("#### Memória somática agregada")

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.download_button(
                    "CSV: memória agregada",
                    memoria_agregada.to_csv(index=False).encode("utf-8"),
                    "memoria_somatica_agregada.csv",
                    "text/csv"
                )

            with m2:
                st.download_button(
                    "LaTeX: memória agregada +",
                    df_to_latex(
                        formatar_memoria_agregada_latex(memoria_agregada, "positivos", top_n=10)
                    ).encode("utf-8"),
                    "tabela_memoria_agregada_positiva_latex.tex",
                    "text/plain"
                )

            with m3:
                st.download_button(
                    "LaTeX: memória agregada -",
                    df_to_latex(
                        formatar_memoria_agregada_latex(memoria_agregada, "negativos", top_n=10)
                    ).encode("utf-8"),
                    "tabela_memoria_agregada_negativa_latex.tex",
                    "text/plain"
                )

            with m4:
                st.download_button(
                    "LaTeX: memória agregada intensa",
                    df_to_latex(
                        formatar_memoria_agregada_latex(memoria_agregada, "intensos", top_n=10)
                    ).encode("utf-8"),
                    "tabela_memoria_agregada_intensa_latex.tex",
                    "text/plain"
                )

            pacote_latex = montar_latex_todas_tabelas(
                resumo_comparativo, ganhos, testes, estabilizacao,
                features_comp, contexto_comp, memoria_somatica
            )
            st.download_button(
                "LaTeX: pacote com todas as tabelas resumidas",
                pacote_latex.encode("utf-8"),
                "tabelas_resumidas_artigo.tex",
                "text/plain"
            )

            if not beta_resultados.empty:
                st.download_button("CSV: sensibilidade beta", beta_resultados.to_csv(index=False).encode("utf-8"), "sensibilidade_beta.csv", "text/csv")
                st.download_button("LaTeX: sensibilidade beta", df_to_latex(beta_resultados.round(4)).encode("utf-8"), "tabela_sensibilidade_beta_latex.tex", "text/plain")

else:
    st.markdown("### Visão inicial")
    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            panel_header("Saída esperada", "após execução")
            st.markdown(
                """
                <div class="small-muted">
                    Após executar a simulação, o painel exibirá indicadores comparativos, curvas de evolução,
                    análise por contexto, memória somática final, teste estatístico, estabilização e arquivos CSV/LaTeX.
                </div>
                """,
                unsafe_allow_html=True
            )
    with p2:
        with st.container(border=True):
            panel_header("Modelos comparados", "baseline")
            st.markdown(
                """
                <div class="small-muted">
                    O baseline é o Q-Learning tradicional. O modelo proposto adiciona uma memória somática
                    atualizada por sinais discretos de aceitação, correção e rejeição. O estado inclui luminosidade,
                    tipo de ambiente, período e atividade. Na escolha da ação,
                    o Q-Learning Somático combina Q normalizado e memória somática por meio da utilidade U'(s,a).
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown(
    """
    <div class="footer">
        Simulador experimental para avaliação comparativa entre Q-Learning tradicional e Q-Learning Somático.
    </div>
    """,
    unsafe_allow_html=True
)

