# ==============================================================================
# ARQUIVO: teste_pontuacao.py
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(layout="wide", page_title="Diagnóstico Pontuação")

# --- FUNÇÃO DE SEGURANÇA: RENOMEAR DUPLICATAS ---
def tornar_colunas_unicas(lista_colunas):
    """Recebe ['Data', 'Data', 'Nome'] e retorna ['Data', 'Data.1', 'Nome']"""
    seen = {}
    nova_lista = []
    for col in lista_colunas:
        c = str(col).strip()
        if c in seen:
            seen[c] += 1
            nova_lista.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            nova_lista.append(c)
    return nova_lista

# --- 1. CONEXÃO ---
@st.cache_resource
def conectar():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        client = gspread.authorize(creds)
        return client.open_by_url(st.secrets.get("SHEET_URL"))
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        return None

# --- 2. LEITURA DA PLANILHA ---
def carregar_dados():
    ss = conectar()
    if not ss: return None, None, None

    # LIDERANÇA
    try:
        df_mapa = pd.DataFrame(ss.worksheet("Liderança").get_all_records())
        for c in df_mapa.columns: df_mapa[c] = df_mapa[c].astype(str).str.strip()
    except: df_mapa = pd.DataFrame()

    # NOTAS (SEPARAÇÃO)
    df_g = pd.DataFrame()
    df_l = pd.DataFrame()
    
    try:
        ws = ss.worksheet("Notas")
        raw = ws.get_all_values()
        
        if raw:
            # Acha o buraco
            idx_corte = -1
            for i, row in enumerate(raw):
                if not any(str(cell).strip() for cell in row):
                    idx_corte = i
                    break
            
            # Tabela 1
            if idx_corte == -1: 
                t1 = raw
            else: 
                t1 = raw[:idx_corte]
            
            if len(t1) > 1:
                headers = tornar_colunas_unicas(t1[0])
                df_g = pd.DataFrame(t1[1:], columns=headers)

            # Tabela 2
            if idx_corte != -1:
                start_t2 = -1
                for j in range(idx_corte, len(raw)):
                    if any(str(cell).strip() for cell in raw[j]):
                        start_t2 = j; break
                
                if start_t2 != -1 and len(raw) > start_t2 + 1:
                    t2 = raw[start_t2:]
                    headers2 = tornar_colunas_unicas(t2[0])
                    df_l = pd.DataFrame(t2[1:], columns=headers2)
                    
    except Exception as e: st.error(f"Erro ao ler: {e}")

    return df_g, df_l, df_mapa

# --- 3. LIMPEZA ---
def limpar_df_matriz(df):
    if df.empty: return df
    for col in df.columns:
        if col.lower() != "encarregado":
            try:
                # Remove sufixo .1 para tentar converter
                data_limpa = col.split('.')[0]
                pd.to_datetime(data_limpa, dayfirst=True) 
                
                df[col] = df[col].astype(str).str.replace(',', '.').replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except: pass
    if "Encarregado" in df.columns:
        df["Encarregado"] = df["Encarregado"].astype(str).str.strip()
    return df

# --- 4. INTERFACE E CÁLCULOS ---
st.title("🧪 Diagnóstico de Pontuação")

df_geral, df_lideres, df_mapa = carregar_dados()
df_geral = limpar_df_matriz(df_geral)
df_lideres = limpar_df_matriz(df_lideres)

# Slider de Data (CORREÇÃO DO ERRO NaT)
todas_datas = []
cols_possiveis = list(df_geral.columns) + list(df_lideres.columns)
for c in cols_possiveis:
    try: 
        data_limpa = c.split('.')[0]
        dt = pd.to_datetime(data_limpa, dayfirst=True, errors='coerce') # Coerce gera NaT se falhar
        if pd.notnull(dt): # Só adiciona se for data válida
            todas_datas.append(dt.date())
    except: pass

if todas_datas:
    min_d, max_d = min(todas_datas), max(todas_datas)
    rng = st.slider("Período", min_value=min_d, max_value=max_d, value=(min_d, max_d))
    ts_s, ts_e = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
else:
    st.error("Nenhuma coluna de data encontrada nas tabelas.")
    st.stop()

# --- CÁLCULOS ---
def calcular_soma(df, nome_tabela):
    if df.empty: return pd.DataFrame()
    cols_validas = []
    for c in df.columns:
        try:
            data_limpa = c.split('.')[0]
            dt = pd.to_datetime(data_limpa, dayfirst=True, errors='coerce')
            if pd.notnull(dt) and ts_s <= dt <= ts_e: cols_validas.append(c)
        except: pass
    
    if not cols_validas:
        st.warning(f"{nome_tabela}: Nenhuma coluna de data no período selecionado.")
        return pd.DataFrame()
    
    res = df.copy()
    res['Total'] = res[cols_validas].sum(axis=1)
    return res

# 1. Geral
rank_g = calcular_soma(df_geral, "Geral")

# 2. Líderes (Merge)
pts_liderados = calcular_soma(df_lideres, "Liderados")
rank_l = pd.DataFrame()

if not pts_liderados.empty and not df_mapa.empty:
    df_mapa['Liderado'] = df_mapa['Liderado'].str.lower()
    pts_liderados['Encarregado'] = pts_liderados['Encarregado'].str.lower()
    
    merged = pd.merge(df_mapa, pts_liderados, left_on='Liderado', right_on='Encarregado', how='inner')
    rank_l = merged.groupby('Lider')['Total'].sum().reset_index().rename(columns={'Lider':'Encarregado'})

# 3. Total
dfs = []
if not rank_g.empty: dfs.append(rank_g[['Encarregado', 'Total']])
if not rank_l.empty: dfs.append(rank_l[['Encarregado', 'Total']])
rank_t = pd.concat(dfs).groupby('Encarregado')['Total'].sum().reset_index() if dfs else pd.DataFrame()

# --- GRÁFICOS ---
def plot(df, tit):
    if df.empty: return
    df = df[df['Total'] > 0].sort_values('Total', ascending=True)
    fig = px.bar(df, x='Total', y='Encarregado', text='Total', orientation='h', title=f"<b>{tit}</b>", color='Total', color_continuous_scale='Blues')
    fig.update_layout(template='plotly_white', coloraxis_showscale=False, yaxis_title=None, height=300+(len(df)*25))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
plot(rank_g, "1. Pontuação Geral")
st.write("---")
plot(rank_l, "2. Pontuação de Líderes")
st.write("---")
plot(rank_t, "3. Pontuação Total")