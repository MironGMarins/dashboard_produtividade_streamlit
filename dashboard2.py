# ==============================================================================
# IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
from datetime import date, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(layout="wide", page_title="Dashboard de Produtividade")

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def aplicar_heatmap_vermelho(df):
    return df.style.background_gradient(cmap='Reds', vmin=0, vmax=10, axis=None).format(precision=0)

def aplicar_heatmap_verde(df):
    return df.style.background_gradient(cmap='Greens', vmin=0, vmax=20, axis=None).format(precision=1)

def converter_data_hibrida(series):
    datas_iso = pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')
    falhas = datas_iso.isna()
    if falhas.any():
        datas_br = pd.to_datetime(series[falhas], dayfirst=True, errors='coerce')
        datas_iso = datas_iso.fillna(datas_br)
    return datas_iso

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

# ==============================================================================
# CARREGAMENTO DE DADOS
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados_completos():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.readonly"]
    try:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    except: return None, None, None, None, None, None

    client = gspread.authorize(creds)
    url = st.secrets.get("SHEET_URL")
    spreadsheet = client.open_by_url(url)
    
    # 1. TAREFAS
    try:
        ws_dados = spreadsheet.worksheet("Total BaseCamp para Notas")
        df_dados = pd.DataFrame(ws_dados.get_all_records())
    except: df_dados = pd.DataFrame()

    # 2. EQUIPES
    try:
        ws_eq = spreadsheet.worksheet("Equipes")
        df_equipe = pd.DataFrame(ws_eq.get_all_records())
    except: df_equipe = pd.DataFrame()

    # 3. LIDERANÇA
    try:
        ws_lid = spreadsheet.worksheet("Liderança")
        df_lideranca = pd.DataFrame(ws_lid.get_all_records())
        if not df_lideranca.empty:
            for col in df_lideranca.columns: df_lideranca[col] = df_lideranca[col].astype(str).str.strip()
    except: df_lideranca = pd.DataFrame()

    # 4. PONTUAÇÃO (USANDO A LÓGICA DO SEU ARQUIVO DE TESTE)
    df_ng = pd.DataFrame(); df_nl = pd.DataFrame()
    try:
        ws_notas = spreadsheet.worksheet("Notas")
        raw_data = ws_notas.get_all_values()
        if raw_data:
            indice_quebra = -1
            for i, row in enumerate(raw_data):
                if not any(str(cell).strip() for cell in row):
                    indice_quebra = i; break
            
            # Tabela 1
            if indice_quebra == -1: 
                t1 = raw_data
            else:
                t1 = raw_data[:indice_quebra]

            if len(t1) > 1:
                headers = tornar_colunas_unicas(t1[0])
                df_ng = pd.DataFrame(t1[1:], columns=headers)

            # Tabela 2
            if indice_quebra != -1:
                start_t2 = -1
                for j in range(indice_quebra, len(raw_data)):
                    if any(str(cell).strip() for cell in raw_data[j]):
                        start_t2 = j; break
                if start_t2 != -1 and len(raw_data) > start_t2 + 1:
                    t2 = raw_data[start_t2:]
                    headers2 = tornar_colunas_unicas(t2[0])
                    df_nl = pd.DataFrame(t2[1:], columns=headers2)
    except: pass

    # 5. HISTÓRICO
    try:
        ws_hist = spreadsheet.worksheet("HistoricoDiario")
        df_hist = pd.DataFrame(ws_hist.get_all_records())
        if not df_hist.empty:
            df_hist.columns = df_hist.columns.astype(str).str.strip()
            if 'Data Final' in df_hist.columns: df_hist.rename(columns={'Data Final': 'Data'}, inplace=True)
            if 'Data' in df_hist.columns:
                df_hist['Data'] = converter_data_hibrida(df_hist['Data'])
                df_hist.dropna(subset=['Data'], inplace=True)
                df_hist.sort_values('Data', inplace=True)
                for col in ['Total_Fechadas', 'Total_Tarefas']:
                    if col in df_hist.columns: df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce').fillna(0)
            else: df_hist = pd.DataFrame()
    except: df_hist = pd.DataFrame()

    # --- PROCESSAMENTO TAREFAS ---
    if not df_dados.empty:
        df_dados.columns = df_dados.columns.astype(str).str.strip()
        for col in ['Data Inicial', 'Data Final']:
            if col in df_dados.columns: df_dados[col] = converter_data_hibrida(df_dados[col])
        
        if 'Data Final' in df_dados.columns:
            df_dados['Status_Tarefa'] = np.where(df_dados['Data Final'].isnull(), 'Aberto', 'Executado')
            df_dados['Data_Filtro'] = df_dados['Data Final'].fillna(pd.Timestamp.now().normalize())
        else:
            df_dados['Status_Tarefa'] = 'Desconhecido'; df_dados['Data_Filtro'] = pd.Timestamp.now()

        if 'Encarregado' in df_dados.columns: df_dados['Encarregado'] = df_dados['Encarregado'].astype(str).str.strip().replace('', 'Em Branco')
        else: df_dados['Encarregado'] = 'Em Branco'
        
        col_tk = 'Nome Task' if 'Nome Task' in df_dados.columns else 'Tarefa'
        if col_tk in df_dados.columns: df_dados['Nome Task'] = df_dados[col_tk].astype(str).str.strip()
        else: df_dados['Nome Task'] = 'Sem Nome'

        if not df_equipe.empty:
            df_dados = pd.merge(df_dados, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
            if 'Status' in df_dados.columns: df_dados.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)
            if 'Status_Funcionario' in df_dados.columns: df_dados['Status_Funcionario'].fillna('Outros', inplace=True)

        df_dados['Mes_Ano'] = df_dados['Data_Filtro'].dt.strftime('%m/%Y')
        meses_pt = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
        df_dados['Mes_Ano_Abrev'] = df_dados['Data_Filtro'].dt.month.map(meses_pt) + '/' + df_dados['Data_Filtro'].dt.strftime('%y')
        df_dados['Dia'] = df_dados['Data_Filtro'].dt.day
        df_dados['Dia_Semana'] = df_dados['Data_Filtro'].dt.dayofweek 
        df_dados['Nome Dia Semana'] = df_dados['Dia_Semana'].map({0:'seg', 1:'ter', 2:'qua', 3:'qui', 4:'sex', 5:'sab', 6:'dom'})
        df_dados['Semana do Mês'] = (df_dados['Data_Filtro'].dt.day - 1) // 7 + 1
        df_dados['Data_Sexta'] = df_dados['Data_Filtro'] - pd.to_timedelta(df_dados['Dia_Semana'], unit='d') + pd.to_timedelta(4, unit='d')
        df_dados['Semana_Ref'] = df_dados['Data_Sexta'].dt.strftime('%d/%m/%Y')

    return df_dados, df_ng, df_nl, df_lideranca, df_hist, df_equipe

# ==============================================================================
# GRÁFICOS
# ==============================================================================
def criar_grafico_tarefas_por_mes(df):
    if df.empty: return go.Figure().update_layout(title="Sem dados", template='plotly_white')
    df_agg = df.groupby(['Mes_Ano_Abrev', 'Mes_Ano']).agg(Contagem=('Status_Tarefa', 'count')).reset_index().sort_values('Mes_Ano')
    fig = px.bar(df_agg, x='Mes_Ano_Abrev', y='Contagem', text='Contagem', title="<b>Tarefas por Mês</b>", color='Contagem', color_continuous_scale='Blues')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False)
    return fig

def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Evolução Temporal (Principal)</b>")
    ordem_dias = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']
    
    def recortar_zeros_pontas(series_dados):
        lista = series_dados.tolist(); i_i = -1; i_f = -1
        for i, v in enumerate(lista):
            if v > 0:
                if i_i == -1: i_i = i
                i_f = i
        if i_i == -1: return [np.nan] * len(lista)
        return [v if i_i <= i <= i_f else np.nan for i, v in enumerate(lista)]
    
    # Dados Agrupados Globais
    df_dia_total = df.groupby('Dia')['Status_Tarefa'].count().reset_index(name='Contagem').sort_values('Dia')
    df_semana_total = df.groupby('Semana do Mês')['Status_Tarefa'].count().reset_index(name='Contagem').sort_values('Semana do Mês')
    df_diasemana_full = df.groupby(['Mes_Ano', 'Mes_Ano_Abrev', 'Semana do Mês', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    df_diasemana_total = df.groupby('Nome Dia Semana')['Status_Tarefa'].count().reindex(ordem_dias).fillna(0).reset_index(name='Contagem')
    df_diasemana_total['Contagem'] = recortar_zeros_pontas(df_diasemana_total['Contagem'])
    
    mes_map = df[['Mes_Ano', 'Mes_Ano_Abrev']].drop_duplicates().sort_values('Mes_Ano')
    opcoes_meses = mes_map['Mes_Ano_Abrev'].tolist()

    fig = go.Figure()
    
    # Traces Globais
    fig.add_trace(go.Scatter(x=df_dia_total['Dia'], y=df_dia_total['Contagem'], name='Global (Dias)', visible=True, line=dict(color='royalblue', width=4), mode='lines+markers+text', text=df_dia_total['Contagem'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df_semana_total['Semana do Mês'], y=df_semana_total['Contagem'], name='Global (Semanas)', visible=False, line=dict(color='royalblue', width=4), mode='lines+markers+text', text=df_semana_total['Contagem'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df_diasemana_total['Nome Dia Semana'], y=df_diasemana_total['Contagem'], name='Global (Dia Semana)', visible=False, line=dict(color='royalblue', width=4), mode='lines+markers+text', text=df_diasemana_total['Contagem'], textposition='top center'))

    offset_dia = 3
    for mes in opcoes_meses:
        d = df[df['Mes_Ano_Abrev'] == mes].groupby('Dia')['Status_Tarefa'].count().reset_index(name='Contagem').sort_values('Dia')
        fig.add_trace(go.Scatter(x=d['Dia'], y=d['Contagem'], name=mes, visible=True, mode='lines+markers+text', text=d['Contagem'], textposition='top center'))
    count_dia = len(opcoes_meses)

    offset_semana = offset_dia + count_dia
    for mes in opcoes_meses:
        d = df[df['Mes_Ano_Abrev'] == mes].groupby('Semana do Mês')['Status_Tarefa'].count().reset_index(name='Contagem').sort_values('Semana do Mês')
        fig.add_trace(go.Scatter(x=d['Semana do Mês'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem'], textposition='top center'))
    count_semana = len(opcoes_meses)

    offset_diasemana = offset_semana + count_semana
    diasemana_trace_map = [] 
    for mes in opcoes_meses:
        d_mes = df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes].groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index(name='Contagem')
        d_mes['Contagem'] = recortar_zeros_pontas(d_mes['Contagem'])
        fig.add_trace(go.Scatter(x=d_mes['Nome Dia Semana'], y=d_mes['Contagem'], name=f"{mes} Agregado", visible=False, mode='lines+markers+text', text=d_mes['Contagem'], textposition='top center'))
        
        semanas_do_mes = sorted(df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes]['Semana do Mês'].unique())
        for sem in semanas_do_mes:
            d_sem = df_diasemana_full[(df_diasemana_full['Mes_Ano_Abrev'] == mes) & (df_diasemana_full['Semana do Mês'] == sem)].groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index(name='Contagem')
            d_sem['Contagem'] = recortar_zeros_pontas(d_sem['Contagem'])
            fig.add_trace(go.Scatter(x=d_sem['Nome Dia Semana'], y=d_sem['Contagem'], name=f"{mes} Sem {sem}", visible=False, mode='lines+markers+text', text=d_sem['Contagem'], textposition='top center'))
        diasemana_trace_map.append({'mes': mes, 'num_semanas': len(semanas_do_mes)})

    total_traces = len(fig.data)
    
    # Botões
    vis_dia = [False]*total_traces; vis_dia[0] = True
    for i in range(count_dia): vis_dia[offset_dia + i] = True
    buttons_dia = [dict(label="Global (Meses)", method="update", args=[{"visible": vis_dia}])]
    for i, mes in enumerate(opcoes_meses):
        vis = [False]*total_traces; vis[offset_dia + i] = True
        buttons_dia.append(dict(label=mes, method="update", args=[{"visible": vis}]))

    vis_sem = [False]*total_traces; vis_sem[1] = True
    for i in range(count_semana): vis_sem[offset_semana + i] = True
    buttons_sem = [dict(label="Global (Meses)", method="update", args=[{"visible": vis_sem}])]
    for i, mes in enumerate(opcoes_meses):
        vis = [False]*total_traces; vis[offset_semana + i] = True
        buttons_sem.append(dict(label=mes, method="update", args=[{"visible": vis}]))

    vis_ds = [False]*total_traces; vis_ds[2] = True
    idx = offset_diasemana
    for item in diasemana_trace_map: vis_ds[idx] = True; idx += 1 + item['num_semanas']
    buttons_ds = [dict(label="Global (Médias)", method="update", args=[{"visible": vis_ds}])]
    temp_idx = offset_diasemana
    for item in diasemana_trace_map:
        vis_mes = [False]*total_traces; vis_mes[temp_idx] = True
        buttons_ds.append(dict(label=f"{item['mes']} Agregado", method="update", args=[{"visible": vis_mes}]))
        temp_idx += 1
        for s in range(item['num_semanas']):
             vis_s = [False]*total_traces; vis_s[temp_idx] = True
             buttons_ds.append(dict(label=f"{item['mes']} Sem {s+1}", method="update", args=[{"visible": vis_s}]))
             temp_idx += 1

    for i in range(total_traces): fig.data[i].visible = vis_dia[i]

    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", x=0.99, y=1.2, buttons=[
            dict(label="Dia do Mês", method="update", args=[{"visible": vis_dia}, {"updatemenus[1].buttons": buttons_dia, "xaxis.title": "Dia", "xaxis.type": "linear"}]),
            dict(label="Semana do Mês", method="update", args=[{"visible": vis_sem}, {"updatemenus[1].buttons": buttons_sem, "xaxis.title": "Semana", "xaxis.type": "linear"}]),
            dict(label="Dia da Semana", method="update", args=[{"visible": vis_ds}, {"updatemenus[1].buttons": buttons_ds, "xaxis.title": "Dia da Semana", "xaxis.type": "category", "xaxis.categoryarray": ordem_dias}])
        ]), dict(direction="down", x=0.01, y=1.2, showactive=True, buttons=buttons_dia)],
        title="<b>Evolução Temporal (Principal)</b>", margin=dict(t=140), template='plotly_white'
    )
    fig.update_xaxes(type='linear')
    return fig

def criar_graficos_volume_detalhado(df):
    if df.empty: return go.Figure(), go.Figure()
    v_p = df['Encarregado'].value_counts().reset_index(); v_p.columns=['Encarregado','Qtd']
    v_p = v_p.sort_values('Qtd', ascending=True) 
    f1 = px.bar(v_p, x='Qtd', y='Encarregado', orientation='h', text='Qtd', title="<b>Tarefas por Pessoa</b>")
    f1.update_layout(template='plotly_white')
    v_s = df['Status_Tarefa'].value_counts().reset_index(); v_s.columns=['Status','Qtd']
    f2 = px.pie(v_s, names='Status', values='Qtd', title="<b>Status Geral</b>", hole=0.4, color_discrete_map={'Executado':'blue','Aberto':'red'})
    f2.update_traces(textinfo='value+percent')
    return f1, f2

def criar_grafico_historico_semanal(df_hist, semana_str=None):
    if df_hist is None or df_hist.empty: return go.Figure().update_layout(title="Sem dados", template='plotly_white'), None
    if semana_str and semana_str != "Todos":
        try: sexta = pd.to_datetime(semana_str, format='%d/%m/%Y'); segunda = sexta - timedelta(days=4); titulo = f"<b>Progresso ({segunda.strftime('%d/%m')} - {sexta.strftime('%d/%m')})</b>"
        except: segunda = pd.Timestamp.min; sexta = pd.Timestamp.max; titulo = "Erro"
    else:
        hoje = pd.Timestamp.now().normalize(); segunda = hoje - timedelta(days=hoje.dayofweek); sexta = segunda + timedelta(days=4); titulo = "<b>Progresso Atual</b>"
    df_filt = df_hist[(df_hist['Data'] >= segunda) & (df_hist['Data'] <= sexta)].copy()
    if df_filt.empty: return go.Figure().update_layout(title=f"{titulo}<br><i>Sem dados.</i>", template='plotly_white'), None
    df_filt['Data'] = df_filt['Data'].dt.normalize(); df_filt = df_filt.drop_duplicates(subset=['Data'], keep='last').sort_values('Data')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filt['Data'], y=df_filt['Total_Tarefas'], name='Total', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=df_filt['Data'], y=df_filt['Total_Fechadas'], name='Fechadas', line=dict(color='green', width=3)))
    fig.update_layout(title=titulo, template='plotly_white', legend=dict(orientation="h", y=1.1))
    dias = pd.date_range(start=segunda, end=sexta, freq='D')
    fig.update_xaxes(tickmode='array', tickvals=dias, ticktext=[d.strftime('%d/%m') for d in dias]); fig.update_yaxes(rangemode="tozero")
    return fig, df_filt.iloc[-1]

def criar_grafico_historico_mensal(df_hist):
    if df_hist is None or df_hist.empty: return go.Figure().update_layout(title="Sem dados", template='plotly_white'), None
    hoje = pd.Timestamp.now(); inicio = hoje.replace(day=1); fim = inicio + pd.DateOffset(months=1)
    df_mes = df_hist[(df_hist['Data'] >= inicio) & (df_hist['Data'] < fim)].copy()
    if df_mes.empty: return go.Figure().update_layout(title="Sem dados mês", template='plotly_white'), None
    df_mes['Data'] = df_mes['Data'].dt.normalize(); df_mes = df_mes.drop_duplicates(subset=['Data'], keep='last').sort_values('Data')
    df_mes['Delta_F'] = df_mes['Total_Fechadas'].diff().fillna(df_mes['Total_Fechadas'])
    df_mes.loc[df_mes['Delta_F'] < 0, 'Delta_F'] = df_mes.loc[df_mes['Delta_F'] < 0, 'Total_Fechadas']
    df_mes['Acum_F'] = df_mes['Delta_F'].cumsum()
    df_mes['Acum_T'] = df_mes['Acum_F'] + (df_mes['Total_Tarefas'] - df_mes['Total_Fechadas'])
    meses_pt = {1:'Janeiro', 2:'Fevereiro', 3:'Março', 4:'Abril', 5:'Maio', 6:'Junho', 7:'Julho', 8:'Agosto', 9:'Setembro', 10:'Outubro', 11:'Novembro', 12:'Dezembro'}
    nome_mes = meses_pt.get(inicio.month, inicio.strftime('%B'))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mes['Data'], y=df_mes['Acum_T'], name='Total', line=dict(color='red'))); fig.add_trace(go.Scatter(x=df_mes['Data'], y=df_mes['Acum_F'], name='Fechadas', line=dict(color='green')))
    fig.update_layout(title=f"<b>Evolução Mensal ({nome_mes})</b>", template='plotly_white', legend=dict(orientation="h", y=1.1))
    return fig, df_mes.iloc[-1]

# ==============================================================================
# APP
# ==============================================================================
st.title("Dashboard de Produtividade")
df_tarefas, df_ng, df_nl, df_lideranca, df_hist, df_equipes = carregar_dados_completos()

# --- LIMPEZA (USANDO A LÓGICA DO SEU ARQUIVO DE TESTE) ---
def limpar_df_pontuacao(df):
    if df.empty: return df
    for col in df.columns:
        if col.lower() != "encarregado":
            try:
                # Remove sufixo .1 para tentar converter
                data_limpa = str(col).split('.')[0]
                pd.to_datetime(data_limpa, dayfirst=True) 
                
                df[col] = df[col].astype(str).str.replace(',', '.').replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except: pass
    if "Encarregado" in df.columns:
        df["Encarregado"] = df["Encarregado"].astype(str).str.strip()
        # Merge com equipe para manter a lógica original do dashboard de filtrar status
        if not df_equipes.empty:
            df = pd.merge(df, df_equipes, how='left', left_on='Encarregado', right_on='Nome')
            if 'Status' in df.columns: df.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)
    return df

# Aplica limpeza aqui fora (e apenas aqui)
df_ng = limpar_df_pontuacao(df_ng)
df_nl = limpar_df_pontuacao(df_nl)

# --- DEFINIÇÃO DA REGUA DE TEMPO (SLIDER) ---
todas_datas = []

# 1. Datas vindas das Tarefas
if df_tarefas is not None and not df_tarefas.empty:
    if 'Data_Filtro' in df_tarefas.columns:
        todas_datas.extend(df_tarefas['Data_Filtro'].dropna().tolist())

# 2. Datas vindas da Pontuação
cols_ng = df_ng.columns if df_ng is not None else []
cols_nl = df_nl.columns if df_nl is not None else []

for col_name in list(cols_ng) + list(cols_nl):
    try:
        data_limpa = str(col_name).strip().split('.')[0] # Strip adicionado para segurança
        dt = pd.to_datetime(data_limpa, dayfirst=True, errors='coerce')
        if pd.notnull(dt):
            todas_datas.append(dt)
    except: pass

if todas_datas:
    min_d = min(todas_datas).date()
    max_d = max(todas_datas).date()
else:
    # Fallback seguro para evitar que o dashboard quebre
    min_d = date.today() - timedelta(days=30)
    max_d = date.today()

with st.sidebar:
    st.image("media portal logo.png", width=200)
    st.header("Filtros")
    l_enc = sorted(df_tarefas['Encarregado'].unique()) if df_tarefas is not None else []
    sel_enc = st.multiselect("Encarregado", ["Todos"]+l_enc, default="Todos")
    l_cont = ["Todos"] + sorted(df_tarefas['Status_Funcionario'].unique().tolist()) if df_tarefas is not None and 'Status_Funcionario' in df_tarefas.columns else ["Todos"]
    sel_cont = st.selectbox("Status (Contrato)", l_cont)
    
    d_range = st.slider("Período (Pontuação/Geral)", min_value=min_d, max_value=max_d, value=(min_d, max_d))

if df_tarefas is not None:
    df_f = df_tarefas.copy()
    if "Todos" not in sel_enc: df_f = df_f[df_f['Encarregado'].isin(sel_enc)]
    if sel_cont != "Todos" and 'Status_Funcionario' in df_f.columns: df_f = df_f[df_f['Status_Funcionario'] == sel_cont]
    
    # Filtro global de data para tarefas
    ts_s = pd.to_datetime(d_range[0])
    ts_e = pd.to_datetime(d_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    df_f_per = df_f[(df_f['Data_Filtro'] >= ts_s) & (df_f['Data_Filtro'] <= ts_e)]

    fn = sel_enc if "Todos" not in sel_enc else None
    fs = sel_cont if sel_cont != "Todos" else None

    c1, c2 = st.columns(2)
    c1.metric("Total Tarefas", len(df_f_per))
    c2.metric("Período", f"{(ts_e - ts_s).days} dias")
    st.divider()

    t1, t2, t3, t4, t5 = st.tabs(["Semana", "Mês", "Backlog", "Geral", "🏆 Pontuação"])

    with t1: # SEMANA
        st.subheader("Acompanhamento Semanal")
        l_sem = []
        idx = 0
        if 'Semana_Ref' in df_tarefas.columns:
            raw_weeks = df_tarefas['Semana_Ref'].unique()
            l_sem = sorted(raw_weeks, key=lambda x: pd.to_datetime(x, dayfirst=True), reverse=True)
            hoje = pd.Timestamp.now().normalize(); sex = hoje - pd.to_timedelta(hoje.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
            str_sex = sex.strftime('%d/%m/%Y')
            if str_sex in l_sem: idx = l_sem.index(str_sex)
        
        sel_sem = st.selectbox("📅 Semana:", l_sem, index=idx)
        f_h, m_h = criar_grafico_historico_semanal(df_hist, sel_sem)
        st.plotly_chart(f_h, use_container_width=True)
        st.markdown("---")
        df_fs = df_f[df_f['Semana_Ref'] == sel_sem]
        df_ex = df_fs[df_fs['Status_Tarefa'] == 'Executado']
        if not df_ex.empty:
            try:
                piv = pd.pivot_table(df_ex, index='Encarregado', columns='Nome Dia Semana', values='ID', aggfunc='count', fill_value=0)
                piv = piv.reindex(columns=['seg','ter','qua','qui','sex','sab','dom'], fill_value=0)
                st.write("##### Tarefas por Dia"); st.dataframe(aplicar_heatmap_vermelho(piv), use_container_width=True)
            except: pass
        st.write("##### Lista Detalhada")
        cols = ['Nome Task', 'Link', 'Data Final']; cfg = {"Link": st.column_config.LinkColumn("Link", display_text="Abrir"), "Data Final": st.column_config.DateColumn(format="DD/MM/YYYY")}
        for enc in sorted(df_fs['Encarregado'].unique()):
            d_e = df_fs[df_fs['Encarregado'] == enc]
            ab = d_e[d_e['Status_Tarefa'] == 'Aberto']; fe = d_e[d_e['Status_Tarefa'] == 'Executado']
            with st.expander(f"{enc} | Total: {len(d_e)} (🔴 {len(ab)} | 🟢 {len(fe)})"):
                if not ab.empty: st.caption("🔴 Abertas"); st.dataframe(ab[cols], use_container_width=True, hide_index=True, column_config=cfg)
                if not fe.empty: st.caption("🟢 Fechadas"); st.dataframe(fe[cols], use_container_width=True, hide_index=True, column_config=cfg)

    with t2: # MÊS
        st.subheader("Evolução Mensal")
        f_m, m_m = criar_grafico_historico_mensal(df_hist)
        if m_m is not None:
            t = m_m.get('Acumulado_Total', 0); f = m_m.get('Acumulado_Fechadas', 0); g = m_m.get('Backlog', 0)
            c1, c2, c3 = st.columns(3); c1.metric("Acumulado", f"{t:.0f}"); c2.metric("Gap", f"{g:.0f}"); c3.metric("Fechadas", f"{f:.0f}")
        st.plotly_chart(f_m, use_container_width=True)
        st.markdown("---")
        hoje = pd.Timestamp.now(); mes_str = hoje.strftime('%m/%Y')
        meses_pt = {1:'Janeiro', 2:'Fevereiro', 3:'Março', 4:'Abril', 5:'Maio', 6:'Junho', 7:'Julho', 8:'Agosto', 9:'Setembro', 10:'Outubro', 11:'Novembro', 12:'Dezembro'}
        nome_mes_pt = meses_pt.get(hoje.month, hoje.strftime('%B'))
        df_bm = df_f[df_f['Mes_Ano'] == mes_str]
        if not df_bm.empty:
            df_em = df_bm[df_bm['Status_Tarefa'] == 'Executado'].copy()
            if not df_em.empty:
                df_em['Dia'] = df_em['Data_Filtro'].dt.day
                try:
                    piv_m = pd.pivot_table(df_em, index='Encarregado', columns='Dia', values='ID', aggfunc='count', fill_value=0)
                    piv_m = piv_m[sorted(piv_m.columns)]
                    st.write(f"##### Tarefas Concluídas em {nome_mes_pt}"); st.dataframe(aplicar_heatmap_vermelho(piv_m), use_container_width=True)
                except: pass
            st.markdown("---")
            st.write("##### Lista Detalhada de Tarefas")
            for enc in sorted(df_bm['Encarregado'].unique()):
                d_e = df_bm[df_bm['Encarregado'] == enc]
                ab = d_e[d_e['Status_Tarefa'] == 'Aberto']; fe = d_e[d_e['Status_Tarefa'] == 'Executado']
                with st.expander(f"{enc} | Mês: {len(d_e)} (🔴 {len(ab)} | 🟢 {len(fe)})"):
                     if not ab.empty: st.caption("🔴 Abertas"); st.dataframe(ab[['Nome Task','Link']], use_container_width=True, hide_index=True, column_config={"Link": st.column_config.LinkColumn("Link", display_text="Abrir")})
                     if not fe.empty: st.caption("🟢 Fechadas"); st.dataframe(fe[['Nome Task','Link','Data Final']], use_container_width=True, hide_index=True, column_config={"Link": st.column_config.LinkColumn("Link", display_text="Abrir"), "Data Final": st.column_config.DateColumn(format="DD/MM/YYYY")})

    with t3: # BACKLOG
        st.subheader("Gestão de Backlog")
        df_bk = df_tarefas[df_tarefas['Status_Tarefa'] == 'Aberto'].copy()
        if "Todos" not in sel_enc: df_bk = df_bk[df_bk['Encarregado'].isin(sel_enc)]
        st.metric("Pendente Global", len(df_bk))
        sem = df_bk[df_bk['Encarregado'] == 'Em Branco']; com = df_bk[df_bk['Encarregado'] != 'Em Branco']
        cols = {'Link': st.column_config.LinkColumn("Link", display_text="Abrir"), 'Data Inicial': st.column_config.DateColumn(format="DD/MM/YYYY")}
        with st.expander(f"⚫ Sem Dono ({len(sem)})", expanded=True):
            if not sem.empty: st.dataframe(sem[['Nome Task','Data Inicial','Link']], use_container_width=True, hide_index=True, column_config=cols)
        with st.expander(f"🔴 Com Responsável ({len(com)})"):
            if not com.empty: st.dataframe(com[['Encarregado','Nome Task','Data Inicial','Link']].sort_values('Encarregado'), use_container_width=True, hide_index=True, column_config=cols)
        df_c = df_f_per[df_f_per['Status_Tarefa'] == 'Executado']
        with st.expander(f"🟢 Arquivo Fechado ({len(df_c)})"):
             if not df_c.empty: st.dataframe(df_c[['Encarregado','Nome Task','Data Final','Link']].sort_values('Data Final', ascending=False), use_container_width=True, hide_index=True, column_config={'Link': st.column_config.LinkColumn("Link", display_text="Abrir"), 'Data Final': st.column_config.DateColumn(format="DD/MM/YYYY")})

    with t4: # GERAL
        st.subheader("Análise de Volume")
        c_top1, c_top2 = st.columns(2)
        with c_top1: st.plotly_chart(criar_grafico_tarefas_por_mes(df_f), use_container_width=True)
        with c_top2: st.plotly_chart(criar_grafico_principal(df_f), use_container_width=True)
        st.markdown("---")
        fp, fs = criar_graficos_volume_detalhado(df_f_per)
        c_bot1, c_bot2 = st.columns(2)
        c_bot1.plotly_chart(fp, use_container_width=True)
        c_bot2.plotly_chart(fs, use_container_width=True)

    with t5: # PONTUAÇÃO
        st.subheader("Ranking de Pontuação")

        # --- AQUI ESTA A CORREÇÃO PRINCIPAL: COMPARAÇÃO APENAS POR .date() ---
        def calcular_soma(df, nome_tabela):
            if df is None or df.empty: return pd.DataFrame()
            cols_validas = []
            
            # Pega datas LIMPAS do slider (sem hora)
            d_inicio = d_range[0] # já é date
            d_fim = d_range[1]    # já é date

            for c in df.columns:
                try:
                    data_limpa = str(c).strip().split('.')[0]
                    dt = pd.to_datetime(data_limpa, dayfirst=True, errors='coerce')
                    
                    # Compara usando apenas a parte da DATA (dia/mês/ano)
                    # Isso ignora qualquer problema de hora 00:00 vs 23:59
                    if pd.notnull(dt) and d_inicio <= dt.date() <= d_fim:
                        cols_validas.append(c)
                except: pass
            
            if not cols_validas:
                return pd.DataFrame()
            
            res = df.copy()
            if fn: res = res[res['Encarregado'].isin(fn)]
            if fs and 'Status_Funcionario' in res.columns: res = res[res['Status_Funcionario'] == fs]
            
            res['Total'] = res[cols_validas].sum(axis=1)
            return res

        def plot(df, tit):
            if df.empty: 
                st.info(f"{tit}: Sem dados no período.")
                return
            
            df = df[df['Total'] > 0].sort_values('Total', ascending=True)
            if df.empty:
                st.info(f"{tit}: Sem pontuação maior que zero no período.")
                return

            fig = px.bar(
                df, 
                x='Total', 
                y='Encarregado', 
                text='Total', 
                orientation='h', 
                title=f"<b>{tit}</b>", 
                color='Total', 
                color_continuous_scale='Blues'
            )
            fig.update_layout(template='plotly_white', coloraxis_showscale=False, yaxis_title=None, height=300+(len(df)*25))
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        rank_g = calcular_soma(df_ng, "Geral")
        plot(rank_g, "1. Pontuação Geral")
        st.markdown("---")

        pts_liderados = calcular_soma(df_nl, "Liderados")
        rank_l = pd.DataFrame()

        if not pts_liderados.empty and not df_lideranca.empty:
            df_lideranca['Liderado'] = df_lideranca['Liderado'].astype(str).str.strip().str.lower()
            pts_liderados['Encarregado_Lower'] = pts_liderados['Encarregado'].astype(str).str.strip().str.lower()
            
            merged = pd.merge(df_lideranca, pts_liderados, left_on='Liderado', right_on='Encarregado_Lower', how='inner')
            rank_l = merged.groupby('Lider')['Total'].sum().reset_index().rename(columns={'Lider': 'Encarregado'})
            
            if fn: rank_l = rank_l[rank_l['Encarregado'].isin(fn)]
            
        plot(rank_l, "2. Pontuação de Líderes")
        st.markdown("---")

        dfs_total = []
        if not rank_g.empty: dfs_total.append(rank_g[['Encarregado', 'Total']])
        if not rank_l.empty: dfs_total.append(rank_l[['Encarregado', 'Total']])
        
        if dfs_total:
            rank_t = pd.concat(dfs_total).groupby('Encarregado')['Total'].sum().reset_index()
            plot(rank_t, "3. Pontuação Total")
        else:
            st.info("Sem dados para o Total.")

else:
    st.error("Não foi possível carregar os dados. Verifique o Gerenciador.")