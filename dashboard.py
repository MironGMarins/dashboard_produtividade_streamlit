# ==============================================================================
# IMPORTS CONSOLIDADOS
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
st.set_page_config(
    layout="wide",
    page_title="Dashboard de Produtividade"
)

# ==============================================================================
# FUNÇÕES AUXILIARES DE LIMPEZA E UTILS
# ==============================================================================
def converter_data_robusta(series):
    """
    Converte uma série de dados para datetime forçando o padrão brasileiro (Dia/Mês)
    para formatos ambíguos, mas aceitando ISO (Ano-Mês-Dia) corretamente.
    """
    # Garante que é string e remove espaços
    series = series.astype(str).str.strip()
    # Remove lixo comum
    series = series.replace(['nan', 'None', '', 'NaT', '0'], np.nan)
    # dayfirst=True é a chave: resolve 04/05/2025 como 04 de Maio, não 05 de Abril.
    return pd.to_datetime(series, dayfirst=True, errors='coerce')

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

def recortar_zeros_pontas(series_dados):
    """Substitui zeros iniciais e finais por NaN para gráficos de linha."""
    lista = series_dados.tolist()
    idx_inicio = -1
    idx_fim = -1
    for i, v in enumerate(lista):
        if v > 0:
            if idx_inicio == -1: idx_inicio = i
            idx_fim = i
    
    if idx_inicio == -1: return [np.nan] * len(lista)
    
    resultado = []
    for i, v in enumerate(lista):
        if i < idx_inicio or i > idx_fim: resultado.append(np.nan)
        else: resultado.append(v)
    return resultado

# ==============================================================================
# FUNÇÃO DE CARREGAMENTO DE DADOS (CACHE)
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados_completos():
    # --- Autenticação Segura ---
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=scopes)
    except (FileNotFoundError, KeyError):
        try:
            creds = Credentials.from_service_account_file("google_credentials.json", scopes=scopes)
        except Exception:
            st.error("Credenciais não encontradas em st.secrets ou arquivo local.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    client = gspread.authorize(creds)
    url_da_planilha = st.secrets.get("SHEET_URL", 'https://docs.google.com/spreadsheets/d/1juyOfIh0ZqsfJjN0p3gD8pKaAIX0R6IAPG9vysl7yWI/edit#gid=901870248')
    
    try:
        spreadsheet = client.open_by_url(url_da_planilha)
    except Exception as e:
        st.error(f"Erro ao abrir planilha: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # --- Nomes das Abas ---
    nome_aba_dados = "Total BaseCamp para Notas" 
    nome_aba_equipes = "Equipes"
    nome_aba_pontuacao = "Notas"
    nome_aba_lideranca = "Liderança"
    nome_aba_backlog = "Backlog" 
    nome_aba_source = "Total BaseCamp"
    nome_aba_historico = "HistoricoDiario"
    
    # Inicializa DataFrames vazios
    df_dados = pd.DataFrame()
    df_equipe = pd.DataFrame()
    df_notas_tabela1 = pd.DataFrame()
    df_notas_tabela2 = pd.DataFrame()
    df_lideranca = pd.DataFrame()
    df_backlog = pd.DataFrame()
    df_source = pd.DataFrame() 
    df_source_analise = pd.DataFrame()
    df_historico = pd.DataFrame()

    try:
        # --- Carregar Abas ---
        try: df_dados = pd.DataFrame(spreadsheet.worksheet(nome_aba_dados).get_all_records())
        except: pass
        try: df_equipe = pd.DataFrame(spreadsheet.worksheet(nome_aba_equipes).get_all_records()) 
        except: pass
        try: df_lideranca = pd.DataFrame(spreadsheet.worksheet(nome_aba_lideranca).get_all_records())
        except: pass
        try: df_backlog = pd.DataFrame(spreadsheet.worksheet(nome_aba_backlog).get_all_records())
        except: pass
        try: df_source = pd.DataFrame(spreadsheet.worksheet(nome_aba_source).get_all_records())
        except: pass
        try: df_historico = pd.DataFrame(spreadsheet.worksheet(nome_aba_historico).get_all_records())
        except: pass 

        # ==============================================================================
        # --- Carregar AMBAS as tabelas da aba "Notas" ---
        # ==============================================================================
        try:
            all_values_notas = spreadsheet.worksheet(nome_aba_pontuacao).get_all_values()
            primeira_linha_branca_index = -1
            for i, row in enumerate(all_values_notas):
                if not row or all(cell == '' for cell in row):
                    primeira_linha_branca_index = i
                    break
            
            if primeira_linha_branca_index == -1:
                dados_tabela_superior = all_values_notas
                dados_tabela_inferior = [] 
            else:
                dados_tabela_superior = all_values_notas[:primeira_linha_branca_index]
                dados_tabela_inferior_inicio = -1
                for i, row in enumerate(all_values_notas[primeira_linha_branca_index + 1:], start=primeira_linha_branca_index + 1):
                    if row and any(cell != '' for cell in row):
                        dados_tabela_inferior_inicio = i
                        break
                if dados_tabela_inferior_inicio != -1:
                    dados_tabela_inferior = all_values_notas[dados_tabela_inferior_inicio:]
                else:
                    dados_tabela_inferior = []

            if len(dados_tabela_superior) > 1:
                headers_sup = tornar_colunas_unicas(dados_tabela_superior[0])
                df_notas_tabela1 = pd.DataFrame(dados_tabela_superior[1:], columns=headers_sup)
            elif len(dados_tabela_superior) == 1:
                 df_notas_tabela1 = pd.DataFrame(columns=dados_tabela_superior[0])

            if len(dados_tabela_inferior) > 1:
                headers_inf = tornar_colunas_unicas(dados_tabela_inferior[0])
                df_notas_tabela2 = pd.DataFrame(dados_tabela_inferior[1:], columns=headers_inf)
            elif len(dados_tabela_inferior) == 1:
                 df_notas_tabela2 = pd.DataFrame(columns=headers_inf[0])
        except: pass
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not df_equipe.empty and 'Status' in df_equipe.columns:
        df_equipe.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)

    # --- PREPARAÇÃO DOS DADOS (df_analise) ---
    df_grafico = df_dados.copy()
    
    # REMOVIDO PESO DA LISTA DE NUMERICOS
    colunas_para_numerico = ['Pablo', 'Leonardo', 'Itiel', 'Ítalo']
    for col in colunas_para_numerico:
        if col not in df_grafico.columns: df_grafico[col] = 0
        else: df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce').fillna(0)

    # --- APLICAÇÃO DA CONVERSÃO ROBUSTA DE DATAS ---
    if 'Data Inicial' in df_grafico.columns:
        df_grafico['Data Inicial'] = converter_data_robusta(df_grafico['Data Inicial'])
    if 'Data Final' in df_grafico.columns:
        df_grafico['Data Final'] = converter_data_robusta(df_grafico['Data Final'])
        df_grafico['Status_Tarefa'] = np.where(df_grafico['Data Final'].isnull(), 'Aberto', 'Executado')
        data_hoje = pd.Timestamp.now().normalize()
        df_grafico['Data Final (aberta)'] = df_grafico['Data Final'].fillna(data_hoje)
    else:
        df_grafico['Status_Tarefa'] = 'Desconhecido'
        df_grafico['Data Final (aberta)'] = pd.Timestamp.now().normalize()
    
    if 'Encarregado' in df_grafico.columns:
        df_grafico['Encarregado'] = df_grafico['Encarregado'].astype(str).str.strip().replace('', 'Em Branco')
    else: df_grafico['Encarregado'] = 'Em Branco'

    if 'Nome Task' in df_grafico.columns:
        df_grafico['Nome Task'] = df_grafico['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
    else: df_grafico['Nome Task'] = 'Sem Nome'

    data_inicio_analise = pd.Timestamp.now().normalize()
    if 'Data Inicial' in df_grafico.columns:
        val_min = df_grafico['Data Inicial'].min()
        if pd.notna(val_min): data_inicio_analise = val_min
    
    # Extender calendário
    data_fim_analise = pd.Timestamp.now().normalize() + pd.Timedelta(days=365)
    
    # --- Tabela Calendário ---
    data_inicio_calendario = data_inicio_analise
    if not df_source.empty and 'Data Inicial' in df_source.columns:
        data_inicio_source = converter_data_robusta(df_source['Data Inicial']).min()
        if pd.notna(data_inicio_source) and data_inicio_source < data_inicio_analise:
            data_inicio_calendario = data_inicio_source

    tabela_calendario = pd.DataFrame({"Date": pd.date_range(start=data_inicio_calendario, end=data_fim_analise, freq='D')})
    tabela_calendario['Ano'] = tabela_calendario['Date'].dt.year
    meses_pt = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    tabela_calendario['Nome Mês'] = tabela_calendario['Date'].dt.month.map(meses_pt)
    tabela_calendario['Mes_Ano_Abrev'] = tabela_calendario['Nome Mês'] + '/' + tabela_calendario['Date'].dt.strftime('%y')
    tabela_calendario['Ano-Mês'] = tabela_calendario['Date'].dt.strftime('%Y-%m')
    tabela_calendario['Dia'] = tabela_calendario['Date'].dt.day
    tabela_calendario['Dia da Semana_ISO'] = tabela_calendario['Date'].dt.dayofweek
    dias_pt_map = {0: 'seg', 1: 'ter', 2: 'qua', 3: 'qui', 4: 'sex', 5: 'sab', 6: 'dom'}
    tabela_calendario['Nome Dia Semana'] = tabela_calendario['Dia da Semana_ISO'].map(dias_pt_map)
    tabela_calendario['Data_Inicio_Semana'] = tabela_calendario['Date'] - pd.to_timedelta(tabela_calendario['Dia da Semana_ISO'], unit='d')
    tabela_calendario['Data_Sexta_Feira'] = tabela_calendario['Data_Inicio_Semana'] + pd.to_timedelta(4, unit='d')
    tabela_calendario['Nome_da_Semana'] = tabela_calendario['Data_Sexta_Feira'].dt.strftime('%d/%m/%Y')
    tabela_calendario['Semana_Ano'] = tabela_calendario['Data_Sexta_Feira'].dt.strftime('%Y-%U') 
    tabela_calendario['Semana do Mês'] = (tabela_calendario['Date'].dt.dayofweek + (tabela_calendario['Date'].dt.day - 1)).floordiv(7) + 1
    tabela_calendario['Dia da Semana'] = tabela_calendario['Dia da Semana_ISO'] + 1

    df_analise_temp = pd.merge(df_grafico, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date')
    if 'Date' in df_analise_temp.columns: df_analise_temp = df_analise_temp.drop(columns=['Date'])
    
    if not df_equipe.empty:
        df_analise = pd.merge(df_analise_temp, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
        if 'Status_Funcionario' in df_analise.columns: df_analise['Status_Funcionario'].fillna('Outros', inplace=True)
    else:
        df_analise = df_analise_temp
        df_analise['Status_Funcionario'] = 'Outros'
    
    # --- PREPARAÇÃO DOS DADOS (df_source_analise) ---
    if not df_source.empty:
        df_source_proc = df_source.copy()
        df_source_proc['Data Inicial'] = converter_data_robusta(df_source_proc.get('Data Inicial', pd.Series()))
        df_source_proc['Data Final'] = converter_data_robusta(df_source_proc.get('Data Final', pd.Series()))
        df_source_proc['Status_Tarefa'] = np.where(df_source_proc['Data Final'].isnull(), 'Aberto', 'Executado')
        df_source_proc['Data Final (aberta)'] = df_source_proc['Data Final'].fillna(pd.Timestamp.now().normalize())
        
        if 'Encarregado' in df_source_proc.columns: df_source_proc['Encarregado'] = df_source_proc['Encarregado'].astype(str).str.strip().replace('', 'Em Branco')
        if 'Nome Task' in df_source_proc.columns: df_source_proc['Nome Task'] = df_source_proc['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
        
        df_source_analise = pd.merge(df_source_proc, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date')
        if 'Date' in df_source_analise.columns: df_source_analise = df_source_analise.drop(columns=['Date'])
        
        if not df_equipe.empty:
            df_source_analise = pd.merge(df_source_analise, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
            if 'Status_Funcionario' in df_source_analise.columns: df_source_analise['Status_Funcionario'].fillna('Outros', inplace=True)
    
    # --- PREPARAÇÃO DOS DADOS (df_backlog) ---
    if not df_backlog.empty:
        df_backlog['Data Inicial'] = converter_data_robusta(df_backlog.get('Data Inicial', pd.Series()))
        df_backlog['Data Final'] = converter_data_robusta(df_backlog.get('Data Final', pd.Series()))
        df_backlog['Status_Backlog'] = np.where(df_backlog['Data Final'].isnull(), 'Aberto', 'Fechado')
        if 'Encarregado' in df_backlog.columns: df_backlog['Encarregado'] = df_backlog['Encarregado'].astype(str).str.strip().replace('', 'Em Branco') 
        if 'Nome Task' in df_backlog.columns: df_backlog['Nome Task'] = df_backlog['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
        
        if not df_equipe.empty:
            df_backlog = pd.merge(df_backlog, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
            if 'Status_Funcionario' in df_backlog.columns: df_backlog['Status_Funcionario'].fillna('Outros', inplace=True)
        else:
            df_backlog['Status_Funcionario'] = 'Outros'

    # --- PREPARAÇÃO DOS DADOS (df_historico) ---
    if not df_historico.empty:
        if 'Data Final' in df_historico.columns: df_historico.rename(columns={'Data Final': 'Data'}, inplace=True)
        if 'Data' in df_historico.columns:
            df_historico['Data'] = converter_data_robusta(df_historico['Data'])
            if 'Total_Fechadas' in df_historico.columns: df_historico['Total_Fechadas'] = pd.to_numeric(df_historico['Total_Fechadas'], errors='coerce').fillna(0)
            if 'Total_Tarefas' in df_historico.columns: df_historico['Total_Tarefas'] = pd.to_numeric(df_historico['Total_Tarefas'], errors='coerce').fillna(0)
            df_historico.dropna(subset=['Data'], inplace=True)

    return df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca, df_equipe, df_backlog, df_source_analise, df_historico

# ==============================================================================
# FUNÇÕES GRÁFICAS ESTRUTURADAS
# ==============================================================================
def criar_grafico_historico_semanal(df_historico, semana_selecionada_str=None):
    if df_historico is None or df_historico.empty:
        return go.Figure().update_layout(title="Sem dados históricos", template='plotly_white'), None

    if semana_selecionada_str:
        try:
            data_referencia = pd.to_datetime(semana_selecionada_str, format='%d/%m/%Y')
            fim_semana = data_referencia 
            inicio_semana = fim_semana - pd.Timedelta(days=4)
            titulo_grafico = f"<b>Progresso ({inicio_semana.strftime('%d/%m')} a {fim_semana.strftime('%d/%m')})</b>"
        except ValueError:
            titulo_grafico = "Erro na Data"
            inicio_semana = pd.Timestamp.min; fim_semana = pd.Timestamp.max
    else:
        inicio_semana = pd.Timestamp.min; fim_semana = pd.Timestamp.max
        titulo_grafico = "Selecione uma semana"

    df_filt = df_historico[(df_historico['Data'] >= inicio_semana) & (df_historico['Data'] <= fim_semana)].sort_values('Data')
    
    if df_filt.empty:
        return go.Figure().update_layout(title=f"{titulo_grafico}<br><i>Sem registros diários para esta semana.</i>", template='plotly_white'), None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filt['Data'], y=df_filt['Total_Tarefas'], mode='lines+markers+text', name='Total', line=dict(color='red', width=3), text=df_filt['Total_Tarefas'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df_filt['Data'], y=df_filt['Total_Fechadas'], mode='lines+markers+text', name='Fechadas', line=dict(color='green', width=3), text=df_filt['Total_Fechadas'], textposition='bottom center'))
    fig.update_layout(title=titulo_grafico, template='plotly_white', legend=dict(orientation="h", y=1.1))
    
    range_semana = pd.date_range(start=inicio_semana, end=fim_semana)
    dias_pt = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']
    tick_vals = range_semana
    tick_text = [f"{dias_pt[d.dayofweek]} ({d.strftime('%d/%m')})" for d in range_semana]
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text)
    
    return fig, df_filt.iloc[-1]

def criar_grafico_historico_mensal(df_historico):
    if df_historico is None or df_historico.empty: return go.Figure(), None
    hoje = pd.Timestamp.now().normalize(); inicio = hoje.replace(day=1); fim = (inicio + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    df_mes = df_historico[(df_historico['Data'] >= inicio) & (df_historico['Data'] <= fim)].sort_values('Data')
    if df_mes.empty: return go.Figure().update_layout(title="Sem dados mês atual", template='plotly_white'), None

    df_c = df_mes.copy()
    df_c['Delta_F'] = df_c['Total_Fechadas'].diff().fillna(df_c['Total_Fechadas'])
    df_c.loc[df_c['Delta_F'] < 0, 'Delta_F'] = df_c.loc[df_c['Delta_F'] < 0, 'Total_Fechadas']
    df_c['Mensal_Fechadas'] = df_c['Delta_F'].cumsum()
    df_c['Backlog'] = df_c['Total_Tarefas'] - df_c['Total_Fechadas']
    df_c['Mensal_Tarefas'] = df_c['Mensal_Fechadas'] + df_c['Backlog']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_c['Data'], y=df_c['Mensal_Tarefas'], name='Total (Mês)', line=dict(color='red'), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_c['Data'], y=df_c['Mensal_Fechadas'], name='Fechadas (Mês)', line=dict(color='green'), mode='lines+markers'))
    fig.update_layout(title=f"<b>Progresso Mensal ({inicio.strftime('%B')})</b>", template='plotly_white', legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(tickformat="%d/%m")
    return fig, df_c.iloc[-1]

def criar_grafico_produtividade_mensal(df):
    if df.empty: return go.Figure()
    agg = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev']).agg(qtd=('ID', 'count')).reset_index().sort_values('Ano-Mês')
    fig = px.bar(agg, x='Mes_Ano_Abrev', y='qtd', text='qtd', title="<b>Produtividade Mensal</b>")
    fig.update_layout(template='plotly_white')
    return fig

def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Gráfico Principal</b>")
    
    ordem_dias = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']
    
    df_dia = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Dia']).size().reset_index(name='Contagem')
    df_dia_total = df_dia.groupby('Dia')['Contagem'].sum().reset_index()
    
    df_semana = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês']).size().reset_index(name='Contagem')
    df_semana_total = df_semana.groupby('Semana do Mês')['Contagem'].sum().reset_index()

    df_diasemana_full = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    
    df_diasemana_total = df_diasemana_full.groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index()
    df_diasemana_total['Contagem'] = recortar_zeros_pontas(df_diasemana_total['Contagem'])
    df_diasemana_total = df_diasemana_total.dropna(subset=['Contagem'])

    mes_map = df[['Ano-Mês', 'Mes_Ano_Abrev']].drop_duplicates().sort_values('Ano-Mês')
    opcoes_meses = mes_map['Mes_Ano_Abrev'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_dia_total['Dia'], y=df_dia_total['Contagem'], name='Soma (Dias)', visible=True, mode='lines+markers+text', text=df_dia_total['Contagem'], line=dict(color='royalblue', width=3))) 
    fig.add_trace(go.Scatter(x=df_semana_total['Semana do Mês'], y=df_semana_total['Contagem'], name='Soma (Semanas)', visible=False, mode='lines+markers+text', text=df_semana_total['Contagem'], line=dict(color='royalblue', width=3))) 
    fig.add_trace(go.Scatter(x=df_diasemana_total['Nome Dia Semana'], y=df_diasemana_total['Contagem'], name='Soma (Dia Semana)', visible=False, mode='lines+markers+text', text=df_diasemana_total['Contagem'], line=dict(color='royalblue', width=3))) 
    
    offset_dia = 3
    for mes in opcoes_meses:
        d = df_dia[df_dia['Mes_Ano_Abrev'] == mes].sort_values('Dia')
        d['Contagem'] = d['Contagem'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=d['Dia'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem']))
    count_dia = len(opcoes_meses)

    offset_semana = offset_dia + count_dia
    for mes in opcoes_meses:
        d = df_semana[df_semana['Mes_Ano_Abrev'] == mes].sort_values('Semana do Mês')
        d['Contagem'] = d['Contagem'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=d['Semana do Mês'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem']))
    count_semana = len(opcoes_meses)

    offset_diasemana = offset_semana + count_semana
    diasemana_trace_map = [] 
    
    for mes in opcoes_meses:
        d_mes = df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes].groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index()
        d_mes['Contagem'] = recortar_zeros_pontas(d_mes['Contagem'])
        d_mes = d_mes.dropna(subset=['Contagem'])
        fig.add_trace(go.Scatter(x=d_mes['Nome Dia Semana'], y=d_mes['Contagem'], name=f"{mes} Agregado", visible=False, mode='lines+markers+text', text=d_mes['Contagem']))
        
        semanas_do_mes = sorted(df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes]['Semana do Mês'].unique())
        for sem in semanas_do_mes:
            d_sem = df_diasemana_full[(df_diasemana_full['Mes_Ano_Abrev'] == mes) & (df_diasemana_full['Semana do Mês'] == sem)].set_index('Nome Dia Semana').reindex(ordem_dias).fillna(0).reset_index()
            d_sem['Contagem'] = recortar_zeros_pontas(d_sem['Contagem'])
            d_sem = d_sem.dropna(subset=['Contagem'])
            fig.add_trace(go.Scatter(x=d_sem['Nome Dia Semana'], y=d_sem['Contagem'], name=f"{mes} Sem {sem}", visible=False, mode='lines+markers+text', text=d_sem['Contagem']))
        
        diasemana_trace_map.append({'mes': mes, 'num_semanas': len(semanas_do_mes)})

    total_traces = len(fig.data)
    vis_total_agregado_dia = [False] * total_traces
    for i in range(count_dia): vis_total_agregado_dia[offset_dia + i] = True
    buttons_dia = [dict(label="Total Agregado", method="update", args=[{"visible": vis_total_agregado_dia}])]
    for i, mes in enumerate(opcoes_meses):
        vis = [False]*total_traces; vis[offset_dia + i] = True
        buttons_dia.append(dict(label=mes, method="update", args=[{"visible": vis}]))

    vis_total_agregado_semana = [False] * total_traces
    for i in range(count_semana): vis_total_agregado_semana[offset_semana + i] = True
    buttons_semana = [dict(label="Total Agregado", method="update", args=[{"visible": vis_total_agregado_semana}])]
    for i, mes in enumerate(opcoes_meses):
        vis = [False]*total_traces; vis[offset_semana + i] = True
        buttons_semana.append(dict(label=mes, method="update", args=[{"visible": vis}]))

    vis_total_agregado_diasemana = [False] * total_traces
    temp_idx = offset_diasemana
    for item in diasemana_trace_map:
        vis_total_agregado_diasemana[temp_idx] = True 
        temp_idx += 1 + item['num_semanas']
    buttons_diasemana = [dict(label="Total Agregado", method="update", args=[{"visible": vis_total_agregado_diasemana}])]
    
    current_idx = offset_diasemana
    for item in diasemana_trace_map:
        mes = item['mes']
        vis_mes_zoom = [False]*total_traces
        for k in range(item['num_semanas']): vis_mes_zoom[current_idx + 1 + k] = True
        buttons_diasemana.append(dict(label=f"{mes} Agregado", method="update", args=[{"visible": vis_mes_zoom}]))
        current_idx += 1 
        for s in range(item['num_semanas']):
            vis_sem = [False]*total_traces; vis_sem[current_idx] = True
            buttons_diasemana.append(dict(label=f"{mes} Semana {s+1}", method="update", args=[{"visible": vis_sem}]))
            current_idx += 1

    vis_init_dia = [False]*total_traces; vis_init_dia[0] = True
    vis_init_semana = [False]*total_traces; vis_init_semana[1] = True
    vis_init_diasemana = [False]*total_traces; vis_init_diasemana[2] = True

    fig.update_layout(
        updatemenus=[
            dict(type="buttons", direction="right", x=0.99, y=1.25, buttons=[
                dict(label="Dia do Mês", method="update", args=[{"visible": vis_init_dia}, {"updatemenus[1].buttons": buttons_dia, "xaxis.title": "Dia", "xaxis.type": "linear"}]),
                dict(label="Semana do Mês", method="update", args=[{"visible": vis_init_semana}, {"updatemenus[1].buttons": buttons_semana, "xaxis.title": "Semana", "xaxis.type": "linear"}]),
                dict(label="Dia da Semana", method="update", args=[{"visible": vis_init_diasemana}, {"updatemenus[1].buttons": buttons_diasemana, "xaxis.title": "Dia da Semana", "xaxis.type": "category", "xaxis.categoryorder": "array", "xaxis.categoryarray": ordem_dias}])
            ]),
            dict(direction="down", x=0.01, y=1.25, showactive=True, buttons=buttons_dia)
        ],
        title="<b>Gráfico Principal</b>", margin=dict(t=140), template='plotly_white'
    )
    return fig

def criar_grafico_tarefas_funcionarios(df):
    if df.empty: return go.Figure()
    v = df['Encarregado'].value_counts().reset_index(); v.columns=['Encarregado','c']
    fig = px.bar(v, x='c', y='Encarregado', orientation='h', text='c', title="<b>Tarefas por Pessoa</b>")
    fig.update_layout(template='plotly_white', yaxis_categoryorder='total ascending')
    return fig

def criar_grafico_status_tarefas(df):
    if df.empty: return go.Figure()
    v = df['Status_Tarefa'].value_counts().reset_index(); v.columns=['Status','c']
    fig = px.pie(v, names='Status', values='c', title="<b>Status</b>", hole=0.4)
    return fig

# --- GRÁFICOS DE PONTUAÇÃO ---
def criar_grafico_pontuacao_individual(df, nomes, d_ini, d_fim):
    if df is None or df.empty: return go.Figure(), pd.DataFrame()
    df_c = df.copy()
    cols_validas = []
    for c in df_c.columns:
        if str(c).lower() != 'encarregado':
            dt = converter_data_robusta(pd.Series([c]))
            if pd.notna(dt[0]) and d_ini <= dt[0].date() <= d_fim:
                cols_validas.append(c)
                df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(0)
    
    if not cols_validas: return go.Figure().update_layout(title="Sem dados período"), pd.DataFrame()
    
    df_c['Total'] = df_c[cols_validas].sum(axis=1)
    df_f = df_c[df_c['Encarregado'].isin(nomes)].sort_values('Total', ascending=True)
    
    fig = px.bar(df_f, x='Total', y='Encarregado', orientation='h', text='Total', title="<b>Pontuação Individual</b>", color='Total', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', yaxis_title=None)
    return fig, df_f[['Encarregado', 'Total'] + cols_validas].sort_values('Total', ascending=False)

def criar_grafico_pontuacao_lideres(df_mapa, df_pt, nomes, d_ini, d_fim):
    if df_mapa is None or df_pt is None: return go.Figure(), pd.DataFrame(), pd.DataFrame()
    df_c = df_pt.copy()
    cols_validas = []
    for c in df_c.columns:
        if str(c).lower() != 'encarregado':
            dt = converter_data_robusta(pd.Series([c]))
            if pd.notna(dt[0]) and d_ini <= dt[0].date() <= d_fim:
                cols_validas.append(c)
                df_c[c] = pd.to_numeric(df_c[c], errors='coerce').fillna(0)
    
    if not cols_validas: return go.Figure().update_layout(title="Sem dados período"), pd.DataFrame(), pd.DataFrame()
    
    df_c['Pontos'] = df_c[cols_validas].sum(axis=1)
    df_mapa['Lider'] = df_mapa['Lider'].astype(str).str.strip()
    df_mapa['Liderado'] = df_mapa['Liderado'].astype(str).str.strip()
    df_c['Encarregado'] = df_c['Encarregado'].astype(str).str.strip()
    
    merge = pd.merge(df_mapa, df_c[['Encarregado', 'Pontos']], left_on='Liderado', right_on='Encarregado')
    rank = merge.groupby('Lider')['Pontos'].sum().reset_index()
    rank = rank[rank['Lider'].isin(nomes)].sort_values('Pontos', ascending=True)
    
    fig = px.bar(rank, x='Pontos', y='Lider', orientation='h', text='Pontos', title="<b>Pontuação Liderança</b>", color='Pontos', color_continuous_scale='Plasma')
    fig.update_layout(template='plotly_white', yaxis_title=None)
    return fig, rank.sort_values('Pontos', ascending=False), df_c

def criar_grafico_pontuacao_combinada(df_ind, df_lid, df_mapa, nomes, d_ini, d_fim):
    fig_i, df_i = criar_grafico_pontuacao_individual(df_ind, df_ind['Encarregado'].unique() if df_ind is not None else [], d_ini, d_fim)
    fig_l, df_l, _ = criar_grafico_pontuacao_lideres(df_mapa, df_lid, df_mapa['Lider'].unique() if df_mapa is not None else [], d_ini, d_fim)
    
    res = pd.DataFrame(columns=['Pessoa', 'Total'])
    if not df_i.empty: 
        temp = df_i[['Encarregado', 'Total']].rename(columns={'Encarregado':'Pessoa', 'Total':'Ind'})
        res = pd.merge(res, temp, on='Pessoa', how='outer')
    if not df_l.empty:
        temp = df_l[['Lider', 'Pontos']].rename(columns={'Lider':'Pessoa', 'Pontos':'Lid'})
        res = pd.merge(res, temp, on='Pessoa', how='outer')
    
    res = res.fillna(0)
    res['Final'] = res.get('Ind', 0) + res.get('Lid', 0)
    res = res[res['Pessoa'].isin(nomes)].sort_values('Final', ascending=True)
    
    if res.empty: return go.Figure()
    fig = px.bar(res, x='Final', y='Pessoa', orientation='h', text='Final', title="<b>Pontuação Total Combinada</b>", color='Final', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', yaxis_title=None)
    return fig

# ==============================================================================
# APP
# ==============================================================================
st.title("Dashboard de Produtividade")
df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca_mapa, df_equipe, df_backlog, df_source_analise, df_historico = carregar_dados_completos()

# --- DATAS GLOBAIS ---
all_dates = []
if df_analise is not None and not df_analise.empty:
    all_dates.extend(df_analise['Data Final (aberta)'].dropna().dt.date.tolist())
if df_notas_tabela1 is not None:
    for c in df_notas_tabela1.columns:
        dt = converter_data_robusta(pd.Series([c]))
        if pd.notna(dt[0]): all_dates.append(dt[0].date())

if all_dates:
    min_date, max_date = min(all_dates), max(all_dates)
else:
    min_date, max_date = date.today(), date.today()

# --- BARRA LATERAL ---
if 'filtros_iniciados' not in st.session_state:
    st.session_state.encarregado_filtro = ["Todos"]
    st.session_state.contrato_filtro = "Todos"
    st.session_state.status_tarefa_filtro = "Todos"
    st.session_state.semana_filtro = "Todos"
    st.session_state.date_slider = (min_date, max_date)
    st.session_state.filtros_iniciados = True

def limpar():
    st.session_state.encarregado_filtro = ["Todos"]
    st.session_state.date_slider = (min_date, max_date)

with st.sidebar:
    st.title("Filtros")
    if df_analise is not None:
        l_enc = ["Todos"] + sorted(df_analise['Encarregado'].unique())
        st.multiselect("Encarregado", l_enc, key='encarregado_filtro')
        l_status = ["Todos"] + sorted(df_analise['Status_Funcionario'].unique())
        st.selectbox("Contrato", l_status, key='contrato_filtro')
        l_st_tar = ["Todos"] + sorted(df_analise['Status_Tarefa'].unique())
        st.selectbox("Status Tarefa", l_st_tar, key='status_tarefa_filtro')
    st.button("Limpar", on_click=limpar)

# --- FILTRAGEM ---
df_f = df_analise.copy() if df_analise is not None else pd.DataFrame()
if not df_f.empty:
    if "Todos" not in st.session_state.encarregado_filtro: df_f = df_f[df_f['Encarregado'].isin(st.session_state.encarregado_filtro)]
    if st.session_state.contrato_filtro != "Todos": df_f = df_f[df_f['Status_Funcionario'] == st.session_state.contrato_filtro]
    if st.session_state.status_tarefa_filtro != "Todos": df_f = df_f[df_f['Status_Tarefa'] == st.session_state.status_tarefa_filtro]

c1, c2, c3 = st.columns([2, 1, 3])
with c1:
    sems = ["Todos"] + sorted([x for x in df_f['Semana do Mês'].unique() if pd.notna(x)]) if not df_f.empty else []
    st.selectbox("Semana Mês", sems, key='semana_filtro')
with c3:
    st.slider("Período", min_value=min_date, max_value=max_date, key='date_slider')

if st.session_state.semana_filtro != "Todos": df_f = df_f[df_f['Semana do Mês'] == st.session_state.semana_filtro]
d_ini, d_fim = st.session_state.date_slider
if not df_f.empty: df_f = df_f[(df_f['Data Final (aberta)'].dt.date >= d_ini) & (df_f['Data Final (aberta)'].dt.date <= d_fim)]

with c2: st.metric("Total Tarefas", len(df_f))
st.divider()

# --- ABAS ---
t1, t2, t3, t4, t5 = st.tabs(["Semana", "Mês", "Backlog", "Geral", "Pontuação"])

with t1: # SEMANA
    set_semanas = set()
    hj = pd.Timestamp.now().normalize()
    sexta_atual = hj - pd.to_timedelta(hj.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
    set_semanas.add(sexta_atual)
    
    if df_historico is not None and not df_historico.empty:
        dates = df_historico['Data']
        fridays = dates - pd.to_timedelta(dates.dt.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
        set_semanas.update(fridays.unique())
    
    if not df_f.empty:
        dates = df_f['Data Final (aberta)']
        fridays = dates - pd.to_timedelta(dates.dt.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
        set_semanas.update(fridays.unique())

    # ORDENAÇÃO REVERSA (MAIS NOVAS PRIMEIRO)
    lista_semanas = sorted([d for d in list(set_semanas) if pd.notnull(d)], reverse=True)
    lista_str = [d.strftime('%d/%m/%Y') for d in lista_semanas]
    
    # LÓGICA DE SELEÇÃO PADRÃO: Tenta a sexta atual ou a mais próxima no passado
    idx_padrao = 0
    sexta_atual_str = sexta_atual.strftime('%d/%m/%Y')
    if sexta_atual_str in lista_str:
        idx_padrao = lista_str.index(sexta_atual_str)
    else:
        for i, s in enumerate(lista_str):
            if pd.to_datetime(s, dayfirst=True) <= sexta_atual:
                idx_padrao = i; break
    
    sem_sel = st.selectbox("Semana (Sexta-feira referência):", lista_str, index=idx_padrao)
    
    fig_h, last_h = criar_grafico_historico_semanal(df_historico, sem_sel)
    if last_h is not None:
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Total", last_h['Total_Tarefas']); c_m2.metric("Abertas", last_h['Total_Tarefas']-last_h['Total_Fechadas']); c_m3.metric("Fechadas", last_h['Total_Fechadas'])
    st.plotly_chart(fig_h, use_container_width=True)
    
    dt_sel = pd.to_datetime(sem_sel, dayfirst=True)
    if not df_f.empty:
        dates_f = df_f['Data Final (aberta)']
        fridays_f = dates_f - pd.to_timedelta(dates_f.dt.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
        df_sem = df_f[fridays_f.dt.date == dt_sel.date()]
        
        st.markdown("---")
        st.subheader("Tarefas da Semana")
        if not df_sem.empty:
            piv = pd.pivot_table(df_sem[df_sem['Status_Tarefa']=='Executado'], index='Encarregado', columns='Nome Dia Semana', values='ID', aggfunc='count', fill_value=0)
            st.dataframe(piv, use_container_width=True)
        
        for enc in sorted(df_sem['Encarregado'].unique()):
            d_e = df_sem[df_sem['Encarregado'] == enc]
            ab = d_e[d_e['Status_Tarefa']=='Aberto']
            fe = d_e[d_e['Status_Tarefa']=='Executado']
            with st.expander(f"{enc} ({len(d_e)}) - 🔴 {len(ab)} | 🟢 {len(fe)}"):
                if not ab.empty: st.caption("Abertas"); st.dataframe(ab[['Nome Task','Data Inicial']], use_container_width=True, hide_index=True)
                if not fe.empty: st.caption("Fechadas"); st.dataframe(fe[['Nome Task','Data Final']], use_container_width=True, hide_index=True)

with t2: # MÊS
    fig_hm, last_hm = criar_grafico_historico_mensal(df_historico)
    
    if last_hm is not None:
        col_met_m1, col_met_m2, col_met_m3 = st.columns(3)
        col_met_m1.metric("Total Acumulado", f"{last_hm['Mensal_Tarefas']:.0f}")
        col_met_m2.metric("🔴 Gap (Abertas)", f"{last_hm['Mensal_Tarefas'] - last_hm['Mensal_Fechadas']:.0f}")
        col_met_m3.metric("🟢 Fechadas", f"{last_hm['Mensal_Fechadas']:.0f}")

    st.plotly_chart(fig_hm, use_container_width=True)
    
    st.subheader("Detalhes do Mês Atual")
    ini_m = hj.replace(day=1); fim_m = (ini_m + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    if not df_f.empty:
        df_m = df_f[(df_f['Data Final (aberta)'] >= ini_m) & (df_f['Data Final (aberta)'] <= fim_m)]
        if not df_m.empty:
            piv = pd.pivot_table(df_m[df_m['Status_Tarefa']=='Executado'], index='Encarregado', columns=df_m['Data Final (aberta)'].dt.day, values='ID', aggfunc='count', fill_value=0)
            st.dataframe(piv, use_container_width=True)
            
            for enc in sorted(df_m['Encarregado'].unique()):
                d_e = df_m[df_m['Encarregado'] == enc]
                with st.expander(f"{enc} - {len(d_e)} Tarefas"):
                    st.dataframe(d_e[['Nome Task', 'Data Inicial', 'Data Final', 'Status_Tarefa']], use_container_width=True, hide_index=True)

with t3: # BACKLOG
    if df_backlog is not None and not df_backlog.empty:
        bk = df_backlog.copy()
        if "Todos" not in st.session_state.encarregado_filtro: bk = bk[bk['Encarregado'].isin(st.session_state.encarregado_filtro + ['Em Branco'])]
        sem_dono = bk[(bk['Status_Backlog']=='Aberto') & (bk['Encarregado']=='Em Branco')]
        com_dono = bk[(bk['Status_Backlog']=='Aberto') & (bk['Encarregado']!='Em Branco')]
        fechado = bk[bk['Status_Backlog']=='Fechado']
        
        with st.expander(f"⚫ Sem Dono ({len(sem_dono)})"): st.dataframe(sem_dono[['Nome Task','Data Inicial']], use_container_width=True, hide_index=True)
        with st.expander(f"🔴 Com Dono ({len(com_dono)})", expanded=True): st.dataframe(com_dono[['Encarregado','Nome Task','Data Inicial']], use_container_width=True, hide_index=True)
        with st.expander(f"🟢 Fechados ({len(fechado)})"): st.dataframe(fechado[['Encarregado','Nome Task','Data Final']], use_container_width=True, hide_index=True)

with t4: # GERAL
    c_g1, c_g2 = st.columns(2)
    with c_g1: st.plotly_chart(criar_grafico_produtividade_mensal(df_f), use_container_width=True)
    with c_g2: st.plotly_chart(criar_grafico_principal(df_f), use_container_width=True)
    c_g3, c_g4 = st.columns(2)
    with c_g3: st.plotly_chart(criar_grafico_tarefas_funcionarios(df_f), use_container_width=True)
    with c_g4: st.plotly_chart(criar_grafico_status_tarefas(df_f), use_container_width=True)

with t5: # PONTUAÇÃO
    nomes = []
    if df_equipe is not None and not df_equipe.empty:
        if st.session_state.contrato_filtro == "Todos": nomes = df_equipe['Nome'].unique().tolist()
        else: nomes = df_equipe[df_equipe['Status_Funcionario'] == st.session_state.contrato_filtro]['Nome'].unique().tolist()
    
    if "Todos" not in st.session_state.encarregado_filtro:
        if not nomes: nomes = st.session_state.encarregado_filtro
        else: nomes = list(set(nomes) & set(st.session_state.encarregado_filtro))
    
    f_ind, df_ind_t = criar_grafico_pontuacao_individual(df_notas_tabela1, nomes, d_ini, d_fim)
    st.plotly_chart(f_ind, use_container_width=True)
    with st.expander("Dados Individuais"): st.dataframe(df_ind_t, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    f_lid, df_lid_t, _ = criar_grafico_pontuacao_lideres(df_lideranca_mapa, df_notas_tabela2, nomes, d_ini, d_fim)
    st.plotly_chart(f_lid, use_container_width=True)
    
    st.markdown("---")
    f_tot = criar_grafico_pontuacao_combinada(df_notas_tabela1, df_notas_tabela2, df_lideranca_mapa, nomes, d_ini, d_fim)
    st.plotly_chart(f_tot, use_container_width=True)