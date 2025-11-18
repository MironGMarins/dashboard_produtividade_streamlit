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
import holidays
import json
from datetime import date, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Dashboard de Produtividade"
)

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
        creds = Credentials.from_service_account_file("google_credentials.json", scopes=scopes)
    
    client = gspread.authorize(creds)
    url_da_planilha = st.secrets.get("SHEET_URL", 'https://docs.google.com/spreadsheets/d/1juyOfIh0ZqsfJjN0p3gD8pKaAIX0R6IAPG9vysl7yWI/edit#gid=901870248')
    
    spreadsheet = client.open_by_url(url_da_planilha)
    
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
        worksheet_dados = spreadsheet.worksheet(nome_aba_dados)
        df_dados = pd.DataFrame(worksheet_dados.get_all_records())
        
        worksheet_equipe = spreadsheet.worksheet(nome_aba_equipes)
        df_equipe = pd.DataFrame(worksheet_equipe.get_all_records()) 

        worksheet_lideranca = spreadsheet.worksheet(nome_aba_lideranca)
        df_lideranca = pd.DataFrame(worksheet_lideranca.get_all_records())
        
        worksheet_backlog = spreadsheet.worksheet(nome_aba_backlog)
        df_backlog = pd.DataFrame(worksheet_backlog.get_all_records())
        
        worksheet_source = spreadsheet.worksheet(nome_aba_source)
        df_source = pd.DataFrame(worksheet_source.get_all_records())
        
        # --- Carregar Aba "HistoricoDiario" ---
        try:
            worksheet_historico = spreadsheet.worksheet(nome_aba_historico)
            df_historico = pd.DataFrame(worksheet_historico.get_all_records())
        except gspread.exceptions.WorksheetNotFound:
            pass 

        # ==============================================================================
        # --- Carregar AMBAS as tabelas da aba "Notas" ---
        # ==============================================================================
        worksheet_pontuacao = spreadsheet.worksheet(nome_aba_pontuacao)
        all_values_notas = worksheet_pontuacao.get_all_values()
        
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
            headers_sup = dados_tabela_superior[0]
            data_sup = dados_tabela_superior[1:]
            df_notas_tabela1 = pd.DataFrame(data_sup, columns=headers_sup)
        elif len(dados_tabela_superior) == 1:
             df_notas_tabela1 = pd.DataFrame(columns=dados_tabela_superior[0])

        if len(dados_tabela_inferior) > 1:
            headers_inf = dados_tabela_inferior[0]
            data_inf = dados_tabela_inferior[1:]
            df_notas_tabela2 = pd.DataFrame(data_inf, columns=headers_inf)
        elif len(dados_tabela_inferior) == 1:
             df_notas_tabela2 = pd.DataFrame(columns=headers_inf[0])
    
    except gspread.exceptions.WorksheetNotFound as e:
        st.error(f"Erro: A aba '{e.args[0]}' não foi encontrada na planilha. Verifique os nomes.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
    except Exception as e:
        st.error(f"Erro ao carregar dados do Google Sheets: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Renomeia o status da Equipe ANTES de tudo ---
    if not df_equipe.empty and 'Status' in df_equipe.columns:
        df_equipe.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)

    # --- PREPARAÇÃO DOS DADOS (df_analise - Aba 1 e 2) ---
    df_grafico = df_dados.copy()
    colunas_para_numerico = ['Peso', 'Pablo', 'Leonardo', 'Itiel', 'Ítalo']
    for col in colunas_para_numerico:
        if col in df_grafico.columns:
            df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce').fillna(0)

    df_grafico['Data Inicial'] = pd.to_datetime(df_grafico['Data Inicial'], errors='coerce')
    df_grafico['Data Final'] = pd.to_datetime(df_grafico['Data Final'], errors='coerce')
    df_grafico['Status_Tarefa'] = np.where(df_grafico['Data Final'].isnull(), 'Aberto', 'Executado')
    data_hoje = pd.Timestamp.now().normalize()
    df_grafico['Data Final (aberta)'] = df_grafico['Data Final'].fillna(data_hoje)
    
    if 'Encarregado' in df_grafico.columns:
        df_grafico['Encarregado'] = df_grafico['Encarregado'].astype(str).str.strip().replace('', 'Em Branco')
    if 'Nome Task' in df_grafico.columns:
        df_grafico['Nome Task'] = df_grafico['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
    else:
        if not df_grafico.empty:
            st.error("Aba 'Total BaseCamp para Notas' não tem a coluna 'Nome Task'.")
            df_grafico['Nome Task'] = 'Erro: Coluna Faltando'


    data_inicio_analise = df_grafico['Data Inicial'].min() if pd.notna(df_grafico['Data Inicial'].min()) else data_hoje
    data_fim_analise = data_hoje
    
    # --- Tabela Calendário ---
    data_inicio_calendario = data_inicio_analise
    if not df_source.empty:
        if 'Data Inicial' in df_source.columns:
            data_inicio_source = pd.to_datetime(df_source['Data Inicial'], errors='coerce').min()
            if pd.notna(data_inicio_source) and data_inicio_source < data_inicio_analise:
                data_inicio_calendario = data_inicio_source

    tabela_calendario = pd.DataFrame({"Date": pd.date_range(start=data_inicio_calendario, end=data_fim_analise, freq='D')})
    tabela_calendario['Ano'] = tabela_calendario['Date'].dt.year
    
    # --- Mapeamento manual de Meses para PT-BR ---
    meses_pt = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    tabela_calendario['Nome Mês'] = tabela_calendario['Date'].dt.month.map(meses_pt)
    
    tabela_calendario['Mes_Ano_Abrev'] = tabela_calendario['Nome Mês'] + '/' + tabela_calendario['Date'].dt.strftime('%y')
    tabela_calendario['Ano-Mês'] = tabela_calendario['Date'].dt.strftime('%Y-%m')
    tabela_calendario['Dia'] = tabela_calendario['Date'].dt.day
    tabela_calendario['Dia da Semana_ISO'] = tabela_calendario['Date'].dt.dayofweek
    
    # --- Mapeamento manual de Dias para PT-BR ---
    dias_pt_map = {0: 'seg', 1: 'ter', 2: 'qua', 3: 'qui', 4: 'sex', 5: 'sab', 6: 'dom'}
    tabela_calendario['Nome Dia Semana'] = tabela_calendario['Dia da Semana_ISO'].map(dias_pt_map)
    
    tabela_calendario['Data_Inicio_Semana'] = tabela_calendario['Date'] - pd.to_timedelta(tabela_calendario['Dia da Semana_ISO'], unit='d')
    tabela_calendario['Data_Sexta_Feira'] = tabela_calendario['Data_Inicio_Semana'] + pd.to_timedelta(4, unit='d')
    tabela_calendario['Nome_da_Semana'] = tabela_calendario['Data_Sexta_Feira'].dt.strftime('%d/%m/%Y')
    tabela_calendario['Semana_Ano'] = tabela_calendario['Data_Sexta_Feira'].dt.strftime('%Y-%U') 
    tabela_calendario['Semana do Mês'] = (tabela_calendario['Date'].dt.dayofweek + (tabela_calendario['Date'].dt.day - 1)).floordiv(7) + 1
    tabela_calendario['Dia da Semana'] = tabela_calendario['Dia da Semana_ISO'] + 1

    df_analise_temp = pd.merge(df_grafico, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date').drop(columns=['Date'])
    df_analise = pd.merge(df_analise_temp, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
    if 'Status_Funcionario' in df_analise.columns:
        df_analise['Status_Funcionario'].fillna('Outros', inplace=True)
    
    # --- PREPARAÇÃO DOS DADOS (df_source_analise) ---
    if not df_source.empty:
        df_source_proc = df_source.copy()
        df_source_proc['Data Inicial'] = pd.to_datetime(df_source_proc['Data Inicial'], errors='coerce')
        df_source_proc['Data Final'] = pd.to_datetime(df_source_proc['Data Final'], errors='coerce')
        df_source_proc['Status_Tarefa'] = np.where(df_source_proc['Data Final'].isnull(), 'Aberto', 'Executado')
        df_source_proc['Data Final (aberta)'] = df_source_proc['Data Final'].fillna(data_hoje)
        if 'Encarregado' in df_source_proc.columns:
             df_source_proc['Encarregado'] = df_source_proc['Encarregado'].astype(str).str.strip().replace('', 'Em Branco')
        if 'Nome Task' in df_source_proc.columns:
            df_source_proc['Nome Task'] = df_source_proc['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
        
        df_source_analise = pd.merge(df_source_proc, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date').drop(columns=['Date'])
        df_source_analise = pd.merge(df_source_analise, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
        if 'Status_Funcionario' in df_source_analise.columns:
            df_source_analise['Status_Funcionario'].fillna('Outros', inplace=True)
    
    # --- PREPARAÇÃO DOS DADOS (df_backlog - Aba 3) ---
    if not df_backlog.empty:
        df_backlog['Data Inicial'] = pd.to_datetime(df_backlog['Data Inicial'], errors='coerce')
        df_backlog['Data Final'] = pd.to_datetime(df_backlog['Data Final'], errors='coerce') 
        df_backlog['Status_Backlog'] = np.where(df_backlog['Data Final'].isnull(), 'Aberto', 'Fechado')
        if 'Encarregado' in df_backlog.columns:
            df_backlog['Encarregado'] = df_backlog['Encarregado'].astype(str).str.strip().replace('', 'Em Branco') 
        if 'Nome Task' in df_backlog.columns:
            df_backlog['Nome Task'] = df_backlog['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
        df_backlog = pd.merge(df_backlog, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
        if 'Status_Funcionario' in df_backlog.columns:
            df_backlog['Status_Funcionario'].fillna('Outros', inplace=True)

    # --- PREPARAÇÃO DOS DADOS (df_historico) ---
    if not df_historico.empty:
        df_historico.rename(columns={'Data Final': 'Data'}, inplace=True)
        df_historico['Data'] = pd.to_datetime(df_historico['Data'], format='%d/%m/%Y', errors='coerce')
        df_historico['Total_Fechadas'] = pd.to_numeric(df_historico['Total_Fechadas'], errors='coerce').fillna(0)
        df_historico['Total_Tarefas'] = pd.to_numeric(df_historico['Total_Tarefas'], errors='coerce').fillna(0)
        df_historico.dropna(subset=['Data'], inplace=True)

    return df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca, df_equipe, df_backlog, df_source_analise, df_historico

# ==============================================================================
# FUNÇÕES PARA CRIAR OS GRÁFICOS
# ==============================================================================

def criar_grafico_historico_semanal(df_historico, semana_selecionada_str=None):
    # Verifica se o dataframe base existe
    if df_historico is None or df_historico.empty:
        return go.Figure().update_layout(title="<b>Progresso Semanal (Histórico)</b><br><i>Nenhum dado na aba 'HistoricoDiario' ainda.</i>", template='plotly_white'), None

    # --- DEFINIÇÃO DAS DATAS DE INÍCIO E FIM ---
    if semana_selecionada_str:
        # CASO 1: O usuário selecionou uma semana no Dropdown
        try:
            # O dropdown nos dá a data da Sexta-feira (ex: "15/11/2024")
            data_referencia = pd.to_datetime(semana_selecionada_str, format='%d/%m/%Y')
            
            # Recalcula o intervalo (Segunda a Sexta dessa semana específica)
            fim_semana = data_referencia # A data do dropdown já é a sexta
            inicio_semana = fim_semana - pd.Timedelta(days=4) # Retrocede para segunda
            
            titulo_grafico = f"<b>Progresso da Semana ({inicio_semana.strftime('%d/%m')} a {fim_semana.strftime('%d/%m')})</b>"
        except ValueError:
            # Fallback se der erro na conversão
            titulo_grafico = "<b>Erro na Data Selecionada</b>"
            inicio_semana = pd.Timestamp.min
            fim_semana = pd.Timestamp.max
    else:
        # CASO 2: Nenhuma semana selecionada (Comportamento Padrão / Mais Recente)
        df_historico = df_historico.sort_values(by='Data', ascending=False)
        data_mais_recente = df_historico['Data'].iloc[0]
        dia_da_semana_iso = data_mais_recente.dayofweek
        inicio_semana = data_mais_recente - pd.to_timedelta(dia_da_semana_iso, unit='d')
        fim_semana = inicio_semana + pd.to_timedelta(4, unit='d')
        titulo_grafico = "<b>Progresso da Semana Atual (Seg-Sex)</b>"

    # --- FILTRAGEM DOS DADOS ---
    df_semana_historico = df_historico[
        (df_historico['Data'] >= inicio_semana) &
        (df_historico['Data'] <= fim_semana)
    ].sort_values(by='Data', ascending=True)
    
    # Se não tiver dados para a semana selecionada no histórico
    if df_semana_historico.empty:
        msg = f"<b>{titulo_grafico}</b><br><i>Não há registros na aba 'HistoricoDiario' para esta semana específica.</i>"
        return go.Figure().update_layout(title=msg, template='plotly_white'), None

    # --- PLOTAGEM ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_semana_historico['Data'], y=df_semana_historico['Total_Tarefas'], mode='lines+markers+text', name='Total de Tarefas', line=dict(color='red', width=3), text=df_semana_historico['Total_Tarefas'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df_semana_historico['Data'], y=df_semana_historico['Total_Fechadas'], mode='lines+markers+text', name='Tarefas Fechadas', line=dict(color='green', width=3), text=df_semana_historico['Total_Fechadas'], textposition='bottom center'))

    fig.update_layout(title=titulo_grafico, template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    # Eixo X fixo de Segunda a Sexta daquela semana
    range_semana = pd.date_range(start=inicio_semana, end=fim_semana)
    dias_pt = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']
    tick_vals = range_semana
    tick_text = [f"{dias_pt[d.dayofweek]} ({d.strftime('%d/%m')})" for d in range_semana]

    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_text, range=[inicio_semana - pd.Timedelta(days=0.5), fim_semana + pd.Timedelta(days=0.5)])
    fig.update_yaxes(range=[0, 300])
    
    ultimo_registro = df_semana_historico.iloc[-1]
    return fig, ultimo_registro

def criar_grafico_historico_mensal(df_historico):
    if df_historico is None or df_historico.empty:
        return go.Figure().update_layout(title="<b>Progresso do Mês Atual (Histórico)</b><br><i>Nenhum dado na aba 'HistoricoDiario' ainda.</i>", template='plotly_white'), None

    hoje = pd.Timestamp.now().normalize()
    inicio_mes = hoje.replace(day=1)
    proximo_mes = (inicio_mes + pd.DateOffset(months=1))
    fim_mes = proximo_mes - pd.Timedelta(days=1)

    df_mes_historico = df_historico[
        (df_historico['Data'] >= inicio_mes) &
        (df_historico['Data'] <= fim_mes)
    ].sort_values(by='Data', ascending=True)
    
    if df_mes_historico.empty:
        return go.Figure().update_layout(title="<b>Progresso do Mês Atual (Histórico)</b><br><i>Nenhum dado de histórico para o mês atual ainda.</i>", template='plotly_white'), None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mes_historico['Data'], y=df_mes_historico['Total_Tarefas'], mode='lines+markers+text', name='Total de Tarefas', line=dict(color='red', width=3), text=df_mes_historico['Total_Tarefas'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df_mes_historico['Data'], y=df_mes_historico['Total_Fechadas'], mode='lines+markers+text', name='Tarefas Fechadas', line=dict(color='green', width=3), text=df_mes_historico['Total_Fechadas'], textposition='bottom center'))

    fig.update_layout(title=f"<b>Progresso do Mês Atual ({inicio_mes.strftime('%B/%Y')})</b>", template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    fig.update_xaxes(tickformat="%d/%m")
    fig.update_yaxes(range=[0, 300])
    
    ultimo_registro = df_mes_historico.iloc[-1]
    return fig, ultimo_registro

def criar_grafico_produtividade_mensal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Produtividade Mensal</b>")
    df_agregado = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev']).agg(contagem_tarefas=('ID', 'count')).reset_index().sort_values('Ano-Mês')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_agregado['Mes_Ano_Abrev'], y=df_agregado['contagem_tarefas'], name='Quantidade de Tarefas', marker_color='royalblue', text=df_agregado['contagem_tarefas'], textposition='outside'))
    fig.update_layout(title="<b>Produtividade Mensal</b>", template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    if not df_agregado.empty:
        max_y = df_agregado['contagem_tarefas'].max()
        fig.update_yaxes(range=[0, max_y * 1.2])
    return fig

# ==============================================================================
# --- GRÁFICO PRINCIPAL (CORRIGIDO: TRIM DE BORDAS REAIS) ---
# ==============================================================================
def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Gráfico Principal</b>")
    
    ordem_dias = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']

    # --- FUNÇÃO AUXILIAR PARA RECORTAR ZEROS DAS PONTAS ---
    def recortar_zeros_pontas(series_dados):
        """Substitui zeros iniciais e finais por NaN."""
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
    
    # 1. PREPARAÇÃO DOS DADOS
    df_dia = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Dia']).size().reset_index(name='Contagem')
    df_dia_total = df_dia.groupby('Dia')['Contagem'].sum().reset_index()
    
    df_semana = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês']).size().reset_index(name='Contagem')
    df_semana_total = df_semana.groupby('Semana do Mês')['Contagem'].sum().reset_index()

    df_diasemana_full = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    
    df_diasemana_total = df_diasemana_full.groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index()
    # Aplica o recorte para o total global
    df_diasemana_total['Contagem'] = recortar_zeros_pontas(df_diasemana_total['Contagem'])
    # *** MUDANÇA CRUCIAL: Drop NaNs para REMOVER do eixo, não apenas ocultar ***
    df_diasemana_total = df_diasemana_total.dropna(subset=['Contagem'])


    # Ordenação Cronológica dos Meses
    mes_map = df[['Ano-Mês', 'Mes_Ano_Abrev']].drop_duplicates().sort_values('Ano-Mês')
    opcoes_meses = mes_map['Mes_Ano_Abrev'].tolist()

    # 2. CRIAÇÃO DA FIGURA
    fig = go.Figure()
    
    # --- A. TRACES GLOBAIS (SOMAS) ---
    fig.add_trace(go.Scatter(x=df_dia_total['Dia'], y=df_dia_total['Contagem'], name='Soma Total (Dias)', visible=True, mode='lines+markers+text', text=df_dia_total['Contagem'], textposition='top center', line=dict(color='royalblue', width=3))) 
    fig.add_trace(go.Scatter(x=df_semana_total['Semana do Mês'], y=df_semana_total['Contagem'], name='Soma Total (Semanas)', visible=False, mode='lines+markers+text', text=df_semana_total['Contagem'], textposition='top center', line=dict(color='royalblue', width=3))) 
    
    # *** Uso do DF limpo (sem as pontas) para o traço global ***
    fig.add_trace(go.Scatter(x=df_diasemana_total['Nome Dia Semana'], y=df_diasemana_total['Contagem'], name='Soma Total (Dia Semana)', visible=False, mode='lines+markers+text', text=df_diasemana_total['Contagem'], textposition='top center', line=dict(color='royalblue', width=3))) 
    
    # --- B. TRACES DE "DIA DO MÊS" ---
    offset_dia = 3
    for mes in opcoes_meses:
        d = df_dia[df_dia['Mes_Ano_Abrev'] == mes].sort_values('Dia')
        d['Contagem'] = d['Contagem'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=d['Dia'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem']))
    count_dia = len(opcoes_meses)

    # --- C. TRACES DE "SEMANA DO MÊS" ---
    offset_semana = offset_dia + count_dia
    for mes in opcoes_meses:
        d = df_semana[df_semana['Mes_Ano_Abrev'] == mes].sort_values('Semana do Mês')
        d['Contagem'] = d['Contagem'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=d['Semana do Mês'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem']))
    count_semana = len(opcoes_meses)

    # --- D. TRACES DE "DIA DA SEMANA" ---
    offset_diasemana = offset_semana + count_semana
    diasemana_trace_map = [] 
    
    for mes in opcoes_meses:
        # Mês Agregado
        d_mes = df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes].groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index()
        # Recorta e DELETA os NaNs das pontas
        d_mes['Contagem'] = recortar_zeros_pontas(d_mes['Contagem'])
        d_mes = d_mes.dropna(subset=['Contagem'])

        fig.add_trace(go.Scatter(x=d_mes['Nome Dia Semana'], y=d_mes['Contagem'], name=f"{mes} Agregado", visible=False, mode='lines+markers+text', text=d_mes['Contagem']))
        
        # Semanas do Mês
        semanas_do_mes = sorted(df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes]['Semana do Mês'].unique())
        for sem in semanas_do_mes:
            d_sem = df_diasemana_full[(df_diasemana_full['Mes_Ano_Abrev'] == mes) & (df_diasemana_full['Semana do Mês'] == sem)].set_index('Nome Dia Semana').reindex(ordem_dias).fillna(0).reset_index()
            
            # Recorta e DELETA os NaNs das pontas
            d_sem['Contagem'] = recortar_zeros_pontas(d_sem['Contagem'])
            d_sem = d_sem.dropna(subset=['Contagem'])

            fig.add_trace(go.Scatter(x=d_sem['Nome Dia Semana'], y=d_sem['Contagem'], name=f"{mes} Sem {sem}", visible=False, mode='lines+markers+text', text=d_sem['Contagem']))
        
        diasemana_trace_map.append({'mes': mes, 'num_semanas': len(semanas_do_mes)})

    total_traces = len(fig.data)

    # --- 3. CRIAÇÃO DOS MENUS ---

    # Menu 1: Dia do Mês
    vis_total_agregado_dia = [False] * total_traces
    for i in range(count_dia): vis_total_agregado_dia[offset_dia + i] = True
    
    buttons_dia = [dict(label="Total Agregado (Comparar Meses)", method="update", args=[{"visible": vis_total_agregado_dia}])]
    
    for i, mes in enumerate(opcoes_meses):
        vis = [False]*total_traces; vis[offset_dia + i] = True
        buttons_dia.append(dict(label=mes, method="update", args=[{"visible": vis}]))

    # Menu 2: Semana do Mês
    vis_total_agregado_semana = [False] * total_traces
    for i in range(count_semana): vis_total_agregado_semana[offset_semana + i] = True

    buttons_semana = [dict(label="Total Agregado (Comparar Meses)", method="update", args=[{"visible": vis_total_agregado_semana}])]
    
    for i, mes in enumerate(opcoes_meses):
        vis = [False]*total_traces; vis[offset_semana + i] = True
        buttons_semana.append(dict(label=mes, method="update", args=[{"visible": vis}]))

    # Menu 3: Dia da Semana
    vis_total_agregado_diasemana = [False] * total_traces
    temp_idx = offset_diasemana
    for item in diasemana_trace_map:
        vis_total_agregado_diasemana[temp_idx] = True 
        temp_idx += 1 + item['num_semanas']

    buttons_diasemana = [dict(label="Total Agregado (Somas Mensais)", method="update", args=[{"visible": vis_total_agregado_diasemana}])]
    
    current_idx = offset_diasemana
    for item in diasemana_trace_map:
        mes = item['mes']
        vis_mes_zoom = [False]*total_traces
        for k in range(item['num_semanas']):
             vis_mes_zoom[current_idx + 1 + k] = True
        
        buttons_diasemana.append(dict(label=f"{mes} Agregado", method="update", args=[{"visible": vis_mes_zoom}]))
        current_idx += 1 
        
        for s in range(item['num_semanas']):
            vis_sem = [False]*total_traces; vis_sem[current_idx] = True
            buttons_diasemana.append(dict(label=f"{mes} Semana {s+1}", method="update", args=[{"visible": vis_sem}]))
            current_idx += 1

    # --- 4. LAYOUT FINAL ---
    vis_init_dia = [False]*total_traces; vis_init_dia[0] = True
    vis_init_semana = [False]*total_traces; vis_init_semana[1] = True
    vis_init_diasemana = [False]*total_traces; vis_init_diasemana[2] = True

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons", direction="right", x=0.99, xanchor="right", y=1.25,
                buttons=[
                    dict(label="Dia do Mês", method="update", args=[{"visible": vis_init_dia}, {"updatemenus[1].buttons": buttons_dia, "xaxis.title": "Dia", "xaxis.type": "linear", "xaxis.categoryarray": None}]),
                    dict(label="Semana do Mês", method="update", args=[{"visible": vis_init_semana}, {"updatemenus[1].buttons": buttons_semana, "xaxis.title": "Semana", "xaxis.type": "linear", "xaxis.categoryarray": None}]),
                    # --- CONFIGURAÇÃO CRUCIAL: categoryorder='array' para manter Seg,Ter.. na ordem ---
                    # O Plotly só vai desenhar os que existirem nos dados filtrados!
                    dict(label="Dia da Semana", method="update", args=[{"visible": vis_init_diasemana}, {"updatemenus[1].buttons": buttons_diasemana, "xaxis.title": "Dia da Semana", "xaxis.type": "category", "xaxis.categoryorder": "array", "xaxis.categoryarray": ordem_dias}])
                ]
            ),
            dict(direction="down", x=0.01, xanchor="left", y=1.25, showactive=True, buttons=buttons_dia)
        ],
        title=dict(text="<b>Gráfico Principal</b>", y=0.95),
        margin=dict(t=140),
        template='plotly_white'
    )
    return fig

def criar_grafico_status_tarefas(df):
    if df.empty: return go.Figure().update_layout(title="<b>Distribuição por Status</b>")
    df_status = df['Status_Tarefa'].value_counts().reset_index()
    df_status.columns = ['Status_Tarefa', 'Contagem']
    fig = px.pie(df_status, names='Status_Tarefa', values='Contagem', title='<b>Distribuição por Status</b>', hole=0.4, color_discrete_map={'Executado': 'royalblue', 'Aberto': 'firebrick'})
    fig.update_traces(textinfo='percent+label+value')
    return fig

def criar_grafico_tarefas_funcionarios(df):
    if df.empty: return go.Figure().update_layout(title="<b>N° de Tarefas por Funcionário</b>")
    df_funcionarios = df['Encarregado'].value_counts().reset_index()
    df_funcionarios.columns = ['Encarregado', 'Contagem']
    fig = px.bar(df_funcionarios, x='Contagem', y='Encarregado', orientation='h', title="<b>N° de Tarefas por Funcionário</b>", text='Contagem', color='Contagem', color_continuous_scale='Blues')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False, yaxis_categoryorder='total ascending')
    return fig

# ==============================================================================
# --- GRÁFICO 1 (Aba 4): Pontuação Individual (CORRIGIDO) ---
# ==============================================================================
def criar_grafico_pontuacao_individual(df_notas, nomes_para_exibir, start_date, end_date):
    if df_notas is None or df_notas.empty: return go.Figure().update_layout(title="<b>Pontuação Individual</b>"), pd.DataFrame()
    df_proc = df_notas.copy()
    colunas_pontuacao_todas = [col for col in df_proc.columns if col.lower() != 'encarregado']
    
    colunas_pontuacao_filtradas = []
    for col in colunas_pontuacao_todas:
        try:
            data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
            if start_date <= data_coluna <= end_date: 
                colunas_pontuacao_filtradas.append(col)
        except ValueError: pass 
        
    if not colunas_pontuacao_filtradas: return go.Figure().update_layout(title="<b>Pontuação Individual</b>"), pd.DataFrame()
    
    for col in colunas_pontuacao_filtradas: df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
    df_proc['Pontuacao_Total'] = df_proc[colunas_pontuacao_filtradas].sum(axis=1)
    df_grafico = df_proc[df_proc['Encarregado'].isin(nomes_para_exibir)].sort_values(by='Pontuacao_Total', ascending=False)
    fig = px.bar(df_grafico, x='Pontuacao_Total', y='Encarregado', orientation='h', title="<b>Ranking de Pontuação Individual</b>", text='Pontuacao_Total', color='Pontuacao_Total', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False, yaxis_categoryorder='total ascending')
    fig.update_traces(texttemplate='%{text:.0f}')
    colunas_tabela = ['Encarregado', 'Pontuacao_Total'] + colunas_pontuacao_filtradas
    return fig, df_grafico[colunas_tabela]

def criar_grafico_pontuacao_lideres(df_mapa_lideres, df_pontos_liderados, nomes_para_exibir, start_date, end_date):
    if df_mapa_lideres is None or df_mapa_lideres.empty or df_pontos_liderados is None or df_pontos_liderados.empty: return go.Figure().update_layout(title="<b>Pontuação de Líderes</b>"), pd.DataFrame(), pd.DataFrame()
    df_pontos = df_pontos_liderados.copy()
    colunas_pontuacao_todas = [col for col in df_pontos.columns if col.lower() != 'encarregado']
    
    colunas_pontuacao_filtradas = []
    for col in colunas_pontuacao_todas:
        try:
            data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
            if start_date <= data_coluna <= end_date: colunas_pontuacao_filtradas.append(col)
        except ValueError: pass 
        
    if not colunas_pontuacao_filtradas: return go.Figure().update_layout(title="<b>Pontuação de Líderes</b>"), pd.DataFrame(), pd.DataFrame(columns=['Encarregado'] + colunas_pontuacao_filtradas)
    
    for col in colunas_pontuacao_filtradas: df_pontos[col] = pd.to_numeric(df_pontos[col], errors='coerce').fillna(0)
    df_pontos['Pontuacao_Total_Liderado'] = df_pontos[colunas_pontuacao_filtradas].sum(axis=1)
    df_liderados_pontos_detalhe = df_pontos[['Encarregado', 'Pontuacao_Total_Liderado'] + colunas_pontuacao_filtradas]
    df_liderados_pontos_detalhe['Encarregado'] = df_liderados_pontos_detalhe['Encarregado'].astype(str).str.strip()
    df_mapa_lideres['Lider'] = df_mapa_lideres['Lider'].astype(str).str.strip(); df_mapa_lideres['Liderado'] = df_mapa_lideres['Liderado'].astype(str).str.strip()
    df_merge = pd.merge(df_mapa_lideres, df_liderados_pontos_detalhe[['Encarregado', 'Pontuacao_Total_Liderado']], left_on='Liderado', right_on='Encarregado')
    df_final_lideres = df_merge.groupby('Lider')['Pontuacao_Total_Liderado'].sum().reset_index().rename(columns={'Pontuacao_Total_Liderado': 'Pontuacao_Total_Lider'})
    df_lideres_visiveis = df_final_lideres[df_final_lideres['Lider'].isin(nomes_para_exibir)].sort_values(by='Pontuacao_Total_Lider', ascending=False)
    fig = px.bar(df_lideres_visiveis, x='Pontuacao_Total_Lider', y='Lider', orientation='h', title="<b>Ranking de Pontuação (Apenas Liderança)</b>", text='Pontuacao_Total_Lider', color='Pontuacao_Total_Lider', color_continuous_scale='Plasma')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False, yaxis_categoryorder='total ascending')
    fig.update_traces(texttemplate='%{text:.0f}')
    return fig, df_lideres_visiveis, df_liderados_pontos_detalhe

def criar_grafico_pontuacao_combinada(df_notas_enc, df_notas_liderados, df_mapa_lideres, nomes_para_exibir, start_date, end_date):
    df_individuais = pd.DataFrame(columns=['Pessoa', 'Pontuacao_Individual'])
    if df_notas_enc is not None and not df_notas_enc.empty:
        df_proc_enc = df_notas_enc.copy()
        col_pont_enc_todas = [col for col in df_proc_enc.columns if col.lower() != 'encarregado']
        col_pont_enc_filtradas = []
        for col in col_pont_enc_todas:
            try:
                data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
                if start_date <= data_coluna <= end_date: col_pont_enc_filtradas.append(col)
            except ValueError: pass
        if col_pont_enc_filtradas:
            for col in col_pont_enc_filtradas: df_proc_enc[col] = pd.to_numeric(df_proc_enc[col], errors='coerce').fillna(0)
            df_proc_enc['Pontuacao_Individual'] = df_proc_enc[col_pont_enc_filtradas].sum(axis=1)
            df_individuais = df_proc_enc[['Encarregado', 'Pontuacao_Individual']].rename(columns={'Encarregado': 'Pessoa'})
            df_individuais['Pessoa'] = df_individuais['Pessoa'].astype(str).str.strip()
        else:
             df_individuais = df_proc_enc[['Encarregado']].rename(columns={'Encarregado': 'Pessoa'}); df_individuais['Pontuacao_Individual'] = 0; df_individuais['Pessoa'] = df_individuais['Pessoa'].astype(str).str.strip()
    df_lideres_final = pd.DataFrame(columns=['Pessoa', 'Pontuacao_Lideranca'])
    if (df_notas_liderados is not None and not df_notas_liderados.empty and df_mapa_lideres is not None and not df_mapa_lideres.empty):
        df_pontos = df_notas_liderados.copy()
        col_pont_lid_todas = [col for col in df_pontos.columns if col.lower() != 'encarregado']
        col_pont_lid_filtradas = []
        for col in col_pont_lid_todas:
            try:
                data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
                if start_date <= data_coluna <= end_date: col_pont_lid_filtradas.append(col)
            except ValueError: pass
        if col_pont_lid_filtradas:
            for col in col_pont_lid_filtradas: df_pontos[col] = pd.to_numeric(df_pontos[col], errors='coerce').fillna(0)
            df_pontos['Pontuacao_Total_Liderado'] = df_pontos[col_pont_lid_filtradas].sum(axis=1)
            df_pontos_total = df_pontos[['Encarregado', 'Pontuacao_Total_Liderado']]
            df_mapa_lideres['Lider'] = df_mapa_lideres['Lider'].astype(str).str.strip(); df_mapa_lideres['Liderado'] = df_mapa_lideres['Liderado'].astype(str).str.strip(); df_pontos_total['Encarregado'] = df_pontos_total['Encarregado'].astype(str).str.strip()
            df_merge = pd.merge(df_mapa_lideres, df_pontos_total, left_on='Liderado', right_on='Encarregado')
            df_lideres_soma = df_merge.groupby('Lider')['Pontuacao_Total_Liderado'].sum().reset_index()
            df_lideres_final = df_lideres_soma.rename(columns={'Lider': 'Pessoa', 'Pontuacao_Total_Liderado': 'Pontuacao_Lideranca'})
        else:
            df_lideres_final = df_mapa_lideres[['Lider']].rename(columns={'Lider': 'Pessoa'}); df_lideres_final['Pontuacao_Lideranca'] = 0; df_lideres_final = df_lideres_final.drop_duplicates()
    df_combinado = pd.merge(df_individuais, df_lideres_final, on='Pessoa', how='outer').fillna(0)
    df_combinado['Pontuacao_Total_Combinada'] = df_combinado['Pontuacao_Individual'] + df_combinado['Pontuacao_Lideranca']
    df_grafico = df_combinado[df_combinado['Pontuacao_Total_Combinada'] > 0]
    df_grafico_filtrado = df_grafico[df_grafico['Pessoa'].isin(nomes_para_exibir)]
    fig = px.bar(df_grafico_filtrado, x='Pontuacao_Total_Combinada', y='Pessoa', orientation='h', title="<b>Ranking Geral de Pontuação (Individual + Liderança)</b>", text='Pontuacao_Total_Combinada', color='Pontuacao_Total_Combinada', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False, yaxis_categoryorder='total ascending')
    fig.update_traces(texttemplate='%{text:.0f}')
    return fig

# ==============================================================================
# CORPO PRINCIPAL DO DASHBOARD
# ==============================================================================
st.title("Dashboard de Produtividade")
df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca_mapa, df_equipe, df_backlog, df_source_analise, df_historico = carregar_dados_completos()

if (df_analise is not None and not df_analise.empty):
    min_date = df_analise['Data Final (aberta)'].min().date()
    max_date = df_analise['Data Final (aberta)'].max().date()
elif (df_notas_tabela1 is not None and not df_notas_tabela1.empty):
    colunas_data_notas = [col for col in df_notas_tabela1.columns if col.lower() != 'encarregado']
    datas_convertidas = []
    for col in colunas_data_notas:
        try: datas_convertidas.append(pd.to_datetime(col, format='%Y-%m-%d').date())
        except ValueError: pass 
    if datas_convertidas: min_date = min(datas_convertidas); max_date = max(datas_convertidas)
    else: min_date = date.today(); max_date = date.today()
else: min_date = date.today(); max_date = date.today()

if (df_analise is not None and not df_analise.empty):
    def limpar_filtros():
        st.session_state.encarregado_filtro = ["Todos"]
        st.session_state.contrato_filtro = "Todos"
        st.session_state.status_tarefa_filtro = "Todos"
        st.session_state.semana_filtro = "Todos"
        st.session_state.peso_filtro = "Todos"
        st.session_state.date_slider = (min_date, max_date)

    if 'filtros_iniciados' not in st.session_state:
        limpar_filtros(); st.session_state.filtros_iniciados = True

    with st.sidebar:
        # st.image("media portal logo.png", width=200) # Descomente se tiver a imagem
        st.title("Filtros")
        encarregados_disponiveis = ["Todos"] + sorted(df_analise['Encarregado'].unique())
        st.multiselect("Encarregado", encarregados_disponiveis, key='encarregado_filtro')
        contratos_disponiveis = ["Todos"] + df_analise['Status_Funcionario'].unique().tolist()
        st.selectbox("Status (Contrato)", contratos_disponiveis, key='contrato_filtro')
        status_tarefas = ["Todos"] + df_analise['Status_Tarefa'].unique().tolist()
        st.selectbox("Status da Tarefa", status_tarefas, key='status_tarefa_filtro')
        st.markdown("---"); st.button("Limpar Filtros 🗑️", on_click=limpar_filtros)

    df_filtrado_aba1 = df_analise.copy()
    if "Todos" not in st.session_state.encarregado_filtro: df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Encarregado'].isin(st.session_state.encarregado_filtro)]
    if st.session_state.contrato_filtro != "Todos": df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Status_Funcionario'] == st.session_state.contrato_filtro]
    if st.session_state.status_tarefa_filtro != "Todos": df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Status_Tarefa'] == st.session_state.status_tarefa_filtro]
    
    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([2, 2, 1, 1, 4])
    with top_col1:
        semanas_disponiveis = ["Todos"] + sorted([i for i in df_filtrado_aba1['Semana do Mês'].unique() if i is not np.nan])
        st.selectbox("Semana do Mês", semanas_disponiveis, key='semana_filtro')
    with top_col2:
        pesos_disponiveis = ["Todos"] + sorted(df_filtrado_aba1['Peso'].astype(int).unique())
        st.selectbox("Peso da Tarefa", pesos_disponiveis, key='peso_filtro')
    with top_col5:
        st.slider("Intervalo de Datas (para Abas 4 e 5)", min_value=min_date, max_value=max_date, key='date_slider')
    
    if st.session_state.semana_filtro != "Todos": df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Semana do Mês'] == st.session_state.semana_filtro]
    if st.session_state.peso_filtro != "Todos": df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Peso'] == st.session_state.peso_filtro]
    
    start_date, end_date = st.session_state.date_slider
    df_filtrado_aba1 = df_filtrado_aba1[(df_filtrado_aba1['Data Final (aberta)'].dt.date >= start_date) & (df_filtrado_aba1['Data Final (aberta)'].dt.date <= end_date)]

    with top_col3: st.metric("Tarefas", f"{df_filtrado_aba1.shape[0]:,}")
    with top_col4: pass 
    st.markdown("---") 

    # --- ATUALIZAÇÃO DA ORDEM DAS ABAS E INCLUSÃO DA ABA MÊS ---
    aba_semana, aba_mes, aba_backlog, aba_geral, aba_pontuacao = st.tabs(["Semana", "Mês", "Backlog", "Atividade Geral", "Pontuação Geral"])

    with aba_semana:
        st.header("Análise Detalhada por Semana")
        
        # 1. PREPARAÇÃO DO FILTRO DE SEMANAS (Movemos para cima)
        df_analise_filtrado_aba2 = df_analise.copy()
        if "Todos" not in st.session_state.encarregado_filtro: 
            df_analise_filtrado_aba2 = df_analise_filtrado_aba2[df_analise_filtrado_aba2['Encarregado'].isin(st.session_state.encarregado_filtro)]
        if st.session_state.contrato_filtro != "Todos": 
            df_analise_filtrado_aba2 = df_analise_filtrado_aba2[df_analise_filtrado_aba2['Status_Funcionario'] == st.session_state.contrato_filtro]
        
        df_analise_executado = df_analise_filtrado_aba2[df_analise_filtrado_aba2['Status_Tarefa'] == 'Executado']
        semanas_df = df_analise_executado[['Nome_da_Semana', 'Semana_Ano']].drop_duplicates().sort_values(by='Semana_Ano', ascending=False)
        semanas_lista = semanas_df['Nome_da_Semana'].tolist()
        
        if not semanas_lista: 
            st.info("Nenhuma tarefa executada encontrada para os filtros selecionados.")
        else:
            # 2. O SELETOR AGORA VEM PRIMEIRO
            semana_selecionada = st.selectbox("Selecione uma Semana (data da Sexta-feira):", semanas_lista, key="aba2_semana_select")
            
            # 3. GERAR O GRÁFICO SINCRONIZADO COM A SELEÇÃO
            # Passamos 'semana_selecionada' para a função
            fig_historico, ultimo_registro_historico = criar_grafico_historico_semanal(df_historico, semana_selecionada)
            
            # Exibir Métricas baseadas no histórico daquela semana
            if ultimo_registro_historico is not None:
                total_tarefas = ultimo_registro_historico['Total_Tarefas']
                total_fechadas = ultimo_registro_historico['Total_Fechadas']
                total_abertas = total_tarefas - total_fechadas
                st.markdown(f"### Resumo da Semana Selecionada ({semana_selecionada})")
                col_met1, col_met2, col_met3 = st.columns(3)
                col_met1.metric("Total de Tarefas (Histórico)", total_tarefas)
                col_met2.metric("🔴 Abertas (Histórico)", total_abertas)
                col_met3.metric("🟢 Fechadas (Histórico)", total_fechadas)
            
            st.plotly_chart(fig_historico, use_container_width=True)
            st.markdown("---")

            # 4. TABELA PIVÔ (Lógica mantida, já usa a semana_selecionada)
            df_semana_full_FECHADAS = df_analise_executado[df_analise_executado['Nome_da_Semana'] == semana_selecionada]
            try:
                pivot = pd.pivot_table(df_semana_full_FECHADAS, index='Encarregado', columns='Nome Dia Semana', values='ID', aggfunc='count', fill_value=0)
                colunas_dias_ordem = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']
                pivot = pivot.reindex(columns=colunas_dias_ordem, fill_value=0)
                data_sexta = pd.to_datetime(semana_selecionada, format='%d/%m/%Y')
                data_segunda = data_sexta - pd.Timedelta(days=4); data_domingo = data_sexta + pd.Timedelta(days=2)
                st.subheader(f"Contagem de Tarefas Concluídas ({data_segunda.strftime('%d/%m')} a {data_domingo.strftime('%d/%m/%Y')})")
                st.dataframe(pivot, use_container_width=True)

                st.markdown("---")
                st.subheader("Detalhes das Tarefas da Semana (Abertas e Fechadas)")
                df_semana_COMPLETA = df_analise_filtrado_aba2[df_analise_filtrado_aba2['Nome_da_Semana'] == semana_selecionada].copy()
                encarregados_da_semana = sorted(df_semana_COMPLETA['Encarregado'].unique())
                colunas_mostrar_abertas = ['Nome Task', 'Link', 'Data Inicial']
                colunas_mostrar_fechadas = ['Nome Task', 'Link', 'Data Final']
                column_config = {"Link": st.column_config.LinkColumn("Link", display_text="Abrir ↗"), "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"), "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")}

                for encarregado in encarregados_da_semana:
                    df_enc = df_semana_COMPLETA[df_semana_COMPLETA['Encarregado'] == encarregado]
                    df_abertas = df_enc[df_enc['Status_Tarefa'] == 'Aberto'].sort_values(by='Data Inicial', ascending=True)
                    df_fechadas = df_enc[df_enc['Status_Tarefa'] == 'Executado'].sort_values(by='Data Final', ascending=True)
                    count_abertas = len(df_abertas); count_fechadas = len(df_fechadas); count_total = count_abertas + count_fechadas
                    with st.expander(f"{encarregado}  |  Total: {count_total} (🔴 Abertas: {count_abertas}, 🟢 Fechadas: {count_fechadas})"):
                        st.markdown("##### 🔴 Tarefas Abertas")
                        if df_abertas.empty: st.text("Nenhuma tarefa aberta esta semana.")
                        else: st.dataframe(df_abertas[colunas_mostrar_abertas], use_container_width=True, column_config=column_config, hide_index=True)
                        st.markdown("---")
                        st.markdown("##### 🟢 Tarefas Fechadas")
                        if df_fechadas.empty: st.text("Nenhuma tarefa fechada esta semana.")
                        else: st.dataframe(df_fechadas[colunas_mostrar_fechadas], use_container_width=True, column_config=column_config, hide_index=True)
            except Exception as e: st.error(f"Erro ao gerar a tabela pivô: {e}")

    with aba_mes:
        st.header("Análise Detalhada por Mês")
        # Gráfico de Histórico Mensal
        fig_historico_mes, ultimo_registro_mes = criar_grafico_historico_mensal(df_historico)
        
        if ultimo_registro_mes is not None:
            total_tarefas_mes = ultimo_registro_mes['Total_Tarefas']
            total_fechadas_mes = ultimo_registro_mes['Total_Fechadas']
            total_abertas_mes = total_tarefas_mes - total_fechadas_mes
            st.markdown("### Resumo do Mês Atual (até o momento)")
            col_met_m1, col_met_m2, col_met_m3 = st.columns(3)
            col_met_m1.metric("Total de Tarefas no Mês", total_tarefas_mes)
            col_met_m2.metric("🔴 Abertas", total_abertas_mes)
            col_met_m3.metric("🟢 Fechadas", total_fechadas_mes)

        st.plotly_chart(fig_historico_mes, use_container_width=True)
        st.markdown("---")

        # Lógica para Tabela Pivô do Mês Atual
        df_analise_filtrado_aba_mes = df_analise.copy()
        if "Todos" not in st.session_state.encarregado_filtro: df_analise_filtrado_aba_mes = df_analise_filtrado_aba_mes[df_analise_filtrado_aba_mes['Encarregado'].isin(st.session_state.encarregado_filtro)]
        if st.session_state.contrato_filtro != "Todos": df_analise_filtrado_aba_mes = df_analise_filtrado_aba_mes[df_analise_filtrado_aba_mes['Status_Funcionario'] == st.session_state.contrato_filtro]

        # Filtrar apenas Executadas no Mês Atual
        hoje = pd.Timestamp.now().normalize()
        inicio_mes_atual = hoje.replace(day=1)
        proximo_mes_dt = (inicio_mes_atual + pd.DateOffset(months=1))
        fim_mes_atual = proximo_mes_dt - pd.Timedelta(days=1)

        df_analise_executado_mes = df_analise_filtrado_aba_mes[
            (df_analise_filtrado_aba_mes['Status_Tarefa'] == 'Executado') &
            (df_analise_filtrado_aba_mes['Data Final (aberta)'] >= inicio_mes_atual) &
            (df_analise_filtrado_aba_mes['Data Final (aberta)'] <= fim_mes_atual)
        ].copy()

        if df_analise_executado_mes.empty:
            st.info(f"Nenhuma tarefa executada encontrada neste mês ({inicio_mes_atual.strftime('%B/%Y')}) para os filtros selecionados.")
        else:
            try:
                # Pivot por DIA DO MÊS (1 a 31)
                df_analise_executado_mes['Dia_do_Mes'] = df_analise_executado_mes['Data Final (aberta)'].dt.day
                pivot_mes = pd.pivot_table(df_analise_executado_mes, index='Encarregado', columns='Dia_do_Mes', values='ID', aggfunc='count', fill_value=0)
                
                # Garantir que todas as colunas (dias) apareçam ordenadas
                dias_existentes = sorted(pivot_mes.columns.tolist())
                pivot_mes = pivot_mes.reindex(columns=dias_existentes, fill_value=0)

                st.subheader(f"Contagem de Tarefas Concluídas - {inicio_mes_atual.strftime('%B/%Y')}")
                st.dataframe(pivot_mes, use_container_width=True)

                st.markdown("---")
                st.subheader("Detalhes das Tarefas do Mês Atual")
                
                # Detalhes (Abertas e Fechadas no Mês)
                # Para "Abertas", consideramos aquelas criadas ou pendentes no mês? 
                # Simplificação: Vamos mostrar todas do mês com base na Data Final (executadas) e Data Inicial (abertas que começaram este mês)
                
                df_mes_detalhes = df_analise_filtrado_aba_mes[
                    ((df_analise_filtrado_aba_mes['Status_Tarefa'] == 'Executado') & (df_analise_filtrado_aba_mes['Data Final (aberta)'] >= inicio_mes_atual) & (df_analise_filtrado_aba_mes['Data Final (aberta)'] <= fim_mes_atual)) |
                    ((df_analise_filtrado_aba_mes['Status_Tarefa'] == 'Aberto') & (df_analise_filtrado_aba_mes['Data Inicial'] >= inicio_mes_atual))
                ].copy()

                encarregados_do_mes = sorted(df_mes_detalhes['Encarregado'].unique())
                colunas_mostrar_abertas = ['Nome Task', 'Link', 'Data Inicial']
                colunas_mostrar_fechadas = ['Nome Task', 'Link', 'Data Final']
                column_config = {"Link": st.column_config.LinkColumn("Link", display_text="Abrir ↗"), "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"), "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")}

                for encarregado in encarregados_do_mes:
                    df_enc = df_mes_detalhes[df_mes_detalhes['Encarregado'] == encarregado]
                    df_abertas = df_enc[df_enc['Status_Tarefa'] == 'Aberto'].sort_values(by='Data Inicial', ascending=True)
                    df_fechadas = df_enc[df_enc['Status_Tarefa'] == 'Executado'].sort_values(by='Data Final', ascending=True)
                    count_abertas = len(df_abertas); count_fechadas = len(df_fechadas); count_total = count_abertas + count_fechadas
                    
                    with st.expander(f"{encarregado}  |  Total: {count_total} (🔴 Abertas: {count_abertas}, 🟢 Fechadas: {count_fechadas})"):
                        st.markdown("##### 🔴 Tarefas Abertas (Iniciadas neste mês)")
                        if df_abertas.empty: st.text("Nenhuma tarefa aberta iniciada este mês.")
                        else: st.dataframe(df_abertas[colunas_mostrar_abertas], use_container_width=True, column_config=column_config, hide_index=True)
                        st.markdown("---")
                        st.markdown("##### 🟢 Tarefas Fechadas (Neste mês)")
                        if df_fechadas.empty: st.text("Nenhuma tarefa fechada neste mês.")
                        else: st.dataframe(df_fechadas[colunas_mostrar_fechadas], use_container_width=True, column_config=column_config, hide_index=True)

            except Exception as e: st.error(f"Erro ao gerar a tabela do mês: {e}")

    with aba_backlog:
        st.header("Backlog de Tarefas por Status")
        df_backlog_filtrado = df_backlog.copy()
        if "Todos" not in st.session_state.encarregado_filtro: df_backlog_filtrado = df_backlog_filtrado[df_backlog_filtrado['Encarregado'].isin(st.session_state.encarregado_filtro + ["Em Branco"])] 
        if st.session_state.contrato_filtro != "Todos": df_backlog_filtrado = df_backlog_filtrado[df_backlog_filtrado['Status_Funcionario'].isin([st.session_state.contrato_filtro, "Outros"])]
        
        if df_backlog.empty: st.error("Aba 'Backlog' não foi carregada ou está vazia.")
        else:
            colunas_backlog_para_mostrar = ['Nome Task', 'Encarregado', 'Link', 'Lista', 'Data Inicial', 'Data Final', 'ID']
            if 'Nome Task' not in df_backlog_filtrado.columns:
                colunas_backlog_para_mostrar = ['ID', 'Encarregado', 'Link', 'Lista', 'Data Inicial', 'Data Final']
                st.warning("Coluna 'Nome Task' não encontrada na aba 'Backlog'. Exibindo 'ID'.")
            column_config_backlog = {"Link": st.column_config.LinkColumn("Tarefa Link", display_text="Abrir ↗"), "Lista": st.column_config.LinkColumn("Lista Link", display_text="Abrir ↗"), "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"), "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")}
            
            df_abertas_sem_resp = df_backlog_filtrado[(df_backlog_filtrado['Status_Backlog'] == 'Aberto') & (df_backlog_filtrado['Encarregado'] == 'Em Branco')]
            with st.expander(f"⚫ Tarefas Abertas (Em Branco) - {len(df_abertas_sem_resp)}", expanded=True): st.dataframe(df_abertas_sem_resp[colunas_backlog_para_mostrar], use_container_width=True, column_config={**column_config_backlog, "Encarregado": None}, hide_index=True)
            
            df_abertas_com_resp = df_backlog_filtrado[(df_backlog_filtrado['Status_Backlog'] == 'Aberto') & (df_backlog_filtrado['Encarregado'] != 'Em Branco')]
            with st.expander(f"🔴 Tarefas Abertas (Com Responsável) - {len(df_abertas_com_resp)}", expanded=True): st.dataframe(df_abertas_com_resp[colunas_backlog_para_mostrar].sort_values(by="Encarregado"), use_container_width=True, column_config=column_config_backlog, hide_index=True)

            df_fechadas = df_backlog_filtrado[(df_backlog_filtrado['Status_Backlog'] == 'Fechado')]
            with st.expander(f"🟢 Tarefas Fechadas (Com ou Sem Responsável) - {len(df_fechadas)}", expanded=False): st.dataframe(df_fechadas[colunas_backlog_para_mostrar].sort_values(by="Data Final", ascending=False), use_container_width=True, column_config=column_config_backlog, hide_index=True)

    with aba_geral:
        st.header("Visão Geral da Atividade")
        col_geral1, col_geral2 = st.columns(2)
        with col_geral1: fig_prod_mensal = criar_grafico_produtividade_mensal(df_filtrado_aba1); st.plotly_chart(fig_prod_mensal, use_container_width=True)
        with col_geral2: fig_principal = criar_grafico_principal(df_filtrado_aba1); st.plotly_chart(fig_principal, use_container_width=True)
        st.markdown("---"); st.header("Análise de Equipe e Status")
        col_equipe1, col_equipe2 = st.columns(2)
        with col_equipe1: fig_tarefas = criar_grafico_tarefas_funcionarios(df_filtrado_aba1); st.plotly_chart(fig_tarefas, use_container_width=True)
        with col_equipe2: fig_status = criar_grafico_status_tarefas(df_filtrado_aba1); st.plotly_chart(fig_status, use_container_width=True)
        
    with aba_pontuacao:
        nomes_status_filtrados = []
        if df_equipe is not None and not df_equipe.empty:
            if st.session_state.contrato_filtro == "Todos": nomes_status_filtrados = df_equipe['Nome'].unique().tolist()
            else: nomes_status_filtrados = df_equipe[df_equipe['Status_Funcionario'] == st.session_state.contrato_filtro]['Nome'].unique().tolist()
        nomes_encarregado_filtrados = []
        if "Todos" in st.session_state.encarregado_filtro:
            if df_equipe is not None and not df_equipe.empty: nomes_encarregado_filtrados = df_equipe['Nome'].unique().tolist()
            else: nomes_encarregado_filtrados = df_analise['Encarregado'].unique().tolist()
        else: nomes_encarregado_filtrados = st.session_state.encarregado_filtro
        lista_nomes_final_para_exibir = list(set(nomes_status_filtrados) & set(nomes_encarregado_filtrados))
        
        st.header("Ranking de Pontuação Individual")
        fig_pontuacao_individual, df_tabela_individual = criar_grafico_pontuacao_individual(df_notas_tabela1, lista_nomes_final_para_exibir, start_date, end_date)
        st.plotly_chart(fig_pontuacao_individual, use_container_width=True)
        with st.expander("Ver tabela de dados (Pontuação Individual)"):
            if not df_tabela_individual.empty: st.dataframe(df_tabela_individual, use_container_width=True, hide_index=True)
            else: st.info("Nenhum dado de pontuação individual para exibir com os filtros atuais.")
        st.markdown("---")
        
        st.header("Ranking de Pontuação (Apenas Liderança)")
        fig_pontuacao_lideres, df_lideres_visiveis, df_liderados_pontos_detalhe = criar_grafico_pontuacao_lideres(df_lideranca_mapa, df_notas_tabela2, lista_nomes_final_para_exibir, start_date, end_date)
        st.plotly_chart(fig_pontuacao_lideres, use_container_width=True)
        with st.expander("Ver tabelas de dados (Detalhes por Líder)"):
            if not df_lideres_visiveis.empty and not df_liderados_pontos_detalhe.empty:
                lideres_para_exibir_lista = df_lideres_visiveis['Lider'].tolist()
                for lider in lideres_para_exibir_lista:
                    st.subheader(f"Detalhes da Pontuação: {lider}")
                    liderados_deste_lider = df_lideranca_mapa[df_lideranca_mapa['Lider'] == lider]['Liderado'].tolist()
                    df_tabela_lider = df_liderados_pontos_detalhe[df_liderados_pontos_detalhe['Encarregado'].isin(liderados_deste_lider)]
                    df_tabela_lider_sorted = df_tabela_lider.sort_values(by='Pontuacao_Total_Liderado', ascending=False)
                    st.dataframe(df_tabela_lider_sorted, use_container_width=True, hide_index=True)
            else: st.info("Nenhum dado de pontuação de liderança para exibir com os filtros atuais.")
        st.markdown("---")

        st.header("Ranking Geral de Pontuação (Individual + Liderança)")
        fig_pontuacao_combinada = criar_grafico_pontuacao_combinada(df_notas_tabela1, df_notas_tabela2, df_lideranca_mapa, lista_nomes_final_para_exibir, start_date, end_date)
        st.plotly_chart(fig_pontuacao_combinada, use_container_width=True)
        st.info("Este gráfico soma a pontuação individual e de liderança no período selecionado.")
else:
    if (df_notas_tabela1 is not None):
        st.warning("Não foi possível carregar os dados da aba 'Total BaseCamp para Notas', mas as abas de pontuação foram carregadas.")
        st.info("A aba 'Pontuação Geral' pode estar funcional, mas 'Atividade Geral' está desabilitada.")
        aba_pontuacao_fallback = st.tabs(["Pontuação Geral"])[0]
        if aba_pontuacao_fallback:
            with aba_pontuacao_fallback:
                if df_equipe is not None and not df_equipe.empty: nomes_para_exibir_fallback = df_equipe['Nome'].unique().tolist()
                elif df_notas_tabela1 is not None: nomes_para_exibir_fallback = df_notas_tabela1['Encarregado'].unique().tolist()
                else: nomes_para_exibir_fallback = []
                st.header("Ranking de Pontuação Individual")
                fig_pontuacao_individual, df_tabela_individual_fb = criar_grafico_pontuacao_individual(df_notas_tabela1, nomes_para_exibir_fallback, min_date, max_date)
                st.plotly_chart(fig_pontuacao_individual, use_container_width=True)
                with st.expander("Ver tabela de dados (Pontuação Individual)"): st.dataframe(df_tabela_individual_fb, use_container_width=True, hide_index=True)
                st.markdown("---")
                st.header("Ranking de Pontuação (Apenas Liderança)")
                fig_pontuacao_lideres, df_lideres_visiveis_fb, df_liderados_pontos_detalhe_fb = criar_grafico_pontuacao_lideres(df_lideranca_mapa, df_notas_tabela2, nomes_para_exibir_fallback, min_date, max_date)
                st.plotly_chart(fig_pontuacao_lideres, use_container_width=True)
                with st.expander("Ver tabelas de dados (Detalhes por Líder)"):
                    if not df_lideres_visiveis_fb.empty and not df_liderados_pontos_detalhe_fb.empty:
                        lideres_para_exibir_lista_fb = df_lideres_visiveis_fb['Lider'].tolist()
                        for lider in lideres_para_exibir_lista_fb:
                            st.subheader(f"Detalhes da Pontuação: {lider}")
                            liderados_deste_lider_fb = df_lideranca_mapa[df_lideranca_mapa['Lider'] == lider]['Liderado'].tolist()
                            df_tabela_lider_fb = df_liderados_pontos_detalhe_fb[df_liderados_pontos_detalhe_fb['Encarregado'].isin(liderados_deste_lider_fb)]
                            df_tabela_lider_sorted_fb = df_tabela_lider_fb.sort_values(by='Pontuacao_Total_Liderado', ascending=False)
                            st.dataframe(df_tabela_lider_sorted_fb, use_container_width=True, hide_index=True)
                st.markdown("---")
                st.header("Ranking Geral de Pontuação (Individual + Liderança)")
                fig_pontuacao_combinada = criar_grafico_pontuacao_combinada(df_notas_tabela1, df_notas_tabela2, df_lideranca_mapa, nomes_para_exibir_fallback, min_date, max_date)
                st.plotly_chart(fig_pontuacao_combinada, use_container_width=True)
    else:
        st.error("Não foi possível carregar nenhum dado para exibir o dashboard.")