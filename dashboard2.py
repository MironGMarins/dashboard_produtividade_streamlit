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
    series = series.astype(str).str.strip()
    series = series.replace(['nan', 'None', '', 'NaT', '0', '#N/A', 'nan'], np.nan)
    return pd.to_datetime(series, dayfirst=True, errors='coerce')

def tornar_colunas_unicas(lista_colunas):
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

def aplicar_heatmap_vermelho(df):
    return df.style.background_gradient(cmap='Reds', axis=None).format(precision=0)

# ==============================================================================
# FUNÇÃO DE CARREGAMENTO DE DADOS (CACHE)
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados_completos():
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
    
    nome_aba_dados = "Total BaseCamp para Notas" 
    nome_aba_equipes = "Equipes"
    nome_aba_pontuacao = "Notas"
    nome_aba_lideranca = "Liderança"
    nome_aba_backlog = "Backlog" 
    nome_aba_source = "Total BaseCamp"
    nome_aba_historico = "HistoricoDiario"
    
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

    if not df_equipe.empty:
        if 'Status' in df_equipe.columns:
            df_equipe.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)
        # --- ATUALIZAÇÃO: Conversão da Data de Saída na aba Equipes ---
        if 'Data de Saída' in df_equipe.columns:
            df_equipe['Data de Saída'] = converter_data_robusta(df_equipe['Data de Saída'])

    for df_temp in [df_dados, df_equipe, df_lideranca, df_backlog, df_source, df_historico]:
            if not df_temp.empty:
                df_temp.columns = df_temp.columns.astype(str).str.strip()

    df_grafico = df_dados.copy()
    colunas_para_numerico = ['Pablo', 'Leonardo', 'Itiel', 'Ítalo']
    for col in colunas_para_numerico:
        if col not in df_grafico.columns: df_grafico[col] = 0
        else: df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce').fillna(0)

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
    
    data_fim_analise = pd.Timestamp.now().normalize() + pd.Timedelta(days=365)
    
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
    
    if not df_source.empty:
        df_source_proc = df_source.copy()
        if 'Data Inicial' in df_source_proc.columns: df_source_proc['Data Inicial'] = converter_data_robusta(df_source_proc.get('Data Inicial', pd.Series()))
        if 'Data Final' in df_source_proc.columns: df_source_proc['Data Final'] = converter_data_robusta(df_source_proc.get('Data Final', pd.Series()))
        df_source_proc['Status_Tarefa'] = np.where(df_source_proc['Data Final'].isnull(), 'Aberto', 'Executado')
        df_source_proc['Data Final (aberta)'] = df_source_proc['Data Final'].fillna(pd.Timestamp.now().normalize())
        
        if 'Encarregado' in df_source_proc.columns: df_source_proc['Encarregado'] = df_source_proc['Encarregado'].astype(str).str.strip().replace('', 'Em Branco')
        if 'Nome Task' in df_source_proc.columns: df_source_proc['Nome Task'] = df_source_proc['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
        
        # Merge seguro
        if 'Data Final (aberta)' in df_source_proc.columns:
            df_source_analise = pd.merge(df_source_proc, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date')
            if 'Date' in df_source_analise.columns: df_source_analise = df_source_analise.drop(columns=['Date'])
        else:
            df_source_analise = df_source_proc
        
        if not df_equipe.empty:
            df_source_analise = pd.merge(df_source_analise, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
            if 'Status_Funcionario' in df_source_analise.columns: df_source_analise['Status_Funcionario'].fillna('Outros', inplace=True)
    
    if not df_backlog.empty:
        if 'Data Inicial' in df_backlog.columns: df_backlog['Data Inicial'] = converter_data_robusta(df_backlog.get('Data Inicial', pd.Series()))
        if 'Data Final' in df_backlog.columns: df_backlog['Data Final'] = converter_data_robusta(df_backlog.get('Data Final', pd.Series()))
        df_backlog['Status_Backlog'] = np.where(df_backlog['Data Final'].isnull(), 'Aberto', 'Fechado')
        if 'Encarregado' in df_backlog.columns: df_backlog['Encarregado'] = df_backlog['Encarregado'].astype(str).str.strip().replace('', 'Em Branco') 
        if 'Nome Task' in df_backlog.columns: df_backlog['Nome Task'] = df_backlog['Nome Task'].astype(str).str.strip().replace('', 'Vazio')
        if not df_equipe.empty:
            df_backlog = pd.merge(df_backlog, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
            if 'Status_Funcionario' in df_backlog.columns: df_backlog['Status_Funcionario'].fillna('Outros', inplace=True)
        else:
            df_backlog['Status_Funcionario'] = 'Outros'

    if not df_historico.empty:
        if 'Data Final' in df_historico.columns: df_historico.rename(columns={'Data Final': 'Data'}, inplace=True)
        if 'Data' in df_historico.columns:
            df_historico['Data'] = converter_data_robusta(df_historico['Data'])
            if 'Total_Fechadas' in df_historico.columns: df_historico['Total_Fechadas'] = pd.to_numeric(df_historico['Total_Fechadas'], errors='coerce').fillna(0)
            if 'Total_Tarefas' in df_historico.columns: df_historico['Total_Tarefas'] = pd.to_numeric(df_historico['Total_Tarefas'], errors='coerce').fillna(0)
            df_historico.dropna(subset=['Data'], inplace=True)
    
    if not df_analise.empty:
        if 'Data Inicial' in df_analise.columns: 
            df_analise['Data Inicial'] = df_analise['Data Inicial'].dt.date
        if 'Data Final' in df_analise.columns: 
            df_analise['Data Final'] = df_analise['Data Final'].dt.date
            
    if not df_backlog.empty:
        if 'Data Inicial' in df_backlog.columns: 
            df_backlog['Data Inicial'] = df_backlog['Data Inicial'].dt.date
        if 'Data Final' in df_backlog.columns: 
            df_backlog['Data Final'] = df_backlog['Data Final'].dt.date

    return df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca, df_equipe, df_backlog, df_source_analise, df_historico

# ==============================================================================
# FUNÇÕES GRÁFICAS ESTRUTURADAS
# ==============================================================================
def criar_grafico_historico_semanal(df_historico, semana_selecionada_str=None):
    if df_historico is None or df_historico.empty: return go.Figure().update_layout(title="Sem dados históricos", template='plotly_white'), None
    if semana_selecionada_str:
        try:
            data_referencia = pd.to_datetime(semana_selecionada_str, format='%d/%m/%Y')
            fim_semana = data_referencia 
            inicio_semana = fim_semana - pd.Timedelta(days=4)
            titulo_grafico = f"<b>Progresso ({inicio_semana.strftime('%d/%m')} a {fim_semana.strftime('%d/%m')})</b>"
        except ValueError:
            titulo_grafico = "Erro na Data"; inicio_semana = pd.Timestamp.min; fim_semana = pd.Timestamp.max
    else:
        inicio_semana = pd.Timestamp.min; fim_semana = pd.Timestamp.max; titulo_grafico = "Selecione uma semana"

    df_filt = df_historico[(df_historico['Data'] >= inicio_semana) & (df_historico['Data'] <= fim_semana)].sort_values('Data')
    if df_filt.empty: return go.Figure().update_layout(title=f"{titulo_grafico}<br><i>Sem registros diários para esta semana.</i>", template='plotly_white'), None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filt['Data'], y=df_filt['Total_Tarefas'], mode='lines+markers+text', name='Total', line=dict(color='red', width=3), text=df_filt['Total_Tarefas'], textposition='top center'))
    fig.add_trace(go.Scatter(x=df_filt['Data'], y=df_filt['Total_Fechadas'], mode='lines+markers+text', name='Fechadas', line=dict(color='green', width=3), text=df_filt['Total_Fechadas'], textposition='bottom center'))
    fig.update_layout(title=titulo_grafico, template='plotly_white', legend=dict(orientation="h", y=1.1))
    range_semana = pd.date_range(start=inicio_semana, end=fim_semana)
    dias_pt = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex']
    tick_text = [f"{dias_pt[d.dayofweek]} ({d.strftime('%d/%m')})" for d in range_semana]
    fig.update_xaxes(tickmode='array', tickvals=range_semana, ticktext=tick_text)
    return fig, df_filt.iloc[-1]

def criar_grafico_historico_mensal(df_historico, data_referencia=None):
    if df_historico is None or df_historico.empty: return go.Figure(), None
    
    if data_referencia is None:
        hoje = pd.Timestamp.now().normalize()
        inicio = hoje.replace(day=1)
    else:
        inicio = pd.Timestamp(data_referencia).normalize().replace(day=1)
        
    fim = (inicio + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    
    df_mes = df_historico[(df_historico['Data'] >= inicio) & (df_historico['Data'] <= fim)].sort_values('Data')
    
    meses_full = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Jun', 7: 'Jul', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
    nome_mes_pt = meses_full.get(inicio.month, inicio.strftime('%B')).lower()

    if df_mes.empty: 
        return go.Figure().update_layout(title=f"<b>Evolução Diária ({nome_mes_pt})</b> - Sem dados", template='plotly_white'), None

    df_c = df_mes.copy()
    df_c['Delta_F'] = df_c['Total_Fechadas'].diff().fillna(df_c['Total_Fechadas'])
    df_c.loc[df_c['Delta_F'] < 0, 'Delta_F'] = df_c.loc[df_c['Delta_F'] < 0, 'Total_Fechadas']
    df_c['Mensal_Fechadas'] = df_c['Delta_F'].cumsum()
    df_c['Backlog'] = df_c['Total_Tarefas'] - df_c['Total_Fechadas']
    df_c['Mensal_Tarefas'] = df_c['Mensal_Fechadas'] + df_c['Backlog']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_c['Data'], y=df_c['Mensal_Tarefas'], name='Total (Mês)', mode='lines+markers', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df_c['Data'], y=df_c['Mensal_Fechadas'], name='Fechadas (Mês)', mode='lines+markers', line=dict(color='green')))
    fig.update_layout(title=f"<b>Evolução Diária ({nome_mes_pt})</b>", template='plotly_white', legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(tickformat="%d/%m")
    return fig, df_c.iloc[-1]

def criar_grafico_produtividade_mensal(df):
    if df.empty: return go.Figure()
    agg = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev']).agg(qtd=('ID', 'count')).reset_index().sort_values('Ano-Mês')
    fig = px.bar(agg, x='Mes_Ano_Abrev', y='qtd', text='qtd', title="<b>Produtividade Mensal</b>")
    fig.update_layout(template='plotly_white', margin=dict(l=50, r=50, t=40, b=40))
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
    fig.add_trace(go.Scatter(x=df_dia_total['Dia'], y=df_dia_total['Contagem'], name='Soma (Dias)', visible=True, mode='lines+markers+text', text=df_dia_total['Contagem'], textposition='top center', line=dict(color='royalblue', width=3))) 
    fig.add_trace(go.Scatter(x=df_semana_total['Semana do Mês'], y=df_semana_total['Contagem'], name='Soma (Semanas)', visible=False, mode='lines+markers+text', text=df_semana_total['Contagem'], textposition='top center', line=dict(color='royalblue', width=3))) 
    fig.add_trace(go.Scatter(x=df_diasemana_total['Nome Dia Semana'], y=df_diasemana_total['Contagem'], name='Soma (Dia Semana)', visible=False, mode='lines+markers+text', text=df_diasemana_total['Contagem'], textposition='top center', line=dict(color='royalblue', width=3))) 
    
    offset_dia = 3
    for mes in opcoes_meses:
        d = df_dia[df_dia['Mes_Ano_Abrev'] == mes].sort_values('Dia')
        d['Contagem'] = d['Contagem'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=d['Dia'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem'], textposition='top center'))
    count_dia = len(opcoes_meses)

    offset_semana = offset_dia + count_dia
    for mes in opcoes_meses:
        d = df_semana[df_semana['Mes_Ano_Abrev'] == mes].sort_values('Semana do Mês')
        d['Contagem'] = d['Contagem'].replace(0, np.nan)
        fig.add_trace(go.Scatter(x=d['Semana do Mês'], y=d['Contagem'], name=mes, visible=False, mode='lines+markers+text', text=d['Contagem'], textposition='top center'))
    count_semana = len(opcoes_meses)

    offset_diasemana = offset_semana + count_semana
    diasemana_trace_map = [] 
    for mes in opcoes_meses:
        d_mes = df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes].groupby('Nome Dia Semana')['Contagem'].sum().reindex(ordem_dias).fillna(0).reset_index()
        d_mes['Contagem'] = recortar_zeros_pontas(d_mes['Contagem'])
        d_mes = d_mes.dropna(subset=['Contagem'])
        fig.add_trace(go.Scatter(x=d_mes['Nome Dia Semana'], y=d_mes['Contagem'], name=f"{mes} Agregado", visible=False, mode='lines+markers+text', text=d_mes['Contagem'], textposition='top center'))
        semanas_do_mes = sorted(df_diasemana_full[df_diasemana_full['Mes_Ano_Abrev'] == mes]['Semana do Mês'].unique())
        for sem in semanas_do_mes:
            d_sem = df_diasemana_full[(df_diasemana_full['Mes_Ano_Abrev'] == mes) & (df_diasemana_full['Semana do Mês'] == sem)].set_index('Nome Dia Semana').reindex(ordem_dias).fillna(0).reset_index()
            d_sem['Contagem'] = recortar_zeros_pontas(d_sem['Contagem'])
            d_sem = d_sem.dropna(subset=['Contagem'])
            fig.add_trace(go.Scatter(x=d_sem['Nome Dia Semana'], y=d_sem['Contagem'], name=f"{mes} Sem {sem}", visible=False, mode='lines+markers+text', text=d_sem['Contagem'], textposition='top center'))
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
        title={'text': "<b>Gráfico Principal</b>", 'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        height=600,
        margin=dict(l=50, r=50, t=120, b=50),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[
            dict(type="buttons", direction="right", x=1.0, y=1.2, xanchor="right", yanchor="top", buttons=[
                dict(label="Dia do Mês", method="update", args=[{"visible": vis_init_dia}, {"updatemenus[1].buttons": buttons_dia, "xaxis.title": "Dia", "xaxis.type": "linear", "xaxis.categoryarray": None}]),
                dict(label="Semana do Mês", method="update", args=[{"visible": vis_init_semana}, {"updatemenus[1].buttons": buttons_semana, "xaxis.title": "Semana", "xaxis.type": "linear", "xaxis.categoryarray": None}]),
                dict(label="Dia da Semana", method="update", args=[{"visible": vis_init_diasemana}, {"updatemenus[1].buttons": buttons_diasemana, "xaxis.title": "Dia da Semana", "xaxis.type": "category", "xaxis.categoryorder": "array", "xaxis.categoryarray": ordem_dias}])
            ]),
            dict(direction="down", x=0.0, y=1.2, xanchor="left", yanchor="top", showactive=True, buttons=buttons_dia)
        ],
        xaxis=dict(title="Tempo", showgrid=False, showline=True, linecolor='black'),
        yaxis=dict(title="Quantidade", showgrid=True, gridcolor='lightgray')
    )
    return fig

def criar_grafico_tarefas_funcionarios(df):
    if df.empty: return go.Figure()
    v = df['Encarregado'].value_counts().reset_index(); v.columns=['Encarregado','c']
    fig = px.bar(v, x='c', y='Encarregado', orientation='h', text='c', title="<b>Tarefas por Pessoa</b>")
    fig.update_layout(template='plotly_white', yaxis_categoryorder='total ascending')
    return fig

def criar_grafico_status_tarefas(df):
    if df.empty: return go.Figure().update_layout(title="<b>Distribuição por Status</b>")
    v = df['Status_Tarefa'].value_counts().reset_index()
    v.columns = ['Status', 'Contagem']
    fig = px.pie(v, names='Status', values='Contagem', title="<b>Distribuição por Status</b>", hole=0.4, 
                 color='Status', 
                 color_discrete_map={'Executado': 'royalblue', 'Aberto': 'firebrick', 'Desconhecido': 'gray'})
    fig.update_traces(textinfo='value+percent')
    return fig

# --- NOVO GRÁFICO DE CRESCIMENTO (PRODUTIVIDADE) COM DATA DE SAÍDA ---
def criar_grafico_crescimento_acumulado(df, lista_encarregados):
    if df.empty or not lista_encarregados:
        return go.Figure().update_layout(title="Sem dados ou nenhum encarregado selecionado", template='plotly_white')
    
    df_ex = df[df['Status_Tarefa'] == 'Executado'].copy()
    if df_ex.empty:
        return go.Figure().update_layout(title="Nenhuma tarefa executada no período", template='plotly_white')

    df_ex['Data'] = pd.to_datetime(df_ex['Data Final (aberta)']).dt.date
    
    d_min = df_ex['Data'].min()
    d_max = df_ex['Data'].max()
    if pd.isna(d_min) or pd.isna(d_max): return go.Figure()

    idx = pd.date_range(d_min, d_max)
    
    fig = go.Figure()
    
    # --- CÁLCULO DA MÉDIA DINÂMICA (COM SAÍDA) ---
    # 1. Data de Entrada (Primeira tarefa no período)
    first_appearance = df_ex.groupby('Encarregado')['Data'].min()
    
    # 2. Data de Saída (Se existir na coluna, senão None)
    exit_dates = pd.Series(pd.NaT, index=first_appearance.index)
    if 'Data de Saída' in df_ex.columns:
        # Pega a data de saída (assume que é igual em todas as linhas do mesmo encarregado)
        raw_exit = df_ex.groupby('Encarregado')['Data de Saída'].first()
        # Converte para .dt.date para comparação
        for enc, dt in raw_exit.items():
            if pd.notna(dt):
                # Se for timestamp, pega date(). Se for date, usa direto.
                try: exit_dates[enc] = dt.date()
                except: exit_dates[enc] = dt

    active_counts_list = []
    
    # Loop dia a dia para calcular Headcount preciso
    for current_day_ts in idx:
        current_date = current_day_ts.date()
        
        count_active = 0
        for enc in first_appearance.index:
            start_date = first_appearance[enc]
            exit_date = exit_dates.get(enc, pd.NaT)
            
            # Está ativo SE: Já começou E (Não saiu OU Saiu no futuro)
            # Regra: "descontados APÓS essa data". Então no dia da saída ainda conta.
            if start_date <= current_date:
                if pd.isna(exit_date) or current_date <= exit_date:
                    count_active += 1
        
        active_counts_list.append(count_active)
            
    active_team_size = pd.Series(active_counts_list, index=idx)
    active_team_size = active_team_size.replace(0, 1) # Evita divisão por zero

    daily_total_tasks = df_ex.groupby('Data').size()
    daily_total_tasks.index = pd.to_datetime(daily_total_tasks.index)
    daily_total_tasks = daily_total_tasks.reindex(idx, fill_value=0)
    
    daily_avg_productivity = daily_total_tasks / active_team_size
    s_media_acumulada = daily_avg_productivity.cumsum()
    
    fig.add_trace(go.Scatter(
        x=s_media_acumulada.index, 
        y=s_media_acumulada.values, 
        name='Média Dinâmica (Equipe Ativa)', 
        line=dict(color='black', width=4, dash='dot'),
        mode='lines',
        hovertemplate='Data: %{x}<br>Média Acumulada: %{y:.1f}<br>Equipe Ativa: %{customdata} pessoas<extra></extra>',
        customdata=active_team_size
    ))
    
    colors = px.colors.qualitative.Plotly
    for i, nome in enumerate(lista_encarregados):
        df_u = df_ex[df_ex['Encarregado'] == nome]
        if df_u.empty: continue
        
        s_u = df_u.groupby('Data').size()
        s_u.index = pd.to_datetime(s_u.index)
        s_u = s_u.reindex(idx, fill_value=0).cumsum()
        
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=s_u.index, 
            y=s_u.values, 
            name=nome, 
            mode='lines+markers',
            line=dict(color=c, width=2)
        ))
        
    fig.update_layout(
        title="<b>Curva de Produtividade Acumulada (Entregas)</b>",
        template='plotly_white',
        xaxis=dict(title="Tempo"),
        yaxis=dict(title="Tarefas Entregues (Acumulado)"),
        hovermode="x unified"
    )
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

all_dates = []
if df_analise is not None and not df_analise.empty:
    if 'Data Final (aberta)' in df_analise.columns:
        all_dates.extend(pd.to_datetime(df_analise['Data Final (aberta)']).dt.date.dropna().tolist())

if df_notas_tabela1 is not None:
    for c in df_notas_tabela1.columns:
        dt = converter_data_robusta(pd.Series([c]))
        if pd.notna(dt[0]): all_dates.append(dt[0].date())

if all_dates:
    min_date, max_date = min(all_dates), max(all_dates)
else:
    min_date, max_date = date.today(), date.today()

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

t1, t2, t3, t4, t5, t6 = st.tabs(["Semana", "Mês", "Produtividade", "Backlog", "Geral", "Pontuação"])

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
        dates = pd.to_datetime(df_f['Data Final (aberta)'])
        fridays = dates - pd.to_timedelta(dates.dt.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
        set_semanas.update(fridays.unique())

    lista_semanas = sorted([d for d in list(set_semanas) if pd.notnull(d)], reverse=True)
    lista_str = [d.strftime('%d/%m/%Y') for d in lista_semanas]
    
    idx_padrao = 0
    sexta_atual_str = sexta_atual.strftime('%d/%m/%Y')
    if sexta_atual_str in lista_str:
        idx_padrao = lista_str.index(sexta_atual_str)
    else:
        for i, s in enumerate(lista_str):
            if pd.to_datetime(s, dayfirst=True) <= sexta_atual:
                idx_padrao = i; break
    
    sem_sel = st.selectbox("Semana (Sexta-feira referência):", lista_str, index=idx_padrao)
    dt_sel = pd.to_datetime(sem_sel, dayfirst=True)
    d_seg = dt_sel - pd.Timedelta(days=4)
    d_dom = dt_sel + pd.Timedelta(days=2)
    str_periodo = f"({d_seg.strftime('%d/%m')} a {d_dom.strftime('%d/%m')})"

    st.subheader(f"Progresso da Semana {str_periodo}")
    
    fig_h, last_h = criar_grafico_historico_semanal(df_historico, sem_sel)
    if last_h is not None:
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Total", last_h['Total_Tarefas']); c_m2.metric("Abertas", last_h['Total_Tarefas']-last_h['Total_Fechadas']); c_m3.metric("Fechadas", last_h['Total_Fechadas'])
    st.plotly_chart(fig_h, use_container_width=True)
    
    if not df_f.empty:
        dates_f = pd.to_datetime(df_f['Data Final (aberta)'])
        fridays_f = dates_f - pd.to_timedelta(dates_f.dt.dayofweek, unit='d') + pd.to_timedelta(4, unit='d')
        df_sem = df_f[fridays_f.dt.date == dt_sel.date()]
        
        if not df_sem.empty:
            piv = pd.pivot_table(df_sem[df_sem['Status_Tarefa']=='Executado'], index='Encarregado', columns='Nome Dia Semana', values='ID', aggfunc='count', fill_value=0)
            ordem_dias = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']
            piv = piv.reindex(columns=ordem_dias, fill_value=0)
            st.subheader("Resumo da Semana")
            st.dataframe(aplicar_heatmap_vermelho(piv), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Detalhes da Semana")
        
        for enc in sorted(df_sem['Encarregado'].unique()):
            d_e = df_sem[df_sem['Encarregado'] == enc]
            ab = d_e[d_e['Status_Tarefa']=='Aberto']
            fe = d_e[d_e['Status_Tarefa']=='Executado']
            with st.expander(f"{enc} ({len(d_e)}) - 🔴 {len(ab)} | 🟢 {len(fe)}"):
                column_config_semana = {
                    "Link": st.column_config.LinkColumn("Link", display_text="Abrir ↗"),
                    "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"),
                    "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")
                }
                if not ab.empty: 
                    st.caption("Abertas")
                    st.dataframe(ab[['Nome Task','Data Inicial', 'Link']], use_container_width=True, hide_index=True, column_config=column_config_semana)
                if not fe.empty: 
                    st.caption("Fechadas")
                    st.dataframe(fe[['Nome Task','Data Final', 'Link']], use_container_width=True, hide_index=True, column_config=column_config_semana)

with t2: # MÊS
    if not df_f.empty and 'Data Final (aberta)' in df_f.columns:
        df_f['Periodo_Mes_Ref'] = df_f['Data Final (aberta)'].dt.to_period('M')
        periodos_unicos_mes = sorted(df_f['Periodo_Mes_Ref'].dropna().unique(), reverse=True)
        
        if not periodos_unicos_mes:
             periodos_unicos_mes = [pd.Timestamp.now().to_period('M')]

        meses_full_list = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Jun', 7: 'Jul', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
        opcoes_formatadas_mes = [f"{meses_full_list[p.month]} {p.year}" for p in periodos_unicos_mes]

        hj_periodo = pd.Timestamp.now().to_period('M')
        idx_default_mes = 0
        if hj_periodo in periodos_unicos_mes:
            idx_default_mes = periodos_unicos_mes.index(hj_periodo)
        
        sel_mes_aba2_str = st.selectbox("Selecione o Mês para Visualizar:", opcoes_formatadas_mes, index=idx_default_mes)
        idx_sel_mes = opcoes_formatadas_mes.index(sel_mes_aba2_str)
        periodo_selecionado_mes = periodos_unicos_mes[idx_sel_mes]
        data_ref_mes_grafico = periodo_selecionado_mes.start_time

        nome_mes_header = meses_full_list.get(periodo_selecionado_mes.month, '').lower()
        st.markdown(f"### Progresso do Mês ({nome_mes_header})")

        fig_hm, last_hm = criar_grafico_historico_mensal(df_historico, data_referencia=data_ref_mes_grafico)
        
        if last_hm is not None:
            col_met_m1, col_met_m2, col_met_m3 = st.columns(3)
            col_met_m1.metric("Total Acumulado", f"{last_hm['Mensal_Tarefas']:.0f}")
            col_met_m2.metric("🔴 Gap (Abertas)", f"{last_hm['Mensal_Tarefas'] - last_hm['Mensal_Fechadas']:.0f}")
            col_met_m3.metric("🟢 Fechadas", f"{last_hm['Mensal_Fechadas']:.0f}")

        st.plotly_chart(fig_hm, use_container_width=True)
        
        df_m = df_f[df_f['Periodo_Mes_Ref'] == periodo_selecionado_mes].copy()
        
        if not df_m.empty:
            piv = pd.pivot_table(df_m[df_m['Status_Tarefa']=='Executado'], index='Encarregado', columns=df_m['Data Final (aberta)'].dt.day, values='ID', aggfunc='count', fill_value=0)
            st.subheader("Resumo do Mês")
            st.dataframe(aplicar_heatmap_vermelho(piv), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Detalhes do Mês")

            column_config_mes = {
                "Link": st.column_config.LinkColumn("Link", display_text="Abrir ↗"),
                "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"),
                "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")
            }
            
            for enc in sorted(df_m['Encarregado'].unique()):
                d_e = df_m[df_m['Encarregado'] == enc]
                ab = d_e[d_e['Status_Tarefa'] == 'Aberto'].sort_values(by='Data Inicial', ascending=True)
                fe = d_e[d_e['Status_Tarefa'] == 'Executado'].sort_values(by='Data Final', ascending=True)
                
                count_ab = len(ab); count_fe = len(fe); count_total = count_ab + count_fe

                with st.expander(f"{enc} ({count_total}) - 🔴 {count_ab} | 🟢 {count_fe}"):
                    if not ab.empty:
                        st.caption("🔴 Abertas")
                        st.dataframe(ab[['Nome Task', 'Data Inicial', 'Link']], use_container_width=True, hide_index=True, column_config=column_config_mes)
                    if not fe.empty:
                        st.caption("🟢 Fechadas")
                        st.dataframe(fe[['Nome Task', 'Data Final', 'Link']], use_container_width=True, hide_index=True, column_config=column_config_mes)
    else:
        st.info("Sem dados suficientes para gerar a visualização mensal.")

with t3: # PRODUTIVIDADE
    st.header("Curva de Produtividade (Acumulada)")
    
    modo_visualizacao = st.radio(
        "Escolha o escopo de tempo:",
        ["📅 Visão Mensal", "📈 Visão Geral (Histórico Completo)"],
        horizontal=True
    )
    
    st.markdown("---")

    if not df_f.empty and 'Data Final (aberta)' in df_f.columns:
        
        if modo_visualizacao == "📅 Visão Mensal":
            df_f['Periodo_Mes'] = df_f['Data Final (aberta)'].dt.to_period('M')
            periodos_unicos = sorted(df_f['Periodo_Mes'].dropna().unique(), reverse=True)
            
            if not periodos_unicos:
                st.info("Não há datas válidas para gerar a lista de meses.")
                df_prod_plot = pd.DataFrame()
            else:
                meses_full_prod = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Jun', 7: 'Jul', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
                opcoes_formatadas = [f"{meses_full_prod[p.month]} {p.year}" for p in periodos_unicos]
                
                hj_periodo = pd.Timestamp.now().to_period('M')
                idx_default = 0
                if hj_periodo in periodos_unicos:
                    idx_default = periodos_unicos.index(hj_periodo)
                    
                sel_mes_str = st.selectbox("Selecione o Mês de Referência:", opcoes_formatadas, index=idx_default)
                idx_sel = opcoes_formatadas.index(sel_mes_str)
                periodo_selecionado = periodos_unicos[idx_sel]
                
                df_prod_plot = df_f[df_f['Periodo_Mes'] == periodo_selecionado].copy()
                titulo_legenda = f"crescimento diário em **{sel_mes_str}**"

        else:
            df_prod_plot = df_f.copy()
            titulo_legenda = "crescimento acumulado de **todo o período**"

        if not df_prod_plot.empty:
            todos_enc = sorted(df_prod_plot['Encarregado'].unique())
            sel_enc_prod = st.multiselect(
                "Selecione Encarregados para Comparar:", 
                options=todos_enc,
                default=todos_enc 
            )
            
            st.caption(f"Exibindo {titulo_legenda}")
            fig_cresc = criar_grafico_crescimento_acumulado(df_prod_plot, sel_enc_prod)
            st.plotly_chart(fig_cresc, use_container_width=True)
        else:
            st.info("Sem dados de tarefas executadas para o período selecionado com os filtros atuais.")

    else:
        st.info("Sem dados de tarefas disponíveis para gerar a curva de produtividade.")

with t4: # BACKLOG (Movido para 4ª)
    if df_backlog is not None and not df_backlog.empty:
        bk = df_backlog.copy()
        if "Todos" not in st.session_state.encarregado_filtro: bk = bk[bk['Encarregado'].isin(st.session_state.encarregado_filtro + ['Em Branco'])]
        sem_dono = bk[(bk['Status_Backlog']=='Aberto') & (bk['Encarregado']=='Em Branco')]
        com_dono = bk[(bk['Status_Backlog']=='Aberto') & (bk['Encarregado']!='Em Branco')]
        fechado = bk[bk['Status_Backlog']=='Fechado']
        
        column_config_backlog = {
            "Link": st.column_config.LinkColumn("Tarefa Link", display_text="Abrir ↗"),
            "Lista": st.column_config.LinkColumn("Lista Link", display_text="Abrir ↗"),
            "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"),
            "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")
        }
        cols_bk = ['Nome Task', 'Data Inicial', 'Data Final', 'Link']
        if 'Lista' in bk.columns: cols_bk.insert(1, 'Lista')
        
        with st.expander(f"⚫ Sem Dono ({len(sem_dono)})"): 
            st.dataframe(sem_dono[cols_bk], use_container_width=True, hide_index=True, column_config=column_config_backlog)
        with st.expander(f"🔴 Com Dono ({len(com_dono)})", expanded=True): 
            cols_bk_com_dono = ['Encarregado'] + cols_bk
            st.dataframe(com_dono[cols_bk_com_dono].sort_values('Encarregado'), use_container_width=True, hide_index=True, column_config=column_config_backlog)
        with st.expander(f"🟢 Fechados ({len(fechado)})"): 
            cols_bk_fechado = ['Encarregado'] + cols_bk
            st.dataframe(fechado[cols_bk_fechado].sort_values('Data Final', ascending=False), use_container_width=True, hide_index=True, column_config=column_config_backlog)

with t5: # GERAL (Movido para 5ª)
    # GAP SMALL APLICADO
    c_g1, c_g2 = st.columns(2, gap="small")
    with c_g1: st.plotly_chart(criar_grafico_produtividade_mensal(df_f), use_container_width=True)
    with c_g2: st.plotly_chart(criar_grafico_principal(df_f), use_container_width=True)
    c_g3, c_g4 = st.columns(2)
    with c_g3: st.plotly_chart(criar_grafico_tarefas_funcionarios(df_f), use_container_width=True)
    with c_g4: st.plotly_chart(criar_grafico_status_tarefas(df_f), use_container_width=True)

with t6: # PONTUAÇÃO (Movido para 6ª)
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