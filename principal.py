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
from datetime import date

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Teste do Gráfico Principal"
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
        st.error("Credenciais do Google não encontradas. Verifique seus secrets ou o arquivo local.")
        return None
    
    client = gspread.authorize(creds)
    url_da_planilha = st.secrets.get("SHEET_URL", 'https://docs.google.com/spreadsheets/d/1juyOfIh0ZqsfJjN0p3gD8pKaAIX0R6IAPG9vysl7yWI/edit#gid=901870248')
    
    try:
        spreadsheet = client.open_by_url(url_da_planilha)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Planilha não encontrada. Verifique a URL.")
        return None

    # --- Carregamento das Abas ---
    nome_aba_dados = "Total BaseCamp para Notas" 
    nome_aba_equipes = "Equipes"
    
    try:
        worksheet_dados = spreadsheet.worksheet(nome_aba_dados)
        worksheet_equipe = spreadsheet.worksheet(nome_aba_equipes)
    except gspread.exceptions.WorksheetNotFound as e:
        st.error(f"Aba da planilha não encontrada: {e}")
        return None

    df_dados = pd.DataFrame(worksheet_dados.get_all_records())
    df_equipe = pd.DataFrame(worksheet_equipe.get_all_records())
    
    # --- PREPARAÇÃO DOS DADOS ---
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

    data_inicio = df_grafico['Data Inicial'].min() if pd.notna(df_grafico['Data Inicial'].min()) else pd.Timestamp.now()
    data_fim = pd.Timestamp.now()
    tabela_calendario = pd.DataFrame({"Date": pd.date_range(start=data_inicio, end=data_fim, freq='D')})
    tabela_calendario['Ano'] = tabela_calendario['Date'].dt.year
    tabela_calendario['Nome Mês'] = tabela_calendario['Date'].dt.strftime('%b').str.capitalize()
    tabela_calendario['Mes_Ano_Abrev'] = tabela_calendario['Nome Mês'] + '/' + tabela_calendario['Date'].dt.strftime('%y')
    tabela_calendario['Ano-Mês'] = tabela_calendario['Date'].dt.strftime('%Y-%m')
    tabela_calendario['Dia'] = tabela_calendario['Date'].dt.day
    tabela_calendario['Dia da Semana'] = tabela_calendario['Date'].dt.dayofweek + 1
    tabela_calendario['Nome Dia Semana'] = tabela_calendario['Date'].dt.dayofweek.map({0:'seg', 1:'ter', 2:'qua', 3:'qui', 4:'sex', 5:'sab', 6:'dom'})
    tabela_calendario['Semana do Mês'] = (tabela_calendario['Date'].dt.day - 1) // 7 + 1


    df_analise_temp = pd.merge(df_grafico, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date').drop(columns=['Date'])
    
    df_equipe.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)
    df_analise = pd.merge(df_analise_temp, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
    df_analise['Status_Funcionario'].fillna('Outros', inplace=True)

    return df_analise

# ==============================================================================
# FUNÇÃO PARA CRIAR O GRÁFICO PRINCIPAL
# ==============================================================================
def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Gráfico Principal</b>", template='plotly_white')
    
    def criar_figura_com_menu(df_contagem, df_agregado, col_x, col_filtro, nome_agregado, titulo, xaxis_titulo, yaxis_titulo, xaxis_extra=None):
        figura = go.Figure()
        figura.add_trace(go.Scatter(x=df_agregado[col_x], y=df_agregado['Contagem'], name=nome_agregado, mode='lines+markers+text', text=df_agregado['Contagem'], textposition='top center'))
        opcoes_filtro = sorted(df_contagem[col_filtro].unique())
        for opcao in opcoes_filtro:
            df_filtrado = df_contagem[df_contagem[col_filtro] == opcao]
            if not df_filtrado.empty:
                figura.add_trace(go.Scatter(x=df_filtrado[col_x], y=df_filtrado['Contagem'], name=opcao, mode='lines+markers+text', text=df_filtrado['Contagem'], textposition='top center', visible=False))
        
        botoes = [{'label': nome_agregado, 'method': 'update', 'args': [{'visible': [i == 0 for i in range(len(figura.data))]}]}]
        for i, trace in enumerate(figura.data[1:], 1):
            visibilidade_args = [False] * len(figura.data)
            visibilidade_args[i] = True
            botoes.append({'label': trace.name, 'method': 'update', 'args': [{'visible': visibilidade_args}]})
        
        # ### <<< ALTERAÇÃO AQUI >>> ###
        # Os valores de 'x' e 'y' foram ajustados para mover o menu dropdown.
        # A estrutura `updatemenus=[dict(...)]` foi garantida para evitar erros.
        figura.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=botoes,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=-0.0,        # Movido mais para a esquerda
                    xanchor="left",
                    y=1.19,        # Movido mais para cima
                    yanchor="top"
                )
            ],
            title_text=titulo,
            xaxis_title=xaxis_titulo,
            yaxis_title=yaxis_titulo
        )

        if xaxis_extra: figura.update_layout(xaxis=xaxis_extra)
        figura.update_traces(textfont=dict(size=10, color='#444'))
        return figura

    contagem_diaria = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Dia']).size().reset_index(name='Contagem')
    agregado_todos_dias = contagem_diaria.groupby('Dia')['Contagem'].sum().reset_index()
    fig_dia = criar_figura_com_menu(contagem_diaria, agregado_todos_dias, 'Dia', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Dia do Mês</b>', 'Dia do Mês', 'Qtd. Atividades')

    contagem_semanal = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês']).size().reset_index(name='Contagem')
    agregado_todas_semanas = contagem_semanal.groupby('Semana do Mês')['Contagem'].sum().reset_index()
    fig_semana = criar_figura_com_menu(contagem_semanal, agregado_todas_semanas, 'Semana do Mês', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Semana do Mês</b>', 'Semana do Mês', 'Qtd. Atividades', xaxis_extra=dict(type='category'))

    contagem_diaria_semana = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês', 'Dia da Semana', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    contagem_diaria_semana['Filtro'] = contagem_diaria_semana['Mes_Ano_Abrev'] + " / Semana " + contagem_diaria_semana['Semana do Mês'].astype(str)
    agregado_todos_dias_semana = contagem_diaria_semana.groupby(['Dia da Semana', 'Nome Dia Semana'])['Contagem'].sum().reset_index()
    fig_dia_semana = criar_figura_com_menu(contagem_diaria_semana, agregado_todos_dias_semana, 'Nome Dia Semana', 'Filtro', 'Total Agregado', '<b>Contagem por Dia da Semana</b>', 'Dia da Semana', 'Qtd. Atividades', xaxis_extra=dict(categoryorder='array', categoryarray=['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']))
    
    fig_master = go.Figure()
    for trace in fig_dia.data: fig_master.add_trace(trace)
    for trace in fig_semana.data: fig_master.add_trace(trace)
    for trace in fig_dia_semana.data: fig_master.add_trace(trace)

    num_traces_fig_dia = len(fig_dia.data)
    num_traces_fig_semana = len(fig_semana.data)
    
    def criar_argumentos_botao(fig_original, offset, active_button_index):
        visibilidade = [False] * len(fig_master.data)
        visibilidade[offset] = True
        layout_update = {
            "title.text": fig_original.layout.title.text,
            "xaxis": fig_original.layout.xaxis,
            "yaxis": fig_original.layout.yaxis,
            "updatemenus[1].buttons": fig_original.layout.updatemenus[0].buttons,
            "updatemenus[1].active": 0,
            "updatemenus[0].active": active_button_index
        }
        return [{"visible": visibilidade}, layout_update]

    args1 = criar_argumentos_botao(fig_dia, 0, 0)
    args2 = criar_argumentos_botao(fig_semana, num_traces_fig_dia, 1)
    args3 = criar_argumentos_botao(fig_dia_semana, num_traces_fig_dia + num_traces_fig_semana, 2)
    
    botoes_principais_config = dict(
        type="buttons",
        direction="right",
        x=0.95,
        xanchor="right",
        y=1.152,
        yanchor="top",
        buttons=[
            dict(label="Dia do Mês", method="update", args=args1),
            dict(label="Semana do Mês", method="update", args=args2),
            dict(label="Dia da Semana", method="update", args=args3)
        ]
    )
    
    fig_master.update_layout(
        template='plotly_white',
        title=fig_dia.layout.title,
        xaxis=fig_dia.layout.xaxis,
        yaxis=fig_dia.layout.yaxis,
        updatemenus=[
            botoes_principais_config,      # Menu 0: Botões Principais
            fig_dia.layout.updatemenus[0]  # Menu 1: Dropdown (inicialmente do primeiro gráfico)
        ]
    )
    
    fig_master.update_traces(visible=False)
    fig_master.data[0].visible = True
    
    fig_master.add_annotation(text="Selecione uma visualização:", xref="paper", yref="paper", x=0.852, y=1.16, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=14))
    return fig_master

# ==============================================================================
# CORPO PRINCIPAL DO APP
# ==============================================================================
st.title("Teste do Gráfico Principal")

df_analise = carregar_dados_completos()

if df_analise is not None and not df_analise.empty:
    fig_principal = criar_grafico_principal(df_analise)
    st.plotly_chart(fig_principal, use_container_width=True)
else:
    st.warning("Não foi possível carregar os dados ou o DataFrame está vazio.")