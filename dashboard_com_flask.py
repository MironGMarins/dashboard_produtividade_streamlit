# ==============================================================================
# PASSO 1: IMPORTS CONSOLIDADOS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials
import holidays

# ==============================================================================
# PASSO 2: FUNÇÃO DE CARREGAMENTO E PREPARAÇÃO DOS DADOS (O "MOTOR")
# Esta função continua em cache para alta performance.
# ==============================================================================
@st.cache_data(ttl=600)
def carregar_dados_completos():
    # --- Autenticação e Carregamento do Google Sheets ---
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file("google_credentials.json", scopes=scopes)
    client = gspread.authorize(creds)
    url_da_planilha = 'https://docs.google.com/spreadsheets/d/1juyOfIh0ZqsfJjN0p3gD8pKaAIX0R6IAPG9vysl7yWI/edit#gid=901870248'
    spreadsheet = client.open_by_url(url_da_planilha)
    worksheet = spreadsheet.get_worksheet(2)
    df = pd.DataFrame(worksheet.get_all_records())

    # --- Tratamento e Criação de Colunas ---
    df_grafico = df.copy()
    df_grafico['Peso'] = pd.to_numeric(df_grafico['Peso'], errors='coerce').fillna(0)
    df_grafico['Data Inicial'] = pd.to_datetime(df_grafico['Data Inicial'], errors='coerce')
    df_grafico['Data Final'] = pd.to_datetime(df_grafico['Data Final'], errors='coerce')
    df_grafico['Status'] = np.where(df_grafico['Data Final'].isnull(), 'Aberto', 'Executado')
    data_hoje = pd.Timestamp.now().normalize()
    df_grafico['Data Final (aberta)'] = df_grafico['Data Final'].fillna(data_hoje)

    # --- Criação da Tabela Calendário ---
    data_inicio = df_grafico['Data Inicial'].min()
    data_fim = pd.Timestamp.now()
    tabela_calendario = pd.DataFrame({"Date": pd.date_range(start=data_inicio, end=data_fim, freq='D')})
    tabela_calendario['Ano'] = tabela_calendario['Date'].dt.year
    tabela_calendario['Nome Mês'] = tabela_calendario['Date'].dt.strftime('%b').str.capitalize()
    tabela_calendario['Mes_Ano_Abrev'] = tabela_calendario['Nome Mês'] + '/' + tabela_calendario['Date'].dt.strftime('%y')
    tabela_calendario['Ano-Mês'] = tabela_calendario['Date'].dt.strftime('%Y-%m')
    tabela_calendario['Dia'] = tabela_calendario['Date'].dt.day
    tabela_calendario['Dia da Semana'] = tabela_calendario['Date'].dt.dayofweek + 1
    tabela_calendario['Nome Dia Semana'] = tabela_calendario['Dia da Semana'].map({1:'seg', 2:'ter', 3:'qua', 4:'qui', 5:'sex', 6:'sab', 7:'dom'})
    tabela_calendario['Semana do Mês'] = (tabela_calendario['Date'].dt.day + tabela_calendario['Date'].dt.dayofweek - 1) // 7 + 1

    # --- União (Merge) Final ---
    df_analise = pd.merge(df_grafico, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date').drop(columns=['Date'])
    
    return df_analise

# ==============================================================================
# PASSO 3: FUNÇÃO PARA CRIAR O PAINEL PLOTLY AVANÇADO
# Toda a complexidade dos gráficos do Colab vive aqui dentro.
# ==============================================================================
def criar_painel_plotly(df_para_grafico):
    # --- Preparação de dados para os gráficos ---
    contagem_diaria = df_para_grafico.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Dia']).size().reset_index(name='Contagem')
    agregado_todos_dias = contagem_diaria.groupby('Dia')['Contagem'].sum().reset_index()

    contagem_semanal = df_para_grafico.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês']).size().reset_index(name='Contagem')
    agregado_todas_semanas = contagem_semanal.groupby('Semana do Mês')['Contagem'].sum().reset_index()

    contagem_diaria_semana = df_para_grafico.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês', 'Dia da Semana', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    contagem_diaria_semana['Filtro'] = contagem_diaria_semana['Mes_Ano_Abrev'] + " / Semana " + contagem_diaria_semana['Semana do Mês'].astype(str)
    agregado_todos_dias_semana = contagem_diaria_semana.groupby(['Dia da Semana', 'Nome Dia Semana'])['Contagem'].sum().reset_index()

    # --- Função interna para criar figuras individuais com menus ---
    def criar_figura_com_menu(df_contagem, df_agregado, col_x, col_filtro, nome_agregado, titulo, xaxis_titulo, yaxis_titulo, xaxis_extra=None):
        figura = go.Figure()
        figura.add_trace(go.Scatter(x=df_agregado[col_x], y=df_agregado['Contagem'], name=nome_agregado, mode='lines+markers+text', text=df_agregado['Contagem'], textposition='top center'))
        opcoes_filtro = df_contagem.sort_values('Ano-Mês')[col_filtro].unique()
        for opcao in opcoes_filtro:
            df_filtrado = df_contagem[df_contagem[col_filtro] == opcao]
            if not df_filtrado.empty:
                figura.add_trace(go.Scatter(x=df_filtrado[col_x], y=df_filtrado['Contagem'], name=opcao, mode='lines+markers+text', text=df_filtrado['Contagem'], textposition='top center', visible=False))
        botoes = [{'label': nome_agregado, 'method': 'update', 'args': [{'visible': [i == 0 for i in range(len(figura.data))]}]}]
        for i, trace in enumerate(figura.data[1:], 1):
            visibilidade = [False] * len(figura.data)
            visibilidade[i] = True
            botoes.append({'label': trace.name, 'method': 'update', 'args': [{'visible': visibilidade}]})
        figura.update_layout(updatemenus=[dict(active=0, buttons=botoes, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=1.15, yanchor="top")], title_text=titulo, xaxis_title=xaxis_titulo, yaxis_title=yaxis_titulo)
        if xaxis_extra: figura.update_layout(xaxis=xaxis_extra)
        figura.update_traces(textfont=dict(size=10, color='#444'))
        return figura

    # --- Criação das 3 figuras base ---
    fig = criar_figura_com_menu(contagem_diaria, agregado_todos_dias, 'Dia', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Dia do Mês</b>', 'Dia do Mês', 'Qtd. Atividades')
    fig_semana = criar_figura_com_menu(contagem_semanal, agregado_todas_semanas, 'Semana do Mês', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Semana do Mês</b>', 'Semana do Mês', 'Qtd. Atividades', xaxis_extra=dict(type='category'))
    fig_dia_semana = criar_figura_com_menu(contagem_diaria_semana, agregado_todos_dias_semana, 'Nome Dia Semana', 'Filtro', 'Total Agregado', '<b>Contagem por Dia da Semana</b>', 'Dia da Semana', 'Qtd. Atividades', xaxis_extra=dict(categoryorder='array', categoryarray=['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']))

    # --- Montagem da Figura Mestra ---
    fig_master = go.Figure()
    for trace in fig.data: fig_master.add_trace(trace)
    for trace in fig_semana.data: fig_master.add_trace(trace)
    for trace in fig_dia_semana.data: fig_master.add_trace(trace)

    num_traces_fig, num_traces_fig_semana, num_traces_fig_dia_semana = len(fig.data), len(fig_semana.data), len(fig_dia_semana.data)
    num_total_traces = num_traces_fig + num_traces_fig_semana + num_traces_fig_dia_semana

    def criar_menu_corrigido(fig_original, offset, num_total_traces):
        botoes_corrigidos = []
        for botao in fig_original.layout.updatemenus[0].buttons:
            visibilidade_curta = botao['args'][0]['visible']
            visibilidade_longa = [False] * num_total_traces
            for i, visivel in enumerate(visibilidade_curta):
                if i + offset < num_total_traces: visibilidade_longa[i + offset] = visivel
            botoes_corrigidos.append(dict(label=botao['label'], method='update', args=[{'visible': visibilidade_longa}]))
        return [dict(active=0, buttons=botoes_corrigidos, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=1.15, yanchor="top")]

    updatemenus_fig, updatemenus_fig_semana, updatemenus_fig_dia_semana = criar_menu_corrigido(fig, 0, num_total_traces), criar_menu_corrigido(fig_semana, num_traces_fig, num_total_traces), criar_menu_corrigido(fig_dia_semana, num_traces_fig + num_traces_fig_semana, num_total_traces)
    
    args1 = [{"visible": [i == 0 for i in range(num_total_traces)]}, {"title": fig.layout.title, "xaxis": fig.layout.xaxis, "yaxis": fig.layout.yaxis, "updatemenus": updatemenus_fig}]
    args2 = [{"visible": [i == num_traces_fig for i in range(num_total_traces)]}, {"title": fig_semana.layout.title, "xaxis": fig_semana.layout.xaxis, "yaxis": fig_semana.layout.yaxis, "updatemenus": updatemenus_fig_semana}]
    args3 = [{"visible": [i == (num_traces_fig + num_traces_fig_semana) for i in range(num_total_traces)]}, {"title": fig_dia_semana.layout.title, "xaxis": fig_dia_semana.layout.xaxis, "yaxis": fig_dia_semana.layout.yaxis, "updatemenus": updatemenus_fig_dia_semana}]

    fig_master.update_layout(
        updatemenus=[dict(type="buttons", direction="right", active=0, x=1, xanchor="right", y=1.2, yanchor="top", buttons=[
            dict(label="Dia do Mês", method="update", args=args1),
            dict(label="Semana do Mês", method="update", args=args2),
            dict(label="Dia da Semana", method="update", args=args3)])])

    fig_master.update_layout(title=args1[1]['title'], xaxis=args1[1]['xaxis'], yaxis=args1[1]['yaxis'], template='plotly_white')
    fig_master.update_traces(visible=False); fig_master.data[0].visible = True
    fig_master.add_annotation(text="Selecione uma visualização:", xref="paper", yref="paper", x=0.99, y=1.15, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=14))
    
    return fig_master

# ==============================================================================
# PASSO 4: MONTAGEM DA INTERFACE DO STREAMLIT (A "VITRINE")
# ==============================================================================
df_analise = carregar_dados_completos()

st.sidebar.header("Filtros")
encarregados_unicos = sorted(df_analise['Encarregado'].unique())
encarregado_selecionado = st.sidebar.multiselect("Encarregado:", options=encarregados_unicos, default=encarregados_unicos)
status_unicos = df_analise['Status'].unique()
status_selecionado = st.sidebar.multiselect("Status da Tarefa:", options=status_unicos, default=status_unicos)

# Aplica os filtros do Streamlit ANTES de criar os gráficos
df_filtrado = df_analise.query("Encarregado == @encarregado_selecionado & Status == @status_selecionado")

# Gera e exibe o painel de controle interativo
if not df_filtrado.empty:
    painel_de_controle = criar_painel_plotly(df_filtrado)
    st.plotly_chart(painel_de_controle, use_container_width=True)
else:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")

# Exibe a tabela de dados detalhados
st.markdown("### Dados Detalhados")
st.dataframe(df_filtrado)