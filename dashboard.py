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
from datetime import date

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
    
    # --- Carregamento das Abas ---
    nome_aba_dados = "Total BaseCamp para Notas" 
    nome_aba_equipes = "Equipes"
    worksheet_dados = spreadsheet.worksheet(nome_aba_dados)
    worksheet_equipe = spreadsheet.worksheet(nome_aba_equipes)

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
    tabela_calendario['Semana do Mês'] = (tabela_calendario['Date'].dt.dayofweek + (tabela_calendario['Date'].dt.day - 1)).floordiv(7) + 1

    df_analise_temp = pd.merge(df_grafico, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date').drop(columns=['Date'])
    
    df_equipe.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)
    df_analise = pd.merge(df_analise_temp, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
    df_analise['Status_Funcionario'].fillna('Outros', inplace=True)

    return df_analise

# ==============================================================================
# FUNÇÕES PARA CRIAR OS GRÁFICOS
# ==============================================================================
def criar_grafico_produtividade_mensal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Produtividade Mensal</b>")
    
    df_agregado = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev']).agg(
        contagem_tarefas=('ID', 'count'),
        soma_peso=('Peso', 'sum')
    ).reset_index().sort_values('Ano-Mês')

    fig = go.Figure()
    
    # Adiciona as barras (sem texto)
    fig.add_trace(go.Bar(
        x=df_agregado['Mes_Ano_Abrev'], 
        y=df_agregado['contagem_tarefas'], 
        name='Quantidade de Tarefas', 
        marker_color='royalblue'
    ))
    
    # Adiciona a linha (sem texto)
    fig.add_trace(go.Scatter(
        x=df_agregado['Mes_Ano_Abrev'], 
        y=df_agregado['soma_peso'], 
        name='Soma de Peso', 
        mode='lines+markers', 
        line=dict(color='firebrick')
    ))

    # Cria as anotações customizadas
    anotacoes = []
    for index, row in df_agregado.iterrows():
        # Anotação para a COLUNA (deslocada para a direita)
        anotacoes.append(dict(x=row['Mes_Ano_Abrev'], y=row['contagem_tarefas'], text=str(row['contagem_tarefas']), showarrow=False, xshift=12, yshift=10, font=dict(color='royalblue')))
        # Anotação para a LINHA (deslocada para a esquerda)
        anotacoes.append(dict(x=row['Mes_Ano_Abrev'], y=row['soma_peso'], text=str(int(row['soma_peso'])), showarrow=False, xshift=-12, yshift=10, font=dict(color='firebrick')))

    fig.update_layout(
        title="<b>Produtividade Mensal</b>", 
        template='plotly_white', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=anotacoes # Adiciona as anotações ao gráfico
    )
    
    # Adiciona espaço no topo para os rótulos
    if not df_agregado.empty:
        max_y = df_agregado[['contagem_tarefas', 'soma_peso']].max().max()
        fig.update_yaxes(range=[0, max_y * 1.2])

    return fig

def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Gráfico Principal</b>")
    
    # --- PASSO 1: Função auxiliar para criar as figuras base ---
    def criar_figura_com_menu(df_contagem, df_agregado, col_x, col_filtro, nome_agregado, titulo, xaxis_titulo, yaxis_titulo, xaxis_extra=None):
        figura = go.Figure()
        figura.add_trace(go.Scatter(x=df_agregado[col_x], y=df_agregado['Contagem'], name=nome_agregado, mode='lines+markers+text', text=df_agregado['Contagem'], textposition='top center'))
        
        # ==============================================================================
        # ### CORREÇÃO FINAL AQUI: Ordenação inteligente do menu dropdown ###
        # ==============================================================================
        # Verifica se a coluna 'Semana do Mês' existe antes de tentar ordenar por ela.
        if 'Semana do Mês' in df_contagem.columns:
            # Se existir (caso do gráfico "Dia da Semana"), ordena por ano/mês E semana.
            opcoes_filtro = df_contagem.sort_values(['Ano-Mês', 'Semana do Mês'])[col_filtro].unique()
        else:
            # Senão (casos de "Dia do Mês" e "Semana do Mês"), ordena apenas por ano/mês.
            opcoes_filtro = df_contagem.sort_values(['Ano-Mês'])[col_filtro].unique()
        
        for opcao in opcoes_filtro:
            df_filtrado = df_contagem[df_contagem[col_filtro] == opcao]
            if not df_filtrado.empty:
                if col_x == 'Nome Dia Semana':
                    df_filtrado = df_filtrado.sort_values(by='Dia da Semana')
                else:
                    df_filtrado = df_filtrado.sort_values(by=col_x)

                figura.add_trace(go.Scatter(x=df_filtrado[col_x], y=df_filtrado['Contagem'], name=opcao, mode='lines+markers+text', text=df_filtrado['Contagem'], textposition='top center', visible=False))
        
        botoes = [{'label': nome_agregado, 'method': 'update', 'args': [{'visible': [i == 0 for i in range(len(figura.data))]}]}]
        for i, trace in enumerate(figura.data[1:], 1):
            visibilidade_args = [False] * len(figura.data); visibilidade_args[i] = True
            botoes.append({'label': trace.name, 'method': 'update', 'args': [{'visible': visibilidade_args}]})
        
        figura.update_layout(updatemenus=[dict(active=0, buttons=botoes, direction="down", showactive=True)], title_text=titulo, xaxis_title=xaxis_titulo, yaxis_title=yaxis_titulo)
        
        if xaxis_extra: figura.update_layout(xaxis=xaxis_extra)
        figura.update_traces(textfont=dict(size=10, color='#444'))
        return figura

    # --- PASSO 2: Cria as 3 figuras base (sem mudanças aqui) ---
    contagem_diaria = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Dia', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    agregado_todos_dias = contagem_diaria.groupby('Dia')['Contagem'].sum().reset_index().sort_values(by='Dia')
    fig_dia = criar_figura_com_menu(contagem_diaria, agregado_todos_dias, 'Dia', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Dia do Mês</b>', None, None, xaxis_extra=dict(type='linear'))

    contagem_semanal = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês']).size().reset_index(name='Contagem')
    agregado_todas_semanas = contagem_semanal.groupby('Semana do Mês')['Contagem'].sum().reset_index().sort_values(by='Semana do Mês')
    fig_semana = criar_figura_com_menu(contagem_semanal, agregado_todas_semanas, 'Semana do Mês', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Semana do Mês</b>', None, None, xaxis_extra=dict(type='linear'))

    contagem_diaria_semana = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês', 'Dia da Semana', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    contagem_diaria_semana['Filtro'] = contagem_diaria_semana['Mes_Ano_Abrev'] + " / Semana " + contagem_diaria_semana['Semana do Mês'].astype(str)
    agregado_todos_dias_semana = contagem_diaria_semana.groupby(['Dia da Semana', 'Nome Dia Semana'])['Contagem'].sum().reset_index()
    fig_dia_semana = criar_figura_com_menu(contagem_diaria_semana, agregado_todos_dias_semana, 'Nome Dia Semana', 'Filtro', 'Total Agregado', '<b>Contagem por Dia da Semana</b>', None, None, xaxis_extra=dict(categoryorder='array', categoryarray=['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']))
    
    # --- O RESTO DO CÓDIGO PERMANECE EXATAMENTE O MESMO ---
    fig_master = go.Figure()
    for trace in fig_dia.data: fig_master.add_trace(trace)
    for trace in fig_semana.data: fig_master.add_trace(trace)
    for trace in fig_dia_semana.data: fig_master.add_trace(trace)

    num_traces_fig_dia = len(fig_dia.data)
    num_traces_fig_semana = len(fig_semana.data)
    num_traces_fig_dia_semana = len(fig_dia_semana.data)
    
    def criar_argumentos_botao(fig_original, offset, active_button_index):
        visibilidade_principal = [False] * len(fig_master.data); visibilidade_principal[offset] = True
        novos_botoes_dropdown = []
        botoes_originais = fig_original.layout.updatemenus[0].buttons
        for i, botao_original in enumerate(botoes_originais):
            nova_visibilidade_dropdown = [False] * len(fig_master.data)
            if i == 0 and active_button_index in [0, 1]:
                num_traces_grupo = num_traces_fig_dia if active_button_index == 0 else num_traces_fig_semana
                for j in range(1, num_traces_grupo): nova_visibilidade_dropdown[j + offset] = True
            elif i == 0 and active_button_index == 2:
                for j in range(1, num_traces_fig_dia_semana): nova_visibilidade_dropdown[j + offset] = True
            else:
                indice_global_correto = i + offset; nova_visibilidade_dropdown[indice_global_correto] = True
            novos_botoes_dropdown.append(dict(label=botao_original['label'], method='update', args=[{'visible': nova_visibilidade_dropdown}]))
        layout_update = {
            "title.text": fig_original.layout.title.text, "xaxis": fig_original.layout.xaxis, "yaxis": fig_original.layout.yaxis,
            "updatemenus[1].buttons": novos_botoes_dropdown, "updatemenus[1].active": 0, "updatemenus[0].active": active_button_index
        }
        return [{"visible": visibilidade_principal}, layout_update]

    args1 = criar_argumentos_botao(fig_dia, 0, 0)
    args2 = criar_argumentos_botao(fig_semana, num_traces_fig_dia, 1)
    args3 = criar_argumentos_botao(fig_dia_semana, num_traces_fig_dia + num_traces_fig_semana, 2)
    
    botoes_principais_config = dict(type="buttons", direction="right", x=0.99, xanchor="right", y=1.275, yanchor="top", buttons=[dict(label="Dia do Mês", method="update", args=args1), dict(label="Semana do Mês", method="update", args=args2), dict(label="Dia da Semana", method="update", args=args3)])
    menu_suspenso_config = fig_dia.layout.updatemenus[0]
    menu_suspenso_config.x = -0.01; menu_suspenso_config.xanchor = "left"; menu_suspenso_config.y = 1.275; menu_suspenso_config.yanchor = "top"
    
    fig_master.update_layout(
        template='plotly_white',
        title=dict(text=fig_dia.layout.title.text, y=0.93, x=0.001, xanchor='left', yanchor='top'),
        xaxis=args1[1]['xaxis'], yaxis=args1[1]['yaxis'],
        margin=dict(t=130), 
        updatemenus=[botoes_principais_config, menu_suspenso_config]
    )
    
    fig_master.update_traces(visible=False); fig_master.data[0].visible = True
    fig_master.add_annotation(text="Selecione uma visualização:", xref="paper", yref="paper", x=0.79, y=1.335, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=14))
    
    return fig_master

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
    df_funcionarios = df_funcionarios.sort_values('Contagem', ascending=True)
    fig = px.bar(df_funcionarios, x='Contagem', y='Encarregado', orientation='h', title="<b>N° de Tarefas por Funcionário</b>", text='Contagem', color='Contagem', color_continuous_scale='Blues')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False)
    return fig

def criar_grafico_pontuacao(df):
    if df.empty: return go.Figure().update_layout(title="<b>Pontuação</b>")

    if 'Posição' not in df.columns:
        return go.Figure().update_layout(title="<b>Pontuação (Coluna 'Posição' não encontrada)</b>")

    lista_lideres = df[df['Posição'] == 'Lider']['Encarregado'].unique()
    
    df_peso_base = df.groupby('Encarregado')['Peso'].sum().reset_index()

    pontos_lideranca = {}
    for lider in lista_lideres:
        for coluna in df.columns:
            if coluna.upper() == lider.upper():
                pontos_lideranca[lider] = pd.to_numeric(df[coluna], errors='coerce').sum()
                break
    df_pontos_lideranca = pd.DataFrame(list(pontos_lideranca.items()), columns=['Encarregado', 'Pontos_Lideranca'])

    if df_pontos_lideranca.empty:
        df_pontos_lideranca = pd.DataFrame(columns=['Encarregado', 'Pontos_Lideranca'])

    df_agregado_final = pd.merge(df_peso_base, df_pontos_lideranca, on='Encarregado', how='left')
    df_agregado_final['Pontos_Lideranca'].fillna(0, inplace=True)
    df_agregado_final['Soma_Total'] = df_agregado_final['Peso'] + df_agregado_final['Pontos_Lideranca']

    df_total_view = df_agregado_final.sort_values('Soma_Total', ascending=True)
    df_funcionarios_view = df_agregado_final.sort_values('Peso', ascending=True)
    df_lideranca_view = df_agregado_final[df_agregado_final['Pontos_Lideranca'] > 0].sort_values('Pontos_Lideranca', ascending=True)

    fig = go.Figure(data=[
        go.Bar(x=df_total_view['Soma_Total'], y=df_total_view['Encarregado'], orientation='h', text=df_total_view['Soma_Total'].astype(int), textposition='outside', name='Total'),
        go.Bar(x=df_funcionarios_view['Peso'], y=df_funcionarios_view['Encarregado'], orientation='h', text=df_funcionarios_view['Peso'].astype(int), textposition='outside', name='Funcionários', visible=False),
        go.Bar(x=df_lideranca_view['Pontos_Lideranca'], y=df_lideranca_view['Encarregado'], orientation='h', text=df_lideranca_view['Pontos_Lideranca'].astype(int), textposition='outside', name='Liderança', visible=False)
    ])
    
    botoes_posicao = [
        dict(label="Total", method="restyle", args=[{"visible": [True, False, False]}]),
        dict(label="Funcionários", method="restyle", args=[{"visible": [False, True, False]}]),
        dict(label="Liderança", method="restyle", args=[{"visible": [False, False, True]}]),
    ]

    fig.update_layout(
        title_text="<b>Pontuação (Soma de Peso) por Funcionário</b>",
        template='plotly_white', yaxis_title=None,
        updatemenus=[dict(type="buttons", direction="right", active=0, x=1, xanchor="right", y=1.15, yanchor="top", buttons=botoes_posicao)]
    )
    return fig


# ==============================================================================
# CORPO PRINCIPAL DO DASHBOARD (INTERFACE STREAMLIT)
# ==============================================================================
st.title("Dashboard de Produtividade")
df_analise = carregar_dados_completos()

if df_analise is not None and not df_analise.empty:

    min_date = df_analise['Data Final (aberta)'].min().date()
    max_date = df_analise['Data Final (aberta)'].max().date()

    def limpar_filtros():
        st.session_state.encarregado_filtro = ["Todos"]
        st.session_state.contrato_filtro = "Todos"
        st.session_state.status_tarefa_filtro = "Todos"
        st.session_state.semana_filtro = "Todos"
        st.session_state.peso_filtro = "Todos"
        st.session_state.date_slider = (min_date, max_date)

    if 'filtros_iniciados' not in st.session_state:
        limpar_filtros()
        st.session_state.filtros_iniciados = True

    with st.sidebar:
        st.image("media portal logo.png", width=200)
        st.title("Filtros")
        
        encarregados_disponiveis = ["Todos"] + sorted(df_analise['Encarregado'].unique())
        st.multiselect("Encarregado", encarregados_disponiveis, key='encarregado_filtro')

        contratos_disponiveis = ["Todos"] + df_analise['Status_Funcionario'].unique().tolist()
        st.selectbox("Contrato", contratos_disponiveis, key='contrato_filtro')

        status_tarefas = ["Todos"] + df_analise['Status_Tarefa'].unique().tolist()
        st.selectbox("Status da Tarefa", status_tarefas, key='status_tarefa_filtro')

        st.markdown("---")
        st.button("Limpar Filtros 🗑️", on_click=limpar_filtros)

    df_filtrado = df_analise.copy()
    if "Todos" not in st.session_state.encarregado_filtro:
        df_filtrado = df_filtrado[df_filtrado['Encarregado'].isin(st.session_state.encarregado_filtro)]
    if st.session_state.contrato_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Status_Funcionario'] == st.session_state.contrato_filtro]
    if st.session_state.status_tarefa_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Status_Tarefa'] == st.session_state.status_tarefa_filtro]
        
    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([2, 2, 1, 1, 4])

    with top_col1:
        semanas_disponiveis = ["Todos"] + sorted([i for i in df_filtrado['Semana do Mês'].unique() if i is not np.nan])
        st.selectbox("Semana do Mês", semanas_disponiveis, key='semana_filtro')
    
    with top_col2:
        pesos_disponiveis = ["Todos"] + sorted(df_filtrado['Peso'].astype(int).unique())
        st.selectbox("Peso da Tarefa", pesos_disponiveis, key='peso_filtro')

    with top_col5:
        st.slider("Intervalo de Datas (Data Final)", min_value=min_date, max_value=max_date, key='date_slider')

    if st.session_state.semana_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Semana do Mês'] == st.session_state.semana_filtro]
    if st.session_state.peso_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Peso'] == st.session_state.peso_filtro]
    
    start_date, end_date = st.session_state.date_slider
    df_filtrado = df_filtrado[(df_filtrado['Data Final (aberta)'].dt.date >= start_date) & (df_filtrado['Data Final (aberta)'].dt.date <= end_date)]

    with top_col3:
        st.metric("Tarefas", f"{df_filtrado.shape[0]:,}")
    with top_col4:
        st.metric("Soma de Peso", f"{int(df_filtrado['Peso'].sum()):,}")
        
    st.markdown("---") 

    # --- CORPO PRINCIPAL: GRÁFICOS (SEM ABAS) ---
    st.markdown("### Visão Geral")
    col_geral1, col_geral2 = st.columns(2)
    with col_geral1:
        fig_prod_mensal = criar_grafico_produtividade_mensal(df_filtrado)
        st.plotly_chart(fig_prod_mensal, use_container_width=True)
    with col_geral2:
        fig_principal = criar_grafico_principal(df_filtrado)
        st.plotly_chart(fig_principal, use_container_width=True)

    st.markdown("---")
    st.markdown("### Análise de Equipe")
    col_equipe1, col_equipe2 = st.columns(2)
    with col_equipe1:
        fig_tarefas = criar_grafico_tarefas_funcionarios(df_filtrado)
        st.plotly_chart(fig_tarefas, use_container_width=True)
    with col_equipe2:
        fig_pontuacao = criar_grafico_pontuacao(df_filtrado)
        st.plotly_chart(fig_pontuacao, use_container_width=True)

    st.markdown("---")
    st.markdown("### Status Geral das Tarefas")
    col_status1, col_status2, col_status3 = st.columns([1,2,1])
    with col_status2:
        fig_status = criar_grafico_status_tarefas(df_filtrado)
        st.plotly_chart(fig_status, use_container_width=True)

else:
    st.error("Não foi possível carregar os dados para exibir o dashboard.")