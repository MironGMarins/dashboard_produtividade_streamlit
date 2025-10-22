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
    except (KeyError, FileNotFoundError):
        st.error("Credenciais do Google (gcp_service_account) não encontradas. Verifique seu arquivo .streamlit/secrets.toml.")
        return None
    
    client = gspread.authorize(creds)
    
    url_da_planilha = st.secrets.get("SHEET_URL")
    if not url_da_planilha:
        st.error("URL da planilha (SHEET_URL) não encontrada no arquivo secrets.toml.")
        return None
        
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
    
    # Identifica líderes e colunas numéricas
    encarregados = df_grafico['Encarregado'].astype(str).unique()
    encarregados_lower = {str(e).lower() for e in encarregados}
    lideres = [col for col in df_grafico.columns if str(col).lower() in encarregados_lower]
    
    colunas_para_numerico = ['Peso'] + lideres
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
    month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
    tabela_calendario['Nome Mês'] = tabela_calendario['Date'].dt.month.map(month_map)
    tabela_calendario['Mes_Ano_Abrev'] = tabela_calendario['Nome Mês'] + '/' + tabela_calendario['Date'].dt.strftime('%y')
    tabela_calendario['Ano-Mês'] = tabela_calendario['Date'].dt.strftime('%Y-%m')
    tabela_calendario['Dia'] = tabela_calendario['Date'].dt.day
    tabela_calendario['Dia da Semana'] = tabela_calendario['Date'].dt.dayofweek # Mantém 0 para Segunda
    dias_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sab', 6: 'Dom'}
    tabela_calendario['Nome Dia Semana'] = tabela_calendario['Dia da Semana'].map(dias_map)
    
    tabela_calendario['Week_Start_Date'] = tabela_calendario['Date'] - pd.to_timedelta(tabela_calendario['Date'].dt.dayofweek, unit='d')
    tabela_calendario['Semana do Mês'] = tabela_calendario.groupby(tabela_calendario['Date'].dt.to_period('M'))['Week_Start_Date'].rank(method='dense').astype(int)


    df_analise_temp = pd.merge(df_grafico, tabela_calendario, how='left', left_on=pd.to_datetime(df_grafico['Data Final (aberta)'].dt.date), right_on=pd.to_datetime(tabela_calendario['Date'].dt.date)).drop(columns=['Date', 'key_0'])
    
    df_equipe.rename(columns={'Status': 'Status_Funcionario', 'Nome': 'Encarregado'}, inplace=True)
    df_analise = pd.merge(df_analise_temp, df_equipe, how='left', on='Encarregado')
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
    fig.add_trace(go.Bar(x=df_agregado['Mes_Ano_Abrev'], y=df_agregado['contagem_tarefas'], name='Quantidade de Tarefas', marker_color='royalblue'))
    fig.add_trace(go.Scatter(x=df_agregado['Mes_Ano_Abrev'], y=df_agregado['soma_peso'], name='Soma de Peso', mode='lines+markers', line=dict(color='firebrick')))

    anotacoes = []
    for index, row in df_agregado.iterrows():
        anotacoes.append(dict(x=row['Mes_Ano_Abrev'], y=row['contagem_tarefas'], text=str(row['contagem_tarefas']), showarrow=False, xshift=12, yshift=10, font=dict(color='royalblue')))
        anotacoes.append(dict(x=row['Mes_Ano_Abrev'], y=row['soma_peso'], text=str(int(row['soma_peso'])), showarrow=False, xshift=-12, yshift=10, font=dict(color='firebrick')))

    fig.update_layout(
        title="<b>Produtividade Mensal</b>", template='plotly_white', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=anotacoes
    )
    
    if not df_agregado.empty:
        max_y = df_agregado[['contagem_tarefas', 'soma_peso']].max().max()
        fig.update_yaxes(range=[0, max_y * 1.2])

    return fig

def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Gráfico Principal</b>")

    # --- PALETA DE CORES EM GRADIENTE REVISADA ---
    MONTH_COLORS_SCALES = {
        'Jan': px.colors.sequential.Blues, 
        'Fev': px.colors.sequential.Greens,
        'Mar': px.colors.sequential.Oranges, 
        'Abr': px.colors.sequential.Purples,
        'Mai': px.colors.sequential.Reds, 
        'Jun': px.colors.sequential.Teal,
        'Jul': px.colors.sequential.Mint,
        'Ago': px.colors.sequential.YlOrBr, 
        'Set': px.colors.sequential.Burg,
        'Out': px.colors.sequential.Electric,
        'Nov': px.colors.sequential.Cividis,
        'Dez': px.colors.sequential.Magenta,
    }
    
    def criar_figura_com_menu(df_contagem, df_agregado, col_x, col_filtro, nome_agregado, titulo, xaxis_titulo, yaxis_titulo, xaxis_extra=None, apply_custom_coloring=False):
        figura = go.Figure()
        
        hover_template_agregado = 'Dia: %{x}<br>Qtd: %{y}<extra></extra>'
        figura.add_trace(go.Scatter(x=df_agregado[col_x], y=df_agregado['Contagem'], name=nome_agregado, mode='lines+markers+text', text=df_agregado['Contagem'], textposition='top center', hovertemplate=hover_template_agregado))

        if 'Semana do Mês' in df_contagem.columns:
            opcoes_filtro = df_contagem.sort_values(['Ano-Mês', 'Semana do Mês'])[col_filtro].unique()
        else:
            opcoes_filtro = df_contagem.sort_values(['Ano-Mês'])[col_filtro].unique()
        
        for opcao in opcoes_filtro:
            df_filtrado = df_contagem[df_contagem[col_filtro] == opcao]
            if not df_filtrado.empty:
                if col_x == 'Nome Dia Semana': df_filtrado = df_filtrado.sort_values(by='Dia da Semana')
                else: df_filtrado = df_filtrado.sort_values(by=col_x)
                
                custom_data_filtrado = df_filtrado['Nome Dia Semana'] if 'Nome Dia Semana' in df_filtrado.columns else None
                hover_template_filtrado = '<b>%{customdata}</b><br>Dia: %{x}<br>Qtd: %{y}<extra></extra>' if custom_data_filtrado is not None else 'Dia: %{x}<br>Qtd: %{y}<extra></extra>'
                
                line_args = {}
                if apply_custom_coloring:
                    try:
                        parts = opcao.replace('/', '').split()
                        mes_abrev = parts[0][:3]
                        semana_num = int(parts[-1])
                        # Usa a paleta de gradiente
                        color_scale = MONTH_COLORS_SCALES.get(mes_abrev, px.colors.sequential.gray)
                        # Mapeia a semana (1-5) para um índice na escala de cores
                        color_index = int(((semana_num - 1) / 4.0) * (len(color_scale) - 1))
                        # Garante que o índice esteja dentro dos limites e evita as cores mais claras
                        color_index = max(2, min(len(color_scale) - 2, color_index)) 
                        line_args['line'] = dict(color=color_scale[color_index])
                    except (IndexError, ValueError):
                        pass

                figura.add_trace(go.Scatter(x=df_filtrado[col_x], y=df_filtrado['Contagem'], name=opcao, mode='lines+markers+text', text=df_filtrado['Contagem'], textposition='top center', visible=False, customdata=custom_data_filtrado, hovertemplate=hover_template_filtrado, **line_args))
        
        botoes = []
        # --- LÓGICA DE BOTÕES CORRIGIDA ---
        vis_total_agregado = [False] * len(figura.data)
        for i in range(1, len(figura.data)): vis_total_agregado[i] = True
            
        if apply_custom_coloring: # Lógica especial para "Dia da Semana"
            botoes.append({'label': nome_agregado, 'method': 'update', 'args': [{'visible': vis_total_agregado}]})

            meses_unicos = df_contagem.sort_values('Ano-Mês')['Mes_Ano_Abrev'].unique()
            trace_map = {trace.name: i for i, trace in enumerate(figura.data)}
            
            for mes in meses_unicos:
                vis_mes = [False] * len(figura.data)
                semanas_do_mes = [opcao for opcao in opcoes_filtro if opcao.startswith(mes)]
                for semana in semanas_do_mes:
                    if semana in trace_map: vis_mes[trace_map[semana]] = True
                botoes.append({'label': f"{mes} - Todas as Semanas", 'method': 'update', 'args': [{'visible': vis_mes}]})
                
                for semana in semanas_do_mes:
                    if semana in trace_map:
                        vis_individual = [False] * len(figura.data)
                        vis_individual[trace_map[semana]] = True
                        botoes.append({'label': semana, 'method': 'update', 'args': [{'visible': vis_individual}]})
        else: # Lógica para "Dia do Mês" e "Semana do Mês"
            # CORREÇÃO: "Todos os Meses" agora mostra todas as linhas individuais
            botoes.append({'label': nome_agregado, 'method': 'update', 'args': [{'visible': vis_total_agregado}]})
            for i, trace in enumerate(figura.data[1:], 1):
                visibilidade_args = [False] * len(figura.data); visibilidade_args[i] = True
                botoes.append({'label': trace.name, 'method': 'update', 'args': [{'visible': visibilidade_args}]})

        figura.update_layout(updatemenus=[dict(active=0, buttons=botoes, direction="down", showactive=True)], title_text=titulo, xaxis_title=xaxis_titulo, yaxis_title=yaxis_titulo)
        if xaxis_extra: figura.update_layout(xaxis=xaxis_extra)
        figura.update_traces(textfont=dict(size=10, color='#444'))
        return figura

    contagem_diaria = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Dia', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    agregado_todos_dias = contagem_diaria.groupby('Dia')['Contagem'].sum().reset_index().sort_values(by='Dia')
    fig_dia = criar_figura_com_menu(contagem_diaria, agregado_todos_dias, 'Dia', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Dia do Mês</b>', None, None, xaxis_extra=dict(type='linear'))
    
    contagem_semanal = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês']).size().reset_index(name='Contagem')
    agregado_todas_semanas = contagem_semanal.groupby('Semana do Mês')['Contagem'].sum().reset_index().sort_values(by='Semana do Mês')
    fig_semana = criar_figura_com_menu(contagem_semanal, agregado_todas_semanas, 'Semana do Mês', 'Mes_Ano_Abrev', 'Todos os Meses', '<b>Contagem por Semana do Mês</b>', None, None, xaxis_extra=dict(type='linear'))

    contagem_diaria_semana = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev', 'Semana do Mês', 'Dia da Semana', 'Nome Dia Semana']).size().reset_index(name='Contagem')
    contagem_diaria_semana['Filtro'] = contagem_diaria_semana['Mes_Ano_Abrev'] + " / Semana " + contagem_diaria_semana['Semana do Mês'].astype(str)
    agregado_todos_dias_semana = contagem_diaria_semana.groupby(['Dia da Semana', 'Nome Dia Semana'])['Contagem'].sum().reset_index()
    fig_dia_semana = criar_figura_com_menu(contagem_diaria_semana, agregado_todos_dias_semana, 'Nome Dia Semana', 'Filtro', 'Total Agregado', '<b>Contagem por Dia da Semana</b>', None, None, xaxis_extra=dict(categoryorder='array', categoryarray=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']), apply_custom_coloring=True)
    
    fig_master = go.Figure()
    for trace in fig_dia.data: fig_master.add_trace(trace)
    for trace in fig_semana.data: fig_master.add_trace(trace)
    for trace in fig_dia_semana.data: fig_master.add_trace(trace)

    num_traces_fig_dia = len(fig_dia.data)
    num_traces_fig_semana = len(fig_semana.data)
    
    def criar_argumentos_botao(fig_original, offset, active_button_index):
        visibilidade_principal = [False] * len(fig_master.data); visibilidade_principal[offset] = True
        novos_botoes_dropdown = []
        botoes_originais = fig_original.layout.updatemenus[0].buttons
        for botao_original in botoes_originais:
            visibilidade_local = botao_original['args'][0]['visible']
            visibilidade_global = [False] * len(fig_master.data)
            for i_local, is_visible in enumerate(visibilidade_local):
                if is_visible:
                    i_global = i_local + offset
                    if i_global < len(visibilidade_global):
                        visibilidade_global[i_global] = True
            novos_botoes_dropdown.append(dict(label=botao_original['label'], method='update', args=[{'visible': visibilidade_global}]))
        layout_update = {"title.text": fig_original.layout.title.text, "xaxis": fig_original.layout.xaxis, "yaxis": fig_original.layout.yaxis, "updatemenus[1].buttons": novos_botoes_dropdown, "updatemenus[1].active": 0, "updatemenus[0].active": active_button_index}
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
        margin=dict(t=130, b=80), # Aumenta a margem inferior para dar espaço para a legenda
        updatemenus=[botoes_principais_config, menu_suspenso_config],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2, # Posição abaixo do eixo X
            xanchor="center",
            x=0.5
        )
    )
    
    fig_master.update_traces(visible=False); fig_master.data[0].visible = True
    fig_master.add_annotation(text="Selecione uma visualização:", xref="paper", yref="paper", x=0.79, y=1.335, xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=14))
    
    return fig_master

def criar_grafico_status_tarefas(df):
    if df.empty: return go.Figure().update_layout(title="<b>Distribuição por Status</b>")
    df_status = df['Status_Tarefa'].value_counts().reset_index()
    fig = px.pie(df_status, names='Status_Tarefa', values='count', title='<b>Distribuição por Status</b>', hole=0.4, color_discrete_map={'Executado': 'royalblue', 'Aberto': 'firebrick'})
    fig.update_traces(textinfo='percent+label+value')
    return fig

def criar_grafico_tarefas_funcionarios(df):
    if df.empty: return go.Figure().update_layout(title="<b>N° de Tarefas por Funcionário</b>")
    df_funcionarios = df['Encarregado'].value_counts().reset_index()
    df_funcionarios = df_funcionarios.sort_values('count', ascending=True)
    fig = px.bar(df_funcionarios, x='count', y='Encarregado', orientation='h', title="<b>N° de Tarefas por Funcionário</b>", text_auto=True, color='count', color_continuous_scale='Blues')
    fig.update_layout(template='plotly_white', yaxis_title=None, coloraxis_showscale=False)
    return fig

def criar_grafico_pontuacao(df):
    if df.empty or 'Posição' not in df.columns: return go.Figure().update_layout(title="<b>Pontuação</b>")

    df_lideres_equipe = df[df['Posição'] == 'Lider']['Encarregado'].unique()
    
    df_peso_base = df.groupby('Encarregado')['Peso'].sum().reset_index()

    pontos_lideranca = {}
    for lider in df_lideres_equipe:
        lider_col_name = next((col for col in df.columns if col.lower() == lider.lower()), None)
        if lider_col_name:
            pontos_lideranca[lider] = pd.to_numeric(df[lider_col_name], errors='coerce').sum()
    df_pontos_lideranca = pd.DataFrame(list(pontos_lideranca.items()), columns=['Encarregado', 'Pontos_Lideranca'])

    df_agregado_final = pd.merge(df_peso_base, df_pontos_lideranca, on='Encarregado', how='left').fillna(0)
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
        dict(label="Total", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Funcionários", method="update", args=[{"visible": [False, True, False]}]),
        dict(label="Liderança", method="update", args=[{"visible": [False, False, True]}]),
    ]

    fig.update_layout(
        title_text="<b>Pontuação (Soma de Peso) por Funcionário</b>", template='plotly_white', yaxis_title=None,
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
        semanas_disponiveis = ["Todos"] + sorted([int(i) for i in df_filtrado['Semana do Mês'].unique() if pd.notna(i)])
        st.selectbox("Semana do Mês", semanas_disponiveis, key='semana_filtro')
    
    with top_col2:
        pesos_disponiveis = ["Todos"] + sorted([int(i) for i in df_filtrado['Peso'].unique() if pd.notna(i) and i > 0])
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