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
    
    # --- Nomes das Abas ---
    nome_aba_dados = "Total BaseCamp para Notas" 
    nome_aba_equipes = "Equipes"
    nome_aba_pontuacao = "Notas"
    nome_aba_lideranca = "Liderança"
    nome_aba_backlog = "Backlog" 
    
    # Inicializa DataFrames vazios
    df_dados = pd.DataFrame()
    df_equipe = pd.DataFrame()
    df_notas_tabela1 = pd.DataFrame()
    df_notas_tabela2 = pd.DataFrame()
    df_lideranca = pd.DataFrame()
    df_backlog = pd.DataFrame()

    try:
        # --- Carregar Abas de Atividade e Equipes ---
        worksheet_dados = spreadsheet.worksheet(nome_aba_dados)
        worksheet_equipe = spreadsheet.worksheet(nome_aba_equipes)
        df_dados = pd.DataFrame(worksheet_dados.get_all_records())
        df_equipe = pd.DataFrame(worksheet_equipe.get_all_records()) 

        # --- Carregar Aba "Liderança" ---
        worksheet_lideranca = spreadsheet.worksheet(nome_aba_lideranca)
        df_lideranca = pd.DataFrame(worksheet_lideranca.get_all_records())
        
        # --- Carregar Aba "Backlog" ---
        worksheet_backlog = spreadsheet.worksheet(nome_aba_backlog)
        df_backlog = pd.DataFrame(worksheet_backlog.get_all_records())

        # ==============================================================================
        # --- Carregar AMBAS as tabelas da aba "Notas" ---
        # ==============================================================================
        worksheet_pontuacao = spreadsheet.worksheet(nome_aba_pontuacao)
        all_values_notas = worksheet_pontuacao.get_all_values()
        
        primeira_linha_branca_index = -1
        # Encontra a primeira linha totalmente em branco
        for i, row in enumerate(all_values_notas):
            if not row or all(cell == '' for cell in row):
                primeira_linha_branca_index = i
                break
        
        # Se não achou linha em branco, a aba inteira é a Tabela 1
        if primeira_linha_branca_index == -1:
            dados_tabela_superior = all_values_notas
            dados_tabela_inferior = [] # Tabela 2 não existe
        else:
            dados_tabela_superior = all_values_notas[:primeira_linha_branca_index]
            
            # Procura o início da próxima tabela
            dados_tabela_inferior_inicio = -1
            for i, row in enumerate(all_values_notas[primeira_linha_branca_index + 1:], start=primeira_linha_branca_index + 1):
                if row and any(cell != '' for cell in row):
                    dados_tabela_inferior_inicio = i
                    break
            
            if dados_tabela_inferior_inicio != -1:
                dados_tabela_inferior = all_values_notas[dados_tabela_inferior_inicio:]
            else:
                dados_tabela_inferior = [] # Tabela 2 não existe

        # Processa Tabela 1 (Superior)
        if len(dados_tabela_superior) > 1:
            headers_sup = dados_tabela_superior[0]
            data_sup = dados_tabela_superior[1:]
            df_notas_tabela1 = pd.DataFrame(data_sup, columns=headers_sup)
        elif len(dados_tabela_superior) == 1:
             df_notas_tabela1 = pd.DataFrame(columns=dados_tabela_superior[0])

        # Processa Tabela 2 (Inferior)
        if len(dados_tabela_inferior) > 1:
            headers_inf = dados_tabela_inferior[0]
            data_inf = dados_tabela_inferior[1:]
            df_notas_tabela2 = pd.DataFrame(data_inf, columns=headers_inf)
        elif len(dados_tabela_inferior) == 1:
             df_notas_tabela2 = pd.DataFrame(columns=headers_inf[0])
        # ==============================================================================
    
    except gspread.exceptions.WorksheetNotFound as e:
        st.error(f"Erro: A aba '{e.args[0]}' não foi encontrada na planilha. Verifique os nomes.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() 
    except Exception as e:
        st.error(f"Erro ao carregar dados do Google Sheets: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- PREPARAÇÃO DOS DADOS (df_analise) ---
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
    
    # --- Colunas de Semana Adicionadas (baseada na Sexta-feira) ---
    tabela_calendario['Dia da Semana_ISO'] = tabela_calendario['Date'].dt.dayofweek # 0=Segunda, 4=Sexta
    tabela_calendario['Nome Dia Semana'] = tabela_calendario['Dia da Semana_ISO'].map({0:'seg', 1:'ter', 2:'qua', 3:'qui', 4:'sex', 5:'sab', 6:'dom'})
    
    # Calcula o início da semana (Segunda-feira)
    tabela_calendario['Data_Inicio_Semana'] = tabela_calendario['Date'] - pd.to_timedelta(tabela_calendario['Dia da Semana_ISO'], unit='d')
    # Calcula a Sexta-feira dessa semana
    tabela_calendario['Data_Sexta_Feira'] = tabela_calendario['Data_Inicio_Semana'] + pd.to_timedelta(4, unit='d')
    
    # O nome da semana agora é a data da sexta-feira
    tabela_calendario['Nome_da_Semana'] = tabela_calendario['Data_Sexta_Feira'].dt.strftime('%d/%m/%Y')
    
    # Coluna de ordenação
    tabela_calendario['Semana_Ano'] = tabela_calendario['Data_Sexta_Feira'].dt.strftime('%Y-%U') 
    
    tabela_calendario['Semana do Mês'] = (tabela_calendario['Date'].dt.dayofweek + (tabela_calendario['Date'].dt.day - 1)).floordiv(7) + 1
    # Mantém a coluna antiga para compatibilidade
    tabela_calendario['Dia da Semana'] = tabela_calendario['Dia da Semana_ISO'] + 1


    df_analise_temp = pd.merge(df_grafico, tabela_calendario, how='left', left_on='Data Final (aberta)', right_on='Date').drop(columns=['Date'])
    
    # Renomeia o status ANTES de fazer merge
    df_equipe.rename(columns={'Status': 'Status_Funcionario'}, inplace=True)
    df_analise = pd.merge(df_analise_temp, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
    df_analise['Status_Funcionario'].fillna('Outros', inplace=True)
    
    # --- PREPARAÇÃO DOS DADOS (df_backlog) ---
    if not df_backlog.empty:
        df_backlog['Data Inicial'] = pd.to_datetime(df_backlog['Data Inicial'], errors='coerce')
        # --- CORREÇÃO: "Data final" para "Data Final" ---
        df_backlog['Data Final'] = pd.to_datetime(df_backlog['Data Final'], errors='coerce') 
        df_backlog['Status_Backlog'] = np.where(df_backlog['Data Final'].isnull(), 'Aberto', 'Fechado')
        df_backlog['Encarregado'] = df_backlog['Encarregado'].astype(str).str.strip().replace('', 'Sem Responsável') # <-- Define um nome
        # Junta com a equipe para permitir filtragem por Status
        df_backlog = pd.merge(df_backlog, df_equipe, how='left', left_on='Encarregado', right_on='Nome')
        # Se 'Sem Responsável', o status fica 'Outros'
        df_backlog['Status_Funcionario'].fillna('Outros', inplace=True)


    # --- Retorna os SEIS DataFrames ---
    return df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca, df_equipe, df_backlog

# ==============================================================================
# FUNÇÕES PARA CRIAR OS GRÁFICOS (Sem alterações)
# ==============================================================================

def criar_grafico_produtividade_mensal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Produtividade Mensal</b>")
    
    df_agregado = df.groupby(['Ano-Mês', 'Mes_Ano_Abrev']).agg(
        contagem_tarefas=('ID', 'count')
    ).reset_index().sort_values('Ano-Mês')

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_agregado['Mes_Ano_Abrev'], 
        y=df_agregado['contagem_tarefas'], 
        name='Quantidade de Tarefas', 
        marker_color='royalblue',
        text=df_agregado['contagem_tarefas'], 
        textposition='outside'
    ))
    
    fig.update_layout(
        title="<b>Produtividade Mensal</b>", 
        template='plotly_white', 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    if not df_agregado.empty:
        max_y = df_agregado['contagem_tarefas'].max()
        fig.update_yaxes(range=[0, max_y * 1.2])

    return fig

def criar_grafico_principal(df):
    if df.empty: return go.Figure().update_layout(title="<b>Gráfico Principal</b>")
    
    def criar_figura_com_menu(df_contagem, df_agregado, col_x, col_filtro, nome_agregado, titulo, xaxis_titulo, yaxis_titulo, xaxis_extra=None):
        figura = go.Figure()
        
        custom_data_agregado = df_agregado['Nome Dia Semana'] if 'Nome Dia Semana' in df_agregado.columns else None
        hover_template_agregado = '<b>%{customdata}</b><br>Dia: %{x}<br>Qtd: %{y}<extra></extra>' if custom_data_agregado is not None else 'Dia: %{x}<br>Qtd: %{y}<extra></extra>'
        figura.add_trace(go.Scatter(
            x=df_agregado[col_x], y=df_agregado['Contagem'], name=nome_agregado,
            mode='lines+markers+text', text=df_agregado['Contagem'], textposition='top center',
            customdata=custom_data_agregado, hovertemplate=hover_template_agregado
        ))

        if 'Semana do Mês' in df_contagem.columns:
            opcoes_filtro = df_contagem.sort_values(['Ano-Mês', 'Semana do Mês'])[col_filtro].unique()
        else:
            opcoes_filtro = df_contagem.sort_values(['Ano-Mês'])[col_filtro].unique()
        
        for opcao in opcoes_filtro:
            df_filtrado = df_contagem[df_contagem[col_filtro] == opcao]
            if not df_filtrado.empty:
                if col_x == 'Nome Dia Semana':
                    df_filtrado = df_filtrado.sort_values(by='Dia da Semana')
                else:
                    df_filtrado = df_filtrado.sort_values(by=col_x)

                custom_data_filtrado = df_filtrado['Nome Dia Semana'] if 'Nome Dia Semana' in df_filtrado.columns else None
                hover_template_filtrado = '<b>%{customdata}</b><br>Dia: %{x}<br>Qtd: %{y}<extra></extra>' if custom_data_filtrado is not None else 'Dia: %{x}<br>Qtd: %{y}<extra></extra>'
                figura.add_trace(go.Scatter(
                    x=df_filtrado[col_x], y=df_filtrado['Contagem'], name=opcao,
                    mode='lines+markers+text', text=df_filtrado['Contagem'], textposition='top center', visible=False,
                    customdata=custom_data_filtrado, hovertemplate=hover_template_filtrado
                ))
        
        botoes = [{'label': nome_agregado, 'method': 'update', 'args': [{'visible': [i == 0 for i in range(len(figura.data))]}]}]
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
    fig_dia_semana = criar_figura_com_menu(contagem_diaria_semana, agregado_todos_dias_semana, 'Nome Dia Semana', 'Filtro', 'Total Agregado', '<b>Contagem por Dia da Semana</b>', None, None, xaxis_extra=dict(categoryorder='array', categoryarray=['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']))
    
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
    fig = px.bar(df_funcionarios, x='Contagem', y='Encarregado', orientation='h', title="<b>N° de Tarefas por Funcionário</b>", text='Contagem', color='Contagem', color_continuous_scale='Blues')
    fig.update_layout(
        template='plotly_white', 
        yaxis_title=None, 
        coloraxis_showscale=False,
        yaxis_categoryorder='total ascending' # Menor na base, maior no topo
    )
    return fig

# ==============================================================================
# --- GRÁFICO 1 (Aba 4): Pontuação Individual ---
# ==============================================================================
def criar_grafico_pontuacao_individual(df_notas, nomes_para_exibir, start_date, end_date):
    """
    Cria um gráfico de barras com a soma total da pontuação da aba "Notas" (Tabela 1).
    Filtra as *colunas de data* com base no slider.
    Filtra a *exibição de linhas* com base na lista 'nomes_para_exibir'.
    Retorna a figura E o dataframe filtrado e ordenado para a tabela.
    """
    if df_notas is None or df_notas.empty:
        fig = go.Figure().update_layout(title="<b>Pontuação Individual (Encarregados)</b><br><i>Tabela 1 da aba 'Notas' não encontrada ou vazia.</i>")
        return fig, pd.DataFrame()
    
    df_proc = df_notas.copy()
    colunas_pontuacao_todas = [col for col in df_proc.columns if col.lower() != 'encarregado']
    
    colunas_pontuacao_filtradas = []
    for col in colunas_pontuacao_todas:
        try:
            data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
            if start_date <= data_coluna <= end_date:
                colunas_pontuacao_filtradas.append(col)
        except ValueError:
            pass 
    
    if not colunas_pontuacao_filtradas:
        fig = go.Figure().update_layout(title="<b>Pontuação Individual (Encarregados)</b><br><i>Nenhuma coluna de pontuação encontrada no período selecionado.</i>")
        return fig, pd.DataFrame(columns=['Encarregado', 'Pontuacao_Total'] + colunas_pontuacao_filtradas)

    for col in colunas_pontuacao_filtradas:
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)
    
    df_proc['Pontuacao_Total'] = df_proc[colunas_pontuacao_filtradas].sum(axis=1)
    
    # Filtra por nomes ANTES de criar o gráfico
    df_grafico = df_proc[df_proc['Encarregado'].isin(nomes_para_exibir)]
    
    # Ordena para o gráfico e para a tabela
    df_grafico_final = df_grafico.sort_values(by='Pontuacao_Total', ascending=False)


    fig = px.bar(
        df_grafico_final, 
        x='Pontuacao_Total', 
        y='Encarregado', 
        orientation='h', 
        title="<b>Ranking de Pontuação Individual</b>", 
        text='Pontuacao_Total',
        color='Pontuacao_Total',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        template='plotly_white', 
        yaxis_title=None, 
        coloraxis_showscale=False,
        yaxis_categoryorder='total ascending' 
    )
    fig.update_traces(texttemplate='%{text:.0f}') 
    
    # Prepara o DF para a tabela (com colunas reordenadas)
    colunas_tabela = ['Encarregado', 'Pontuacao_Total'] + colunas_pontuacao_filtradas
    df_para_tabela = df_grafico_final[colunas_tabela]
    
    return fig, df_para_tabela

# ==============================================================================
# --- GRÁFICO 2 (Aba 4): Pontuação de Liderança ---
# ==============================================================================
def criar_grafico_pontuacao_lideres(df_mapa_lideres, df_pontos_liderados, nomes_para_exibir, start_date, end_date):
    """
    Cria um gráfico de ranking de líderes e os dados detalhados para as tabelas.
    Retorna: fig, df_lideres_visiveis (para o loop), df_liderados_pontos_detalhe (para as tabelas)
    """
    if df_mapa_lideres is None or df_mapa_lideres.empty:
        fig = go.Figure().update_layout(title="<b>Pontuação de Líderes</b><br><i>Aba 'Liderança' não encontrada ou vazia.</i>")
        return fig, pd.DataFrame(), pd.DataFrame()
    if df_pontos_liderados is None or df_pontos_liderados.empty:
        fig = go.Figure().update_layout(title="<b>Pontuação de Líderes</b><br><i>Tabela 2 da aba 'Notas' não encontrada ou vazia.</i>")
        return fig, pd.DataFrame(), pd.DataFrame()

    # 1. Processar a tabela de pontos dos liderados (TODOS)
    df_pontos = df_pontos_liderados.copy()
    colunas_pontuacao_todas = [col for col in df_pontos.columns if col.lower() != 'encarregado']
    
    colunas_pontuacao_filtradas = []
    for col in colunas_pontuacao_todas:
        try:
            data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
            if start_date <= data_coluna <= end_date:
                colunas_pontuacao_filtradas.append(col)
        except ValueError:
            pass 

    if not colunas_pontuacao_filtradas:
        fig = go.Figure().update_layout(title="<b>Pontuação de Líderes</b><br><i>Nenhuma coluna de pontuação encontrada no período selecionado.</i>")
        return fig, pd.DataFrame(), pd.DataFrame(columns=['Encarregado', 'Pontuacao_Total_Liderado'] + colunas_pontuacao_filtradas)

    for col in colunas_pontuacao_filtradas:
        df_pontos[col] = pd.to_numeric(df_pontos[col], errors='coerce').fillna(0)
    
    # 2. Somar os pontos para cada liderado (TODOS)
    df_pontos['Pontuacao_Total_Liderado'] = df_pontos[colunas_pontuacao_filtradas].sum(axis=1)
    
    df_liderados_pontos_detalhe = df_pontos[['Encarregado', 'Pontuacao_Total_Liderado'] + colunas_pontuacao_filtradas]
    df_liderados_pontos_detalhe['Encarregado'] = df_liderados_pontos_detalhe['Encarregado'].astype(str).str.strip()
    
    df_pontos_total = df_liderados_pontos_detalhe[['Encarregado', 'Pontuacao_Total_Liderado']]

    # 3. Juntar com o mapa da aba "Liderança"
    df_mapa_lideres['Lider'] = df_mapa_lideres['Lider'].astype(str).str.strip() 
    df_mapa_lideres['Liderado'] = df_mapa_lideres['Liderado'].astype(str).str.strip()

    df_merge = pd.merge(
        df_mapa_lideres, 
        df_pontos_total, 
        left_on='Liderado',
        right_on='Encarregado'
    )

    # 4. Somar os pontos por Líder (TODOS OS LÍDERES)
    df_final_lideres = df_merge.groupby('Lider')['Pontuacao_Total_Liderado'].sum().reset_index()
    df_final_lideres = df_final_lideres.rename(columns={'Pontuacao_Total_Liderado': 'Pontuacao_Total_Lider'})

    # 5. Filtra o dataframe final antes de plotar
    df_grafico_filtrado = df_final_lideres[df_final_lideres['Lider'].isin(nomes_para_exibir)]
    
    df_lideres_visiveis = df_grafico_filtrado.sort_values(by='Pontuacao_Total_Lider', ascending=False)

    # 6. Criar o Gráfico
    fig = px.bar(
        df_lideres_visiveis, # Usa o DF já filtrado e ordenado
        x='Pontuacao_Total_Lider',
        y='Lider',
        orientation='h',
        title="<b>Ranking de Pontuação (Apenas Liderança)</b>",
        text='Pontuacao_Total_Lider',
        color='Pontuacao_Total_Lider',
        color_continuous_scale='Plasma'
    )
    
    fig.update_layout(
        template='plotly_white',
        yaxis_title=None,
        coloraxis_showscale=False,
        yaxis_categoryorder='total ascending'
    )
    fig.update_traces(texttemplate='%{text:.0f}')
    
    return fig, df_lideres_visiveis, df_liderados_pontos_detalhe

# ------------------------------------------------------------------------------
# --- GRÁFICO 3 (Aba 4): Pontuação Combinada ---
# ------------------------------------------------------------------------------
def criar_grafico_pontuacao_combinada(df_notas_enc, df_notas_liderados, df_mapa_lideres, nomes_para_exibir, start_date, end_date):
    """
    Cria um gráfico de barras com a SOMA da pontuação individual (Tabela 1)
    e da pontuação de liderança (Tabela 2 + Mapa).
    Filtra as *colunas de data* com base no slider.
    Filtra a *exibição de linhas* com base na lista 'nomes_para_exibir'.
    """
    
    # --- 1. Calcular Pontos Individuais (da Tabela 1 - TODOS) ---
    df_individuais = pd.DataFrame(columns=['Pessoa', 'Pontuacao_Individual'])
    if df_notas_enc is not None and not df_notas_enc.empty:
        df_proc_enc = df_notas_enc.copy()
        col_pont_enc_todas = [col for col in df_proc_enc.columns if col.lower() != 'encarregado']
        
        # Filtra colunas de data
        col_pont_enc_filtradas = []
        for col in col_pont_enc_todas:
            try:
                data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
                if start_date <= data_coluna <= end_date:
                    col_pont_enc_filtradas.append(col)
            except ValueError: pass

        if col_pont_enc_filtradas:
            for col in col_pont_enc_filtradas:
                df_proc_enc[col] = pd.to_numeric(df_proc_enc[col], errors='coerce').fillna(0)
            df_proc_enc['Pontuacao_Individual'] = df_proc_enc[col_pont_enc_filtradas].sum(axis=1)
            df_individuais = df_proc_enc[['Encarregado', 'Pontuacao_Individual']].rename(columns={'Encarregado': 'Pessoa'})
            df_individuais['Pessoa'] = df_individuais['Pessoa'].astype(str).str.strip()
        else:
             df_individuais = df_proc_enc[['Encarregado']].rename(columns={'Encarregado': 'Pessoa'})
             df_individuais['Pontuacao_Individual'] = 0
             df_individuais['Pessoa'] = df_individuais['Pessoa'].astype(str).str.strip()


    # --- 2. Calcular Pontos de Liderança (da Tabela 2 + Mapa - TODOS) ---
    df_lideres_final = pd.DataFrame(columns=['Pessoa', 'Pontuacao_Lideranca']) # Default vazio
    if (df_notas_liderados is not None and not df_notas_liderados.empty and 
        df_mapa_lideres is not None and not df_mapa_lideres.empty):
        
        df_pontos = df_notas_liderados.copy()
        col_pont_lid_todas = [col for col in df_pontos.columns if col.lower() != 'encarregado']
        
        # Filtra colunas de data
        col_pont_lid_filtradas = []
        for col in col_pont_lid_todas:
            try:
                data_coluna = pd.to_datetime(col, format='%Y-%m-%d').date()
                if start_date <= data_coluna <= end_date:
                    col_pont_lid_filtradas.append(col)
            except ValueError: pass

        if col_pont_lid_filtradas:
            for col in col_pont_lid_filtradas:
                df_pontos[col] = pd.to_numeric(df_pontos[col], errors='coerce').fillna(0)
            
            df_pontos['Pontuacao_Total_Liderado'] = df_pontos[col_pont_lid_filtradas].sum(axis=1)
            df_pontos_total = df_pontos[['Encarregado', 'Pontuacao_Total_Liderado']]

            # Limpa os nomes para o merge
            df_mapa_lideres['Lider'] = df_mapa_lideres['Lider'].astype(str).str.strip()
            df_mapa_lideres['Liderado'] = df_mapa_lideres['Liderado'].astype(str).str.strip()
            df_pontos_total['Encarregado'] = df_pontos_total['Encarregado'].astype(str).str.strip()

            df_merge = pd.merge(
                df_mapa_lideres, 
                df_pontos_total, 
                left_on='Liderado', 
                right_on='Encarregado'
            )
            
            df_lideres_soma = df_merge.groupby('Lider')['Pontuacao_Total_Liderado'].sum().reset_index()
            df_lideres_final = df_lideres_soma.rename(columns={'Lider': 'Pessoa', 'Pontuacao_Total_Liderado': 'Pontuacao_Lideranca'})
        else:
            df_lideres_final = df_mapa_lideres[['Lider']].rename(columns={'Lider': 'Pessoa'})
            df_lideres_final['Pontuacao_Lideranca'] = 0
            df_lideres_final = df_lideres_final.drop_duplicates()


    # --- 3. Combinar Pontuações (Individual + Liderança - TODOS) ---
    df_combinado = pd.merge(
        df_individuais,
        df_lideres_final,
        on='Pessoa',
        how='outer'
    ).fillna(0)

    df_combinado['Pontuacao_Total_Combinada'] = df_combinado['Pontuacao_Individual'] + df_combinado['Pontuacao_Lideranca']
    
    # Filtra quem tem 0 pontos
    df_grafico = df_combinado[df_combinado['Pontuacao_Total_Combinada'] > 0]

    # 4. Filtra o dataframe final antes de plotar
    df_grafico_filtrado = df_grafico[df_grafico['Pessoa'].isin(nomes_para_exibir)]

    # 5. Criar o Gráfico
    fig = px.bar(
        df_grafico_filtrado, 
        x='Pontuacao_Total_Combinada', 
        y='Pessoa', 
        orientation='h', 
        title="<b>Ranking Geral de Pontuação (Individual + Liderança)</b>", 
        text='Pontuacao_Total_Combinada',
        color='Pontuacao_Total_Combinada',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        template='plotly_white', 
        yaxis_title=None, 
        coloraxis_showscale=False,
        yaxis_categoryorder='total ascending' # Ranking (maior no topo)
    )
    fig.update_traces(texttemplate='%{text:.0f}') 
    return fig


# ==============================================================================
# CORPO PRINCIPAL DO DASHBOARD (INTERFACE STREAMLIT)
# ==============================================================================
st.title("Dashboard de Produtividade")
# --- MUDANÇA: Carrega os SEIS dataframes ---
df_analise, df_notas_tabela1, df_notas_tabela2, df_lideranca_mapa, df_equipe, df_backlog = carregar_dados_completos()

# ==============================================================================
# --- MUDANÇA: Lógica de definição de data do slider ---
# ==============================================================================
# Define o range do slider. Prioriza a aba de atividade (df_analise)
if (df_analise is not None and not df_analise.empty):
    min_date = df_analise['Data Final (aberta)'].min().date()
    max_date = df_analise['Data Final (aberta)'].max().date()
# Fallback se "Notas" foi carregada mas "Atividade Geral" não
elif (df_notas_tabela1 is not None and not df_notas_tabela1.empty):
    colunas_data_notas = [col for col in df_notas_tabela1.columns if col.lower() != 'encarregado']
    datas_convertidas = []
    for col in colunas_data_notas:
        try:
            datas_convertidas.append(pd.to_datetime(col, format='%Y-%m-%d').date())
        except ValueError:
            pass 
    
    if datas_convertidas:
        min_date = min(datas_convertidas)
        max_date = max(datas_convertidas)
    else:
        min_date = date.today()
        max_date = date.today()
else: # Fallback se tudo falhar
    min_date = date.today()
    max_date = date.today()
# ==============================================================================
# --- FIM DA MUDANÇA ---
# ==============================================================================


# --- Verifica se o df_analise (principal) foi carregado para exibir a Aba 1 ---
if (df_analise is not None and not df_analise.empty):

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
        st.selectbox("Status (Contrato)", contratos_disponiveis, key='contrato_filtro')

        status_tarefas = ["Todos"] + df_analise['Status_Tarefa'].unique().tolist()
        st.selectbox("Status da Tarefa", status_tarefas, key='status_tarefa_filtro')

        st.markdown("---")
        st.button("Limpar Filtros 🗑️", on_click=limpar_filtros)

    # --- Lógica de filtragem (aplica-se apenas ao df_analise para a Aba 1) ---
    df_filtrado_aba1 = df_analise.copy()
    if "Todos" not in st.session_state.encarregado_filtro:
        df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Encarregado'].isin(st.session_state.encarregado_filtro)]
    if st.session_state.contrato_filtro != "Todos":
        df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Status_Funcionario'] == st.session_state.contrato_filtro]
    if st.session_state.status_tarefa_filtro != "Todos":
        df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Status_Tarefa'] == st.session_state.status_tarefa_filtro]
        
    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns([2, 2, 1, 1, 4])

    with top_col1:
        semanas_disponiveis = ["Todos"] + sorted([i for i in df_filtrado_aba1['Semana do Mês'].unique() if i is not np.nan])
        st.selectbox("Semana do Mês", semanas_disponiveis, key='semana_filtro')
    
    with top_col2:
        pesos_disponiveis = ["Todos"] + sorted(df_filtrado_aba1['Peso'].astype(int).unique())
        st.selectbox("Peso da Tarefa", pesos_disponiveis, key='peso_filtro')

    with top_col5:
        st.slider("Intervalo de Datas (para Abas 1, 3 e 4)", min_value=min_date, max_value=max_date, key='date_slider')

    if st.session_state.semana_filtro != "Todos":
        df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Semana do Mês'] == st.session_state.semana_filtro]
    if st.session_state.peso_filtro != "Todos":
        df_filtrado_aba1 = df_filtrado_aba1[df_filtrado_aba1['Peso'] == st.session_state.peso_filtro]
    
    start_date, end_date = st.session_state.date_slider
    df_filtrado_aba1 = df_filtrado_aba1[(df_filtrado_aba1['Data Final (aberta)'].dt.date >= start_date) & (df_filtrado_aba1['Data Final (aberta)'].dt.date <= end_date)]

    with top_col3:
        st.metric("Tarefas", f"{df_filtrado_aba1.shape[0]:,}")
    with top_col4:
        # Métrica de Soma de Peso removida
        pass 
        
    st.markdown("---") 

    # ==============================================================================
    # --- CORPO PRINCIPAL COM ABAS ---
    # ==============================================================================
    
    aba1, aba2, aba3, aba4 = st.tabs([
        "Atividade Geral", 
        "Semana", 
        "Backlog", 
        "Pontuação Geral"
    ])

    # --- Aba 1: Atividade Geral ---
    with aba1:
        st.header("Visão Geral da Atividade")
        
        col_geral1, col_geral2 = st.columns(2)
        with col_geral1:
            fig_prod_mensal = criar_grafico_produtividade_mensal(df_filtrado_aba1)
            st.plotly_chart(fig_prod_mensal, use_container_width=True)
        with col_geral2:
            fig_principal = criar_grafico_principal(df_filtrado_aba1)
            st.plotly_chart(fig_principal, use_container_width=True)

        st.markdown("---")
        st.header("Análise de Equipe e Status")
        col_equipe1, col_equipe2 = st.columns(2)
        with col_equipe1:
            fig_tarefas = criar_grafico_tarefas_funcionarios(df_filtrado_aba1)
            st.plotly_chart(fig_tarefas, use_container_width=True)
        with col_equipe2:
            fig_status = criar_grafico_status_tarefas(df_filtrado_aba1)
            st.plotly_chart(fig_status, use_container_width=True)
        
    # --- Aba 2: Semana ---
    with aba2:
        st.header("Análise Detalhada por Semana")
        
        # ==============================================================================
        # --- MUDANÇA: Lógica da Aba Semanal (Baseada em 'Total Basecamp...') ---
        # ==============================================================================
        
        # 1. Obter a lista de semanas do df_analise
        df_analise_executado = df_analise[df_analise['Status_Tarefa'] == 'Executado']
        
        # Usa 'Semana_Ano' para ordenação correta e 'Nome_da_Semana' (Sexta-feira) para exibição
        semanas_df = df_analise_executado[['Nome_da_Semana', 'Semana_Ano']].drop_duplicates().sort_values(by='Semana_Ano', ascending=False)
        semanas_lista = semanas_df['Nome_da_Semana'].tolist()
        
        if not semanas_lista:
            st.info("Nenhuma tarefa executada encontrada nos dados.")
        else:
            # --- MUDANÇA: Adicionada 'key' para corrigir bug de scroll ---
            semana_selecionada = st.selectbox("Selecione uma Semana (data da Sexta-feira):", semanas_lista, key="aba2_semana_select")
            
            # Filtra o df de atividades para a semana (Nome_da_Semana == Sexta-feira)
            df_semana_full = df_analise_executado[df_analise_executado['Nome_da_Semana'] == semana_selecionada]
            
            if df_semana_full.empty:
                st.info("Nenhuma tarefa encontrada para esta semana.")
            else:
                # 4. Criar a Tabela Pivô
                try:
                    pivot = pd.pivot_table(
                        df_semana_full,
                        index='Encarregado',
                        columns='Nome Dia Semana',
                        values='ID',  # Usa 'ID' para contagem
                        aggfunc='count', # 'count' é a contagem de tarefas
                        fill_value=0
                    )
                    
                    # Garante que todos os dias da semana estejam presentes
                    colunas_dias_ordem = ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom']
                    pivot = pivot.reindex(columns=colunas_dias_ordem, fill_value=0)
                    
                    # Calcula o início (Seg) e fim (Dom) da semana para o título
                    data_sexta = pd.to_datetime(semana_selecionada, format='%d/%m/%Y')
                    data_segunda = data_sexta - pd.Timedelta(days=4)
                    data_domingo = data_sexta + pd.Timedelta(days=2)
                    
                    st.subheader(f"Contagem de Tarefas Concluídas ({data_segunda.strftime('%d/%m')} a {data_domingo.strftime('%d/%m/%Y')})")
                    st.info("A tabela abaixo mostra a *contagem* de tarefas que cada pessoa concluiu em cada dia da semana selecionada.")
                    st.dataframe(pivot, use_container_width=True)

                    # --- 5. Ferramenta de Drill-Down ---
                    st.markdown("---")
                    st.subheader("Detalhar Tarefas da Célula")
                    st.write("Selecione um encarregado e um dia da tabela acima para ver as tarefas em detalhes.")
                    
                    col_detalhe1, col_detalhe2 = st.columns(2)
                    with col_detalhe1:
                        encarregados_na_semana = pivot.index.tolist()
                        # --- MUDANÇA: Adicionada 'key' para corrigir bug de scroll ---
                        encarregado_detalhe = st.selectbox("Selecione um Encarregado:", [""] + encarregados_na_semana, key="aba2_encarregado_select")
                    
                    with col_detalhe2:
                        # --- MUDANÇA: Adicionada 'key' para corrigir bug de scroll ---
                        dia_detalhe = st.selectbox("Selecione um Dia:", [""] + colunas_dias_ordem, key="aba2_dia_select")

                    if encarregado_detalhe and dia_detalhe:
                        df_detalhe = df_semana_full[
                            (df_semana_full['Encarregado'] == encarregado_detalhe) &
                            (df_semana_full['Nome Dia Semana'] == dia_detalhe)
                        ]
                        
                        num_tarefas = len(df_detalhe)
                        if num_tarefas > 0:
                            st.success(f"Encontradas {num_tarefas} tarefas para '{encarregado_detalhe}' na '{dia_detalhe}'.")
                            # ==============================================================================
                            # --- MUDANÇA: "Peso" removido, LinkColumn adicionado ---
                            # ==============================================================================
                            colunas_para_mostrar = ['ID', 'Nome Task', 'Link', 'Data Final']
                            
                            # Configuração da coluna de link
                            column_config = {
                                "Link": st.column_config.LinkColumn("Link da Tarefa", display_text="Abrir ↗")
                            }
                            
                            if 'Nome Task' not in df_detalhe.columns:
                                st.error("Erro: A coluna 'Nome Task' não foi encontrada. Verifique o nome na aba 'Total BaseCamp para Notas'.")
                                colunas_para_mostrar = ['ID', 'Link', 'Data Final']
                                st.dataframe(
                                    df_detalhe[colunas_para_mostrar], 
                                    use_container_width=True,
                                    column_config=column_config # Aplica a configuração
                                )
                            else:
                                st.dataframe(
                                    df_detalhe[colunas_para_mostrar], 
                                    use_container_width=True,
                                    column_config=column_config # Aplica a configuração
                                )
                            # ==============================================================================
                            # --- FIM DA MUDANÇA ---
                            # ==============================================================================
                        else:
                            st.warning(f"Nenhuma tarefa encontrada para '{encarregado_detalhe}' na '{dia_detalhe}'.")

                except Exception as e:
                    st.error(f"Erro ao gerar a tabela pivô: {e}")
        # ==============================================================================
        # --- FIM DA LÓGICA DA ABA SEMANAL ---
        # ==============================================================================


    # --- Aba 3: Backlog ---
    with aba3:
        st.header("Backlog de Tarefas por Status")
        
        # ==============================================================================
        # --- NOVO: Lógica da Aba Backlog (3 Porções) ---
        # ==============================================================================
        df_backlog_filtrado = df_backlog.copy()
        
        # 1. Aplica filtros da barra lateral (Exceto Status da Tarefa)
        if "Todos" not in st.session_state.encarregado_filtro:
            df_backlog_filtrado = df_backlog_filtrado[df_backlog_filtrado['Encarregado'].isin(st.session_state.encarregado_filtro + ["Sem Responsável"])] # Inclui 'Sem Responsável'
        if st.session_state.contrato_filtro != "Todos":
            df_backlog_filtrado = df_backlog_filtrado[df_backlog_filtrado['Status_Funcionario'] == st.session_state.contrato_filtro]
        
        # 2. ==============================================================================
        # --- MUDANÇA: Filtro de data do slider REMOVIDO desta aba ---
        # ==============================================================================
        
        if df_backlog.empty:
             st.error("Aba 'Backlog' não foi carregada ou está vazia.")
        else:
            # Colunas para exibir
            colunas_backlog_para_mostrar = ['Nome Task', 'Encarregado', 'Link', 'Lista', 'Data Inicial', 'Data Final', 'ID']
            # Verificação de 'Nome Task'
            if 'Nome Task' not in df_backlog_filtrado.columns:
                colunas_backlog_para_mostrar = ['ID', 'Encarregado', 'Link', 'Lista', 'Data Inicial', 'Data Final']
                st.warning("Coluna 'Nome Task' não encontrada na aba 'Backlog'. Exibindo 'ID'.")

            # Configuração das colunas
            column_config_backlog = {
                "Link": st.column_config.LinkColumn("Tarefa Link", display_text="Abrir ↗"),
                "Lista": st.column_config.LinkColumn("Lista Link", display_text="Abrir ↗"),
                "Data Inicial": st.column_config.DateColumn("Data Inicial", format="DD/MM/YYYY"),
                "Data Final": st.column_config.DateColumn("Data Final", format="DD/MM/YYYY")
            }

            # --- Seção 1: Abertas Sem Responsável ---
            df_abertas_sem_resp = df_backlog_filtrado[
                (df_backlog_filtrado['Status_Backlog'] == 'Aberto') &
                (df_backlog_filtrado['Encarregado'] == 'Sem Responsável')
            ]
            with st.expander(f"⚫ Tarefas Abertas (Sem Responsável) - {len(df_abertas_sem_resp)}", expanded=True):
                st.dataframe(
                    df_abertas_sem_resp[colunas_backlog_para_mostrar], 
                    use_container_width=True, 
                    column_config={
                        **column_config_backlog, 
                        "Encarregado": None # Oculta a coluna
                    }, 
                    hide_index=True
                )

            # --- Seção 2: Abertas Com Responsável ---
            df_abertas_com_resp = df_backlog_filtrado[
                (df_backlog_filtrado['Status_Backlog'] == 'Aberto') &
                (df_backlog_filtrado['Encarregado'] != 'Sem Responsável')
            ]
            with st.expander(f"🔴 Tarefas Abertas (Com Responsável) - {len(df_abertas_com_resp)}", expanded=True):
                st.dataframe(
                    df_abertas_com_resp[colunas_backlog_para_mostrar].sort_values(by="Encarregado"), 
                    use_container_width=True, 
                    column_config=column_config_backlog, 
                    hide_index=True
                )

            # --- Seção 3: Fechadas (Todas) ---
            df_fechadas = df_backlog_filtrado[
                (df_backlog_filtrado['Status_Backlog'] == 'Fechado')
            ]
            with st.expander(f"🟢 Tarefas Fechadas (Com ou Sem Responsável) - {len(df_fechadas)}", expanded=False):
                st.dataframe(
                    df_fechadas[colunas_backlog_para_mostrar].sort_values(by="Data Final", ascending=False), 
                    use_container_width=True, 
                    column_config=column_config_backlog, 
                    hide_index=True
                )
        # ==============================================================================
        # --- FIM DA LÓGICA DA ABA BACKLOG ---
        # ==============================================================================


    # --- Aba 4: Pontuação Geral ---
    with aba4:
        # ==============================================================================
        # --- Lógica de Filtro para a Aba de Pontuação ---
        # ==============================================================================
        
        # 1. Pega os nomes da aba "Equipes" com base no filtro de Status
        nomes_status_filtrados = []
        if df_equipe is not None and not df_equipe.empty:
            if st.session_state.contrato_filtro == "Todos":
                nomes_status_filtrados = df_equipe['Nome'].unique().tolist()
            else:
                nomes_status_filtrados = df_equipe[df_equipe['Status_Funcionario'] == st.session_state.contrato_filtro]['Nome'].unique().tolist()
        
        # 2. Pega os nomes do filtro multiselect "Encarregado"
        nomes_encarregado_filtrados = []
        if "Todos" in st.session_state.encarregado_filtro:
            # Se 'Todos' está selecionado, pegamos todos os nomes da aba Equipes (ou uma lista grande)
            if df_equipe is not None and not df_equipe.empty:
                 nomes_encarregado_filtrados = df_equipe['Nome'].unique().tolist()
            else:
                 nomes_encarregado_filtrados = df_analise['Encarregado'].unique().tolist()
        else:
            nomes_encarregado_filtrados = st.session_state.encarregado_filtro
            
        # 3. A lista final de nomes é a INTERSECÇÃO dos dois filtros
        lista_nomes_final_para_exibir = list(set(nomes_status_filtrados) & set(nomes_encarregado_filtrados))
        
        # ==============================================================================
        # --- Gráficos da Aba 4 (Reordenados e com Tabelas) ---
        # ==============================================================================

        # --- Gráfico 1: Individual ---
        st.header("Ranking de Pontuação Individual")
        fig_pontuacao_individual, df_tabela_individual = criar_grafico_pontuacao_individual(
            df_notas_tabela1,
            lista_nomes_final_para_exibir,
            start_date, end_date 
        )
        st.plotly_chart(fig_pontuacao_individual, use_container_width=True)
        
        # --- Tabela 1: Detalhes Individuais ---
        with st.expander("Ver tabela de dados (Pontuação Individual)"):
            if not df_tabela_individual.empty:
                st.dataframe(df_tabela_individual, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum dado de pontuação individual para exibir com os filtros atuais.")
        
        st.markdown("---")
        
        # --- Gráfico 2: Liderança ---
        st.header("Ranking de Pontuação (Apenas Liderança)")
        fig_pontuacao_lideres, df_lideres_visiveis, df_liderados_pontos_detalhe = criar_grafico_pontuacao_lideres(
            df_lideranca_mapa, 
            df_notas_tabela2,
            lista_nomes_final_para_exibir,
            start_date, end_date 
        )
        st.plotly_chart(fig_pontuacao_lideres, use_container_width=True)

        # --- Tabelas 2: Detalhes por Líder ---
        with st.expander("Ver tabelas de dados (Detalhes por Líder)"):
            if not df_lideres_visiveis.empty and not df_liderados_pontos_detalhe.empty:
                lideres_para_exibir_lista = df_lideres_visiveis['Lider'].tolist()
                
                for lider in lideres_para_exibir_lista:
                    st.subheader(f"Detalhes da Pontuação: {lider}")
                    liderados_deste_lider = df_lideranca_mapa[df_lideranca_mapa['Lider'] == lider]['Liderado'].tolist()
                    
                    df_tabela_lider = df_liderados_pontos_detalhe[df_liderados_pontos_detalhe['Encarregado'].isin(liderados_deste_lider)]
                    df_tabela_lider_sorted = df_tabela_lider.sort_values(by='Pontuacao_Total_Liderado', ascending=False)
                    
                    st.dataframe(df_tabela_lider_sorted, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum dado de pontuação de liderança para exibir com os filtros atuais.")

        st.markdown("---")

        # --- Gráfico 3: Combinado ---
        st.header("Ranking Geral de Pontuação (Individual + Liderança)")
        fig_pontuacao_combinada = criar_grafico_pontuacao_combinada(
            df_notas_tabela1, 
            df_notas_tabela2, 
            df_lideranca_mapa,
            lista_nomes_final_para_exibir,
            start_date, end_date 
        )
        st.plotly_chart(fig_pontuacao_combinada, use_container_width=True)
        st.info("Este gráfico soma a pontuação individual e de liderança no período selecionado.")
    
else:
    # --- Fallback se 'df_analise' falhar, mas 'df_notas' carregar ---
    if (df_notas_tabela1 is not None):
        st.warning("Não foi possível carregar os dados da aba 'Total BaseCamp para Notas', mas as abas de pontuação foram carregadas.")
        st.info("A aba 'Pontuação Geral' pode estar funcional, mas 'Atividade Geral' está desabilitada.")
        
        # Mostra apenas a aba 4 se os dados de pontuação existirem
        aba4_tabs = st.tabs(["Pontuação Geral"])
        
        if aba4_tabs:
            with aba4_tabs[0]:
                
                # Na falha, não há filtros, então exibimos todos
                if df_equipe is not None and not df_equipe.empty:
                    nomes_para_exibir_fallback = df_equipe['Nome'].unique().tolist()
                elif df_notas_tabela1 is not None:
                    nomes_para_exibir_fallback = df_notas_tabela1['Encarregado'].unique().tolist()
                else:
                    nomes_para_exibir_fallback = []

                # --- Gráficos (Ordem Corrigida) ---
                st.header("Ranking de Pontuação Individual")
                fig_pontuacao_individual, df_tabela_individual_fb = criar_grafico_pontuacao_individual(
                    df_notas_tabela1,
                    nomes_para_exibir_fallback,
                    min_date, max_date 
                )
                st.plotly_chart(fig_pontuacao_individual, use_container_width=True)
                with st.expander("Ver tabela de dados (Pontuação Individual)"):
                    st.dataframe(df_tabela_individual_fb, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                st.header("Ranking de Pontuação (Apenas Liderança)")
                fig_pontuacao_lideres, df_lideres_visiveis_fb, df_liderados_pontos_detalhe_fb = criar_grafico_pontuacao_lideres(
                    df_lideranca_mapa, 
                    df_notas_tabela2,
                    nomes_para_exibir_fallback,
                    min_date, max_date 
                )
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
                fig_pontuacao_combinada = criar_grafico_pontuacao_combinada(
                    df_notas_tabela1, 
                    df_notas_tabela2, 
                    df_lideranca_mapa,
                    nomes_para_exibir_fallback,
                    min_date, max_date 
                )
                st.plotly_chart(fig_pontuacao_combinada, use_container_width=True)
    
    else:
        st.error("Não foi possível carregar nenhum dado para exibir o dashboard.")