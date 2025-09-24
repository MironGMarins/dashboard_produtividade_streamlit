# api.py (Versão 2 - Servindo os Dados)

from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials


# ==============================================================================
# FUNÇÃO DE CARREGAMENTO E PREPARAÇÃO DOS DADOS
# Esta é a mesma lógica que aperfeiçoamos no notebook.
# ==============================================================================
def preparar_dados_completos():
    # --- Autenticação e Carregamento do Google Sheets ---
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file("google_credentials.json", scopes=scopes)
    client = gspread.authorize(creds)
    url_da_planilha = st.secrets["SHEET_URL"]
    spreadsheet = client.open_by_url(url_da_planilha)
    worksheet = spreadsheet.get_worksheet(2)
    df = pd.DataFrame(worksheet.get_all_records())

    # --- Tratamento e Criação de Colunas (df_grafico) ---
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
# CONFIGURAÇÃO DO SERVIDOR FLASK
# ==============================================================================
app = Flask(__name__)
CORS(app)

# --- Criação dos Endpoints da API ---

@app.route("/")
#def hello_world():
    #return "<p>O backend está funcionando. Acesse /api/dados para ver os dados.</p>"
def home():
    # O Flask vai procurar por 'index.html' na pasta 'templates'
    return render_template('index.html')

# ESTE É O NOSSO NOVO ENDPOINT DE DADOS
@app.route("/api/dados")
def servir_dados():
    print("Requisição recebida! Preparando os dados...")
    # 1. Chama a função para preparar o DataFrame completo
    df_final = preparar_dados_completos()

    # 2. Converte o DataFrame para o formato JSON (orient='records' é o ideal para JavaScript)
    dados_json = df_final.to_json(orient='records', date_format='iso')
    print("Dados preparados e enviados como JSON.")
    
    # 3. Retorna os dados como uma resposta da API
    return dados_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)