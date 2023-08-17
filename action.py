# AÇÔES DISPONÍVEIS #
# Cada função/método corresponde a uma ação (por momento apenas simulada)
import re
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

class Action:
    def date(self, ctx):
        time = datetime.now().strftime('%H:%M')
        ctx.storage.set_belief("horary", time)
        print("###> update horary <###")
        
    def get_symbol(self, ctx):
        # nome = ctx.storage.get_belief("symbol")
        stock_symbols = ["AAPL", "GOOGL", "TSLA"]
        current_symbol = ctx.storage.get_belief("symbol")

        if current_symbol in stock_symbols:
            index = stock_symbols.index(current_symbol)
            next_index = (index + 1) % len(stock_symbols)
            next_symbol = stock_symbols[next_index]
            
            ctx.storage.set_belief("symbol", next_symbol)
        else:
            ctx.storage.set_belief("symbol", stock_symbols[0])         
            
    def get_min(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        print(symbol)
        initial = '2020-01-01'
        end = '2023-08-10' # time = datetime.datetime.now().strftime('%Y/%m/%d')

        # Baixar os dados da ação usando a biblioteca yfinance
        dados_acao = yf.download(symbol, start=initial, end=end)

        # Extrair os valores de baixa (Low) da ação e remodelar para entrada no escalador
        cotacao = dados_acao['Low'].to_numpy().reshape(-1, 1)

        # Definir o tamanho dos dados de treinamento
        tamanho_dados_treinamento = int(len(cotacao) * 0.8)

        # Escalar os dados entre 0 e 1 usando MinMaxScaler
        escalador = MinMaxScaler(feature_range=(0, 1))
        dados_entre_0_e_1_treinamento = escalador.fit_transform(cotacao[:tamanho_dados_treinamento, :])
        dados_entre_0_e_1_teste = escalador.transform(cotacao[tamanho_dados_treinamento:, :])

        # Concatenar os dados de treinamento e teste escalados
        dados_entre_0_e_1 = np.concatenate((dados_entre_0_e_1_treinamento, dados_entre_0_e_1_teste))

        # Criar os dados de treinamento (input) e alvo (output)
        dados_para_treinamento = dados_entre_0_e_1[:tamanho_dados_treinamento, :]

        treinamento_x = []
        treinamento_y = []

        for i in range(60, len(dados_para_treinamento)):
            treinamento_x.append(dados_para_treinamento[i - 60: i, 0])
            treinamento_y.append(dados_para_treinamento[i, 0])

        # Converter as listas em arrays e dar reshape para 3D
        treinamento_x, treinamento_y = np.array(treinamento_x), np.array(treinamento_y)
        treinamento_x = treinamento_x.reshape(treinamento_x.shape[0], treinamento_x.shape[1], 1)

        # Construir o modelo LSTM
        modelo = Sequential()
        modelo.add(LSTM(50, return_sequences=True, input_shape=(treinamento_x.shape[1], 1)))
        modelo.add(LSTM(50, return_sequences=False))
        modelo.add(Dense(25))
        modelo.add(Dense(1))

        # Compilar o modelo
        modelo.compile(optimizer="adam", loss="mean_squared_error")

        # Treinar o modelo
        modelo.fit(treinamento_x, treinamento_y, batch_size=1, epochs=1)

        # Preparar os dados de teste
        dados_teste = dados_entre_0_e_1[tamanho_dados_treinamento - 60:, :]
        teste_x = []

        for i in range(60, len(dados_teste)):
            teste_x.append(dados_teste[i - 60: i, 0])

        teste_x = np.array(teste_x)
        teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

        # Fazer previsões usando o modelo
        predicoes = modelo.predict(teste_x)
        predicoes = escalador.inverse_transform(predicoes)

        # Calcular o erro médio quadrático (RMSE)
        rmse = np.sqrt(np.mean((predicoes - cotacao[tamanho_dados_treinamento:]) ** 2))

        # Criar DataFrame para análise de resultados
        df_teste = pd.DataFrame({"Low": dados_acao['Low'].iloc[tamanho_dados_treinamento:],
                                "predicoes": predicoes.reshape(len(predicoes))})

        # Análise de acertos e expectativa de lucro
        df_teste['variacao_percentual_acao'] = df_teste['Low'].pct_change()
        df_teste['variacao_percentual_modelo'] = df_teste['predicoes'].pct_change()

        df_teste.dropna(inplace=True)

        df_teste['var_acao_maior_menor_que_zero'] = df_teste['variacao_percentual_acao'] > 0
        df_teste['var_modelo_maior_menor_que_zero'] = df_teste['variacao_percentual_modelo'] > 0

        df_teste['acertou_o_lado'] = df_teste['var_acao_maior_menor_que_zero'] == df_teste['var_modelo_maior_menor_que_zero']

        # Criar coluna de variação percentual absoluta da ação
        df_teste['variacao_percentual_acao_abs'] = df_teste['variacao_percentual_acao'].abs()

        acertou_lado = df_teste['acertou_o_lado'].sum() / len(df_teste['acertou_o_lado'])
        errou_lado = 1 - acertou_lado

        media_lucro = df_teste.groupby('acertou_o_lado')['variacao_percentual_acao_abs'].mean()

        exp_mat_lucro = acertou_lado * media_lucro[1] - media_lucro[0] * errou_lado

        ganho_sobre_perda = media_lucro[1] / media_lucro[0]

        # Imprimir resultados
        print("Média de lucro:", media_lucro)
        print("Ganho sobre perda:", ganho_sobre_perda)
        print("Taxa de acerto do lado:", acertou_lado)
        print("Expectativa de lucro:", exp_mat_lucro * 100)
        #criando um código que você passa 60 dias e ele devolve a cotação
        #resumindo: vamos descobrir o preço da petrobras de hoje/amanha com esse modelo

        data_hoje = datetime.now().strftime("%d/%m/%Y")

        #se quiser escolher um dia, basta fazer assim

        data_hoje = datetime.now() - timedelta(days = 1)

        if data_hoje.hour > 18:

            final = data_hoje
            inicial = datetime.now() - timedelta(days = 252)

        else:
            final = data_hoje - timedelta(days = 1)
            inicial = datetime.now() - timedelta(days = 252)

        #nao vai botar outra ação aqui hein kkkkkkkk
        cotacoes = yf.download(symbol, initial, end)
        ultimos_60_dias = cotacoes['Low'].iloc[-60:].values.reshape(-1, 1)

        ultimos_60_dias_escalado = escalador.transform(ultimos_60_dias)

        teste_x = []
        teste_x.append(ultimos_60_dias_escalado)
        teste_x = np.array(teste_x)
        teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

        previsao_de_preco = modelo.predict(teste_x)
        previsao_de_preco = escalador.inverse_transform(previsao_de_preco)

        ############### ENTRA NO BANCO E COLOCA A CARTEIRA (SERIALIZER)#####################################################################
        print(previsao_de_preco)
        ctx.storage.set_belief(f"price_min_{symbol}", float(previsao_de_preco))
        ctx.storage.set_belief(f"price_min_check_{symbol}", True)
        self.up_min(ctx)
        
    def check_min(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        if ctx.storage.get_belief(f"price_min_check_{symbol}") == True:
            ctx.storage.set_belief(f"price_min_check", False)
        else:
            ctx.storage.set_belief(f"price_min_check", True)         
        
    def get_max(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        print(symbol)
        initial = '2020-01-01'
        end = '2023-08-10' # time = datetime.datetime.now().strftime('%Y/%m/%d')

        # Baixar os dados da ação usando a biblioteca yfinance
        dados_acao = yf.download(symbol, start=initial, end=end)

        # Extrair os valores de baixa (High) da ação e remodelar para entrada no escalador
        cotacao = dados_acao['High'].to_numpy().reshape(-1, 1)

        # Definir o tamanho dos dados de treinamento
        tamanho_dados_treinamento = int(len(cotacao) * 0.8)

        # Escalar os dados entre 0 e 1 usando MinMaxScaler
        escalador = MinMaxScaler(feature_range=(0, 1))
        dados_entre_0_e_1_treinamento = escalador.fit_transform(cotacao[:tamanho_dados_treinamento, :])
        dados_entre_0_e_1_teste = escalador.transform(cotacao[tamanho_dados_treinamento:, :])

        # Concatenar os dados de treinamento e teste escalados
        dados_entre_0_e_1 = np.concatenate((dados_entre_0_e_1_treinamento, dados_entre_0_e_1_teste))

        # Criar os dados de treinamento (input) e alvo (output)
        dados_para_treinamento = dados_entre_0_e_1[:tamanho_dados_treinamento, :]

        treinamento_x = []
        treinamento_y = []

        for i in range(60, len(dados_para_treinamento)):
            treinamento_x.append(dados_para_treinamento[i - 60: i, 0])
            treinamento_y.append(dados_para_treinamento[i, 0])

        # Converter as listas em arrays e dar reshape para 3D
        treinamento_x, treinamento_y = np.array(treinamento_x), np.array(treinamento_y)
        treinamento_x = treinamento_x.reshape(treinamento_x.shape[0], treinamento_x.shape[1], 1)

        # Construir o modelo LSTM
        modelo = Sequential()
        modelo.add(LSTM(50, return_sequences=True, input_shape=(treinamento_x.shape[1], 1)))
        modelo.add(LSTM(50, return_sequences=False))
        modelo.add(Dense(25))
        modelo.add(Dense(1))

        # Compilar o modelo
        modelo.compile(optimizer="adam", loss="mean_squared_error")

        # Treinar o modelo
        modelo.fit(treinamento_x, treinamento_y, batch_size=1, epochs=1)

        # Preparar os dados de teste
        dados_teste = dados_entre_0_e_1[tamanho_dados_treinamento - 60:, :]
        teste_x = []

        for i in range(60, len(dados_teste)):
            teste_x.append(dados_teste[i - 60: i, 0])

        teste_x = np.array(teste_x)
        teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

        # Fazer previsões usando o modelo
        predicoes = modelo.predict(teste_x)
        predicoes = escalador.inverse_transform(predicoes)

        # Calcular o erro médio quadrático (RMSE)
        rmse = np.sqrt(np.mean((predicoes - cotacao[tamanho_dados_treinamento:]) ** 2))

        # Criar DataFrame para análise de resultados
        df_teste = pd.DataFrame({"High": dados_acao['High'].iloc[tamanho_dados_treinamento:],
                                "predicoes": predicoes.reshape(len(predicoes))})

        # Análise de acertos e expectativa de lucro
        df_teste['variacao_percentual_acao'] = df_teste['High'].pct_change()
        df_teste['variacao_percentual_modelo'] = df_teste['predicoes'].pct_change()

        df_teste.dropna(inplace=True)

        df_teste['var_acao_maior_menor_que_zero'] = df_teste['variacao_percentual_acao'] > 0
        df_teste['var_modelo_maior_menor_que_zero'] = df_teste['variacao_percentual_modelo'] > 0

        df_teste['acertou_o_lado'] = df_teste['var_acao_maior_menor_que_zero'] == df_teste['var_modelo_maior_menor_que_zero']

        # Criar coluna de variação percentual absoluta da ação
        df_teste['variacao_percentual_acao_abs'] = df_teste['variacao_percentual_acao'].abs()

        acertou_lado = df_teste['acertou_o_lado'].sum() / len(df_teste['acertou_o_lado'])
        errou_lado = 1 - acertou_lado

        media_lucro = df_teste.groupby('acertou_o_lado')['variacao_percentual_acao_abs'].mean()

        exp_mat_lucro = acertou_lado * media_lucro[1] - media_lucro[0] * errou_lado

        ganho_sobre_perda = media_lucro[1] / media_lucro[0]

        # Imprimir resultados
        print("Média de lucro:", media_lucro)
        print("Ganho sobre perda:", ganho_sobre_perda)
        print("Taxa de acerto do lado:", acertou_lado)
        print("Expectativa de lucro:", exp_mat_lucro * 100)
        #criando um código que você passa 60 dias e ele devolve a cotação
        #resumindo: vamos descobrir o preço da petrobras de hoje/amanha com esse modelo

        data_hoje = datetime.now().strftime("%d/%m/%Y")

        #se quiser escolher um dia, basta fazer assim

        data_hoje = datetime.now() - timedelta(days = 1)

        if data_hoje.hour > 18:

            final = data_hoje
            inicial = datetime.now() - timedelta(days = 252)

        else:
            final = data_hoje - timedelta(days = 1)
            inicial = datetime.now() - timedelta(days = 252)

        #nao vai botar outra ação aqui hein kkkkkkkk
        cotacoes = yf.download(symbol, initial, end)
        ultimos_60_dias = cotacoes['High'].iloc[-60:].values.reshape(-1, 1)

        ultimos_60_dias_escalado = escalador.transform(ultimos_60_dias)

        teste_x = []
        teste_x.append(ultimos_60_dias_escalado)
        teste_x = np.array(teste_x)
        teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

        previsao_de_preco = modelo.predict(teste_x)
        previsao_de_preco = escalador.inverse_transform(previsao_de_preco)

        ############### ENTRA NO BANCO E COLOCA A CARTEIRA (SERIALIZER)#####################################################################
        print(previsao_de_preco)
        ctx.storage.set_belief(f"price_max_{symbol}", float(previsao_de_preco))
        ctx.storage.set_belief(f"price_max_check_{symbol}", True)
        self.up_max(ctx)
            
    def check_max(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        if ctx.storage.get_belief(f"price_max_check_{symbol}") == True:
            ctx.storage.set_belief(f"price_max_check", False)
        else:
            ctx.storage.set_belief(f"price_max_check", True)
        
    def trade(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        check_wallet = ctx.storage.get_belief("symbol_buy")
        price_now = ctx.storage.get_belief("price_now")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")
        price_max = ctx.storage.get_belief(f"price_max_{symbol}")
        opening = ctx.storage.get_belief("opening")
        if check_wallet:
            if price_now > price_max:
                ctx.storage.set_belief("sell", True)
            elif opening:
                 ctx.storage.set_belief("sell", False)
            else:
                self.swing(ctx)
        else:
            if price_now < price_min:
                ctx.storage.set_belief("buy", True)
            else:
                ctx.storage.set_belief("buy", False)
                
    def swing(self, ctx):
        ctx.storage.set_belief("msg_swing", True)
                
    def check_price(self, ctx):     
        symbol = ctx.storage.get_belief("symbol")           

        # Inicie o driver do Chrome
        driver = webdriver.Chrome()

        # Defina o tempo de espera implícita
        driver.implicitly_wait(30)

        # Abra o site do Yahoo
        driver.get("https://www.google.com/finance/")

        # Encontrar os campos de email e senha e preenchê-los
        email_field = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz[2]/div/div[3]/div[3]/div/div/div/div[1]/input[2]')
        email_field.send_keys(symbol)
        email_field.send_keys(Keys.RETURN)

        # Encontrar o elemento usando o XPath
        xpath = '//*[@id="yDmH0d"]/c-wiz[3]/div/div[4]/div/main/div[2]/div[1]/div[1]/c-wiz/div/div[1]/div/div[1]/div/div[1]/div/span/div/div'
        element = driver.find_element(By.XPATH, xpath)
        time.sleep(3)

        # Extrair o valor do elemento
        valor = element.text
        print("Valor da div:", valor)
        limpo = ''.join(filter(lambda x: x.isdigit() or x == ',', valor))

        if limpo:
            numero_float = float(limpo.replace(',', '.'))
            ctx.storage.set_belief("price_now", numero_float)
            print(numero_float)
        else:
            ctx.storage.set_belief("price_now", 0)
            print("A string não contém números ou vírgulas!")

        driver.quit()
        
    def sell(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        quant = 1
        
        
        
        ctx.storage.set_belief(f"sell_{symbol}", quant)
        print("feito a ação de vender")
        self.up_sell(ctx)
    
    def buy(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        quant = 1
        
        
        
        ctx.storage.set_belief(f"buy_{symbol}", quant)
        print("feito a ação de comprar")
        self.up_buy(ctx)


######################################################## AQUI COLOCA OS ACESSOS AO BANCO #####################################################################
    def up_min(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://localhost:8000/min/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_symbol': symbol,
            '_price_min': price_min,
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 200:
            print("Pedido POST bem-sucedido!")
            print("Resposta do servidor:")
            print(response.text)
        else:
            print("Erro ao fazer o pedido POST. Código de status:", response.status_code)

    
    def up_max(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        price_max = ctx.storage.get_belief(f"price_max_{symbol}")

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://localhost:8000/max/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_symbol': symbol,
            '_price_max': price_max
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 200:
            print("Pedido POST bem-sucedido!")
            print("Resposta do servidor:")
            print(response.text)
        else:
            print("Erro ao fazer o pedido POST. Código de status:", response.status_code)

    
    def up_buy(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")
        price_now = ctx.storage.get_belief("price_now")
        quant = 1

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://localhost:8000/wallet/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_symbol': symbol,
            '_price_min': price_min,
            '_price_now': price_now,
            '_quant': quant
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 200:
            print("Pedido POST bem-sucedido!")
            print("Resposta do servidor:")
            print(response.text)
        else:
            print("Erro ao fazer o pedido POST. Código de status:", response.status_code)

    
    def up_sell(self,ctx):
        symbol = ctx.storage.get_belief("symbol")

        # URL do endpoint onde você deseja fazer o DELETE
        url = f'http://localhost:8000/wallet/?_symbol={symbol}'

        # Realiza o pedido DELETE
        response = requests.delete(url)

        # Verifica a resposta do servidor
        if response.status_code == 204:
            print("Pedido DELETE bem-sucedido!")
        else:
            print("Erro ao fazer o pedido DELETE. Código de status:", response.status_code)
  
    
    def check_wallet(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        
        # URL do endpoint onde você deseja fazer o GET para verificar se o símbolo está no banco
        url = f'http://localhost:8000/wallet/?_symbol={symbol}'

        # Realiza a requisição GET
        response = requests.get(url)

        # Verifica a resposta do servidor
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                print("O símbolo está contido no banco.")
                ctx.storage.set_belief("symbol_buy", False)
            else:
                print("O símbolo não está no banco.")
                ctx.storage.set_belief("symbol_buy", True)
        else:
            print("Erro ao fazer a requisição GET. Código de status:", response.status_code)