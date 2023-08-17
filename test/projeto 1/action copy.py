# AÇÔES DISPONÍVEIS #
# Cada função/método corresponde a uma ação (por momento apenas simulada)
from datetime import datetime
from socket import create_connection
from time import sleep
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import datetime
from finta import TA



class Action:    
    checked_buy = False
    checked_sell = False
    checked_time = False
                
    def preview_low(self, ctx):
        # # Baixar dados de preços de ações
        # data = yf.download('B3SA3.SA', start='2020-01-01', end='2023-06-20')

        # # Pré-processamento
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_data = scaler.fit_transform(data['Low'].values.reshape(-1,1))

        # # Criação de dados de treinamento e teste
        # train_size = int(len(scaled_data) * 0.8)
        # train_data = scaled_data[0:train_size,:]
        # test_data = scaled_data[train_size - 60:,:]

        # # Criar função para criar conjunto de dados de treinamento
        # def create_dataset(dataset, time_step=1):
        #     dataX, dataY = [], []
        #     for i in range(len(dataset)-time_step-1):
        #         dataX.append(dataset[i:(i+time_step), 0])
        #         dataY.append(dataset[i + time_step, 0])
        #     return np.array(dataX), np.array(dataY)

        # time_step = 60
        # X_train, y_train = create_dataset(train_data, time_step)
        # X_test, ytest = create_dataset(test_data, time_step)

        # # Redimensionar entrada para ser [amostras, passos de tempo, características] que é o requisito do LSTM
        # X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        # X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        # # Criar modelo
        # model=Sequential()
        # model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
        # model.add(GRU(50))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer=Adam(0.01))

        # # Treinar modelo
        # model.fit(X_train,y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64)

        # # Prever valores futuros
        # future_time_steps = 1
        # future_data = scaled_data[-time_step:]
        # future_data = future_data.reshape(1, time_step, 1)

        # # Realizar previsões para os próximos 6 dias
        # future_predictions = []
        # for _ in range(future_time_steps):
        #     future_prediction = model.predict(future_data)
        #     future_predictions.append(future_prediction[0][0])
        #     future_data = np.roll(future_data, -1, axis=1)
        #     future_data[0, -1, 0] = future_prediction

        # # Transformar as previsões futuras de volta à escala original
        # future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # # Calcular o erro médio quadrático (Mean Squared Error - MSE)
        # mse = np.mean((future_predictions - ytest[-future_time_steps:].reshape(-1, 1))**2)
        # print("Erro Médio Quadrático (MSE):", mse)


        # # Imprimir as previsões futuras
        # print("Previsões futuras da máxima:")
        # for i, prediction in enumerate(future_predictions):
        #     date = pd.date_range(start='2023-06-06', periods=future_time_steps, freq='B')[i].date()
        #     dado = {"Data": str(date), "Previsão": float(prediction[0])}
        #     ctx.storage.set_belief("belief_low", dado)
        #     print("Data:", date, "Previsão:", prediction)
        
        print("fui verificar o menor")
        dado = 14.20
        ctx.storage.set_belief("belief_low", dado)

        return True
                
    def price_now(self, ctx):  
        url = 'https://www.google.com/finance/quote/B3SA3:BVMF?sa=X&ved=2ahUKEwjHm-Wtucr-AhWvrJUCHTxCDDwQ3ecFegQINBAY'
        response = requests.get(url)

        soup = BeautifulSoup(response.content, 'html.parser')

        campo_preco = soup.find('div', {'class': 'YMlKec fxKbKc'})
        valor_preco = campo_preco.text.strip()

        ctx.storage.set_belief("belief_now", valor_preco)
        print("verifiquei o preço de agora", valor_preco)
    
    def buy(self, ctx):
        # time.sleep(5)
        # # ação
        # pyautogui.click(x=371, y=140) 
        # time.sleep(3)
        # # pesquisa ação
        # pyautogui.write("B3SA3")
        # time.sleep(2)
        # pyautogui.press('down')
        # time.sleep(2)
        # pyautogui.click(x=1002, y=249)
        # time.sleep(2)
        # # Seta compra Point(x=293, y=176)
        # pyautogui.click(x=293, y=176)
        # time.sleep(2)
        # # Quantidade de ações
        # pyautogui.click(x=549, y=217)
        # pyautogui.click(x=549, y=217)
        # pyautogui.write("1")
        # time.sleep(2)
        # # Setar o preço Point(x=365, y=299)
        # pyautogui.click(x=365, y=299)
        # pyautogui.click(x=365, y=299)
        # pyautogui.write("1100")
        # time.sleep(2)
        # # Confirma Botão de compra
        # pyautogui.click(x=702, y=382)
        # time.sleep(2)
        # # Confirma cotaçao Point(x=615, y=571
        # pyautogui.click(x=615, y=571)
        # time.sleep(2)
        # # Coloca senha )
        # pyautogui.click(x=615, y=389)
        # pyautogui.write("Mggr09@rock")
        # time.sleep(2)
        # # Compra
        # pyautogui.click(x=606, y=461)
        # time.sleep(2)
        
        print("fui comprar")
        self.checked_buy = True
               
    def low_now(self, ctx):
        low = ctx.storage.get_belief("belief_low")
        now = ctx.storage.get_belief("belief_now")
        now_tb = now.replace('R$', '')  # Remove o símbolo de moeda
        print("preço menor", low)
        print("preço agora", float(now_tb))
        if low >= float(now_tb):
            print("o preço da ação é menor que o preço low")
            if self.checked_buy == True:
                return False
            else:
                return True
        else:
            print("o preço da ação é maior que o preço low")
            return False
        
    def buy_checked(self, ctx):
        dado = ctx.storage.get_belief("buy_checked")
        print("compra hoje", dado)
        if self.checked_buy == True and dado == 1:
            ctx.storage.set_belief("buy_checked", dado+1)
            return True
        else:
            return False
            
    def preview_high(self, ctx):
        # # Baixar dados de preços de ações
        # data = yf.download('B3SA3.SA', start='2020-01-01', end='2023-06-20')

        # # Pré-processamento
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_data = scaler.fit_transform(data['High'].values.reshape(-1,1))

        # # Criação de dados de treinamento e teste
        # train_size = int(len(scaled_data) * 0.8)
        # train_data = scaled_data[0:train_size,:]
        # test_data = scaled_data[train_size - 60:,:]

        # # Criar função para criar conjunto de dados de treinamento
        # def create_dataset(dataset, time_step=1):
        #     dataX, dataY = [], []
        #     for i in range(len(dataset)-time_step-1):
        #         dataX.append(dataset[i:(i+time_step), 0])
        #         dataY.append(dataset[i + time_step, 0])
        #     return np.array(dataX), np.array(dataY)

        # time_step = 60
        # X_train, y_train = create_dataset(train_data, time_step)
        # X_test, ytest = create_dataset(test_data, time_step)

        # # Redimensionar entrada para ser [amostras, passos de tempo, características] que é o requisito do LSTM
        # X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        # X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        # # Criar modelo
        # model=Sequential()
        # model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
        # model.add(GRU(50))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer=Adam(0.01))

        # # Treinar modelo
        # model.fit(X_train,y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64)

        # # Prever valores futuros
        # future_time_steps = 1
        # future_data = scaled_data[-time_step:]
        # future_data = future_data.reshape(1, time_step, 1)

        # # Realizar previsões para os próximos 6 dias
        # future_predictions = []
        # for _ in range(future_time_steps):
        #     future_prediction = model.predict(future_data)
        #     future_predictions.append(future_prediction[0][0])
        #     future_data = np.roll(future_data, -1, axis=1)
        #     future_data[0, -1, 0] = future_prediction

        # # Transformar as previsões futuras de volta à escala original
        # future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # # Calcular o erro médio quadrático (Mean Squared Error - MSE)
        # mse = np.mean((future_predictions - ytest[-future_time_steps:].reshape(-1, 1))**2)
        # print("Erro Médio Quadrático (MSE):", mse)


        # # Imprimir as previsões futuras
        # print("Previsões futuras da máxima:")
        # for i, prediction in enumerate(future_predictions):
        #     date = pd.date_range(start='2023-06-06', periods=future_time_steps, freq='B')[i].date()
        #     dado = {"Data": str(date), "Previsão": float(prediction[0])}
        #     ctx.storage.set_belief("belief_high", dado)
        #     print("Data:", date, "Previsão:", prediction)

        print("fui verificar o maior")
        dado = 14.10
        ctx.storage.set_belief("belief_high", dado)
        return True
    
    def sell(self, ctx):
        # time.sleep(5)

        # # ação
        # pyautogui.click(x=371, y=140) 
        # time.sleep(3)
        # # barra de pesquisa ação
        # # pyautogui.click(x=415, y=225)
        # # time.sleep(2)
        # pyautogui.write("B3SA3")
        # time.sleep(2)
        # pyautogui.press('down')
        # time.sleep(2)
        # pyautogui.click(x=1002, y=249)
        # time.sleep(2)
        # # Confirma Botão B3SA3de venda Point(x=507, y=171)
        # pyautogui.click(x=507, y=171)
        # time.sleep(2)
        # # Quantidade de ações
        # pyautogui.click(x=549, y=217)
        # pyautogui.click(x=549, y=217)
        # pyautogui.write("1")
        # time.sleep(2)
        # # Setar o preço Point(x=365, y=299)
        # pyautogui.click(x=365, y=299)
        # pyautogui.click(x=365, y=299)
        # pyautogui.write("1100")
        # time.sleep(2)
        # # Confirma Botão de venda
        # pyautogui.click(x=702, y=382)
        # time.sleep(2)
        # # Confirma cotaçao Point(x=615, y=571
        # pyautogui.click(x=615, y=571)
        # time.sleep(2)
        # # Coloca senha )
        # pyautogui.click(x=615, y=389)
        # pyautogui.write("Mggr09@rock")
        # time.sleep(2)
        # # Compra
        # pyautogui.click(x=606, y=461)
        # time.sleep(2)
        
        print("fui vender")
        self.checked_sell = True
             
    def high_now(self, ctx):
        high = ctx.storage.get_belief("belief_high")
        now = ctx.storage.get_belief("belief_now")
        now_tb = now.replace('R$', '')  # Remove o símbolo de moeda
        print("preço maior", high)
        print("preço agora", float(now_tb))
        if high <= float(now_tb):
            print("o preço da ação é maior que o preço higt")
            ctx.storage.set_belief("belief_higt", True)
            if self.checked_sell == True:
                return False
            else:
                return True
        else:
            print("o preço da ação é menor que o preço higt")
            ctx.storage.set_belief("belief_higt", False)
            return False
        
    def sell_checked(self, ctx):
        dado = ctx.storage.get_belief("sell_checked")
        print("venda hoje", dado)
        if self.checked_buy == True and dado == 1:
            ctx.storage.set_belief("sell_checked", dado+1)
            return True
        else:
            return False
        
    def data_now(self, ctx):
        time = datetime.date.today().strftime('%H:%M:%S')
        if time > "14:30:00":
            print("hora de parar")
            self.checked_time = True
        else:
            self.checked_time = False
            print("continua ai")
            
    def rsi_strategy(ticker):
        # Baixa os dados do Yahoo Finance 
        data = yf.download(ticker, start='2020-01-01', end='2023-06-22')
        
        # Calcula o RSI 
        data['RSI'] = TA.RSI(data)
        
        # Define os limites de compra e venda (aqui utilizamos 30 e 70 como padrão) 
        buy_limit, sell_limit = 30, 70

        # Verifica se o RSI atual é menor que o limite de compra 
        if data['RSI'].iloc[-1] < buy_limit: 
            return f"Comprar: O RSI atual é {data['RSI'].iloc[-1]}, menor que {buy_limit}"  # 1
        elif data['RSI'].iloc[-1] > sell_limit: 
            return f"Vender: O RSI atual é {data['RSI'].iloc[-1]}, maior que {sell_limit}" # 2
        else: 
            return f"Manter: O RSI atual é {data['RSI'].iloc[-1]}, entre {buy_limit} e {sell_limit}" # 3