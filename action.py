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
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
import pandas as pd
import yfinance as yf
from finta import TA
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import MetaTrader5 as mt5
import urllib.request, json

class Action:

    def date(self, ctx):
        # Verifica se o dia atual é um dia útil (0 = segunda-feira, 6 = domingo)
        if datetime.now().weekday() < 5:  # 0-4 são dias úteis (segunda a sexta)
            time = datetime.now().strftime('%H:%M')
            date = datetime.now().strftime("%Y/%m/%d")
            ctx.storage.set_belief("horary", time)
            ctx.storage.set_belief("date", date)
            print("###> update horary <###")
        else:
            print("###> Não é um dia útil. Horary não atualizado. <###")
            
    def get_symbol(self, ctx):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        # Set up the Chrome driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # Navigate to the URL
        url = 'https://www.fundamentus.com.br/resultado.php'
        driver.get(url)

        # Find the table element
        local_tabela = '/html/body/div[1]/div[2]/table'
        elemento = driver.find_element("xpath", local_tabela)
        html_tabela = elemento.get_attribute('outerHTML')
        tabela = pd.read_html(str(html_tabela), thousands='.', decimal=',')[0]

        # Clean up and preprocess the data
        tabela = tabela.set_index("Papel")
        tabela = tabela[['Cotação', 'EV/EBIT', 'ROIC', 'Liq.2meses']]

        tabela['ROIC'] = tabela['ROIC'].str.replace("%", "")
        tabela['ROIC'] = tabela['ROIC'].str.replace(".", "")
        tabela['ROIC'] = tabela['ROIC'].str.replace(",", ".")
        tabela['ROIC'] = tabela['ROIC'].astype(float)

        tabela = tabela[tabela['Liq.2meses'] > 1000000]
        tabela = tabela[tabela['EV/EBIT'] > 0]
        tabela = tabela[tabela['ROIC'] > 0]

        tabela['ranking_ev_ebit'] = tabela['EV/EBIT'].rank(ascending = True)
        tabela['ranking_roic'] = tabela['ROIC'].rank(ascending = False)
        tabela['ranking_total'] = tabela['ranking_ev_ebit'] + tabela['ranking_roic']

        tabela = tabela.sort_values('ranking_total')

        # Exclui a primeira linha onde o valor é "Papel"
        tabela = tabela.iloc[1:]

        # Extrai os valores da coluna "Papel"
        parametros = tabela.head(6).index.tolist()

        print(parametros)
        
        current_symbol = ctx.storage.get_belief("symbol")

        if current_symbol in parametros:
            index = parametros.index(current_symbol)
            next_index = (index + 1) % len(parametros)
            next_symbol = parametros[next_index]
            
            ctx.storage.set_belief("symbol", next_symbol)
        else:
            ctx.storage.set_belief("symbol", parametros[0])  
            
    def get_min(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        print("Este é a Ação que estamos analisando MINIMO: ", symbol)

        api_key = 'F3CLPQNGRS2NIQ9M'
        url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.SAO&outputsize=full&apikey={api_key}"
        with urllib.request.urlopen(url_string) as url:
                    data = json.loads(url.read().decode())
                    # extract stock market data
                    data = data['Time Series (Daily)']
                    dados_acao = pd.DataFrame(columns=['Date','Low','High','Close','Open','Volume'])
                    for k,v in data.items():
                        date = datetime.strptime(k, '%Y-%m-%d')
                        data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                    float(v['4. close']),float(v['1. open']),int(v['5. volume'])]
                        dados_acao.loc[-1,:] = data_row
                        dados_acao.index = dados_acao.index + 1
        dados_acao.set_index('Date',inplace=True)
        dados_acao.sort_index(inplace=True)
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
        # print("Média de lucro:", media_lucro)
        # print("Ganho sobre perda:", ganho_sobre_perda)
        # print("Taxa de acerto do lado:", acertou_lado)
        # print("Expectativa de lucro:", exp_mat_lucro * 100)
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
        ultimos_60_dias = dados_acao['Low'].iloc[-60:].values.reshape(-1, 1)

        ultimos_60_dias_escalado = escalador.transform(ultimos_60_dias)

        teste_x = []
        teste_x.append(ultimos_60_dias_escalado)
        teste_x = np.array(teste_x)
        teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

        previsao_de_preco = modelo.predict(teste_x)
        previsao_de_preco = escalador.inverse_transform(previsao_de_preco)

        ############### ENTRA NO BANCO E COLOCA A CARTEIRA (SERIALIZER)#####################################################################
        print("Esta é a previsão so preço MINIMO", previsao_de_preco)
        ctx.storage.set_belief(f"price_min_{symbol}", float(previsao_de_preco))
        ctx.storage.set_belief(f"price_min_check_{symbol}", True)
        self.up_min(ctx)
        
    def check_min(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        if ctx.storage.get_belief(f"price_min_check_{symbol}") == True:
            print("Checkar Preço MINIMO?: Falso")
            ctx.storage.set_belief(f"price_min_check", False)
        else:
            print("Checkar Preço MINIMO?: True")
            ctx.storage.set_belief(f"price_min_check", True)         
        
    def get_max(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        print("Este é a Ação que estamos analisando MAXIMO: ", symbol)

        api_key = 'F3CLPQNGRS2NIQ9M'
        url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.SAO&outputsize=full&apikey={api_key}"
        with urllib.request.urlopen(url_string) as url:
                    data = json.loads(url.read().decode())
                    # extract stock market data
                    data = data['Time Series (Daily)']
                    dados_acao = pd.DataFrame(columns=['Date','Low','High','Close','Open','Volume'])
                    for k,v in data.items():
                        date = datetime.strptime(k, '%Y-%m-%d')
                        data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                    float(v['4. close']),float(v['1. open']),int(v['5. volume'])]
                        dados_acao.loc[-1,:] = data_row
                        dados_acao.index = dados_acao.index + 1
        dados_acao.set_index('Date',inplace=True)
        dados_acao.sort_index(inplace=True)
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
        # print("Média de lucro:", media_lucro)
        # print("Ganho sobre perda:", ganho_sobre_perda)
        # print("Taxa de acerto do lado:", acertou_lado)
        # print("Expectativa de lucro:", exp_mat_lucro * 100)
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
        ultimos_60_dias = dados_acao['High'].iloc[-60:].values.reshape(-1, 1)

        ultimos_60_dias_escalado = escalador.transform(ultimos_60_dias)

        teste_x = []
        teste_x.append(ultimos_60_dias_escalado)
        teste_x = np.array(teste_x)
        teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

        previsao_de_preco = modelo.predict(teste_x)
        previsao_de_preco = escalador.inverse_transform(previsao_de_preco)

        ############### ENTRA NO BANCO E COLOCA A CARTEIRA (SERIALIZER)#####################################################################
        print("Esta é a previsão do preço MAXIMO: ", previsao_de_preco)
        ctx.storage.set_belief(f"price_max_{symbol}", float(previsao_de_preco))
        ctx.storage.set_belief(f"price_max_check_{symbol}", True)
        self.up_max(ctx)
            
    def check_max(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        if ctx.storage.get_belief(f"price_max_check_{symbol}") == True:
            print("Checkar Preço MAXIMO?: Falso")
            ctx.storage.set_belief(f"price_max_check", False)
        else:
            print("Checkar Preço MAXIMO?: True")
            ctx.storage.set_belief(f"price_max_check", True)
        
    def trade(self, ctx):
        get_metric = ctx.storage.get_belief("metric")
        get_direction = ctx.storage.get_belief("direction")
        get_wallet = ctx.storage.get_belief("symbol_buy")
        
        
        # Definindo as variáveis de entrada e saída fuzzy
        direction = ctrl.Antecedent(np.arange(-1, 2, 1), 'direction')  # -1: Descer, 0: Manter, 1: Subir
        metric = ctrl.Antecedent(np.arange(-1, 2, 1), 'metric')  # -1: Vender, 0: Manter, 1: Comprar
        act = ctrl.Consequent(np.arange(-1, 2, 1), 'act')  # -1: Vender, 0: Manter, 1: Comprar

        # Definindo os conjuntos fuzzy para direction
        direction['down'] = fuzz.trimf(direction.universe, [-1, -1, 0])
        direction['hold'] = fuzz.trimf(direction.universe, [-1, 0, 1])
        direction['up'] = fuzz.trimf(direction.universe, [0, 1, 1])

        # Definindo os conjuntos fuzzy para metric e act
        metric['sell'] = fuzz.trimf(metric.universe, [-1, -1, 0])
        metric['hold'] = fuzz.trimf(metric.universe, [-1, 0, 1])
        metric['buy'] = fuzz.trimf(metric.universe, [0, 1, 1])

        act['sell'] = fuzz.trimf(act.universe, [-1, -1, 0])
        act['hold'] = fuzz.trimf(act.universe, [-1, 0, 1])
        act['buy'] = fuzz.trimf(act.universe, [0, 1, 1])

        # Regras fuzzy
        regra1 = ctrl.Rule(direction['down'] & metric['sell'], act['sell'])
        regra2 = ctrl.Rule(direction['down'] & metric['buy'], act['buy'])
        regra3 = ctrl.Rule(direction['up'] & metric['sell'], act['sell'])
        regra4 = ctrl.Rule(direction['hold'] & metric['hold'], act['hold'])
        regra_fallback = ctrl.Rule(~regra1.antecedent & ~regra2.antecedent & ~regra3.antecedent & ~regra4.antecedent, act['hold'])

        # Sistema de controle fuzzy
        sistema_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra_fallback])
        sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

        # Entrada: direction da act e act anterior
        sistema.input['direction'] = get_direction  # 1: Subir
        sistema.input['metric'] = get_metric  # -1: Vender

        # Calcula a saída do sistema fuzzy
        sistema.compute()

        # Saída: act recomendada
        act_fuzzy = sistema.output['act']
        print("calculo Fuzzy: ", act_fuzzy)
        act_recommended = "sell" if act_fuzzy <= -0.5 else ("hold" if -0.5 < act_fuzzy <= 0.5 else "buy")
        print("Ação Recomendada: ", act_recommended)
        
        if act_recommended == "sell":
            if not get_wallet:
                ctx.storage.set_belief(act_recommended, True)
        elif act_recommended == "hold":
            if get_wallet:
                # ctx.storage.set_belief("buy", True)
                pass
            else:
                ctx.storage.set_belief(act_recommended, True)
        elif act_recommended == 'buy':
            ctx.storage.set_belief(act_recommended, True)
                                          
    def check_price(self, ctx):     
        symbol = ctx.storage.get_belief("symbol")           

        if not mt5.initialize():
            print("Falha ao inicializar o MetaTrader 5.")
            mt5.shutdown()
        else:
            print("Conexão ao MetaTrader 5 estabelecida.")
            
        mt5.symbol_select(symbol)
        price=mt5.symbol_info_tick(symbol).ask
        print("Valor do Preço da Ação AGORA: ", price)
        ctx.storage.set_belief("price_now", price)
  
    def check_upordown(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")
        price_max = ctx.storage.get_belief(f"price_max_{symbol}")    
        
        if price_min > price_max:
            ctx.storage.set_belief("direction", 1) # up
        elif price_max > price_min:
            ctx.storage.set_belief("direction", -1) # down
        else:
            ctx.storage.set_belief("direction", 0) # hold
            
    def check_buyorsell(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        try:
            data = yf.download(symbol + ".SA", period="1y")  # Baixa dados do último ano
            
            # Se os dados estiverem vazios, pule para o próximo symbol
            if data.empty:
                print(f"Sem dados para o symbol {symbol}. Pulando...")
            
            # Calcula médias móveis
            sma_50 = TA.SMA(data, 50)
            sma_200 = TA.SMA(data, 200)
            
            # Bandas de Bollinger
            bollinger = TA.BBANDS(data)
            
            # RSI
            rsi = TA.RSI(data)

            # Sistema de pontos
            points = 0

            # Golden Cross
            if sma_50.iloc[-2] < sma_200.iloc[-2] and sma_50.iloc[-1] > sma_200.iloc[-1]:
                points += 1
            # Death Cross
            elif sma_50.iloc[-2] > sma_200.iloc[-2] and sma_50.iloc[-1] < sma_200.iloc[-1]:
                points -= 1
            # RSI Sobrevendido
            if rsi.iloc[-1] < 30:
                points += 1
            # RSI Sobrecomprado
            elif rsi.iloc[-1] > 70:
                points -= 1
            # Acima da Banda Superior
            if data['Close'].iloc[-1] > bollinger['BB_UPPER'].iloc[-1]:
                points -= 1
            # Abaixo da Banda Inferior
            elif data['Close'].iloc[-1] < bollinger['BB_LOWER'].iloc[-1]:
                points += 1

            # Decisão com base nos pontos
            if points > 0:
                ctx.storage.set_belief("metric", 1) # "BUY"
            elif points < 0:
                ctx.storage.set_belief("metric", -1) # "SELL"
            else:
                ctx.storage.set_belief("metric", 0) # "HOLD"
        
        except Exception as e:
            print(f"Erro ao processar o symbol {symbol}: {e}")
        
    def sell(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        
        if not mt5.initialize():
            print("Falha ao inicializar o MetaTrader 5.")
            mt5.shutdown()
        else:
            print("Conexão ao MetaTrader 5 estabelecida.")
            
        mt5.symbol_select(symbol)
        price=mt5.symbol_info_tick(symbol).bid
        point = mt5.symbol_info(symbol).point
        deviation=20
        lot = 100.0
        positions=mt5.positions_get(symbol=symbol)
        for position in positions:
            price=mt5.symbol_info_tick(symbol).ask
            deviation=20
            request={
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "position": position.ticket,
                "price": price,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result_compra = mt5.order_send(request)
            
        print(result_compra)
            
        # Verifique o valor de retcode e tome ação com base nele
        if result_compra.retcode == 1009:
            print("Ordem de compra enviada com sucesso!")
            ctx.storage.set_belief(f"buy_{symbol}", lot)
            print(f"feito a ação de COMPRAR, o resultado foi {result_compra}")
            ctx.storage.set_belief('sell', False)
            self.up_sell(ctx, lot)
        else:
            print(f"Falha ao enviar a ordem de compra. Código de retorno: {result_compra.retcode}")

    def buy(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        
        if not mt5.initialize():
            print("Falha ao inicializar o MetaTrader 5.")
            mt5.shutdown()
        else:
            print("Conexão ao MetaTrader 5 estabelecida.")
        
        mt5.symbol_select(symbol)
        price=mt5.symbol_info_tick(symbol).ask
        point = mt5.symbol_info(symbol).point
        deviation=20
        lot = 100.0
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": deviation,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result_compra = mt5.order_send(request)
        
        print(result_compra)
        
        # Verifique o valor de retcode e tome ação com base nele
        if result_compra.retcode == 1009:
            print("Ordem de compra enviada com sucesso!")
            ctx.storage.set_belief(f"buy_{symbol}", lot)
            print(f"feito a ação de COMPRAR, o resultado foi {result_compra}")
            ctx.storage.set_belief('buy', False)
            self.up_buy(ctx, lot)
        else:
            print(f"Falha ao enviar a ordem de compra. Código de retorno: {result_compra.retcode}")

######################################################## AQUI COLOCA OS ACESSOS AO BANCO #####################################################################
    def up_min(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://127.0.0.1:8000/min/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_symbol': str(symbol),
            '_price_min': float(price_min),
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 201:
            print("Pedido POST bem-sucedido! MIN")
        else:
            print("Erro ao fazer o pedido POST MIN. Código de status:", response.status_code)
   
    def up_max(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        price_max = ctx.storage.get_belief(f"price_max_{symbol}")

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://127.0.0.1:8000/max/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_symbol': str(symbol),
            '_price_max': float(price_max),
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 201:
            print("Pedido POST bem-sucedido! MAX")
        else:
            print("Erro ao fazer o pedido POST MAX. Código de status:", response.status_code)
    
    def up_buy(self, ctx, lot):
        symbol = ctx.storage.get_belief("symbol")
        price_max = ctx.storage.get_belief(f"price_max_{symbol}")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")
        price_now = ctx.storage.get_belief("price_now")
        quant = lot

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://127.0.0.1:8000/transaction/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_transaction': str(symbol),
            '_operation': "BUY",
            '_quant': int(quant),
            '_price_now': float(price_now),
            '_price_min': float(price_min),
            '_price_max': float(price_max),
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 201:
            print("Pedido POST bem-sucedido! BUY")
        else:
            print("Erro ao fazer o pedido POST BUY. Código de status:", response.status_code)
    
    def up_sell(self,ctx, lot):
        symbol = ctx.storage.get_belief("symbol")
        price_max = ctx.storage.get_belief(f"price_max_{symbol}")
        price_min = ctx.storage.get_belief(f"price_min_{symbol}")
        price_now = ctx.storage.get_belief("price_now")
        quant = lot

        # URL do endpoint onde você deseja fazer o POST
        url = 'http://127.0.0.1:8000/transaction/'

        # Dados que você deseja enviar no corpo do POST (substitua pelos seus dados)
        data = {
            '_transaction': str(symbol),
            '_operation': "SELL",
            '_quant': int(quant),
            '_price_now': float(price_now),
            '_price_min': float(price_min),
            '_price_max': float(price_max),
        }

        # Realiza o pedido POST
        response = requests.post(url, data=data)

        # Verifica a resposta do servidor
        if response.status_code == 201:
            print("Pedido POST bem-sucedido! SELL")
        else:
            print("Erro ao fazer o pedido POST SELL. Código de status:", response.status_code)
    
    def check_wallet(self, ctx):
        symbol = ctx.storage.get_belief("symbol")
        
        # URL do endpoint onde você deseja fazer o GET para verificar se o símbolo está no banco
        url = f'http://127.0.0.1:8000/wallet/?_symbol={symbol}'

        # Realiza a requisição GET
        response = requests.get(url)

        # Verifica a resposta do servidor
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                wallet_entry = data[0]
                if wallet_entry["_quant"] == 0:
                    print(f"O símbolo {symbol} está contido na carteira WALLET com quantidade 0.")
                    ctx.storage.set_belief("symbol_buy", True)
                else:
                    print(f"O símbolo {symbol} está contido na carteira WALLET")
                    ctx.storage.set_belief("symbol_buy", False)
            else:
                print("O símbolo NÂO está na carteira WALLET")
                ctx.storage.set_belief("symbol_buy", True)
        else:
            print("Erro ao fazer a requisição GET WALLET. Código de status:", response.status_code)

    def get_list_symbol(self, parametros):
        response = requests.get("http://127.0.0.1:8000/wallet")
        if response.status_code == 200:
            data = response.json()
            simbolos_diferentes = [item for item in data if item["_symbol"] not in parametros and item["_quant"] != 0]
            return [item["_symbol"] for item in simbolos_diferentes]
        else:
            print("A requisição GET não foi bem-sucedida.")
            return []

    def management_wallet(self, ctx):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        # Set up the Chrome driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # Navigate to the URL
        url = 'https://www.fundamentus.com.br/resultado.php'
        driver.get(url)

        # Find the table element
        local_tabela = '/html/body/div[1]/div[2]/table'
        elemento = driver.find_element("xpath", local_tabela)
        html_tabela = elemento.get_attribute('outerHTML')
        tabela = pd.read_html(str(html_tabela), thousands='.', decimal=',')[0]

        # Clean up and preprocess the data
        tabela = tabela.set_index("Papel")
        tabela = tabela[['Cotação', 'EV/EBIT', 'ROIC', 'Liq.2meses']]

        tabela['ROIC'] = tabela['ROIC'].str.replace("%", "")
        tabela['ROIC'] = tabela['ROIC'].str.replace(".", "")
        tabela['ROIC'] = tabela['ROIC'].str.replace(",", ".")
        tabela['ROIC'] = tabela['ROIC'].astype(float)

        tabela = tabela[tabela['Liq.2meses'] > 1000000]
        tabela = tabela[tabela['EV/EBIT'] > 0]
        tabela = tabela[tabela['ROIC'] > 0]

        tabela['ranking_ev_ebit'] = tabela['EV/EBIT'].rank(ascending = True)
        tabela['ranking_roic'] = tabela['ROIC'].rank(ascending = False)
        tabela['ranking_total'] = tabela['ranking_ev_ebit'] + tabela['ranking_roic']

        tabela = tabela.sort_values('ranking_total')

        # Exclui a primeira linha onde o valor é "Papel"
        tabela = tabela.iloc[1:]

        # Extrai os valores da coluna "Papel"
        parametros = tabela.head(6).index.tolist()

        print(parametros)
            
        if ctx.storage.get_belief("different"):
            lista_de_simbolos = self.get_list_symbol(parametros)
            print("Symbol diferente na carteira: ", lista_de_simbolos)
            ctx.storage.set_belief("different", False)
            if lista_de_simbolos[0] != None:
                ctx.storage.set_belief("symbol", lista_de_simbolos[0])
        else:
            lista_de_simbolos = self.get_list_symbol(parametros)
            current_symbol = ctx.storage.get_belief("symbol")

            if current_symbol in lista_de_simbolos:
                index = lista_de_simbolos.index(current_symbol)
                next_index = (index + 1) % len(lista_de_simbolos)
                next_symbol = lista_de_simbolos[next_index]
                ctx.storage.set_belief("symbol", next_symbol)
            else:
                print("Erro: Símbolo atual não encontrado na lista de símbolos.")                  
                        
    def closing_check(self, ctx):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        # Set up the Chrome driver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # Navigate to the URL
        url = 'https://www.fundamentus.com.br/resultado.php'
        driver.get(url)

        # Find the table element
        local_tabela = '/html/body/div[1]/div[2]/table'
        elemento = driver.find_element("xpath", local_tabela)
        html_tabela = elemento.get_attribute('outerHTML')
        tabela = pd.read_html(str(html_tabela), thousands='.', decimal=',')[0]

        # Clean up and preprocess the data
        tabela = tabela.set_index("Papel")
        tabela = tabela[['Cotação', 'EV/EBIT', 'ROIC', 'Liq.2meses']]

        tabela['ROIC'] = tabela['ROIC'].str.replace("%", "")
        tabela['ROIC'] = tabela['ROIC'].str.replace(".", "")
        tabela['ROIC'] = tabela['ROIC'].str.replace(",", ".")
        tabela['ROIC'] = tabela['ROIC'].astype(float)

        tabela = tabela[tabela['Liq.2meses'] > 1000000]
        tabela = tabela[tabela['EV/EBIT'] > 0]
        tabela = tabela[tabela['ROIC'] > 0]

        tabela['ranking_ev_ebit'] = tabela['EV/EBIT'].rank(ascending = True)
        tabela['ranking_roic'] = tabela['ROIC'].rank(ascending = False)
        tabela['ranking_total'] = tabela['ranking_ev_ebit'] + tabela['ranking_roic']

        tabela = tabela.sort_values('ranking_total')

        # Exclui a primeira linha onde o valor é "Papel"
        tabela = tabela.iloc[1:]

        # Extrai os valores da coluna "Papel"
        parametros = tabela.head(6).index.tolist()

        print(parametros)
        for symbol in parametros:
            ctx.storage.set_belief(f"price_min_check_{symbol}", False)
            
        parametro = self.get_list_symbol(parametros)
        for symbol in parametro:
            ctx.storage.set_belief(f"price_min_check_{symbol}", False)