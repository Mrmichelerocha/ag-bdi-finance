import MetaTrader5 as mt5

symbol = "PETR3"
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

# Verifique o valor de retcode e tome ação com base nele
if result_compra.retcode == 0:
    print("Ordem de compra enviada com sucesso!")
else:
    print(f"Falha ao enviar a ordem de compra. Código de retorno: {result_compra.retcode}")


print(result_compra)

if not mt5.initialize():
    print("Falha ao inicializar o MetaTrader 5.")
    mt5.shutdown()
else:
    print("Conexão ao MetaTrader 5 estabelecida.")
    
account_info = mt5.account_info()
# Verifique se a conexão à conta foi bem-sucedida
if account_info is not None:
    # Acesse o saldo disponível
    balance = account_info.balance
    print(f"Saldo da conta: {balance}")
else:
    print("Não foi possível obter informações da conta.")