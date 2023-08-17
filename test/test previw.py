import requests

# https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=B3SA3&apikey=YNL866OTLJOOW72R

symbol = "B3SA3.SAO"  # símbolo da ação
api_key = "YNL866OTLJOOW72R"  # substituir pelo seu API key da Alpha Vantage

url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
response = requests.get(url)

data = response.json()
current_price = float(data["Global Quote"]["05. price"])
current_previous = float(data["Global Quote"]["08. previous close"])

print(f"O preço abertura da ação da {symbol} é: ${current_price:.2f}")
print(f"O preço previsto da ação da {symbol} é: ${current_previous:.2f}")

