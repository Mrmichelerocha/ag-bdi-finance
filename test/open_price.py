import requests
from bs4 import BeautifulSoup

url = "https://br.financas.yahoo.com/quote/B3SA3.SA?p=B3SA3.SA"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

price_open = soup.find('td', {'data-test': 'OPEN-value'}).span.text
print("Preço de abertura da ação B3SA3.SA:", price_open)
