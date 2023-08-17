import requests
from bs4 import BeautifulSoup


url = 'https://www.google.com/finance/quote/B3SA3:BVMF?sa=X&ved=2ahUKEwjHm-Wtucr-AhWvrJUCHTxCDDwQ3ecFegQINBAY'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

campo_preco = soup.find('div', {'class': 'YMlKec fxKbKc'})
valor_preco = campo_preco.text.strip()

print(valor_preco)