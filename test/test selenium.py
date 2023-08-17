from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

# Coloque o caminho do seu chromedriver aqui
chromedriver_path = "/path/to/chromedriver"

# Inicie o driver do Chrome
driver = webdriver.Chrome(chromedriver_path)

# Defina o tempo de espera implícita
driver.implicitly_wait(10)

# Abra o site do Yahoo
driver.get("https://www.google.com/finance/")

# Encontrar os campos de email e senha e preenchê-los
email_field = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz[2]/div/div[3]/div[3]/div/div/div/div[1]/input[2]')
email_field.send_keys('APPL')
email_field.send_keys(Keys.RETURN)

# Encontrar o elemento usando o XPath
xpath = '//*[@id="yDmH0d"]/c-wiz[3]/div/div[4]/div/main/div[2]/div[1]/div[1]/c-wiz/div/div[1]/div/div[1]/div/div[1]/div/span/div/div'
element = driver.find_element(By.XPATH, xpath)

# Extrair o valor do elemento
valor = element.text
print("Valor da div:", valor)

time.sleep(60)
# senha_field.send_keys('Mgg09@rock')


