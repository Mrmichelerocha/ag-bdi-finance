import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Definindo as variáveis de entrada e saída fuzzy
direcao = ctrl.Antecedent(np.arange(-1, 2, 1), 'direcao')  # -1: Descer, 0: Manter, 1: Subir
acao_anterior = ctrl.Antecedent(np.arange(-1, 2, 1), 'acao_anterior')  # -1: Vender, 0: Manter, 1: Comprar
acao = ctrl.Consequent(np.arange(-1, 2, 1), 'acao')  # -1: Vender, 0: Manter, 1: Comprar

# Definindo os conjuntos fuzzy para direcao
direcao['descer'] = fuzz.trimf(direcao.universe, [-1, -1, 0])
direcao['manter'] = fuzz.trimf(direcao.universe, [-1, 0, 1])
direcao['subir'] = fuzz.trimf(direcao.universe, [0, 1, 1])

# Definindo os conjuntos fuzzy para acao_anterior e acao
acao_anterior['vender'] = fuzz.trimf(acao_anterior.universe, [-1, -1, 0])
acao_anterior['manter'] = fuzz.trimf(acao_anterior.universe, [-1, 0, 1])
acao_anterior['comprar'] = fuzz.trimf(acao_anterior.universe, [0, 1, 1])

acao['vender'] = fuzz.trimf(acao.universe, [-1, -1, 0])
acao['manter'] = fuzz.trimf(acao.universe, [-1, 0, 1])
acao['comprar'] = fuzz.trimf(acao.universe, [0, 1, 1])

# Regras fuzzy
regra1 = ctrl.Rule(direcao['descer'] & acao_anterior['vender'], acao['vender'])
regra2 = ctrl.Rule(direcao['descer'] & acao_anterior['comprar'], acao['comprar'])
regra3 = ctrl.Rule(direcao['subir'] & acao_anterior['vender'], acao['vender'])
regra4 = ctrl.Rule(direcao['manter'] & acao_anterior['manter'], acao['manter'])
regra_fallback = ctrl.Rule(~regra1.antecedent & ~regra2.antecedent & ~regra3.antecedent & ~regra4.antecedent, acao['manter'])

# Sistema de controle fuzzy
sistema_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra_fallback])
sistema = ctrl.ControlSystemSimulation(sistema_ctrl)

# Entrada: direcao da acao e acao anterior
sistema.input['direcao'] = 1  # 1: Subir
sistema.input['acao_anterior'] = 0  # -1: Vender

# Calcula a saída do sistema fuzzy
sistema.compute()

# Saída: acao recomendada
acao_recomendada = sistema.output['acao']
print("NUMERO DO FUZZY:", acao_recomendada)
acao_texto = "Vender" if acao_recomendada <= -0.5 else ("Manter" if -0.5 < acao_recomendada <= 0.5 else "Comprar")
print("Ação Recomendada:", acao_texto)
