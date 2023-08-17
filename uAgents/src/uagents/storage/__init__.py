import json
import os
from typing import Any, Optional
from typing import Tuple
from cosmpy.aerial.wallet import PrivateKey
from uagents.crypto import Identity
import pandas as pd
from uagents.environment import Environment


class KeyValueStore:
    def __init__(self, name: str, cwd: str = None):
        self._data = {}
        self._data_plan = {}
        self._data_belief = {}
        self._data_desire = []
        self._data_intention = []
        self._name = name or "my"

        cwd = cwd or os.getcwd()
        self._path = os.path.join(cwd, f"{self._name}_data.json")

        if os.path.isfile(self._path):
            self._load()

    def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    def has(self, key: str) -> bool:
        return key in self._data

    def set(self, key: str, value: Any):
        self._data[key] = value
        self._save()

    def remove(self, key: str):
        if key in self._data:
            del self._data[key]
            self._save()

    def clear(self):
        self._data.clear()
        self._save()

    def _load(self):
        with open(self._path, "r", encoding="utf-8") as file:
            self._data = json.load(file)

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=4)
            
    ############################################## BELIEF ######################################################
    def _load_belief(self):
        belief_dir = os.path.join(os.getcwd(), "belief")
        belief_file_path = os.path.join(belief_dir, f"{self._name}_belief.json")

        if os.path.isfile(belief_file_path):
            with open(belief_file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return None
    
    def set_belief(self, key: str, value: Any):
        self._data_belief[key] = value
        self._save_belief()
    
    def _save_belief(self):
        belief_dir = os.path.join(os.getcwd(), "belief")
        
        os.makedirs(belief_dir, exist_ok=True)
        
        belief_file_path = os.path.join(belief_dir, f"{self._name}_belief.json")
        
        with open(belief_file_path, "w", encoding="utf-8") as file:
            json.dump(self._data_belief, file, ensure_ascii=False, indent=4)   
            
    def get_belief(self, key: str) -> Optional[Any]:
        belief_data = self._load_belief()
        return belief_data.get(key)
    
    def all_belief(self) -> Optional[Any]:
        return self._load_belief()
    
    
    ############################################## DESIRE ######################################################
    
    def _load_desire(self):
        desire_dir = os.path.join(os.getcwd(), "desire")
        desire_file_path = os.path.join(desire_dir, f"{self._name}_desire.json")

        if os.path.isfile(desire_file_path):
            with open(desire_file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return None
    
    def set_desire(self, key: str):
        self._data_desire.append(key)
        self._save_desire()
    
    def _save_desire(self):
        desire_dir = os.path.join(os.getcwd(), "desire")
        
        os.makedirs(desire_dir, exist_ok=True)
        
        desire_file_path = os.path.join(desire_dir, f"{self._name}_desire.json")
        
        with open(desire_file_path, "w", encoding="utf-8") as file:
            json.dump(self._data_desire, file, ensure_ascii=False, indent=4)   
            
    def get_desire(self) -> Optional[Any]: #atenção ao key
        desire_data = self._load_desire()
        return desire_data.pop()
    
    def all_desire(self) -> Optional[Any]:
        return self._load_desire()
    
    
    ############################################## INTENTION ######################################################
    def _load_intention(self):
        intention_dir = os.path.join(os.getcwd(), "intention")
        intention_file_path = os.path.join(intention_dir, f"{self._name}_intention.json")

        if os.path.isfile(intention_file_path):
            with open(intention_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data
        return None
    
    def set_intention(self, key: str):
        self._data_intention.extend(key)
        self._save_intention()
    
    def _save_intention(self):
        intention_dir = os.path.join(os.getcwd(), "intention")
        
        os.makedirs(intention_dir, exist_ok=True)
        
        intention_file_path = os.path.join(intention_dir, f"{self._name}_intention.json")
        
        with open(intention_file_path, "w", encoding="utf-8") as file:
            json.dump(self._data_intention, file, ensure_ascii=False, indent=4)   
            
    def get_intention(self, key: str) -> Optional[Any]:
        intention_data = self._load_intention()
        return intention_data.get(key)
    
    def all_intention(self) -> Optional[Any]:
        return self._load_intention()
    
    def execute_intetion(self, ctx):
         # enquanto tem açõe empilhadas na intenção
        while self._data_intention:
            next = self._data_intention.pop()
            print(next)
            # se for uma ação, executa ela (ou seja, não há entrada na plan library para 'next')
            if self.get_plan(next) == None:
                next_action = Environment.action()
                if hasattr(next_action, next):
                    action = getattr(next_action, next)
                    action(ctx)
                else: 
                    print("não posso fazer")
            # caso contrário, isso significa que é um objetivo, então olha na plan library o plano para atingir o objetivo e empilha a sequencia de ações
            else:
                 self._data_intention.extend(self.get_plan(next))
    
    
    ############################################## PLAN ######################################################
    def _load_plan(self):
        plan_dir = os.path.join(os.getcwd(), "plan")
        plan_file_path = os.path.join(plan_dir, f"{self._name}_plan.json")

        if os.path.isfile(plan_file_path):
            with open(plan_file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return None
    
    def set_plan(self, goal, scenario, plan):
        self._data_plan[goal] = {'context': scenario, 'plan': plan}
        self._save_plan()
    
    def _save_plan(self):
        plan_dir = os.path.join(os.getcwd(), "plan")
        
        os.makedirs(plan_dir, exist_ok=True)
        
        plan_file_path = os.path.join(plan_dir, f"{self._name}_plan.json")
        
        with open(plan_file_path, "w", encoding="utf-8") as file:
            json.dump(self._data_plan, file, ensure_ascii=False, indent=4) 
            
    # def add_plan(self, goal, prec, plan):
    #     # cria um dicionário indexado por goal, explicitando contexto (pré-condição) e o plano (sequencia de ações ou sub-objetivos)
    #     self._data_plan[goal] = {'context': prec, 'plan': [plan]}   
            
    def get_plan(self, goal) -> Optional[Any]:
        # se o objetivo possui planos para alcançalo
        if goal in self._data_plan:
            # verifico se as pré condições são satisfeitas (estão na Belief Base)
            if(set(self._data_plan[goal]['context'].items()).issubset(self._data_belief.items())):
                # retorno o plano para atingir aquele objetivo
                return self._data_plan[goal]['plan']
        return None
    
    def all_plan(self) -> Optional[Any]:
        return self._load_plan()

def load_all_keys() -> dict:
    private_keys_path = os.path.join(os.getcwd(), "private_keys.json")
    if os.path.exists(private_keys_path):
        with open(private_keys_path, encoding="utf-8") as load_file:
            return json.load(load_file)
    return {}


def save_private_keys(name: str, identity_key: str, wallet_key: str):
    private_keys = load_all_keys()
    private_keys[name] = {"identity_key": identity_key, "wallet_key": wallet_key}

    private_keys_path = os.path.join(os.getcwd(), "private_keys.json")
    with open(private_keys_path, "w", encoding="utf-8") as write_file:
        json.dump(private_keys, write_file, indent=4)


def get_or_create_private_keys(name: str) -> Tuple[str, str]:
    keys = load_all_keys()
    if name in keys.keys():
        private_keys = keys.get(name)
        return private_keys["identity_key"], private_keys["wallet_key"]

    identity_key = Identity.generate().private_key
    wallet_key = PrivateKey().private_key

    save_private_keys(name, identity_key, PrivateKey().private_key)
    return identity_key, wallet_key
