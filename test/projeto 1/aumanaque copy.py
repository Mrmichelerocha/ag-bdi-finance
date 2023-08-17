from uagents.setup import fund_agent_if_low
from uagents import Agent, Context, Model
from uagents.resolver import RulesBasedResolver

# https://colab.research.google.com/drive/1vLtqbvlqAFbfBV0nizucg4yAFC8O_bsr

class Create_agent():  

    BUY_ADDRESS = "agent1qvjktdfvx9zfrd2zsp3ff9pzxen9zlnddnc4hshqx3jza63lct4usq99985"
    SELL_ADDRESS = "agent1q2hxgq6gkhgdtcqce8jex0hjcnkxa05nxppzegzphur9zct5r8kj2u9ycef"
    MIN_ADDRESS = "agent1qvheyshk3pfw3fdpv7tgqwakd4z69vu3hced9trkcvlw087h4ymczmcn58g"
    MAX_ADDRESS = "agent1q2rt0q4vpqkvfqyp0lfp9fk5uwuh9fmw8z3fc95p2695tkffjuwyq90c09h"
    
    resolve=RulesBasedResolver(
        {
            BUY_ADDRESS:    "http://127.0.0.1:8020/submit",
            SELL_ADDRESS:   "http://127.0.0.1:8021/submit",
            MIN_ADDRESS:    "http://127.0.0.1:8022/submit",
            MAX_ADDRESS:    "http://127.0.0.1:8023/submit",
        }
    )
    
    def buy():   
        buy = Agent(
            name="buy",
            port=8020,
            seed="buy secret phrase",
            resolve= Create_agent.resolve,
        )
        
        fund_agent_if_low(buy.wallet.address())
        
        return buy
        
    def sell():
        sell = Agent(
            name="sell",
            port=8021,
            seed="sell secret phrase",
            resolve= Create_agent.resolve,
        )
        
        fund_agent_if_low(sell.wallet.address())
        
        return sell
    
    def min():
        min = Agent(
            name="min",
            port=8022,
            seed="min secret phrase",
            resolve= Create_agent.resolve,
        )
        
        fund_agent_if_low(min.wallet.address())
        
        return min
    
    def max():
        max = Agent(
            name="max",
            port=8023,
            seed="max secret phrase",
            resolve= Create_agent.resolve,
        )
        
        fund_agent_if_low(max.wallet.address())
        
        return max
    