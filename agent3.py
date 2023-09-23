from uagents import Agent, Context, Model
from aumanaque import Create_agent
from action import Action
import time

class Message(Model):
    message: str
    
agent3 = Create_agent.Agent3()
action = Action()
  
@agent3.on_event('startup')
async def plan_event(ctx: Context):
    agent3.belief(ctx, "management_symbol", False)
    agent3.belief(ctx, "different", False)
    agent3.belief(ctx, "symbol", "None")
    agent3.belief(ctx, "price_min_check", False)
    agent3.belief(ctx, "price_max_check", False)
    
    agent3.set_plan_library(ctx, "min", {"price_min_check": True}, ["get_min"])
    agent3.set_plan_library(ctx, "max", {"price_max_check": True}, ["get_max"])
    agent3.set_plan_library(ctx, "day_trade", {"management_symbol": True}, ["trade", "check_buyorsell", "check_upordown", "check_wallet", "check_price"])
    agent3.set_plan_library(ctx, "analytic", {"management_symbol": True}, ["day_trade", "max", "check_max", "min", "check_min", "management_wallet"])
    
@agent3.on_interval(period=10.5)
async def plan_interval(ctx: Context):
    agent3.desire(ctx, 'analytic') if agent3.contexto(ctx, {"management_symbol": True}) else False
    agent3.update_intention(ctx)
    action.sell(ctx) if agent3.contexto(ctx, {"sell": True}) else False
    action.buy(ctx) if agent3.contexto(ctx, {"buy": True}) else False
   
@agent3.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received message from: {msg.message}")
    if msg.message == 'management_opening':
        agent3.belief(ctx, "management_symbol", True)
        agent3.belief(ctx, "different", True)
        agent3.belief(ctx, "symbol", "None")
        agent3.belief(ctx, "price_min_check", True)
        agent3.belief(ctx, "price_max_check", True)
    elif msg.message == 'management_closing':
        agent3.belief(ctx, "management_symbol", False)
        agent3.belief(ctx, "different", False)
        agent3.belief(ctx, "symbol", "None")
        agent3.belief(ctx, "price_min_check", False)
        agent3.belief(ctx, "price_max_check", False)
        action.closing_check(ctx)
        
    
if __name__ == "__main__":
    agent3.run()
