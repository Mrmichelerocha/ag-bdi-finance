from uagents import Agent, Context, Model
from aumanaque import Create_agent
from action import Action
import time

class Message(Model):
    message: str
    
agent2 = Create_agent.Agent2()
action = Action()
  
@agent2.on_event('startup')
async def plan_event(ctx: Context):
    pass
    
@agent2.on_interval(period=3600.5)
async def plan_interval(ctx: Context):
    action.date(ctx)
    await ctx.send(Create_agent.AGENT1_ADDRESS,  Message(message=f'opening')) # if agent2.contexto(ctx, {"horary": "14:32"}) else False
    # await ctx.send(Create_agent.AGENT1_ADDRESS,  Message(message=f'closing')) # if agent2.contexto(ctx, {"horary": "14:32"}) else False

@agent2.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    pass
    
if __name__ == "__main__":
    agent2.run()
