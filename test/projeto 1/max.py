from uagents import Context, Model
from aumanaque import Create_agent
from action import Action

class Message(Model):
    message: str
    
max = Create_agent.max()
action = Action()

@max.on_event("startup")
async def plan_event(ctx: Context):
    ctx.storage.set_belief("sell_checked", 1)
    ctx.storage.set_belief("belief_high", 200.0)
    
@max.on_interval(period=30.5)
async def plan_interval(ctx: Context): 
    action.price_now(ctx)
    action.sell(ctx) if action.high_now(ctx) else False 
    action.data_now(ctx) if action.sell_checked(ctx) == False else False   
    await ctx.send(Create_agent.BUY_ADDRESS, Message(message=f'swing')) if action.checked_time else False 

@max.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received message from: {msg.message}")
    if msg.message == 'buy':
        action.preview_high(ctx) 
        await ctx.send(sender, Message(message="venda")) 
    
if __name__ == "__main__":
    max.run()