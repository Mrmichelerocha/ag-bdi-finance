from uagents import Context, Model
from aumanaque import Create_agent
from action import Action

class Message(Model):
    message: str
    
min = Create_agent.min()
action = Action()

@min.on_event("startup")
async def plan_event(ctx: Context):
    ctx.storage.set_belief("buy_checked", 1)
    action.preview_low(ctx)

@min.on_interval(period=30.5)
async def plan_interval(ctx: Context): 
    action.price_now(ctx)
    action.buy(ctx) if action.low_now(ctx) else False
    await ctx.send(Create_agent.MAX_ADDRESS, Message(message=f'buy')) if action.buy_checked(ctx) else False
    
if __name__ == "__main__":
    min.run()