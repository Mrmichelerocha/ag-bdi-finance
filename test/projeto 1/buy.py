from uagents import Agent, Context, Model
from aumanaque import Create_agent
from action import Action

class Message(Model):
    message: str
    dado: float
    
buy = Create_agent.buy()
action = Action()

@buy.on_interval(period=30.5)
async def plan_interval(ctx: Context):
    print("to no jogo")

@buy.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received message from: {msg.message}")
    if msg.message == 'swing':
        print("recebi sua mensagem")
    
if __name__ == "__main__":
    buy.run()