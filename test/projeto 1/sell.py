from uagents import Agent, Context, Model
from aumanaque import Create_agent
from action import Action

class Message(Model):
    message: str
    
sell = Create_agent.sell()
action = Action()

@sell.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received message from {sender}: {msg.message}")
    # send the response
    await ctx.send(sender, Message(message="Hello there alice.")) 
    
if __name__ == "__main__":
    sell.run()