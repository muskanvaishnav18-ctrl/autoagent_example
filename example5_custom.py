import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
import math
import random
 
load_dotenv()
 
# DEfine custom tool
async def  calculate_circle_area(radius: float) ->str:
    """Calculate the area of  a circle given its radius.""" 
    print(f"Calculating area for radius: {radius}")
    area=math.pi *radius **2
    return f"The area of a circle with radius {radius} is {area:.2f} square units."
 
async def roll_dice(sides: int=6, count: int=1) ->str:
    """Rol dice and return the results."""
    if count<1 or count>10:
        return "Can only roll between 1 and 10 dice at a time. "
    if sides < 2 or sides>100:
        return "Dice must have between 2 and 100 sides."
    results = [random.randint(1, sides) for _ in range(count)]
    total = sum(results)
 
    return f"Rolled {count}d{sides}:{results} (Total: {total})"
 
async def get_random_fact() -> str:
    """Get a random interesting fact."""
    facts=[
        "Ouctopuses have three hearts and blue blood.",
        "A group of flamingos is called a flamboyance",
        "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs",
        "A shrimp's heart is in its head.",
        "Bananas are berries, but strawberries aren't."
    ]
    return random.choice(facts)
 
async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )
 
# cREATE agents with multiple tools
    tool_agent = AssistantAgent(
        name="tool_master",
        model_client=model_client,
        tools=[calculate_circle_area, roll_dice, get_random_fact],
        system_message="""You are a helpful assistant with  access to various tools.
        Use them to help users with calculations, games, and interesting facts. You must call 
        max 2 tools per request. You don't have access to call more than 2 toolt at a time. """,
        max_tool_iterations=2 #Allow multiple tool calls
    )
 
    #FYI : max_tool_iterations is At most 3 iterations of tool calls before stopping the loop
    # The agnet can be configured to execute multiple iterations until model stops
    # generating tool calls or the maximum number of iterations is reached
 
    tasks=["Calculate the area of circle with radius 3.5",
        "Roll 2 six_sided dice",
        "Tell me a random fact",
        """Calculate the area of circle with radius 3.5 and then roll 3 dice with 
        8 sides each and tell me a random fact.""" 
    ]
    for task in tasks:
        print(f"\nTask: {task}")
        result = await tool_agent.run(task=task)
        print(f"Response : {result.messages[-1].content}")
 
asyncio.run(main())   
 