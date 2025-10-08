import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.ui import Console

 
load_dotenv()

async def main() -> None:
    
    model_client=OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
 
    )


    strategist_context = BufferedChatCompletionContext(buffer_size=5)
    analyze_context = BufferedChatCompletionContext(buffer_size=8)
    
    strategist=AssistantAgent(
    name="strategist",
    model_client = model_client,
    model_context= strategist_context,
    system_message="""You are a business strategist.
    Focus a high-level strategy and planning""",
    
    )

    analyst =AssistantAgent(
    name="analyst",
    model_client = model_client,
    model_context= strategist_context,
    system_message="""You are a financial  Analyst.
    provide detailed financial analysis and projection """,
    
    )

    team = RoundRobinGroupChat([strategist,analyst],
                               termination_condition= MaxMessageTermination(8))
    
    print("======Custom Context managment=====")
    result= await team.run(task= """ Analyse the business potential of entering the electric vehicle market
                           in India . Consider both strategic an financial aspects .""")
    
    Console(result)
    print(f"\n Total messages:{len(result.messages)}")
    print(f"stop reason: {result.stop_reason}")

    # await team.reset()

    # print("\n =====NEW CONVERSATION AFTER RESET ======")
    # await  Console(team.run_stream(task="Now analyze the renewable energy sector instead."))
    await model_client.close()

asyncio.run(main())