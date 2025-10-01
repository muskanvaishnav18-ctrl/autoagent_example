import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

from dotenv import load_dotenv

load_dotenv()

api_key= os.getenv("OPENAI_API_KEY")
# print(api_key)

async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model='gpt-4o',
        api_key=api_key
    )

    agent = AssistantAgent("assistant", model_client=model_client)
    print(await agent.run(task="Say 'Hello World' "))
    print("========RESPONSE RECEIVED FROM MODEL================")
    await model_client.close()


asyncio.run(main())