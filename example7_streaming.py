import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel
from typing import List, Literal
from autogen_core.models import ModelInfo
from autogen_agentchat.ui import Console
import random
 
load_dotenv()

async def main() -> None:
    print("Hello How can I help You")
    model_client=OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        model_info= ModelInfo(vision=True, function_calling=True, json_output=True,
        family="unknown", structured_output=True),
        api_key=os.getenv("GOOGLE_API_KEY")
 
    )

    writer_agent=AssistantAgent(
    name="creative_writer",
    model_client = model_client,
    system_message="""You are a creative writer specializing in science fiction . Write enagage
    stories with vivid description and compelling characters.""",
    model_client_stream=True,
    )

    print("Starting creative writing session ...\n")

    await Console(
        writer_agent.run_stream(task="""Write a short science fiction story about a time
                                traveler who discover soemthing unexpected about their post.
                                Make it enagaging and include dialogue"""),
        output_stats=True,
    )

    await model_client.close()

asyncio.run(main())
