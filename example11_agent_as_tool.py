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

async def research_agent_tool(query:str) -> str:    
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
 
    )

    research_agent=AssistantAgent(
    name="Research_apecialist",
    model_client = model_client,
    system_message="""You are a  speacialist research agent.
    provide concise, factual ,research findings""",
    
    )

    result = await research_agent.run(task=f"Reasearch this topic: {query}")
    await model_client.close()
    return result.messages[-1].content

async def main() -> None:
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
 
    )

    coordinator = AssistantAgent(
        "Coordinator",
        model_client= model_client ,
        tools =[research_agent_tool],
        system_message= """ You are a project coordinator.description=Use the research and
        calculatortools to provide comprehensive analysis.""",
    )

    reviewer = AssistantAgent(
        "Reviewer",
        model_client=model_client,
        tools=[research_agent_tool],
        system_message= """You are a project coordinator .
        Use the research and calculator tools to provide comprehensive analysis. """ ,
        
    )

    reviewer= AssistantAgent(
        'Reviewer',
        model_client= model_client,
        system_message= """You are a project reviewer. Evaluate the coordinator's analysis
        and suggest improvements."""
    )

    team = RoundRobinGroupChat(
        [coordinator,reviewer],
        termination_condition=MaxMessageTermination(6)
    )

    print("===AGENT-AS-TOOL PATTERN=====")
    result= await team.run(task="""Analyse the ROI of investing $100,00 in a SaaS startup.
                           research the market and calculate potebtial returns.""")
    
    print(f" \nfinal analysis completed with {len(result.messages)} messages:")

    print("Conversation Transcript:")
    for msg in result.messages:
        print(f"{msg.source}: {msg.content} \n")

    await model_client.close()


asyncio.run(main())