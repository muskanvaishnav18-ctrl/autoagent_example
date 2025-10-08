import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.ui import Console

 
load_dotenv()

async def main() -> str:    
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
 
    )

    requirements_analyst = AssistantAgent(
        "requirements_analyst",
        model_client=model_client,
        handoffs= [Handoff(target= 'architect')],
        system_message="""" You anayze requiremnets and create detailed specificatons.
        when complete , handoff to the architect""",
    )

    architect = AssistantAgent(
        "architect",
        model_client=model_client,
        handoffs= [Handoff(target= 'devloper')],
        system_message="""" You design system architecture based on requirements .
        description when complete , handoff to the developer"""
    )

    developer = AssistantAgent(
        "developer",
        model_client=model_client,
        handoffs= [Handoff(target= 'devloper')],
        system_message="""" You create implementation plans based on architecture.
        when finished , say 'TERMINATE'. """
    )

    handoff_complete = HandoffTermination(target="COMPLETE")
    text_termination =TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(
        [requirements_analyst,architect, developer],
        termination_condition= handoff_complete | text_termination
    )


    print("====sequential workflow with handoof=====")

    await Console(team.run_stream(task="""Design and plan the implementation of a 
                        real-time chat application with useer authentication an message history""")) 
    
    await model_client.close()
asyncio.run(main())    
