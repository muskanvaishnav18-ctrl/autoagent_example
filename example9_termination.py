import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

 
load_dotenv()

async def main() -> None:
    
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
 
    )

    writer_agent=AssistantAgent(
    name="writer",
    model_client = model_client,
    system_message="""You are a creative writer .Write engaging content based on requests and improve it based on feedback""",
    model_client_stream=True,
    )

    criic_agent=AssistantAgent(
    name="Critic",
    model_client = model_client,
    system_message="""You are a creative editor .provide construcive feedback on writing
    you must give at least one feedback and get it improved before approving Respond 
    withh 'APPROVED' when the content meets high standards """,
    
    )

    termination =TextMentionTermination("APPROVED")

    team = RoundRobinGroupChat([writer_agent,criic_agent],
                               termination_condition= termination)
    
    print("======REFLECTION PATTERN EXAMPLE=====")
    await Console(team.run_stream(task='''Write a compelling description for a 
                                  smart fitness watch'''))
    
    await model_client.close()

asyncio.run(main())