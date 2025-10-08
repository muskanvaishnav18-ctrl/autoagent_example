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
        api_key=os.getenv("OPENAI_API_KEY")  # ✅ Use OpenAI key here
    )

    teacher = AssistantAgent(
        name='Teacher',
        model_client=model_client,
        
        system_message=""""You are  a math Teacher. Explain conept clearly and ask follow-up questions """,
        model_client_stream=True,
    )
    student = AssistantAgent(
        name='Student',
        model_client=model_client,
        system_message=""""you are a student learning math.
        Ask question when confused""",
        model_client_stream=True,
    )

    teacher_result = await teacher.run(task="Explain what a probability is to a beginner")
    print("---------------Teacher----------", teacher_result.messages[-1].content)

    student_result= await student.run(task=f"""The teacher said:{teacher_result.messages[-1].content}. please ask
                                      a clarifying question about probabilities""")
    print("-------------Student-----------",student_result.messages[-1].content)


    teacher_result =await teacher.run(task=f"""The student asked:
                                      {student_result.messages[-1].content}. Please provide a detailed
                                      answer.""")
    print("---------------------Teacher----------", teacher_result.messages[-1].content)
    
    await model_client.close()

asyncio.run(main())