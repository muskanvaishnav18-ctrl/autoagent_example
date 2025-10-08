import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
import os
from autogen_core import Image
import PIL
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

api_key= os.getenv("OPENAI_API_KEY")
# print(api_key)

async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model='gpt-4o',#make sure  to use a vision-capable model
        api_key=os.getenv("OPENAI_API_KEY")  # ✅ Use OpenAI key here
    )

    vision_agent = AssistantAgent(
        name='MuskanVaishnav',
        model_client=model_client,
        system_message=""""You are  an expert at describing and analyzing images in detail. """,
        model_client_stream=True,
    )

    image_response = requests.get("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTPQViHijPvRkmF90MRRnH5nQbV8AneGbGNMA&s")
    pil_image= PIL.Image.open(BytesIO(image_response.content))
    img= Image(pil_image)

    multi_model_message= MultiModalMessage(
        content =['describe this image in detail and tell me what mood it convey.', img],
        source ="user"
    )

    result = await vision_agent.run(task=multi_model_message)
    print("Vision Analysis:", result.messages[-1].content)

    await model_client.close()
    await model_client.close()


asyncio.run(main())