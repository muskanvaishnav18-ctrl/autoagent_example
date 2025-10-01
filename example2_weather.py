import asyncio
import os
import requests
from dotenv import load_dotenv

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def get_weather(city: str) -> str:
    """Fetch weather from OpenWeatherMap API"""
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            return f"The weather in {city} is {temp}°C with {description}."
        else:
            return f"Could not fetch weather for {city}. Error: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

async def main() -> None:
    city = input("Enter the city name: ")

    model_client = OpenAIChatCompletionClient(
        model='gpt-4o',
        api_key=os.getenv("OPENAI_API_KEY")  # ✅ Use OpenAI key here
    )

    agent = AssistantAgent(
        name='MuskanVaishnav',
        model_client=model_client,
        tools=[get_weather],
        system_message=""""You are  weather assistant , if user enter any string
          which is invalid city then please enter  a message like invalid city """,
        model_client_stream=True,
    )

    response = agent.run_stream(task=f"What is the weather in {city}?")
    await Console(response)

    await model_client.close()

asyncio.run(main())