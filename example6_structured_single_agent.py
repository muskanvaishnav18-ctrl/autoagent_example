import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel
from typing import List, Literal
from autogen_core.models import ModelInfo
import random
 
load_dotenv()
 
# Define structured output model
class MovieReview(BaseModel):
    title:str
    genre: List[str]
    rating: int #1-10
    sentiment: Literal["positive","negative","mixed"]
    summary: str
    pros: List [str]
    cons: List[str]
    recommendation: str
 
async def main() -> None:
    print("Hello How can I help You")
    movie_name=input("Enter Any Movie--")
    model_client=OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        model_info= ModelInfo(vision=True, function_calling=True, json_output=True,
        family="unknown", structured_output=True),
        api_key=os.getenv("GOOGLE_API_KEY")
 
    )
 
#Movie review agent
    movie_critic_agent=AssistantAgent(
    name="movie_critic",
    model_client = model_client,
    system_message="""You are a professional movie critic. 
    Analyze movies throughly and provide structured reviews.""",
    output_content_type=MovieReview #structured output
    )
 
# Test movie review
    print("=========MOVIEW REVIEW=========")
    movie_result= await movie_critic_agent.run(task="""Review the movie {movie_name}""")
 
    if isinstance(movie_result.messages[-1], StructuredMessage):
        review= movie_result.messages[-1].content
        print(f"==========Title============: {review.title}")
        print(f"==========Genre============: {','.join(review.genre)}")
        print(f"==========Rating============: {review.rating/10}")
        print(f"==========Sentiment============: {review.sentiment}")
        print(f"==========Summary============: {review.summary}")
        print(f"==========Pros============: {review.pros}")
        print(f"==========Cons============: {review.cons}")
        print(f"==========recommended============: {review.recommendation}")
 
    await model_client.close()
 
 
asyncio.run(main())