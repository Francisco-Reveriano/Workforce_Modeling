import asyncio
import os
from openai import OpenAI
from agents import Agent, FileSearchTool, Runner, trace
from IPython.display import display, Markdown
from dotenv import load_dotenv
load_dotenv()


Basic_QA_Agent = Agent(
        name="Basic_QA_Agent",
        instructions="You are a helpful agent. You answer only based on the information in the vector store. Provide all citations with footnotes at the end of the answer.",
        model=os.getenv("LLM_MODEL"),
        tools=[
            FileSearchTool(
                max_num_results=20,
                vector_store_ids=[os.getenv("ASSISTANT_VECTOR_KEY")],
                include_search_results=True,
            )
        ],
    )