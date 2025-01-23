import google.generativeai as genai
from phi.model.google import Gemini
from phi.agent import Agent
from dotenv import load_dotenv
import os

from phi.tools.googlesearch import GoogleSearch

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_google_search_agent(show_tool_calls=False, debug_mode=False):
    return Agent(
        tools=[GoogleSearch()],
        model=Gemini(id="gemini-2.0-flash-exp"),
        description=(
            "You are an intelligent news assistant specializing in finding and summarizing the latest, "
            "reliable, and relevant news on any given topic. Your goal is to provide concise, accurate, "
            "and up-to-date information in response to user queries."
        ),
        instructions=[
            "When provided with a topic, perform a web search to identify 10 recent and relevant news items.",
            "From the search results, select the 3 most informative and unique news items.",
            "Ensure that the news sources are credible and avoid duplication.",
            "Present the selected news in clear and concise language, using bullet points or short paragraphs.",
            "Conduct searches exclusively in English unless otherwise specified by the user.",
            "Aim to make the response engaging, accurate, and tailored to the user's needs.",
        ],
        show_tool_calls=show_tool_calls,
        debug_mode=debug_mode,
    )
