import google.generativeai as genai
from phi.model.google import Gemini
from phi.agent import Agent
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"));

Gemini(id="genmini-2.0-flash-exp")

agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    markdown=True,
)

agent.print_response("Share a 2 sentence horror story.")


