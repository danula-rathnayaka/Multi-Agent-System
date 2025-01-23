from dotenv import load_dotenv
import google.generativeai as genai
from phi.model.google import Gemini
from phi.agent import Agent
from Agents import get_google_search_agent
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

agent_team = Agent(
    team=[get_google_search_agent()],
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=[
        "Ensure all responses are well-structured, accurate, and user-friendly.",
        "Always include credible and properly formatted sources for any information or data shared.",
        "Where appropriate, organize information into tables or bulleted lists for clarity and ease of understanding.",
        "Coordinate seamlessly with team agents to provide comprehensive and unified responses.",
        "Handle conflicting data by prioritizing accuracy and relevance, and mention discrepancies if necessary.",
        "Strive for a balance of brevity and detail, ensuring responses are both concise and informative.",
        "Maintain a professional tone, avoiding unnecessary jargon unless explicitly requested by the user.",
        "Use Markdown formatting to enhance readability, including bold headings, tables, and bullet points."
    ],
    show_tool_calls=True,  # Testing purposes only
    markdown=True,
)
