from agora.agent import Blackboard, Controller
from agora.llm import ClaudeLLMProvider
from agora.roles import TaskAgent, ToolAgent
from agora.specs import load_agent
from agora.tools import tool
from agora.logs import configure_logging

import requests
import xmltodict

@tool
def fetch_news(rss_feed):
    """Fetch news articles from the given RSS feed URL and return their titles as a string."""
    response = requests.get(rss_feed)
    feed = xmltodict.parse(response.content)
    items = feed["rss"]["channel"]["item"]
    news = [{"title": item["title"], "description": item["description"]} for item in items]
    return news

def team():
    configure_logging(trace_file="logs/trace.json")

    blackboard = Blackboard({})
    # Post the RSS feed URL to the blackboard
    blackboard.update("rss_feed", 
        "https://www.theguardian.com/uk/rss")

    # Set up the controller with agents to retrieve news and summarize it
    # Then run the controller to execute the agents in sequence
    controller = Controller(blackboard, "summary")
    controller.add_agent(load_agent("specs/news_retriever.yaml"))
    controller.add_agent(load_agent("specs/summarizer.yaml", 
        llm_provider=ClaudeLLMProvider()))
    controller.run(max_iterations=5)

    # Retrieve and print the summary from the blackboard
    summary = blackboard.get("summary")
    print(summary)

team()