from agora.agent import Agent, Blackboard, Controller
from agora.llm import LLMProvider, ClaudeLLMProvider, ResultParser

import requests
import xmltodict

class SummarizerAgent(Agent):
    """Agent that summarizes text"""

    def can_execute(self, blackboard):
        """Check if the agent's preconditions are met"""
        return blackboard.get("text") is not None and blackboard.get("summary") is None

    def execute(self, blackboard, user_input=None):
        """Execute agent logic"""
        text = blackboard.get("text")
        prompt = f"""Summarize the following text in a few words:
{text}
Output only the summary.
"""
        summary = self.llm_provider.complete(
            [{"role": "user", "content": prompt}])
        blackboard.update("summary", summary)

class NewsRetrieverAgent(Agent):
    """Agent that retrieves news from and RSS feed"""

    def can_execute(self, blackboard):
        """Check if the agent's preconditions are met"""
        return blackboard.get("rss-feed") is not None and blackboard.get("text") is None

    def execute(self, blackboard, user_input=None):
        """Execute agent logic"""
        rss_feed = blackboard.get("rss-feed")
        news = self.fetch_news(rss_feed, n_items=3)
        blackboard.update("text", news)

    def fetch_news(self, rss_feed, n_items=1):
        """Fetch news items from the RSS feed"""
        response = requests.get(rss_feed)
        xml = xmltodict.parse(response.text)
        items = xml["rss"]["channel"]["item"]
        news = ""  # concatenate the news items
        for i in range(min(n_items, len(items))):
            news += items[i]["title"] + "\n"
            news += items[i]["description"] + "\n"
        return news

def test_agent():
    print("test_agent")
    blackboard = Blackboard({
        "text": None,
        "summary": None,
    })
    # Post a text to summarize to the blackboard
    blackboard.update("text", """
In today's other news, two parrots escaped from the local zoo 
and  were spotted flying over the city center. Zoo staff are on the scene 
trying to safely capture them. Check back here for updates.
""")
    # Create the summarizer agent
    # Then run the agent, if it can execute
    agent = SummarizerAgent("Summarizer", ClaudeLLMProvider(
        system="You are an expert summarizer"))
    if agent.can_execute(blackboard):
        assert blackboard.get("text") is not None
        agent.execute(blackboard)
        assert blackboard.get("summary") is not None
        print("Summary:", blackboard.get("summary"))
    else:
        print("Agent cannot execute")

def test_agent_chain():
    print("test_agent_chain")
    blackboard = Blackboard({
        "rss-feed": None,
        "text": None,
        "summary": None,
    })
    # Post the RSS feed URL to the blackboard
    blackboard.update("rss-feed", 
        "https://www.theguardian.com/uk/rss")

    # Set up the controller with agents to retrieve news and summarize it
    # Then run the controller to execute the agents in sequence
    controller = Controller(blackboard, "summary")
    controller.add_agent(NewsRetrieverAgent("NewsRetriever", None))
    controller.add_agent(SummarizerAgent("Summarizer", ClaudeLLMProvider(
        system="You are a news announcer. Make the news sound engaging.")))
    controller.run(max_iterations=3)

    # If everything goes well, there should be a summary on the blackboard
    assert blackboard.get("summary") is not None
    print("News:", blackboard.get("summary"))

test_agent()
test_agent_chain()