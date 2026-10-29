# agora

A lightweight agent framework with a blackboard architecture.

## Install

To install locally call the following from this directory:

```sh
python -m pip install -e .
```

## Use

Each agent must extend the `Agent` class. 

A `Controller` checks which agent should execute next.

Agents communicate via a `Blackboard`.

The following is a minimal example:

```python
class SummarizerAgent(Agent):
    """Agent that summarizes text"""

    def can_execute(self, blackboard):
        """Check if the agent's preconditions are met"""
        return blackboard.get("text") is not None and blackboard.get("summary") is None

    def execute(self, blackboard):
        """Execute agent logic"""
        text = blackboard.get("text")
        prompt = f"""Summarize the following text in a few words:
{text}
Output only the summary.
"""
        summary = self.llm_provider.complete(
            [{"role": "user", "content": prompt}])
        blackboard.update("summary", summary)
```

This agent will run when text has been posted to the blackboard and there is has not been summarized yet. It could run on its own, but to make it more useful let's pair it with another agent that fetches news. The following agent reads business news from an RSS feed: 

```python
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
```

We can feed the output of the news retriever to the summarizer by setting up a controller and add the agents. All that is left to do is to set up the blackboard and update its `rss-feed` slot with the URL of the RSS feed. Then we can start the controller. The controller will stop once the `summary` slot has been filled or after three iterations.

```python
blackboard = Blackboard({
        "rss-feed": None,
        "text": None,
        "summary": None,
    })
    blackboard.update("rss-feed", 
        "https://www.theguardian.com/uk/rss")

    controller = Controller(blackboard, "summary")
    controller.add_agent(NewsRetrieverAgent("NewsRetriever", None))
    controller.add_agent(SummarizerAgent("Summarizer", ClaudeLLMProvider(
        system="You are a news announcer. Make the news sound engaging.")))
    controller.run(max_iterations=3)

    print("News:", blackboard.get("summary"))
```