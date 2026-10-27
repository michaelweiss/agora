from agora.agent import Agent, Blackboard, Controller
from agora.llm import LLMProvider, ClaudeLLMProvider, ResultParser

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

def test_agent():
    print("test_agent")
    blackboard = Blackboard({
        "text": None,
        "summary": None
    })
    blackboard.update("text", """
In today's other news, two parrots escaped from the local zoo 
and  were spotted flying over the city center. Zoo staff are on the scene 
trying to safely capture them. Check back here for updates.
""")
    agent = SummarizerAgent("Summarizer", ClaudeLLMProvider(
        system="You are an expert summarizer"))
    if agent.can_execute(blackboard):
        assert blackboard.get("text") is not None
        agent.execute(blackboard)
        assert blackboard.get("summary") is not None
        print("Summary:", blackboard.get("summary"))
    else:
        print("Agent cannot execute")

test_agent()