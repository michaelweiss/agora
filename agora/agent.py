import threading
import time
import logging

from .llm import LLMProvider, ClaudeLLMProvider, ChatgptLLMProvider

class Blackboard:
    def __init__(self, data):
        self.data = data
        self.lock = threading.Lock()

    def update(self, key, value):
        """Add or update an item on the blackboard"""
        with self.lock:
            self.data[key] = value
            # logging.info(f"Update {key}: {value}")

    def get(self, key, default=None):
        """Get an item from the blackboard"""
        with self.lock:
            value = self.data.get(key, default)
            # logging.info(f"Get {key}: {value}")
            return value
        
    def take(self, key):
        """Get and remove an item from the blackboard"""
        with self.lock:
            value = self.data.pop(key, None)
            # logging.info(f"Take {key}: {value}")
            return value

class Agent:
    def __init__(self, name, llm_provider=None):
        self.name = name
        self.llm_provider = llm_provider

    def can_execute(self, blackboard):
        """Check if the agent's preconditions are met"""
        raise NotImplementedError("Subclasses should implement this")

    def needs_human_input(self, blackboard):
        """Check if the agent needs human input"""
        return None

    def execute(self, blackboard, user_input=None):
        """Execute agent logic"""
        raise NotImplementedError("Subclasses should implement this")

class Controller:
    def __init__(self, blackboard, solution_topic):
        self.blackboard = blackboard
        self.agents = []
        self.solution_topic = solution_topic
        
    def add_agent(self, agent):
        self.agents.append(agent)

    def stream_events(self, max_iterations=5):
        """
        Generator that yields when human input is needed and for status updates.
        """
        iteration = 0
        while self.blackboard.get(self.solution_topic) is None:
            iteration += 1
            for agent in self.agents:
                if agent.can_execute(self.blackboard):
                    input_request = agent.needs_human_input(self.blackboard)
                    if input_request:
                        user_input = yield {
                            "type": "input_needed",
                            "agent": agent.name,
                            **input_request
                        }
                    else:
                        user_input = None
                    agent.execute(self.blackboard, user_input=user_input)
                    yield {
                        "type": "status",
                        "iteration": iteration,
                        "agent": agent.name
                    }
            if iteration >= max_iterations:
                break

    def run(self, max_iterations=5):
        """
        Run until the solution is found or max_iterations has been reached.
        """
        workflow = self.stream_events(max_iterations=5)
        try:
            event = next(workflow)
            while True:
                if event["type"] == "status":
                    print("Agent", event["agent"], "executed")
                    event = next(workflow)
                elif event["type"] == "input_needed":
                    user_input = input(event["prompt"] + ": ")
                    event = workflow.send(user_input)
        except StopIteration:
            pass