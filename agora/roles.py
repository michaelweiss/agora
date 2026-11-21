from .agent import Agent, Blackboard, Controller
from .llm import LLMProvider, ClaudeLLMProvider, ResultParser

import inspect

class TaskAgent(Agent):
    """Task agent that performs a specific task using prompt."""

    def __init__(self, name: str, llm_provider: LLMProvider,
                 in_slots = [], out_slots = [], prompt = ""):
        super().__init__(name)
        self.llm_provider = llm_provider
        self.in_slots = in_slots
        self.out_slots = out_slots
        self.prompt = prompt
        self.parser = ResultParser()

    def can_execute(self, blackboard):
        return all(blackboard.get(slot) is not None for slot in self.in_slots) and \
               all(blackboard.get(slot) is None for slot in self.out_slots)
    
    def execute(self, blackboard: Blackboard, user_input=None):
        inputs = {slot: blackboard.get(slot) for slot in self.in_slots}
        prompt_filled = self.prompt.format(**inputs)
        response = self.llm_provider.complete(
            [{"role": "user", "content": prompt_filled}])
        result = self.parser.parse(response)
        assert "json" in result and len(result["json"]) > 0, "Expected json in output"
        for slot, value in result["json"][0].items():
            assert slot in self.out_slots, f"Unexpected output slot: {slot}"
            blackboard.update(slot, value)

class ToolAgent(Agent):
    """Tool agent that calls a tool."""

    def __init__(self, name: str, in_slots = [], out_slots = [], tool=None):
        super().__init__(name)
        self.in_slots = in_slots
        self.out_slots = out_slots
        self.tool = tool
        if tool is not None:
            self._check_signature(name, tool, in_slots)

    def can_execute(self, blackboard: Blackboard):
        return all(blackboard.get(slot) is not None for slot in self.in_slots) and \
               all(blackboard.get(slot) is None for slot in self.out_slots)

    def execute(self, blackboard: Blackboard, user_input=None):
        inputs = {slot: blackboard.get(slot) for slot in self.in_slots}
        result = self.tool(**inputs)
        assert isinstance(result, dict), "Expected tool to return a dictionary"
        for slot, value in result.items():
            assert slot in self.out_slots, f"Unexpected output slot: {slot}"
            blackboard.update(slot, value)

    def execute(self, blackboard: Blackboard, user_input=None):
        if self.tool is None:
            raise NotImplementedError("ToolAgent must implement the execute method.")
        inputs = {slot: blackboard.get(slot) for slot in self.in_slots}
        result = self.tool(**inputs)
        if len(self.out_slots) == 1:
            result = {self.out_slots[0]: result}
        assert isinstance(result, dict), \
            f"{self.name}: tool with several out_slots must return a dict"
        for slot, value in result.items():
            assert slot in self.out_slots, f"Unexpected output slot: {slot}"
            blackboard.update(slot, value)

    def _check_signature(self, name, tool, in_slots):
        """Fail at load time, if slots and parameters disagree."""
        # Get tool function parameters
        params = inspect.signature(tool).parameters
        unknown = [s for s in in_slots if s not in params]
        missing = [p for p, v in params.items()
            if v.default is inspect.Parameter.empty and p not in in_slots]
        if unknown or missing:
            raise ValueError(f"{name}: in_slots {list(in_slots)} do not match tool parameters")
