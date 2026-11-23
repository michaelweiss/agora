from .agent import Agent, AgentError, Blackboard, Controller
from .llm import LLMProvider, ClaudeLLMProvider, ResultParser
from .logs import trace

import inspect
import logging

logger = logging.getLogger("agora")

def check_and_write_outputs(name, blackboard, outputs, out_slots):
    """Check and write outputs to the blackboard"""
    if not isinstance(outputs, dict):
        raise AgentError(f"{name}: expected a dict of outputs, got {type(outputs).__name__}")
    unexpected = [slot for slot in outputs if slot not in out_slots]
    if unexpected:
        raise AgentError(f"{name}: unexpected output slots {unexpected}")
    missing = [slot for slot in out_slots if slot not in outputs]
    if missing:
        raise AgentError(f"{name}: missing output slots {missing}")
    for slot, value in outputs.items():
        blackboard.update(slot, value)

class TaskAgent(Agent):
    """Task agent that performs a specific task using prompt."""

    def __init__(self, name: str, llm_provider: LLMProvider,
                 in_slots = [], out_slots = [], prompt = ""):
        super().__init__(name, llm_provider)
        self.in_slots = list(in_slots)
        self.out_slots = list(out_slots)
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
        if not result["json"]:
            raise AgentError(f"{self.name}: expected a ```json block in the response")
        outputs = result["json"][0]
        check_and_write_outputs(self.name, blackboard, outputs, self.out_slots)
        trace(self.name, inputs, outputs)

class ToolAgent(Agent):
    """Tool agent that calls a tool."""

    def __init__(self, name: str, in_slots = [], out_slots = [], tool=None):
        super().__init__(name)
        self.in_slots = list(in_slots)
        self.out_slots = list(out_slots)
        self.tool = tool
        if tool is not None:
            self._check_signature(name, tool, in_slots)

    def can_execute(self, blackboard: Blackboard):
        return all(blackboard.get(slot) is not None for slot in self.in_slots) and \
               all(blackboard.get(slot) is None for slot in self.out_slots)

    def execute(self, blackboard: Blackboard, user_input=None):
        if self.tool is None:
            raise NotImplementedError("ToolAgent must implement the execute method.")
        inputs = {slot: blackboard.get(slot) for slot in self.in_slots}
        outputs = self.tool(**inputs)
        if len(self.out_slots) == 1:
            outputs = {self.out_slots[0]: outputs}
        check_and_write_outputs(self.name, blackboard, outputs, self.out_slots)
        trace(self.name, inputs, outputs)

    def _check_signature(self, name, tool, in_slots):
        """Fail at load time, if slots and parameters disagree."""
        # Get tool function parameters
        params = inspect.signature(tool).parameters
        unknown = [s for s in in_slots if s not in params]
        missing = [p for p, v in params.items()
            if v.default is inspect.Parameter.empty and p not in in_slots]
        if unknown or missing:
            raise ValueError(f"{name}: in_slots {list(in_slots)} do not match tool parameters")
