from agora.agent import Agent, Blackboard, Controller
from agora.llm import LLMProvider, ClaudeLLMProvider, ResultParser

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
    """Tool agent that performs a specific tool action using input slots."""

    def __init__(self, name: str, in_slots = [], out_slots = []):
        super().__init__(name)
        self.in_slots = in_slots
        self.out_slots = out_slots

    def can_execute(self, blackboard):
        return all(blackboard.get(slot) is not None for slot in self.in_slots) and \
               all(blackboard.get(slot) is None for slot in self.out_slots)

    def execute(self, blackboard: Blackboard, user_input=None):
        raise NotImplementedError("ToolAgent must implement the execute method.")
