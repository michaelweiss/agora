from agora.agent import Blackboard
from agora.llm import ClaudeLLMProvider
from agora.roles import TaskAgent

import yaml

def create_task_agent(spec_file, llm_provider=None):
    with open(spec_file) as f:
        spec = yaml.safe_load(f)

    return TaskAgent(
        name=spec["role"],
        llm_provider=llm_provider or ClaudeLLMProvider(),
        in_slots=spec["in_slots"],
        out_slots=spec["out_slots"],
        prompt=spec["prompt"],
    )

def make_summarizer():
    return create_task_agent("summarizer.yaml",
        llm_provider=ClaudeLLMProvider())

def test_summarizer():
    summarizer = make_summarizer()
    blackboard = Blackboard({
        "text": None,
        "summary": None,
    })
    blackboard.update("text", """
The term agent is derived from the Latin agere (to do): an agreement 
to act on one's behalf. Such "action on behalf of" implies the authority 
to decide which, if any, action is appropriate.[1][2] Some agents are 
colloquially known as bots, from robot. Source: Wikipedia
""")
    summarizer.execute(blackboard)
    assert blackboard.get("summary") is not None
    print("Summary:", blackboard.get("summary"))

# Run tests
test_summarizer()