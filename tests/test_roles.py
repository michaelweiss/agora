from agora.agent import Blackboard
from agora.llm import ClaudeLLMProvider
from agora.roles import TaskAgent, ToolAgent
from agora.specs import load_agent
from agora.tools import tool

import yaml

def test_task_agent():
    summarizer = load_agent("summarizer.yaml",
        llm_provider=ClaudeLLMProvider())
    blackboard = Blackboard()
    blackboard.update("text", """
The term agent is derived from the Latin agere (to do): an agreement 
to act on one's behalf. Such "action on behalf of" implies the authority 
to decide which, if any, action is appropriate.[1][2] Some agents are 
colloquially known as bots, from robot. Source: Wikipedia
""")
    summarizer.execute(blackboard)
    assert blackboard.get("summary") is not None
    print("Summary:", blackboard.get("summary"))

@tool
def get_weather(location):
    return f"It's sunny in {location} today"

@tool
def complex_tool(input1, input2):
    return {
        "output1": f"Processed {input1}", 
        "output2": f"Processed {input2}"
    }

def test_tool_agent():
    tool_agent = load_agent("weather.yaml")
    blackboard = Blackboard()
    blackboard.update("location", "Ottawa")
    tool_agent.execute(blackboard)
    assert blackboard.get("weather") is not None
    print("Weather:", blackboard.get("weather"))

def test_tool_agent_complex_output():
    complex_tool_agent = load_agent("complex_tool.yaml")
    blackboard = Blackboard()
    blackboard.update("input1", "value1")
    blackboard.update("input2", "value2")
    complex_tool_agent.execute(blackboard)
    assert blackboard.get("output1") is not None
    assert blackboard.get("output2") is not None
    print("Output1:", blackboard.get("output1"))
    print("Output2:", blackboard.get("output2"))

# Run tests
test_task_agent()
test_tool_agent()
test_tool_agent_complex_output()