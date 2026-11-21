"""
Build agents from yaml specs. An agent spec is a mapping with `role`, `in_slots`, 
`out_slots` and either `prompt` (task agent) or `tool` (tool agent).
"""

import yaml

from .llm import ClaudeLLMProvider
from .roles import TaskAgent, ToolAgent
from .tools import lookup

def load_yaml(path):
    """Safely load a yaml file."""
    with open(path) as f:
        return yaml.safe_load(f)

def load_agent(file_path, llm_provider=None):
    """Create an agent from spec."""
    spec = load_yaml(file_path)
    if "in_slots" not in spec or "out_slots" not in spec:
        raise ValueError(f"Spec of '{spec.get('role')}' agent needs 'in_slots' and 'out_slots'")
    kinds = [k for k in ("prompt", "tool") if k in spec]
    if len(kinds) != 1:
        raise ValueError(f"Spec of '{spec.get('role')}' agent needs 'prompt' or 'tool'")
    common = {
        "name": spec["role"],
        "in_slots": spec["in_slots"],
        "out_slots": spec["out_slots"]
    }
    if "tool" in spec:
        return ToolAgent(tool=lookup(spec["tool"]), **common)
    else:
        llm_provider = llm_provider or ClaudeLLMProvider()
        return TaskAgent(prompt=spec["prompt"], llm_provider=llm_provider, **common)