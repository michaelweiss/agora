"""
Tool registry for Python functions. Allows functions to be exposed as 
tools that can be looked up by name.

Usage:
    @tool
    def my_function(...):
        ...
"""
TOOL_REGISTRY = {}

def tool(function):
    """Register fn under its function name and return it unchanged."""
    name = function.__name__
    previous = TOOL_REGISTRY.get(name)
    # Registering the same function again is fine (module reloads, Streamlit
    # reruns), but two different functions under one name is an error
    if previous is not None and _qualified(previous) != _qualified(function):
        raise ValueError(f"Tool '{name}' already registered by {_qualified(previous)}")
    TOOL_REGISTRY[name] = function
    return function

def lookup(name):
    """Look up a tool by name."""
    try:
        return TOOL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown tool '{name}'") from None

def _qualified(fn):
    """Return the name of a function combined with its module name."""
    return f"{fn.__module__}.{fn.__qualname__}"
