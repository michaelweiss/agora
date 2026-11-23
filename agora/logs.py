import json
import logging
import os
from datetime import datetime

logger = logging.getLogger("agora")

class TraceHandler(logging.FileHandler):
    """Writes trace records"""

    def emit(self, record):
        if hasattr(record, "trace"):
            with self.lock:
                self.stream.write(json.dumps(record.trace, default=str) + "\n")
                self.flush()

def configure_logging(level=logging.INFO, trace_file=None):
    """Log messages to the console and optionally to a trace file"""
    logger.setLevel(min(level, logging.INFO) if trace_file else level)
    logger.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)
    if trace_file:
        os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
        logger.addHandler(TraceHandler(trace_file))

def trace(agent, inputs, outputs):
    """Record an agent execution at an observation point. Traces can be replayed in tests"""
    logger.info("%s executed", agent, extra={"trace": {
        "agent": agent,
        "time": datetime.now().isoformat(),
        "inputs": inputs,
        "outputs": outputs}})

def read_trace(trace_file, agent=None):
    """Read the recorded executions. Optionally only those of one agent"""
    with open(trace_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if agent is None or r["agent"] == agent]
