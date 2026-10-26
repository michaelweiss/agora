import json
import logging
import re
import threading
import requests

import anthropic
import openai

# LLM provider interface
class LLMProvider():
    """Base class for LLM providers"""
    
    def complete(self, messages, max_tokens = None, temperature = 0.7):
        """Generate a response from the LLM"""
        raise NotImplementedError("Need to implement complete method")

# Implementation of various LLM providers
class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def complete(self, messages, max_tokens = None, temperature = 0.7):
        last_message = messages[-1]["content"] if messages else ""
        return f"Mock response to: {last_message[:50]}..."
    
class ClaudeLLMProvider(LLMProvider):
    """LLM provider using Anthropic's Claude model"""
    
    def __init__(self, model="claude-haiku-4-5-20251001", max_tokens=1024,
            system="You are a helpful assistant."):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
    
    def complete(self, messages, max_tokens=None, temperature=0.7):
        # anthropic SDK 2.x no longer supports temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            messages=messages
        )
        return response.content[0].text

class ChatgptLLMProvider(LLMProvider):
    """LLM provider using OpenAI's GPT models"""

    def __init__(self, model="gpt-4o-mini", max_tokens=1024, 
            system="You are a helpful assistant. Carefully follow the user's instructions."):
        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
    
    def complete(self, messages, max_tokens=None, temperature=0.7):
        if max_tokens is None:
            max_tokens = self.max_tokens
        full_messages = [{"role": "system", "content": self.system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=full_messages
        )
        return response.choices[0].message.content
    
class LocalLLMProvider(LLMProvider):
    """LLM provider using a local server (e.g., ollama)"""
    
    def __init__(self, model="gemma3:1b", server_url="http://localhost:11434", max_tokens=1024, 
            system="You are a helpful assistant. Carefully follow the user's instructions.", timeout=120):
        self.model = model
        self.server_url = server_url
        self.max_tokens = max_tokens
        self.system = system
        self.timeout = timeout
    
    def complete(self, messages, max_tokens=None, temperature=0.7):
        if max_tokens is None:
            max_tokens = self.max_tokens
        full_messages = [{"role": "system", "content": self.system}] + messages
        
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": full_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logging.error(f"Error from LLM server: {response.status_code} {response.text}")
                return f"Error: {response.status_code} {response.text}"
        except requests.exceptions.Timeout:
            logging.error(f"Request timed out after {self.timeout} seconds")
            return f"Error: Request timed out after {self.timeout} seconds"
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error: {e}")
            return f"Error: Cannot connect to {self.server_url}. Make sure ollama is running."
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return f"Error: {e}"

# Conversation manager for handling history
class ConversationManager:
    """Manages a conversation with an LLM provider"""
    
    def __init__(self, llm_provider, max_history = 10):
        self.llm_provider = llm_provider
        self.max_history = max_history
        self.conversation_history = []
        self._lock = threading.Lock()
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        with self._lock:
            self.conversation_history.append({"role": role, "content": content})
            # Trim history if too long
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
    
    def complete(self, user_message, **llm_kwargs):
        """Generate a response and update the conversation history"""
        self.add_message("user", user_message)
        with self._lock:
            response = self.llm_provider.complete(
                self.conversation_history, **llm_kwargs
            )
        self.add_message("assistant", response)
        return response
    
    def get_history(self):
        """Get a copy of the current conversation history"""
        return self.conversation_history.copy()

# Result parser for extracting JSON from responses
class ResultParser:
    """Parses results from LLM responses"""
    
    def extract_raw_code_blocks(self, response):
        """Extract code blocks fenced by triple backticks from the response"""
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        return matches

    def parse_json(self, code_block):
        """Extract and parse JSON objects from a code block"""
        try:
            return json.loads(code_block)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None
        
    def parse(self, response):
        """Parse the response to extract code blocks and JSON objects"""
        code_blocks = self.extract_raw_code_blocks(response)
        json_objects = []
        python_objects = []
        for block in code_blocks:
            if block.startswith("json"):
                block = block[len("json\n"):].strip()
                json = self.parse_json(block)
                if json is not None:
                    json_objects.append(json)
            elif block.startswith("python"):
                block = block[len("python\n"):].strip()
                python_objects.append(block)
        return {
            "json": json_objects,
            "python": python_objects
        }