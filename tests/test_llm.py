from agora.llm import ResultParser
from agora.llm import ConversationManager, ChatgptLLMProvider, ClaudeLLMProvider

def test_llm_provider():
    print("test_llm_provider")
    llm = ChatgptLLMProvider()
    response = llm.complete([{"role": "user", "content": "Hello!"}])
    print("Response:", response)
    
def test_conversation_manager():
    print("test_conversation_manager")
    llm = ConversationManager(ClaudeLLMProvider(), max_history=3)
    response1 = llm.complete("What is the capital of Canada?")
    print("Response 1:", response1)
    response2 = llm.complete("What's the weather like there in the Fall?")
    print("Response 2:", response2)
    history = llm.get_history()
    assert len(history) == 4  # 2 user + 2 assistant messages

test_llm_provider()
test_conversation_manager()
