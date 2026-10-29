from agora.llm import ChatgptLLMProvider, ClaudeLLMProvider, LocalLLMProvider
from agora.llm import ConversationManager

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
    assert len(history) == 3  # only the last 3 messages are kept

def test_local_llm_provider():
    print("test_local_llm_provider")
    # Connect to the local fm model server on a Mac
    llm = LocalLLMProvider(server_url="http://127.0.0.1:1976/v1", model="system")
    response = llm.complete([{"role": "user", "content": "Write a short fall poem."}])
    print("Response:", response)

test_llm_provider()
test_conversation_manager()
test_local_llm_provider()
