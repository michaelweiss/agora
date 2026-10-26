from agora.llm import ResultParser

SINGLE_JSON_RESPONSE = """
```json
{
    "key": "value",
    "number": 42
}
``` 
"""

MULTIPLE_JSON_RESPONSE = """
```json 
[
    {
        "key": "value1",
        "number": 42
    },
    {
        "key": "value2",
        "number": 24
    }
]
``` 
"""

RAW_CODE_BLOCKS_RESPONSE = """
Here is some text.
```python
def foo():
    return "bar"
```
```json
[
    {"key": "value1"},
    {"key": "value2"}
]
```
"""

EMPTY_JSON_LIST_RESPONSE = """
```json
[]
```
"""

def test_parse_single_json():
    print("test_parse_single_json")
    parser = ResultParser()
    result = parser.parse(SINGLE_JSON_RESPONSE)
    assert len(result["json"]) == 1
    assert result["json"][0]["key"] == "value"
    assert result["json"][0]["number"] == 42

def test_parse_multiple_json():
    print("test_parse_multiple_json")
    parser = ResultParser()
    result = parser.parse(MULTIPLE_JSON_RESPONSE)
    assert len(result["json"]) == 1
    assert len(result["json"][0]) == 2
    assert result["json"][0][0]["key"] == "value1"
    assert result["json"][0][0]["number"] == 42
    assert result["json"][0][1]["key"] == "value2"
    assert result["json"][0][1]["number"] == 24   

def test_parse_raw_code_blocks():
    print("test_parse_raw_code_blocks")
    parser = ResultParser()
    result = parser.extract_raw_code_blocks(RAW_CODE_BLOCKS_RESPONSE)
    print("Parsed result:", result)
    assert len(result) == 2   

def test_parse_empty_json_list():
    print("test_parse_empty_json_list")
    parser = ResultParser()
    result = parser.parse(EMPTY_JSON_LIST_RESPONSE)
    print("Parsed result:", result)
    assert len(result["json"]) == 1
    assert result["json"][0] == []

test_parse_single_json()
test_parse_multiple_json()
test_parse_raw_code_blocks()
test_parse_empty_json_list()
