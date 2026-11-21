from agora.tools import tool, lookup

def test_tool_registration():
    @tool
    def square(x):
        return x * x

    foo = lookup("square")
    assert foo(3) == 9

def test_tool_lookup_failure():
    try:
        lookup("foobar")
    except KeyError as e:
        assert "Unknown tool 'foobar'" in str(e)

test_tool_registration()
test_tool_lookup_failure()