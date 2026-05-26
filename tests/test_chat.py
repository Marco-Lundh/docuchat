import pytest
import chat


@pytest.fixture(autouse=True)
def reset_client():
    chat._client = None
    yield
    chat._client = None


def mock_groq(mocker, reply: str = "mocked answer"):
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = reply
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("chat.Groq", return_value=mock_client)
    mocker.patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    return mock_client


class TestAsk:
    def test_returns_model_response(self, mocker):
        mock_groq(mocker, reply="The answer is 42.")
        result = chat.ask("What is the answer?", ["context"])
        assert result == "The answer is 42."

    def test_prompt_contains_question(self, mocker):
        client = mock_groq(mocker)
        chat.ask("How many vacation days?", ["some context"])
        call = client.chat.completions.create.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "How many vacation days?" in prompt

    def test_prompt_contains_context(self, mocker):
        client = mock_groq(mocker)
        chat.ask("question", ["important context chunk"])
        call = client.chat.completions.create.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "important context chunk" in prompt

    def test_prompt_joins_multiple_chunks(self, mocker):
        client = mock_groq(mocker)
        chat.ask("question", ["chunk one", "chunk two"])
        call = client.chat.completions.create.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "chunk one" in prompt
        assert "chunk two" in prompt

    def test_prompt_separates_chunks(self, mocker):
        client = mock_groq(mocker)
        chat.ask("question", ["first", "second"])
        call = client.chat.completions.create.call_args
        prompt = call.kwargs["messages"][0]["content"]
        assert "---" in prompt

    def test_uses_correct_model(self, mocker):
        client = mock_groq(mocker)
        chat.ask("question", ["ctx"])
        call = client.chat.completions.create.call_args
        assert call.kwargs["model"] == chat.MODEL

    def test_uses_low_temperature(self, mocker):
        client = mock_groq(mocker)
        chat.ask("question", ["ctx"])
        call = client.chat.completions.create.call_args
        assert call.kwargs["temperature"] < 0.5

    def test_handles_empty_context(self, mocker):
        mock_groq(mocker, reply="No context provided.")
        result = chat.ask("question", [])
        assert isinstance(result, str)
