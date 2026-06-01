from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import chat


@pytest.fixture(autouse=True)
def reset_client() -> Generator[None, None, None]:
    chat._get_client.cache_clear()
    yield
    chat._get_client.cache_clear()


def mock_groq(
    mocker: MockerFixture, reply: str = "mocked answer"
) -> MagicMock:
    mock_response = mocker.MagicMock()
    mock_response.choices[0].message.content = reply
    mock_client = mocker.MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("chat.Groq", return_value=mock_client)
    mocker.patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    return mock_client


def test_ask_returns_model_response(mocker: MockerFixture):
    mock_groq(mocker, reply="The answer is 42.")
    result = chat.ask("What is the answer?", ["context"])
    assert result == "The answer is 42."


def test_ask_prompt_contains_question(mocker: MockerFixture):
    client = mock_groq(mocker)
    chat.ask("How many vacation days?", ["some context"])
    call = client.chat.completions.create.call_args
    prompt = call.kwargs["messages"][0]["content"]
    assert "How many vacation days?" in prompt


def test_ask_prompt_contains_context(mocker: MockerFixture):
    client = mock_groq(mocker)
    chat.ask("question", ["important context chunk"])
    call = client.chat.completions.create.call_args
    prompt = call.kwargs["messages"][0]["content"]
    assert "important context chunk" in prompt


def test_ask_prompt_joins_multiple_chunks(mocker: MockerFixture):
    client = mock_groq(mocker)
    chat.ask("question", ["chunk one", "chunk two"])
    call = client.chat.completions.create.call_args
    prompt = call.kwargs["messages"][0]["content"]
    assert "chunk one" in prompt
    assert "chunk two" in prompt


def test_ask_prompt_separates_chunks(mocker: MockerFixture):
    client = mock_groq(mocker)
    chat.ask("question", ["first", "second"])
    call = client.chat.completions.create.call_args
    prompt = call.kwargs["messages"][0]["content"]
    assert "---" in prompt


def test_ask_uses_correct_model(mocker: MockerFixture):
    client = mock_groq(mocker)
    chat.ask("question", ["ctx"])
    call = client.chat.completions.create.call_args
    assert call.kwargs["model"] == chat.MODEL


def test_ask_uses_low_temperature(mocker: MockerFixture):
    client = mock_groq(mocker)
    chat.ask("question", ["ctx"])
    call = client.chat.completions.create.call_args
    assert call.kwargs["temperature"] < 0.5


def test_ask_handles_empty_context(mocker: MockerFixture):
    mock_groq(mocker, reply="No context provided.")
    result = chat.ask("question", [])
    assert isinstance(result, str)


def test_get_client_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        chat._get_client()
