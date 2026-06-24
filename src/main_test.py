import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture


def set_argv(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"] + args)


def setup_chat_loop(
    mocker: MockerFixture,
    index_exists: bool = True,
    questions: list[str] | None = None,
) -> tuple[MagicMock, MagicMock]:
    if questions is None:
        questions = ["quit"]
    mock_index = mocker.patch("main.INDEX_PATH")
    mock_index.exists.return_value = index_exists
    mock_build = mocker.patch("main.build_index")
    mock_prompt = mocker.patch("main.Prompt")
    mock_prompt.ask.side_effect = questions
    mocker.patch("main.retrieve", return_value=["chunk"])
    mock_ask = mocker.patch("main.ask", return_value="an answer")
    mocker.patch("main.console")
    return mock_build, mock_ask


def test_no_args_exits_with_error_code(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    set_argv(monkeypatch, [])
    mock_index = mocker.patch("main.INDEX_PATH")
    mock_index.exists.return_value = False
    with pytest.raises(SystemExit) as exc:
        from main import main

        main()
    assert exc.value.code == 1


def test_reset_only_calls_reset_index(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    set_argv(monkeypatch, ["--reset"])
    mock_reset = mocker.patch("main.reset_index")
    from main import main

    main()
    mock_reset.assert_called_once()


def test_reset_only_does_not_start_chat_loop(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    set_argv(monkeypatch, ["--reset"])
    mocker.patch("main.reset_index")
    mock_chat = mocker.patch("main.chat_loop")
    from main import main

    main()
    mock_chat.assert_not_called()


def test_missing_file_exits_with_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_argv(monkeypatch, [str(tmp_path / "nonexistent.pdf")])
    with pytest.raises(SystemExit) as exc:
        from main import main

        main()
    assert exc.value.code == 1


def test_single_file_starts_chat_loop(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, sample_pdf: str
) -> None:
    set_argv(monkeypatch, [sample_pdf])
    mock_chat = mocker.patch("main.chat_loop")
    from main import main

    main()
    mock_chat.assert_called_once_with(
        [sample_pdf], speak_aloud=False, lang="sv"
    )


def test_multiple_files_passes_all_to_chat_loop(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    sample_pdf: str,
    another_pdf: str,
) -> None:
    set_argv(monkeypatch, [sample_pdf, another_pdf])
    mock_chat = mocker.patch("main.chat_loop")
    from main import main

    main()
    mock_chat.assert_called_once_with(
        [sample_pdf, another_pdf], speak_aloud=False, lang="sv"
    )


def test_multiple_files_exits_if_any_missing(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    sample_pdf: str,
    tmp_path: Path,
) -> None:
    missing = str(tmp_path / "missing.pdf")
    set_argv(monkeypatch, [sample_pdf, missing])
    mocker.patch("main.chat_loop")
    with pytest.raises(SystemExit) as exc:
        from main import main

        main()
    assert exc.value.code == 1


def test_reset_with_file_resets_before_starting_chat(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, sample_pdf: str
) -> None:
    set_argv(monkeypatch, ["--reset", sample_pdf])
    mock_reset = mocker.patch("main.reset_index")
    mock_chat = mocker.patch("main.chat_loop")
    from main import main

    main()
    mock_reset.assert_called_once()
    mock_chat.assert_called_once()


def test_reset_with_file_reset_happens_before_chat_loop(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, sample_pdf: str
) -> None:
    set_argv(monkeypatch, ["--reset", sample_pdf])
    call_order: list[str] = []
    mocker.patch(
        "main.reset_index",
        side_effect=lambda: call_order.append("reset"),
    )
    mocker.patch(
        "main.chat_loop",
        side_effect=lambda *a, **kw: call_order.append("chat"),
    )
    from main import main

    main()
    assert call_order == ["reset", "chat"]


def test_chat_loop_builds_index_when_missing(
    mocker: MockerFixture, sample_pdf: str
) -> None:
    mock_build, _ = setup_chat_loop(mocker, index_exists=False)
    from main import chat_loop

    chat_loop([sample_pdf])
    mock_build.assert_called_once_with([sample_pdf])


def test_chat_loop_skips_build_when_index_exists(
    mocker: MockerFixture, sample_pdf: str
) -> None:
    mock_build, _ = setup_chat_loop(mocker, index_exists=True)
    from main import chat_loop

    chat_loop([sample_pdf])
    mock_build.assert_not_called()


def test_chat_loop_quit_exits_loop(
    mocker: MockerFixture, sample_pdf: str
) -> None:
    setup_chat_loop(mocker, questions=["quit"])
    from main import chat_loop

    chat_loop([sample_pdf])


def test_chat_loop_question_calls_retrieve_and_ask(
    mocker: MockerFixture, sample_pdf: str
) -> None:
    _, mock_ask = setup_chat_loop(mocker, questions=["what is this?", "quit"])
    mock_retrieve = mocker.patch("main.retrieve", return_value=["chunk"])
    from main import chat_loop

    chat_loop([sample_pdf])
    mock_retrieve.assert_called_once_with("what is this?")
    mock_ask.assert_called_once_with("what is this?", ["chunk"])
