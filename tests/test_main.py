import sys
import pytest


def run_main(monkeypatch, args: list[str]):
    monkeypatch.setattr(sys, "argv", ["main.py"] + args)


class TestNoArgs:
    def test_exits_with_error_code(self, monkeypatch):
        run_main(monkeypatch, [])
        with pytest.raises(SystemExit) as exc:
            from main import main
            main()
        assert exc.value.code == 1


class TestResetOnly:
    def test_calls_reset_index(self, monkeypatch, mocker):
        run_main(monkeypatch, ["--reset"])
        mock_reset = mocker.patch("ingest.reset_index")
        from main import main
        main()
        mock_reset.assert_called_once()

    def test_does_not_start_chat_loop(self, monkeypatch, mocker):
        run_main(monkeypatch, ["--reset"])
        mocker.patch("ingest.reset_index")
        mock_chat = mocker.patch("main.chat_loop")
        from main import main
        main()
        mock_chat.assert_not_called()


class TestMissingFile:
    def test_exits_with_error_code(self, monkeypatch, tmp_path):
        run_main(monkeypatch, [str(tmp_path / "nonexistent.pdf")])
        with pytest.raises(SystemExit) as exc:
            from main import main
            main()
        assert exc.value.code == 1


class TestSingleFile:
    def test_starts_chat_loop(self, monkeypatch, mocker, sample_pdf):
        run_main(monkeypatch, [sample_pdf])
        mock_chat = mocker.patch("main.chat_loop")
        from main import main
        main()
        mock_chat.assert_called_once_with([sample_pdf])


class TestMultipleFiles:
    def test_passes_all_files_to_chat_loop(
        self, monkeypatch, mocker, sample_pdf, another_pdf
    ):
        run_main(monkeypatch, [sample_pdf, another_pdf])
        mock_chat = mocker.patch("main.chat_loop")
        from main import main
        main()
        mock_chat.assert_called_once_with([sample_pdf, another_pdf])

    def test_exits_if_any_file_is_missing(
        self, monkeypatch, mocker, sample_pdf, tmp_path
    ):
        missing = str(tmp_path / "missing.pdf")
        run_main(monkeypatch, [sample_pdf, missing])
        mocker.patch("main.chat_loop")
        with pytest.raises(SystemExit) as exc:
            from main import main
            main()
        assert exc.value.code == 1


class TestResetWithFile:
    def test_resets_before_starting_chat(
        self, monkeypatch, mocker, sample_pdf
    ):
        run_main(monkeypatch, ["--reset", sample_pdf])
        mock_reset = mocker.patch("ingest.reset_index")
        mock_chat = mocker.patch("main.chat_loop")
        from main import main
        main()
        mock_reset.assert_called_once()
        mock_chat.assert_called_once()

    def test_reset_happens_before_chat_loop(
        self, monkeypatch, mocker, sample_pdf
    ):
        run_main(monkeypatch, ["--reset", sample_pdf])
        call_order = []
        mocker.patch(
            "ingest.reset_index",
            side_effect=lambda: call_order.append("reset"),
        )
        mocker.patch(
            "main.chat_loop",
            side_effect=lambda _: call_order.append("chat"),
        )
        from main import main
        main()
        assert call_order == ["reset", "chat"]
