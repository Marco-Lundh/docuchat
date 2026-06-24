import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


def test_speak_raises_for_unsupported_lang() -> None:
    from tts import speak

    with pytest.raises(ValueError, match="Unsupported language"):
        speak("hello", lang="de")


def test_speak_calls_gtts_with_correct_lang(tmp_path: Path) -> None:
    mp3 = tmp_path / "out.mp3"

    with (
        patch("tts.gTTS") as mock_gtts,
        patch("tts._play"),
        patch("tts.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value.name = str(mp3)
        mp3.write_bytes(b"")
        from tts import speak

        speak("hej världen", lang="sv")

    mock_gtts.assert_called_once_with(text="hej världen", lang="sv")


def test_speak_saves_and_plays(tmp_path: Path) -> None:
    mp3 = tmp_path / "out.mp3"

    with (
        patch("tts.gTTS") as mock_gtts,
        patch("tts._play") as mock_play,
        patch("tts.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value.name = str(mp3)
        mp3.write_bytes(b"")
        mock_instance = MagicMock()
        mock_gtts.return_value = mock_instance
        from tts import speak

        speak("hello", lang="en")

    mock_instance.save.assert_called_once_with(str(mp3))
    mock_play.assert_called_once_with(mp3)


def test_speak_cleans_up_temp_file(tmp_path: Path) -> None:
    mp3 = tmp_path / "out.mp3"
    mp3.write_bytes(b"")

    with (
        patch("tts.gTTS"),
        patch("tts._play"),
        patch("tts.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value.name = str(mp3)
        from tts import speak

        speak("test", lang="sv")

    assert not mp3.exists()


def test_speak_cleans_up_even_if_play_raises(tmp_path: Path) -> None:
    mp3 = tmp_path / "out.mp3"
    mp3.write_bytes(b"")

    with (
        patch("tts.gTTS"),
        patch("tts._play", side_effect=RuntimeError("playback failed")),
        patch("tts.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value.name = str(mp3)
        from tts import speak

        with pytest.raises(RuntimeError, match="playback failed"):
            speak("test", lang="sv")

    assert not mp3.exists()


def test_play_raises_on_non_windows() -> None:
    from tts import _play

    with patch.object(sys, "platform", "linux"):
        with pytest.raises(RuntimeError, match="only supported on Windows"):
            _play(Path("/tmp/test.mp3"))


def test_play_calls_mci_in_order(tmp_path: Path) -> None:
    mp3 = tmp_path / "track.mp3"
    mp3.write_bytes(b"")
    abs_path = str(mp3.resolve())

    mock_mci = MagicMock()
    with (
        patch.object(sys, "platform", "win32"),
        patch("tts.ctypes") as mock_ctypes,
    ):
        mock_ctypes.windll.winmm.mciSendStringW = mock_mci
        from tts import _play

        _play(mp3)

    assert mock_mci.call_args_list == [
        call(
            f'open "{abs_path}" type mpegvideo alias tts_track', None, 0, None
        ),
        call("play tts_track wait", None, 0, None),
        call("close tts_track", None, 0, None),
    ]
