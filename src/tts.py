import ctypes
import sys
import tempfile
from pathlib import Path

from gtts import gTTS

SUPPORTED_LANGS = {"sv", "en"}


def speak(text: str, lang: str = "sv") -> None:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language: {lang!r}. Use 'sv' or 'en'.")
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        tts.save(str(tmp_path))
        _play(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _play(path: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Audio playback is only supported on Windows.")
    mci = ctypes.windll.winmm.mciSendStringW  # type: ignore[attr-defined]
    abs_path = str(path.resolve())
    mci(f'open "{abs_path}" type mpegvideo alias tts_track', None, 0, None)
    mci("play tts_track wait", None, 0, None)
    mci("close tts_track", None, 0, None)
