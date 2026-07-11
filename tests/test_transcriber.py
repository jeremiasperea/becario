"""Tests del transcriptor faster-whisper (con modelo falso: sin descargas)."""
from becario.infrastructure.whisper_transcriber import FasterWhisperTranscriber


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        return iter(self._segments), {"language": "es"}


class ExplodingModel:
    def transcribe(self, audio, **kwargs):
        raise RuntimeError("boom")


def _with_model(model) -> FasterWhisperTranscriber:
    t = FasterWhisperTranscriber()
    t._model = model  # evita la carga perezosa real
    return t


class TestFasterWhisperTranscriber:
    def test_joins_segments_and_strips(self):
        t = _with_model(FakeModel([FakeSegment(" hola "), FakeSegment("becario ")]))
        assert t.transcribe(b"ogg-bytes") == "hola becario"

    def test_empty_audio_short_circuits(self):
        t = _with_model(ExplodingModel())  # no debería llegar a usarlo
        assert t.transcribe(b"") == ""

    def test_model_failure_returns_empty(self):
        t = _with_model(ExplodingModel())
        assert t.transcribe(b"ogg-bytes") == ""

    def test_no_speech_returns_empty(self):
        t = _with_model(FakeModel([]))
        assert t.transcribe(b"ogg-bytes") == ""
