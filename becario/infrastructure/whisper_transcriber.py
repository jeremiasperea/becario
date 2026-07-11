"""Transcripción de audio local con faster-whisper (implementa Transcriber).

Gratis y sin servicios externos: el modelo Whisper corre en CPU vía
CTranslate2. La primera transcripción descarga el modelo (una sola vez,
queda en ~/.cache/huggingface) y lo carga en memoria; después queda
caliente. El OGG/Opus de las notas de voz de Telegram se decodifica
directo (PyAV), sin necesitar ffmpeg instalado.
"""
from __future__ import annotations

import io
import logging
import threading
import time

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """`transcribe(audio_bytes) -> str` sobre faster-whisper.

    `model_size`: tiny/base/small/medium… ("small" da buen castellano con
    ~700 MB de RAM en int8; "base" es más liviano si la máquina va justa).
    """

    def __init__(self, model_size: str = "small", language: str = "es") -> None:
        self._model_size = model_size
        self._language = language
        self._model = None
        # La carga es perezosa (no frena el arranque del bot) y con lock:
        # dos notas de voz simultáneas no deben cargar el modelo dos veces.
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                t0 = time.time()
                logger.info(
                    "Cargando modelo Whisper %r (la primera vez también lo descarga)…",
                    self._model_size,
                )
                self._model = WhisperModel(
                    self._model_size, device="cpu", compute_type="int8"
                )
                logger.info("Whisper %r listo en %.1f s", self._model_size, time.time() - t0)
            return self._model

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        try:
            model = self._get_model()
            segments, _info = model.transcribe(
                io.BytesIO(audio_bytes),
                language=self._language or None,
                vad_filter=True,  # recorta silencios: menos alucinaciones
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.info("Audio transcripto (%d bytes): %r", len(audio_bytes), text[:120])
            return text
        except Exception as exc:
            logger.error("Fallo transcribiendo audio: %s", exc)
            return ""
