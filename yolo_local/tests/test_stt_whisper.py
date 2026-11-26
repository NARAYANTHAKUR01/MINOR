"""
Automated Unit Tests for Local Open-Source Whisper Speech-to-Text & STTManager.
"""

import math
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from backend.services.stt_service import (
    MockSTTService,
    WhisperSTTService,
    get_stt_manager,
)


class TestWhisperSTTService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_stt_"))
        cls.synthetic_wav = cls.temp_dir / "synthetic_tone.wav"
        cls._create_synthetic_wav(cls.synthetic_wav, duration=0.5, freq=440.0)

        # Check if macOS 'say' command is available to create real speech audio
        cls.speech_wav = None
        if shutil.which("say"):
            wav_path = cls.temp_dir / "speech_test.wav"
            try:
                subprocess.run(
                    ["say", "-o", str(wav_path), "--data-format=LEI16@16000", "find the bottle and cup"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if wav_path.exists():
                    cls.speech_wav = wav_path
            except Exception:
                cls.speech_wav = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @staticmethod
    def _create_synthetic_wav(path: Path, duration: float = 0.5, freq: float = 440.0, sample_rate: int = 16000):
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            n_samples = int(duration * sample_rate)
            for i in range(n_samples):
                val = int(32767.0 * 0.2 * math.sin(2.0 * math.pi * freq * i / sample_rate))
                wf.writeframes(struct.pack("<h", val))

    def test_whisper_availability(self):
        whisper_service = WhisperSTTService(model_size="base")
        self.assertTrue(whisper_service.is_available)
        self.assertEqual(whisper_service.provider_name, "whisper_base")

    def test_whisper_transcribe_file(self):
        whisper_service = WhisperSTTService(model_size="base")
        res = whisper_service.transcribe_detailed(self.synthetic_wav)
        self.assertIn("text", res)
        self.assertIn("processing_time_ms", res)
        self.assertGreater(res["processing_time_ms"], 0)
        self.assertEqual(res["model"], "whisper_base")

    def test_whisper_transcribe_bytes(self):
        whisper_service = WhisperSTTService(model_size="base")
        with open(self.synthetic_wav, "rb") as f:
            audio_bytes = f.read()
        res = whisper_service.transcribe_detailed(audio_bytes)
        self.assertIn("text", res)
        self.assertIn("processing_time_ms", res)

    def test_whisper_real_speech_transcription(self):
        if not self.speech_wav:
            self.skipTest("macOS 'say' not available for speech synthesis test.")

        whisper_service = WhisperSTTService(model_size="base")
        transcription = whisper_service.transcribe(self.speech_wav)
        self.assertTrue(len(transcription) > 0)
        lower_text = transcription.lower()
        self.assertTrue("bottle" in lower_text or "cup" in lower_text or "find" in lower_text)

    def test_stt_manager_defaults_and_status(self):
        manager = get_stt_manager()
        status = manager.get_status()
        self.assertEqual(status["active_provider"], "whisper")
        self.assertTrue(status["is_local"])
        self.assertIn("whisper", status["available_providers"])

    def test_stt_manager_provider_switching(self):
        manager = get_stt_manager()
        original = manager.active_provider_name

        manager.set_active_provider("mock")
        self.assertEqual(manager.active_provider_name, "mock")
        mock_sentence = manager.transcribe(b"")
        self.assertTrue(len(mock_sentence) > 0)

        manager.set_active_provider(original)
        self.assertEqual(manager.active_provider_name, original)

    def test_transcribe_and_extract_end_to_end(self):
        if not self.speech_wav:
            self.skipTest("Speech WAV not available.")

        manager = get_stt_manager()
        sentence, provider, nlp_res = manager.transcribe_and_extract(self.speech_wav)

        self.assertTrue(len(sentence) > 0)
        self.assertEqual(provider, "whisper_base")
        self.assertTrue(nlp_res.success)
        # Should detect bottle or cup
        target_classes = nlp_res.target_classes
        self.assertTrue("bottle" in target_classes or "cup" in target_classes)

    def test_api_stt_status(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app

        client = TestClient(app)
        res = client.get("/api/nlp/stt-status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["active_provider"], "whisper")
        self.assertTrue(data["is_local"])

    def test_api_transcribe_audio_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app

        client = TestClient(app)
        with open(self.synthetic_wav, "rb") as f:
            res = client.post(
                "/api/nlp/transcribe",
                files={"audio_file": ("test.wav", f, "audio/wav")},
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["provider"], "whisper_base")
        self.assertGreater(data["processing_time_ms"], 0)

    def test_api_stream_query_audio_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.api.app import app

        client = TestClient(app)
        with open(self.synthetic_wav, "rb") as f:
            res = client.post(
                "/api/stream/query-audio",
                files={"audio_file": ("test.wav", f, "audio/wav")},
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("speech_text", data)
        self.assertIn("transcribed_sentence", data)


if __name__ == "__main__":
    unittest.main()
