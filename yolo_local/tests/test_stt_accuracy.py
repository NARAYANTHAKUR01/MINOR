"""
Unit test suite validating Speech-to-Text (STT) accuracy enhancements.

Covers:
1. AudioPreprocessor: DC-offset removal, silence trimming, peak gain normalization, and duration safeguarding.
2. STTPostProcessor: ASR artifact cleaning, contraction expansion, and phonetic homophone correction.
3. WhisperSTTService: Beam search decoding, detailed metadata, and audio waveform preprocessing.
4. STTManager: Dual-channel consensus fusion with client speech hints.
5. End-to-end transcription on sample_speech.wav.
"""

from pathlib import Path
import unittest
import numpy as np

from backend.config import settings
from backend.services.stt_service import (
    AudioPreprocessor,
    STTPostProcessor,
    WhisperSTTService,
    get_stt_manager,
)


class TestAudioPreprocessor(unittest.TestCase):
    """Test audio signal preprocessing algorithms for Speech-to-Text accuracy."""

    def test_dc_offset_removal(self):
        """Verify DC offset is removed and waveform is zero-centered."""
        # Create synthetic audio with DC bias of 0.25
        audio = np.full(16000, 0.25, dtype=np.float32)
        audio[4000:8000] += 0.5  # Add signal burst
        processed = AudioPreprocessor.preprocess_waveform(audio, trim_silence=False, normalize_gain=False)
        self.assertAlmostEqual(float(np.mean(processed)), 0.0, places=5)

    def test_peak_gain_normalization(self):
        """Verify low-amplitude audio is scaled to target peak 0.95."""
        # Quiet speech audio with peak 0.05
        audio = np.zeros(16000, dtype=np.float32)
        audio[2000:4000] = 0.05
        processed = AudioPreprocessor.preprocess_waveform(audio, trim_silence=False, normalize_gain=True)
        self.assertAlmostEqual(float(np.max(np.abs(processed))), AudioPreprocessor.TARGET_PEAK, places=4)

    def test_silence_trimming(self):
        """Verify dead silence at head and tail is trimmed."""
        # 3 seconds total: 1.2s silence + 0.6s tone burst + 1.2s silence
        sr = 16000
        audio = np.zeros(sr * 3, dtype=np.float32)
        t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
        audio[int(sr * 1.2) : int(sr * 1.2) + len(t)] = 0.8 * np.sin(2 * np.pi * 440 * t)

        processed = AudioPreprocessor.preprocess_waveform(audio, trim_silence=True, normalize_gain=True)
        # Should be significantly shorter than 3 seconds
        self.assertLess(len(processed), len(audio) * 0.6)
        # Peak must remain close to 0.95
        self.assertAlmostEqual(float(np.max(np.abs(processed))), AudioPreprocessor.TARGET_PEAK, places=3)

    def test_minimum_duration_safeguard(self):
        """Verify ultra-short audio (< 0.4s) is padded to prevent encoder edge artifacts."""
        sr = 16000
        short_audio = np.ones(int(sr * 0.1), dtype=np.float32) * 0.5
        processed = AudioPreprocessor.preprocess_waveform(short_audio, trim_silence=False, normalize_gain=False)
        self.assertGreaterEqual(len(processed), int(sr * AudioPreprocessor.MIN_DURATION_SECONDS))

    def test_preemphasis_filter(self):
        """Verify pre-emphasis high-pass filter emphasizes high-frequency changes."""
        audio = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        filtered = AudioPreprocessor.apply_preemphasis(audio, coeff=0.97)
        self.assertEqual(len(filtered), len(audio))
        # Initial transition should be prominent
        self.assertAlmostEqual(filtered[1], 1.0, places=4)
        # Constant region should be attenuated
        self.assertAlmostEqual(filtered[2], 0.03, places=4)


class TestSTTPostProcessor(unittest.TestCase):
    """Test text cleaning and phonetic homophone normalization for COCO objects."""

    def test_asr_artifact_cleaning(self):
        """Verify stray ellipses, quotation marks, and redundant spaces are cleaned."""
        raw = '  "...where is the bottle?..."  '
        clean = STTPostProcessor.process_transcript(raw)
        self.assertEqual(clean, "where is the bottle?")

    def test_contraction_expansion(self):
        """Verify contractions are expanded for consistent NLP target parsing."""
        cases = [
            ("Where's my cell phone?", "where is my cell phone?"),
            ("What's on the table?", "what is on the table?"),
            ("It's a red apple", "it is a red apple"),
        ]
        for inp, expected in cases:
            self.assertEqual(STTPostProcessor.process_transcript(inp), expected)

    def test_phonetic_homophone_corrections(self):
        """Verify spoken phonetic errors are corrected to standard COCO terms."""
        cases = [
            ("Find the lap atop", "Find the laptop"),
            ("Where is the smart phone", "Where is the cell phone"),
            ("Look at the tee vee", "Look at the tv"),
            ("Can you see the bottle", "Can you see the bottle"),
            ("Where is the study bear", "Where is the teddy bear"),
            ("Where is the cub on the table", "Where is the cup on the table"),
            ("Where is the wineglass", "Where is the wine glass"),
            ("Look at the diningtable", "Look at the dining table"),
            ("Where is the cheir", "Where is the chair"),
        ]
        for inp, expected in cases:
            self.assertEqual(STTPostProcessor.process_transcript(inp), expected)


class TestWhisperSTTServiceAccuracy(unittest.TestCase):
    """Test offline local Whisper transcription with beam search and preprocessing."""

    @classmethod
    def setUpClass(cls):
        cls.stt = WhisperSTTService(model_size="base")

    def test_sample_speech_transcription(self):
        """Verify real sample speech audio is accurately transcribed."""
        sample_path = Path("sample_speech.wav")
        if not sample_path.exists():
            sample_path = Path(__file__).resolve().parent.parent / "sample_speech.wav"
        self.assertTrue(sample_path.exists(), "sample_speech.wav not found")

        detailed = self.stt.transcribe_detailed(str(sample_path))
        self.assertTrue(len(detailed["text"]) > 0)
        self.assertIn("cell phone", detailed["text"].lower())
        self.assertIn("bottle", detailed["text"].lower())
        self.assertEqual(detailed["language"], "en")
        self.assertLess(detailed["processing_time_ms"], 3000.0)

    def test_beam_search_configuration_parameters(self):
        """Verify settings configure beam search size 5 and low no-speech threshold."""
        self.assertEqual(settings.whisper_beam_size, 5)
        self.assertEqual(settings.whisper_best_of, 5)
        self.assertEqual(settings.whisper_patience, 1.0)
        self.assertEqual(settings.whisper_no_speech_threshold, 0.4)
        self.assertTrue(settings.whisper_preprocess_audio)

    def test_model_switching(self):
        """Verify switch_model method switches model configuration cleanly."""
        orig_size = self.stt.model_size
        self.stt.switch_model("base")
        self.assertEqual(self.stt.model_size, "base")


class TestDualChannelConsensus(unittest.TestCase):
    """Test STTManager dual-channel consensus and hint fusion."""

    @classmethod
    def setUpClass(cls):
        cls.manager = get_stt_manager()

    def test_hint_recovery_when_whisper_text_empty(self):
        """Verify hint text recovers target query when audio is empty."""
        res = self.manager.transcribe_with_hint(
            audio_data=b"",
            hint_text="where is my cell phone",
        )
        self.assertEqual(res["text"], "where is my cell phone")
        self.assertEqual(res["fusion_mode"], "hint_fallback")

    def test_hint_enriched_region(self):
        """Verify hint enriches requested region when audio transcription only heard object."""
        sample_path = Path("sample_speech.wav")
        if not sample_path.exists():
            sample_path = Path(__file__).resolve().parent.parent / "sample_speech.wav"

        res = self.manager.transcribe_with_hint(
            audio_data=str(sample_path),
            hint_text="where is my cell phone on the left",
        )
        self.assertIn("left", res["text"])
        self.assertEqual(res["fusion_mode"], "hint_enriched_region")


class TestYOLODetectorAccuracy(unittest.TestCase):
    """Test YOLO detector configuration, resolution, and adaptive illumination enhancements."""

    def test_detector_default_settings(self):
        """Verify default resolution 640 and optimized IoU 0.45."""
        self.assertEqual(settings.default_imgsz, 640)
        self.assertEqual(settings.default_iou_threshold, 0.45)
        self.assertEqual(settings.default_conf_threshold, 0.35)

    def test_adaptive_low_light_enhancement(self):
        """Verify dark frames are enhanced with CLAHE without crashing."""
        from backend.services.detector import get_default_detector
        detector = get_default_detector()
        # Synthetic dark frame
        dark_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        dets, duration, annotated = detector.detect(dark_frame, imgsz=640, enhance_lighting=True)
        self.assertIsInstance(dets, list)
        self.assertGreater(duration, 0)


if __name__ == "__main__":
    unittest.main()
