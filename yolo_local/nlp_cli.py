#!/usr/bin/env python3
"""
CLI Tool for Speech-to-Text & NLP Object Extraction.

Allows testing spoken audio files or natural language queries directly from the terminal
using the local open-source Whisper Speech-to-Text model:
  python nlp_cli.py --audio sample.wav
  python nlp_cli.py --record 4
  python nlp_cli.py --sentence "Can you find my cell phone and two red apples?"
  python nlp_cli.py --interactive
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure backend can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.stt_service import get_stt_manager


def print_stt_and_nlp(sentence: str, stt_meta: dict, res):
    print("=" * 65)
    print(f" 🎙️ STT Provider:       {stt_meta.get('model', 'Whisper')} ({stt_meta.get('device', 'cpu').upper()})")
    print(f" ⏱️ STT Latency:        {stt_meta.get('processing_time_ms', 0):.1f} ms")
    if stt_meta.get("language"):
        print(f" 🌐 Detected Language:  {stt_meta.get('language')}")
    print(f" 🗣️ Transcribed Text:   \"{sentence}\"")
    print(f" 🧹 Cleaned Sentence:   \"{res.cleaned_sentence}\"")
    print(f" 🎯 Detected Intent:    {res.intent.upper()}")
    print(f" ⏱️ NLP Latency:        {res.processing_time_ms:.1f} ms")
    print("-" * 65)

    if res.target_classes:
        print(f" 📦 Target Classes ({len(res.target_classes)}):")
        for cls_name, cls_id in zip(res.target_classes, res.target_class_ids):
            print(f"    • {cls_name.upper()} (COCO ID: {cls_id})")
        targets_str = f'"{res.target_classes[0]}"'
        print(f"\n 🎯 To locate ONLY this target object:")
        print(f"    • Live Camera:  python main.py --source webcam --target {targets_str}")
        print(f"    • Still Image:  python main.py --source image --path bus.jpg --target {targets_str}")
    else:
        print(" 📦 Target Classes:     None detected")

    if res.excluded_classes:
        print(f" 🚫 Excluded / Negated: {res.excluded_classes}")

    if res.extracted_objects:
        print("-" * 65)
        print(" 🔍 Extracted Object Details:")
        for idx, obj in enumerate(res.extracted_objects, 1):
            status = "❌ NEGATED" if obj.negated else "✅ ACTIVE"
            count_str = f"Count: {obj.count}" if obj.count is not None else "Count: 1 (default)"
            attrs_str = f"Attrs: {obj.attributes}" if obj.attributes else "Attrs: None"
            print(
                f"    [{idx}] {obj.class_name} ({status})\n"
                f"        Matched: \"{obj.matched_phrase}\" ({obj.match_type}, conf: {obj.confidence:.2f})\n"
                f"        {count_str} | {attrs_str} | Span: [{obj.start_char}:{obj.end_char}]"
            )
    print("=" * 65 + "\n")


def print_result(res):
    print("=" * 65)
    print(f" 🗣️ Input Sentence:     \"{res.original_sentence}\"")
    print(f" 🧹 Cleaned Sentence:   \"{res.cleaned_sentence}\"")
    print(f" 🎯 Detected Intent:    {res.intent.upper()}")
    print(f" ⏱️ Latency:            {res.processing_time_ms} ms")
    print("-" * 65)

    if res.target_classes:
        print(f" 📦 Target Classes ({len(res.target_classes)}):")
        for cls_name, cls_id in zip(res.target_classes, res.target_class_ids):
            print(f"    • {cls_name.upper()} (COCO ID: {cls_id})")
        targets_str = f'"{res.target_classes[0]}"'
        print(f"\n 🎯 To locate ONLY this target object:")
        print(f"    • Live Camera:  python main.py --source webcam --target {targets_str}")
        print(f"    • Still Image:  python main.py --source image --path bus.jpg --target {targets_str}")
    else:
        print(" 📦 Target Classes:     None detected")

    if res.excluded_classes:
        print(f" 🚫 Excluded / Negated: {res.excluded_classes}")

    if res.extracted_objects:
        print("-" * 65)
        print(" 🔍 Extracted Object Details:")
        for idx, obj in enumerate(res.extracted_objects, 1):
            status = "❌ NEGATED" if obj.negated else "✅ ACTIVE"
            count_str = f"Count: {obj.count}" if obj.count is not None else "Count: 1 (default)"
            attrs_str = f"Attrs: {obj.attributes}" if obj.attributes else "Attrs: None"
            print(
                f"    [{idx}] {obj.class_name} ({status})\n"
                f"        Matched: \"{obj.matched_phrase}\" ({obj.match_type}, conf: {obj.confidence:.2f})\n"
                f"        {count_str} | {attrs_str} | Span: [{obj.start_char}:{obj.end_char}]"
            )
    print("=" * 65 + "\n")


def record_mic_audio(duration: float = 4.0) -> Path | None:
    """Record audio from default mic using FFmpeg on macOS."""
    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg_bin).exists():
        print("❌ FFmpeg not found in PATH or /opt/homebrew/bin/ffmpeg.")
        return None

    tmp_wav = Path(tempfile.gettempdir()) / f"mic_record_{int(time.time())}.wav"
    print(f"🎙️ Recording {duration:.1f}s from microphone... Speak now!")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "avfoundation",
        "-i", ":default",
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        str(tmp_wav),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ Recording completed.")
        return tmp_wav
    except subprocess.CalledProcessError as err:
        print(f"❌ Recording failed: {err}")
        return None


def run_interactive():
    nlp = get_nlp_extractor()
    stt = get_stt_manager()
    status = stt.get_status()

    print("=" * 65)
    print("  🎙️ NLP & Speech-to-Text Terminal Synapse (Interactive Mode)")
    print(f"  Active STT: {status['active_provider']} (Local Model: {status['model_name']})")
    print("  Commands:")
    print("    • Type any text query directly (e.g., 'find my phone and laptop')")
    print("    • ':record [seconds]' to record live audio from microphone")
    print("    • ':audio <file.wav>' to transcribe an existing audio file")
    print("    • ':status' to view STT engine details")
    print("    • 'quit' or 'exit' to terminate")
    print("=" * 65 + "\n")

    while True:
        try:
            line = input("Speech/Query > ").strip()
            if not line:
                continue
            if line.lower() in {"quit", "exit", "q"}:
                print("Exiting NLP CLI.")
                break

            if line.startswith(":status"):
                print(json.dumps(stt.get_status(), indent=2))
                continue

            if line.startswith(":record"):
                parts = line.split()
                dur = float(parts[1]) if len(parts) > 1 else 4.0
                rec_path = record_mic_audio(dur)
                if rec_path and rec_path.exists():
                    print("🧠 Transcribing locally using Whisper STT...")
                    detailed = stt.transcribe_detailed(rec_path)
                    sentence = detailed["text"]
                    res = nlp.extract(sentence)
                    print_stt_and_nlp(sentence, detailed, res)
                    try:
                        rec_path.unlink()
                    except OSError:
                        pass
                continue

            if line.startswith(":audio"):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    print("❌ Usage: :audio <path_to_audio_file>")
                    continue
                audio_path = Path(parts[1].strip("'\""))
                if not audio_path.exists():
                    print(f"❌ File not found: {audio_path}")
                    continue
                print("🧠 Transcribing locally using Whisper STT...")
                detailed = stt.transcribe_detailed(audio_path)
                sentence = detailed["text"]
                res = nlp.extract(sentence)
                print_stt_and_nlp(sentence, detailed, res)
                continue

            # Standard text sentence
            res = nlp.extract(line)
            print_result(res)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting NLP CLI.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Text & NLP Object Extraction using local open-source Whisper."
    )
    parser.add_argument(
        "--sentence", "-s",
        type=str,
        help="Input speech-to-text sentence string",
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        nargs="?",
        const="sample_speech.wav",
        help="Path to speech audio file (.wav, .mp3, etc.). Defaults to sample_speech.wav if omitted.",
    )
    parser.add_argument(
        "--record", "-r",
        type=float,
        nargs="?",
        const=4.0,
        help="Record N seconds of audio from microphone and process (default: 4.0 seconds)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive terminal prompt",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response",
    )
    args = parser.parse_args()

    nlp = get_nlp_extractor()
    stt = get_stt_manager()

    if args.audio is not None:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"❌ Error: Audio file not found at '{audio_path}'", file=sys.stderr)
            if Path("sample_speech.wav").exists():
                print("💡 Tip: A sample audio file is included: run `python nlp_cli.py --audio sample_speech.wav`")
            sys.exit(1)
        detailed = stt.transcribe_detailed(audio_path)
        sentence = detailed["text"]
        res = nlp.extract(sentence)
        if args.json:
            out = {
                "transcription": sentence,
                "stt_metadata": detailed,
                "nlp_result": res.model_dump(),
            }
            print(json.dumps(out, indent=2))
        else:
            print_stt_and_nlp(sentence, detailed, res)
        return

    if args.record is not None:
        rec_path = record_mic_audio(args.record)
        if not rec_path or not rec_path.exists():
            sys.exit(1)
        detailed = stt.transcribe_detailed(rec_path)
        sentence = detailed["text"]
        res = nlp.extract(sentence)
        try:
            rec_path.unlink()
        except OSError:
            pass
        if args.json:
            out = {
                "transcription": sentence,
                "stt_metadata": detailed,
                "nlp_result": res.model_dump(),
            }
            print(json.dumps(out, indent=2))
        else:
            print_stt_and_nlp(sentence, detailed, res)
        return

    if args.sentence:
        res = nlp.extract(args.sentence)
        if args.json:
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print_result(res)
    elif args.interactive or (not args.sentence and not args.audio and args.record is None):
        run_interactive()


if __name__ == "__main__":
    main()
