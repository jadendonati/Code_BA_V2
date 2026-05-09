import argparse
import json
import os
from pathlib import Path

import librosa
from dotenv import load_dotenv
from openai import OpenAI


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY fehlt. Bitte in der .env-Datei setzen.")
    return OpenAI(api_key=api_key)


def transcribe_audio(client: OpenAI, audio_path: Path) -> str:
    with audio_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return transcript.text.strip()


def calculate_prosody(audio_path: Path, transcript_text: str) -> dict:
    duration_seconds = librosa.get_duration(path=str(audio_path))
    word_count = len(transcript_text.split())
    words_per_minute = (word_count / duration_seconds) * 60 if duration_seconds else 0.0

    if words_per_minute < 110:
        speech_speed = "slow"
    elif words_per_minute < 160:
        speech_speed = "normal"
    else:
        speech_speed = "fast"

    return {
        "audio_duration_seconds": round(duration_seconds, 2),
        "word_count": word_count,
        "speech_rate": {
            "words_per_minute": round(words_per_minute, 2),
            "category": speech_speed,
        },
    }


def build_analysis(audio_path: Path, transcript_text: str, prosody: dict) -> dict:
    return {
        "audio_file": audio_path.name,
        "transcript": transcript_text,
        "audio_duration_seconds": prosody["audio_duration_seconds"],
        "word_count": prosody["word_count"],
        "prosody": {
            "speech_rate": prosody["speech_rate"],
        },
    }


def save_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_individual_prosody_jsons(audio_path: Path, prosody: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for feature_name, value in prosody.items():
        payload = {
            "audio_file": audio_path.name,
            "feature_name": feature_name,
            "value": value,
        }
        file_path = output_dir / f"{audio_path.stem}_{feature_name}.json"
        save_json(file_path, payload)
        saved_files.append(file_path)

    return saved_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transkribiert eine Audiodatei und speichert prosodische Merkmale als JSON."
    )
    parser.add_argument(
        "--audio",
        default="audio/postitiv.wav",
        help="Pfad zur Audiodatei relativ zum Projektordner.",
    )
    parser.add_argument(
        "--output",
        help="Pfad fuer das Gesamt-JSON. Standard: json/<audio_stem>_analysis.json.",
    )
    parser.add_argument(
        "--prosody-dir",
        help="Ordner fuer einzelne JSON-Dateien pro prosodischem Merkmal. Standard: json/<audio_stem>_prosody_json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audiodatei nicht gefunden: {audio_path}")

    json_root = Path("json")
    output_path = Path(args.output) if args.output else json_root / f"{audio_path.stem}_analysis.json"
    prosody_dir = (
        Path(args.prosody_dir)
        if args.prosody_dir
        else json_root / f"{audio_path.stem}_prosody_json"
    )

    client = build_client()
    transcript_text = transcribe_audio(client, audio_path)
    prosody = calculate_prosody(audio_path, transcript_text)
    analysis = build_analysis(audio_path, transcript_text, prosody)

    save_json(output_path, analysis)
    saved_prosody_files = save_individual_prosody_jsons(audio_path, prosody, prosody_dir)

    print("Transcript:")
    print(transcript_text)
    print()
    print(f"Gesamt-JSON gespeichert unter: {output_path}")
    print("Einzelne Prosodie-JSONs gespeichert unter:")
    for file_path in saved_prosody_files:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
