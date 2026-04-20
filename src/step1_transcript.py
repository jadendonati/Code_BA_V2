import os
from openai import OpenAI
from dotenv import load_dotenv

# lädt Variablen aus der .env Datei
load_dotenv()

# API Client erstellen
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Pfad zur Audio-Datei
audio_path = "audio/postitiv.wav"

# Audio öffnen und an API senden
with open(audio_path, "rb") as audio_file:

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )

# Ergebnis anzeigen
print("Transcript:")
print(transcript.text)