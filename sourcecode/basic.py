import os
from pyannote.audio import Pipeline
from huggingface_hub import login
import whisper
from pydub import AudioSegment

# ==== CONFIGURATION ====
HUGGINGFACE_TOKEN = "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"  # Replace with your actual token
AUDIO_FILE = "sample/kate_Andrews-trimmed.wav"                      # Input WAV file
WHISPER_MODEL_SIZE = "small"                   # Options: tiny, base, small, medium, large
OUTPUT_DIR = "segments"

# ==== LOGIN TO HUGGING FACE ====
login(token=HUGGINGFACE_TOKEN)

# ==== LOAD DIARIZATION PIPELINE ====
print("[INFO] Loading speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)

# ==== RUN DIARIZATION ====
print("[INFO] Performing speaker diarization...")
diarization = pipeline(AUDIO_FILE)

# ==== LOAD AUDIO ====
print("[INFO] Loading full audio...")
audio = AudioSegment.from_wav(AUDIO_FILE)

# ==== SETUP OUTPUT ====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== LOAD WHISPER ====
print("[INFO] Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL_SIZE)

# ==== PROCESS SEGMENTS ====
print("[INFO] Transcribing segments...")
transcript = []

for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
    segment_audio = audio[turn.start * 1000: turn.end * 1000]
    segment_filename = os.path.join(OUTPUT_DIR, f"segment_{i}_{speaker}.wav")
    segment_audio.export(segment_filename, format="wav")

    result = model.transcribe(segment_filename)
    segment_text = result['text'].strip()

    transcript.append(f"{speaker} [{turn.start:.2f}s - {turn.end:.2f}s]: {segment_text}")
    print(transcript[-1])  # Optional: stream output

# ==== SAVE FINAL TRANSCRIPT ====
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(transcript))

print("\n✅ Diarization and transcription complete. Output saved to transcript.txt")
