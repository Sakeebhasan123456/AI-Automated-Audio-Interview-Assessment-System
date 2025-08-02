import os
from pyannote.audio import Pipeline
from huggingface_hub import login
from pydub import AudioSegment
from datetime import timedelta
from funasr import AutoModel

# === CONFIG ===
HUGGINGFACE_TOKEN = "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"  # Replace with your actual token
AUDIO_FILE = "sample/kate_andrews-trimmed.wav"
OUTPUT_DIR = "segments"
SPEAKER_MAP = {
    "SPEAKER_01": "Interviewer",
    "SPEAKER_00": "Candidate"
    # Extend this map if more speakers
}

# === AUTH & PIPELINES ===
login(token=HUGGINGFACE_TOKEN)
print("[INFO] Loading diarization pipeline...")
diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)

print("[INFO] Loading FunASR model...")
funasr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0"  # Change to "cpu" if no GPU
)

# === LOAD AUDIO ===
print("[INFO] Loading input audio...")
audio = AudioSegment.from_wav(AUDIO_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === RUN DIARIZATION ===
print("[INFO] Running speaker diarization...")
diarization = diarizer(AUDIO_FILE)

# === TRANSCRIBE BY SPEAKER SEGMENT ===
transcript = []

print("[INFO] Transcribing segments using FunASR...")
for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)

    segment_audio = audio[start_ms:end_ms]
    filename = os.path.join(OUTPUT_DIR, f"segment_{i}_{speaker}.wav")
    segment_audio.export(filename, format="wav")

    # Run FunASR on the segment
    res = funasr_model.generate(
        input=filename,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15
    )
    full_text = res[0]['text']

    # Optional: Parse emotion and clean text
    segments = full_text.split('<|en|>')[1:]
    cleaned_text = ""
    for seg in segments:
        parts = seg.split('<|')
        emotion = parts[1].replace('|>', '') if len(parts) > 1 else 'UNKNOWN'
        content = parts[-1].split('|>')[-1].strip()
        cleaned_text += f"[{emotion}] {content} "

    # Assemble transcript entry
    speaker_label = SPEAKER_MAP.get(speaker.upper(), speaker)
    start_time = str(timedelta(seconds=int(turn.start)))
    end_time = str(timedelta(seconds=int(turn.end)))

    transcript.append({
        "speaker": speaker_label,
        "start": start_time,
        "end": end_time,
        "text": cleaned_text.strip()
    })
    print(transcript[-1])

# === OUTPUT CLEAN TRANSCRIPT ===
print("[INFO] Writing transcript to file...")
with open("transcript.txt", "w", encoding="utf-8") as f:
    for seg in transcript:
        f.write(f"{seg['speaker']} [{seg['start']} - {seg['end']}]: {seg['text']}\n")

print("\n✅ Diarization and transcription complete. Output saved to transcript.txt")
