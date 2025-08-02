import librosa
import numpy as np

y, sr = librosa.load("sample/geoffery_bush.wav", sr=None)

# Pitch (Fundamental Frequency)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Energy
energy = np.mean(librosa.feature.rms(y=y))

# Tempo / Speech Rate Estimation (approx)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

print(f"Pitch (mean): {np.nanmean(f0):.2f} Hz")
print(f"Energy (RMS): {energy:.4f}")
print(f"Tempo: {tempo:.2f} BPM (rough speech rate)")
