# ASR

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    # device="cpu"
)

# en
res = model.generate(
    input="sample/geoffery_bush.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
# text = rich_transcription_postprocess(res[0]["text"])
# print(text)
# Extract the text
full_text = res[0]['text']

# Split by segments (each starts with <|en|>)
segments = full_text.split('<|en|>')[1:]  # First element is empty

for segment in segments:
    # Parse each component
    parts = segment.split('<|')
    emotion = parts[1].replace('|>', '') if len(parts) > 1 else 'UNKNOWN'
    text = parts[-1].split('|>')[-1].strip()
    
    # Print with formatting
    print(f"\n[Emotion: {emotion}]")
    print(text)
    print("-" * 50)


# VAD
'''
from funasr import AutoModel

model = AutoModel(model="fsmn-vad")
wav_file = "sample/my_voice_recording.wav"
res = model.generate(input=wav_file)
print(res)
'''

# time-stamp prediction
'''
from funasr import AutoModel

model = AutoModel(model="fa-zh")
wav_file = f"{model.model_path}/example/asr_example.wav"
text_file = f"{model.model_path}/example/text.txt"
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
print()
print(res)
'''

# speech emotion recognition
'''
from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_plus_large")

wav_file = f"{model.model_path}/example/test.wav"

res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)

data = res[0]

# Pair each label with its score and print them
for label, score in zip(data['labels'], data['scores']):
    print(f"{label}: {score}")
'''

'''
[{'key': 'test', 'labels': ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>'], 'scores': [1.0, 4.307079432691596e-12, 7.651620656523583e-12, 1.8212416297291867e-10, 7.213059316502068e-11, 1.3731805914519914e-14, 9.798530042903764e-11, 8.913316751346656e-10, 5.704816692153179e-21]}]
'''
