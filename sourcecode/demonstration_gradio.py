import os
import gradio as gr
from pyannote.audio import Pipeline
from huggingface_hub import login, HfFolder
from pydub import AudioSegment, exceptions as pydub_exceptions
from datetime import timedelta
import tempfile
import traceback # For better error reporting
import re

# Import funasr AFTER potential environment setup if needed
try:
    from funasr import AutoModel
    # from funasr.utils.postprocess_utils import rich_transcription_postprocess # Not used in the provided snippet
except ImportError:
    print("Error: FunASR library not found. Please install it: pip install funasr")
    exit()
except Exception as e:
    print(f"Error importing FunASR: {e}")
    exit()


# === CONFIG & CONSTANTS ===
# --- HUGGING FACE TOKEN ---
# IMPORTANT: Avoid hardcoding tokens. Use environment variables or Gradio Secrets.
# For local testing, you can set it as an environment variable:
# export HUGGING_FACE_HUB_TOKEN='your_token_here'
# Or input it securely when running the script.
# HUGGINGFACE_TOKEN = "hf_..." # REMOVED - Use environment variable or input
HUGGINGFACE_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")

OUTPUT_DIR_BASE = "output_segments" # Base directory for segments
SPEAKER_MAP = {
    "SPEAKER_01": "Interviewer",
    "SPEAKER_00": "Candidate"
    # Extend this map if more speakers are detected by pyannote
}
SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "m4a", "aac"] # Add more if needed by ffmpeg

# === GLOBAL VARIABLES (for loaded models) ===
diarizer = None
transcription_model = None
logged_in = False

# === INITIALIZATION FUNCTION ===
def initialize_models():
    """Loads models once when the script starts."""
    global diarizer, transcription_model, logged_in, HUGGINGFACE_TOKEN

    print("[INFO] Initializing models...")

    # --- Hugging Face Authentication ---
    if not HUGGINGFACE_TOKEN:
        print("\nWARNING: Hugging Face token not found in environment variable HUGGING_FACE_HUB_TOKEN.")
        print("You might encounter rate limits or issues accessing private models.")
        # Optionally prompt for token if running interactively, but less ideal for Gradio apps
        # HUGGINGFACE_TOKEN = input("Please enter your Hugging Face Token: ").strip()
        # if not HUGGINGFACE_TOKEN:
        #     print("ERROR: Hugging Face Token is required.")
        #     return False # Indicate failure

    # Attempt login only if token is available
    if HUGGINGFACE_TOKEN:
        try:
            login(token=HUGGINGFACE_TOKEN)
            HfFolder.save_token(HUGGINGFACE_TOKEN) # Save token for underlying libraries if needed
            logged_in = True
            print("[INFO] Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"[ERROR] Hugging Face login failed: {e}")
            print("Proceeding without authentication. Access to gated models might fail.")
            logged_in = False
    else:
         print("[INFO] Proceeding without Hugging Face authentication.")
         logged_in = False # Explicitly set

    # --- Load Diarization Pipeline ---
    try:
        print("[INFO] Loading diarization pipeline (pyannote/speaker-diarization)...")
        # Use token explicitly if available and needed for the model
        auth_token_param = HUGGINGFACE_TOKEN if logged_in else None
        diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token_param) # Using 3.1 which is often better
        print("[INFO] Diarization pipeline loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load diarization pipeline: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        return False # Indicate failure

    # --- Load Transcription Model (FunASR) ---
    try:
        print("[INFO] Loading transcription model (iic/SenseVoiceSmall)...")
        # Determine device (prefer CUDA if available)
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Using device: {device}")
        except ImportError:
            print("[WARNING] PyTorch not found. Assuming CPU for FunASR.")
            device = "cpu"

        model_dir = "iic/SenseVoiceSmall"
        transcription_model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device
        )
        print("[INFO] Transcription model loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load FunASR transcription model: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        return False # Indicate failure

    print("[INFO] All models initialized successfully.")
    return True # Indicate success


# === CORE PROCESSING FUNCTION ===
def transcribe_interview(audio_filepath):
    """
    Processes the uploaded audio file: converts if necessary, diarizes,
    transcribes segments, and returns formatted transcript.
    """
    global diarizer, transcription_model

    if not diarizer or not transcription_model:
        return "ERROR: Models are not loaded. Please check the console.", []

    if not audio_filepath:
        return "ERROR: No audio file provided.", []

    print(f"[INFO] Processing audio file: {audio_filepath}")

    try:
        # --- Create a temporary directory for this run ---
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "segments")
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Using temporary directory: {temp_dir}")

            # --- Check and Convert Audio ---
            file_ext = os.path.splitext(audio_filepath)[1].lower().replace('.', '')
            wav_audio_path = audio_filepath

            if file_ext != "wav":
                print(f"[INFO] Input is not WAV ({file_ext}). Converting...")
                if file_ext not in SUPPORTED_FORMATS:
                     return f"ERROR: Unsupported audio format '{file_ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}", []

                try:
                    audio = AudioSegment.from_file(audio_filepath, format=file_ext)
                    wav_audio_path = os.path.join(temp_dir, "converted_audio.wav")
                    audio.export(wav_audio_path, format="wav")
                    print(f"[INFO] Audio converted to WAV: {wav_audio_path}")
                except pydub_exceptions.CouldntDecodeError:
                     return f"ERROR: Could not decode audio file. It might be corrupted or an incompatible '{file_ext}'.", []
                except FileNotFoundError:
                    # This might happen if ffmpeg/ffprobe is not installed or not in PATH
                     return ("ERROR: Could not process audio file. Make sure ffmpeg is installed and accessible "
                             "(needed for non-WAV conversion by pydub). Check console for details."), []
                except Exception as e:
                    return f"ERROR: Failed during audio conversion: {e}", []
            else:
                print("[INFO] Input audio is already in WAV format.")


            # --- Load Audio for Segmentation ---
            try:
                audio = AudioSegment.from_wav(wav_audio_path)
            except Exception as e:
                return f"ERROR: Failed to load WAV audio '{wav_audio_path}': {e}", []


            # === RUN DIARIZATION ===
            print("[INFO] Running speaker diarization...")
            try:
                # Send the *path* to the diarizer
                diarization = diarizer(wav_audio_path)
                print("[INFO] Diarization complete.")
            except Exception as e:
                print(f"[ERROR] Diarization failed: {e}")
                print(traceback.format_exc())
                return f"ERROR: Speaker diarization failed. Check console for details. Error: {e}", []


            # === TRANSCRIBE BY SPEAKER SEGMENT ===
            transcript_data = [] # Store structured data [{speaker, start, end, text}]
            print("[INFO] Transcribing segments...")

            # Use diarization.itertracks to get speaker labels directly
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                duration_s = turn.end - turn.start

                # Skip very short segments which might be noise or transcription errors
                if duration_s < 0.2: # Adjust threshold if needed
                    print(f"[WARN] Skipping very short segment ({duration_s:.2f}s) for speaker {speaker}")
                    continue

                try:
                    # --- Export Segment ---
                    segment_audio = audio[start_ms:end_ms]
                    filename = os.path.join(output_dir, f"segment_{len(transcript_data)}_{speaker}.wav")
                    segment_audio.export(filename, format="wav")

                    # --- Transcribe Segment ---
                    # FunASR expects file path
                    res = transcription_model.generate(
                        input=filename,
                        cache={},         # Cache for potential optimizations if needed
                        language="auto",  # Auto language detection
                        use_itn=True,     # Inverse text normalization (e.g., two -> 2)
                        batch_size_s=60,  # Adjust based on VRAM/performance needs
                        merge_vad=True,   # Use VAD results for merging
                        merge_length_s=15 # Merge segments up to 15s
                    )

                                        # === Existing code before this section ===
                    # ...
                    # res = transcription_model.generate(...)
                    # ...

                    # === REPLACE THE OLD PARSING LOGIC WITH THIS ===

                    if not res or 'text' not in res[0]:
                        print(f"[WARN] No transcription result for segment: {filename}")
                        segment_text = "[TRANSCRIPTION FAILED or EMPTY]"
                    else:
                        full_text = res[0]['text']
                        # --- NEW PARSING LOGIC ---
                        final_text_parts = []
                        detected_emotion = "UNKNOWN" # Reset for each pyannote segment

                        # Split by the language tag FunASR uses as a separator (e.g., <|en|>)
                        # Adjust '<|en|>' if your model outputs a different language code tag
                        sub_segments = full_text.split('<|en|>')

                        for i, sub_segment in enumerate(sub_segments):
                            # Skip the first part if it's empty (often occurs before the first tag)
                            if i == 0 and not sub_segment.strip():
                                continue

                            current_part = sub_segment.strip()
                            if not current_part:
                                continue

                            # Attempt to detect emotion ONCE per pyannote segment.
                            # Looks for the first uppercase word within <|...|> tags.
                            if detected_emotion == "UNKNOWN":
                                # Regex explanation: <\|      - Matches the literal '<|'
                                #                  ([A-Z]+) - Captures one or more uppercase letters (Group 1)
                                #                  \|>      - Matches the literal '|>'
                                emotion_match = re.search(r"<\|([A-Z]+)\|>", current_part)
                                if emotion_match:
                                    potential_emotion = emotion_match.group(1)
                                    # Avoid tagging non-emotion keywords if they follow the same pattern
                                    if potential_emotion not in ["SPEECH", "WITHITN"]: # Add any other known non-emotion tags here
                                        detected_emotion = potential_emotion
                                        # print(f"[DEBUG] Detected emotion: {detected_emotion}") # Optional debug print

                            # Remove ALL tags of the format <|...|> to get the clean text
                            # Regex explanation: <\|     - Matches '<|'
                            #                  [^>]+? - Matches one or more characters that are NOT '>' (non-greedy)
                            #                  \|>    - Matches '|>'
                            cleaned_text = re.sub(r"<\|[^>]+?\|>", "", current_part).strip()

                            # Add the cleaned text if it's not empty
                            if cleaned_text:
                                final_text_parts.append(cleaned_text)

                        # Join the cleaned text parts from potentially multiple sub-segments
                        combined_cleaned_text = " ".join(final_text_parts).strip()

                        # Prepend the detected emotion if found
                        if detected_emotion != "UNKNOWN":
                            segment_text = f"[Emotion: {detected_emotion}] {combined_cleaned_text}"
                        else:
                            segment_text = combined_cleaned_text # Use only text if no emotion detected

                        # Handle cases where cleaning might result in empty text
                        if not segment_text:
                            print(f"[WARN] Segment text is empty after cleaning tags. Original chunk: {full_text[:100]}...")
                            segment_text = "[EMPTY AFTER CLEANING TAGS]"
                        # --- END NEW PARSING LOGIC ---

                    # --- The rest of the code appending to transcript_data remains the same ---
                    # Get Speaker Label
                    speaker_label = SPEAKER_MAP.get(speaker, speaker)
                    # Format Timestamps
                    start_time = str(timedelta(seconds=turn.start)).split('.')[0]
                    end_time = str(timedelta(seconds=turn.end)).split('.')[0]
                    # Store Result
                    transcript_data.append({
                        "speaker": speaker_label,
                        "start": start_time,
                        "end": end_time,
                        "text": segment_text # This now uses the properly cleaned text
                    })
                    # print(transcript_data[-1]) # Optional: uncomment to see segment data in console

                    # === Existing code after this section ===

                except Exception as e:
                    print(f"[ERROR] Failed processing segment for speaker {speaker} ({turn.start:.2f}s - {turn.end:.2f}s): {e}")
                    print(traceback.format_exc())
                    # Add a placeholder for the failed segment
                    transcript_data.append({
                        "speaker": SPEAKER_MAP.get(speaker, speaker),
                        "start": str(timedelta(seconds=turn.start)).split('.')[0],
                        "end": str(timedelta(seconds=turn.end)).split('.')[0],
                        "text": f"[ERROR PROCESSING SEGMENT: {e}]"
                    })

            print("[INFO] Transcription of segments complete.")

            # === FORMAT FINAL OUTPUT ===
            formatted_transcript = ""
            for seg in transcript_data:
                formatted_transcript += f"{seg['speaker']} [{seg['start']} - {seg['end']}]: {seg['text']}\n"

            print("[INFO] Process finished successfully.")
            return formatted_transcript.strip(), transcript_data # Return both formatted string and raw data

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during processing: {e}")
        print(traceback.format_exc())
        return f"An unexpected error occurred: {e}. Check console for details.", []


# === INITIALIZE MODELS ON STARTUP ===
models_loaded = initialize_models()


# === GRADIO INTERFACE ===
with gr.Blocks() as demo:
    gr.Markdown("# Audio Interview Transcription (Phase 1 Demo)")
    gr.Markdown(
        "Upload an interview audio recording (.wav, .mp3, .ogg, etc.). "
        "The system will perform speaker diarization (identifying who speaks when), "
        "transcribe the speech for each speaker, and attempt basic emotion recognition."
        "\n\n**Note:** Requires Hugging Face token (set via `HUGGING_FACE_HUB_TOKEN` env var or input) "
        "and `ffmpeg` for non-WAV files."
    )

    if not models_loaded:
        gr.Markdown("**ERROR: Models failed to load. Please check the console output where you ran the script for details. The application cannot proceed.**")
    else:
        with gr.Row():
            audio_input = gr.Audio(label="Upload Audio File", type="filepath") # Get file path

        transcribe_button = gr.Button("Transcribe Interview")

        with gr.Row():
             output_transcript = gr.Textbox(label="Formatted Transcript", lines=15, interactive=False)
        with gr.Row():
            output_raw_json = gr.JSON(label="Raw Transcript Data (JSON)") # Output raw data too

        transcribe_button.click(
            fn=transcribe_interview,
            inputs=audio_input,
            outputs=[output_transcript, output_raw_json]
        )

        gr.Examples(
            examples=[
                # Add paths to example audio files if you have any
                # ["path/to/your/example_interview.wav"],
                # ["path/to/your/example_interview.mp3"],
            ],
            inputs=audio_input,
            outputs=[output_transcript, output_raw_json],
            fn=transcribe_interview,
            cache_examples=False, # Might need to disable caching if models are large or stateful
        )

# === Launch the App ===
if __name__ == "__main__":
    print("\nLaunching Gradio Interface...")
    # Share=True creates a public link (use with caution, especially with tokens)
    # Set server_name="0.0.0.0" to allow access from other devices on your network
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)