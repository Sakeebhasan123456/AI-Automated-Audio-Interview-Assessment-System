import gradio as gr
import os
import tempfile
import json
from collections import namedtuple

# Attempt to import from Phase1 and Phase2
# These files (Phase1.py and Phase2.py) should be in the same directory
# or accessible via PYTHONPATH.
try:
    from Phase1 import transcribe_interview, initialize_models
    print("Successfully imported from Phase1.py")
except ImportError:
    print("Error: Could not import from Phase1.py. Ensure it's in the correct path. Using dummy functions.")
    def initialize_models():
        print("CRITICAL: Phase1.initialize_models is missing. Transcription will fail.")
        return False
    def transcribe_interview(audio_path): # Assuming this might also be async, if not, it's fine
        print("CRITICAL: Phase1.transcribe_interview is missing.")
        return "Error: Phase1.transcribe_interview is missing.", {"error": "Phase1 module not found"}
    # If transcribe_interview is async, the dummy should reflect that too, e.g.:
    # async def transcribe_interview(audio_path): ...

try:
    from Phase2 import run_pipeline_gradio, load_sample_transcript
    print("Successfully imported from Phase2.py")
except ImportError:
    print("Error: Could not import from Phase2.py. Ensure it's in the correct path. Using dummy functions.")
    # This dummy MUST be async if the real one is, to match the `await` in smart_run_pipeline_gradio
    async def run_pipeline_gradio(transcript_file_obj, kb_file_obj):
        transcript_path = transcript_file_obj.name if transcript_file_obj and hasattr(transcript_file_obj, 'name') else "N/A"
        kb_path = kb_file_obj.name if kb_file_obj and hasattr(kb_file_obj, 'name') else "N/A"
        print(f"CRITICAL: Phase2.run_pipeline_gradio (async dummy) is missing. Called with transcript: {transcript_path}, KB: {kb_path}")
        error_msg = "Phase2.run_pipeline_gradio is missing. Assessment will fail."
        return error_msg, {"error": "Phase2 module not found (async dummy)"}, error_msg
    def load_sample_transcript(path):
        print("INFO: Phase2.load_sample_transcript is missing. Cannot create dummy transcript for examples.")
        dummy_data = {
            "metadata": {"source": "dummy_load_sample_transcript"},
            "transcript": [
                {"speaker": "SPEAKER_00", "timestamp_start": "00:00:00.000", "timestamp_end": "00:00:02.000", "text": "This is a dummy transcript."},
                {"speaker": "SPEAKER_01", "timestamp_start": "00:00:02.500", "timestamp_end": "00:00:04.500", "text": "Generated because Phase2.load_sample_transcript was not found."}
            ]
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dummy_data, f, indent=2)
            print(f"Created a basic dummy transcript at: {path}")
        except Exception as e:
            print(f"Failed to create basic dummy transcript: {e}")
        return dummy_data


# === INITIALIZE MODELS ON STARTUP (FROM PHASE 1) ===
print("Initializing transcription models...")
models_loaded = initialize_models()
if models_loaded:
    print("Transcription models loaded successfully.")
else:
    print("Warning: Transcription models failed to load. Phase 1 functionality will be affected.")


# === HELPER FUNCTION FOR PHASE 2 SUBMISSION LOGIC ===
async def smart_run_pipeline_gradio(uploaded_transcript_file,  # From transcript_file_phase2 (gr.File)
                                    phase1_transcript_json_data, # From output_raw_json_phase1 (gr.JSON)
                                    uploaded_kb_file):          # From kb_file_phase2 (gr.File)
    """
    Orchestrates input to Phase 2's run_pipeline_gradio.
    Prioritizes a directly uploaded transcript in Phase 2.
    If not available, uses transcript JSON data from Phase 1's output.
    The original run_pipeline_gradio expects a file-like object with a .name attribute for the transcript.
    """
    final_transcript_arg = None
    temp_file_to_clean_path = None  # Path of the temp file if created

    if uploaded_transcript_file is not None and hasattr(uploaded_transcript_file, 'name'):
        print(f"Using user-uploaded transcript file for Phase 2: {uploaded_transcript_file.name}")
        final_transcript_arg = uploaded_transcript_file
    elif phase1_transcript_json_data is not None:
        print("Using transcript data from Phase 1 output for Phase 2.")
        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json", encoding='utf-8') as tmp_json_file:
                json.dump(phase1_transcript_json_data, tmp_json_file)
                temp_file_to_clean_path = tmp_json_file.name
            FileObjectMock = namedtuple("FileObjectMock", ["name"])
            final_transcript_arg = FileObjectMock(name=temp_file_to_clean_path)
            print(f"Phase 1 data temporarily saved to: {temp_file_to_clean_path}")
        except Exception as e:
            error_message = f"Error saving Phase 1 transcript data to temporary file: {e}"
            print(error_message)
            return "Error processing transcript.", {"error": error_message, "details": str(e)}, error_message
    else:
        message = "No transcript provided: Please upload a transcript in Phase 2 or generate one in Phase 1 first."
        print(message)
        return message, {"error": message}, message

    if final_transcript_arg is None:
        message = "Critical error: Transcript argument became None before calling pipeline."
        print(message)
        return message, {"error": message}, message

    summary_md, report_json, error_text = "Error during pipeline execution.", {"error": "Pipeline did not run as expected."}, "Pipeline error."
    try:
        # Use await here because run_pipeline_gradio is (or might be) async
        summary_md, report_json, error_text = await run_pipeline_gradio(final_transcript_arg, uploaded_kb_file)
    except TypeError as e: # Specifically catch type errors which include coroutine issues
        if "cannot unpack non-iterable coroutine object" in str(e).lower() or "is not iterable" in str(e).lower():
             error_text = f"Error during run_pipeline_gradio execution: Coroutine issue. Ensure 'run_pipeline_gradio' is awaited if async and returns 3 values. Original error: {str(e)}"
        else:
            error_text = f"Error during run_pipeline_gradio execution (TypeError): {str(e)}"
        print(error_text)
        report_json = {"error": error_text, "type": e.__class__.__name__}
        summary_md = f"### Error\n{error_text}"
    except Exception as e:
        error_text = f"Error during run_pipeline_gradio execution: {str(e)}"
        print(error_text)
        report_json = {"error": error_text, "type": e.__class__.__name__}
        summary_md = f"### Error\n{error_text}"
    finally:
        if temp_file_to_clean_path and os.path.exists(temp_file_to_clean_path):
            try:
                os.unlink(temp_file_to_clean_path)
                print(f"Cleaned up temporary file: {temp_file_to_clean_path}")
            except Exception as e_clean:
                print(f"Error cleaning up temporary file {temp_file_to_clean_path}: {e_clean}")
    
    return summary_md, report_json, error_text


# === GRADIO INTERFACE ===
with gr.Blocks(theme=gr.themes.Soft(), title="Comprehensive Audio Interview Assessment System") as demo:
    gr.Markdown("# Comprehensive Audio Interview Assessment System")
    gr.Markdown(
        "This system processes audio interviews in two phases: \n"
        "1.  **Audio Transcription:** Converts spoken audio into a text transcript with speaker diarization. \n"
        "2.  **Transcript Assessment:** Analyzes the generated transcript using RAG to provide an assessment report."
    )

    with gr.Tabs():
        with gr.TabItem("Phase 1: Audio Transcription"):
            gr.Markdown("## Phase 1: Transcribe Audio to Text")
            gr.Markdown(
                "Upload an interview audio recording (.wav, .mp3, .ogg, etc.). "
                "The system will perform speaker diarization (identifying who speaks when), "
                "transcribe the speech for each speaker, and attempt basic emotion recognition. "
                "The output includes a formatted transcript and a raw JSON file. "
                "**The Raw Transcript Data (JSON) can be automatically used in Phase 2 if no other transcript is uploaded there.**"
                "\n\n**Note:** Requires a Hugging Face token (set via `HUGGING_FACE_HUB_TOKEN` environment variable if your models need it) "
                "and `ffmpeg` for non-WAV audio files."
            )

            if not models_loaded:
                gr.Markdown(
                    "**<span style='color:red'>ERROR: Transcription models (from Phase 1) failed to load. "
                    "Please check the console output where you ran the script for details. "
                    "This phase of the application cannot proceed.</span>**"
                )
            else:
                with gr.Row():
                    audio_input_phase1 = gr.Audio(label="Upload Audio File", type="filepath", sources=["upload", "microphone"])

                transcribe_button_phase1 = gr.Button("Transcribe Interview", variant="primary")

                with gr.Row():
                    output_transcript_phase1 = gr.Textbox(
                        label="Formatted Transcript (Phase 1)",
                        lines=15,
                        interactive=False,
                        placeholder="Transcript will appear here..."
                    )
                with gr.Row():
                    output_raw_json_phase1 = gr.JSON(
                        label="Raw Transcript Data (JSON) - Usable in Phase 2"
                    )
                
                # If transcribe_interview is async, this click handler will also work fine.
                transcribe_button_phase1.click(
                    fn=transcribe_interview,
                    inputs=audio_input_phase1,
                    outputs=[output_transcript_phase1, output_raw_json_phase1],
                    show_progress="full"
                )

                gr.Examples(
                    examples=[
                        # ["sample/example_interview.wav"], # Ensure files exist in 'sample' folder
                    ],
                    inputs=audio_input_phase1,
                    outputs=[output_transcript_phase1, output_raw_json_phase1],
                    fn=transcribe_interview,
                    cache_examples="lazy",
                    label="Example Audio Files (Phase 1) - Add paths to local audio files"
                )
                gr.Markdown("Make sure to have a `sample` folder with audio files if you uncomment and use the examples above.")


        with gr.TabItem("Phase 2: Transcript Assessment (RAG)"):
            gr.Markdown("## Phase 2: Assess Interview Transcript using RAG")
            gr.Markdown(
                "Upload an interview transcript (JSON format) or use the transcript generated from Phase 1. "
                "Optionally, provide a Reference Knowledge Base (Markdown/Text file) to generate an assessment report. "
                "The system uses a RAG (Retrieval Augmented Generation) pipeline for this analysis."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    transcript_file_phase2 = gr.File(
                        label="Upload Interview Transcript (JSON - Overrides Phase 1 output if provided)",
                        file_types=[".json"]
                    )
                    kb_file_phase2 = gr.File(
                        label="Reference Knowledge Base (MD/Text - Optional)",
                        file_types=[".md", ".txt"]
                    )
                    submit_button_phase2 = gr.Button("Generate Assessment Report", variant="primary")
                with gr.Column(scale=2):
                    gr.Markdown("### Assessment Summary (MD)")
                    output_summary_md_phase2 = gr.Markdown(label="Report Summary Content")
                    gr.Markdown("### Detailed JSON Report")
                    output_report_json_phase2 = gr.JSON(label="Full Report JSON Data")
                    output_error_text_phase2 = gr.Textbox(
                        label="Errors/Status (Phase 2)",
                        interactive=False,
                        placeholder="Process status and errors will appear here..."
                    )

            submit_button_phase2.click(
                fn=smart_run_pipeline_gradio, # This is now async def
                inputs=[
                    transcript_file_phase2,
                    output_raw_json_phase1,
                    kb_file_phase2
                ],
                outputs=[output_summary_md_phase2, output_report_json_phase2, output_error_text_phase2],
                show_progress="full"
            )

            example_transcript_path = "sample_transcript.json"
            example_kb_path = "sample_reference_kb.md"

            gr.Examples(
                examples=[
                    [example_transcript_path, None, example_kb_path],
                    [example_transcript_path, None, None],
                ],
                inputs=[transcript_file_phase2, output_raw_json_phase1, kb_file_phase2],
                outputs=[output_summary_md_phase2, output_report_json_phase2, output_error_text_phase2],
                fn=smart_run_pipeline_gradio, # Also calls the async wrapper
                cache_examples="lazy",
                label="Example Assessment Inputs (Phase 2) - Uses sample files created on startup"
            )
            gr.Markdown("--- \n *Assessment powered by LangGraph and AI.*")

# === Launch the App ===
if __name__ == "__main__":
    print("\nPreparing sample files for Phase 2 examples (if not present)...")
    sample_transcript_path_main = "sample_transcript.json"
    sample_kb_path_main = "sample_reference_kb.md"

    if not os.path.exists(sample_transcript_path_main):
        if 'load_sample_transcript' in globals() and callable(load_sample_transcript):
            try:
                _ = load_sample_transcript(sample_transcript_path_main)
            except Exception as e:
                print(f"Could not create/load sample_transcript.json using load_sample_transcript: {e}")
        else:
            print("load_sample_transcript function not available. Cannot create sample_transcript.json.")

    if not os.path.exists(sample_kb_path_main):
        try:
            print(f"Creating sample reference KB: {sample_kb_path_main}")
            with open(sample_kb_path_main, "w", encoding='utf-8') as f:
                f.write("# Job Role: Senior Software Engineer\n\n## Key Expectations & Responsibilities\n- Design, develop, test, deploy, maintain and improve software.\n- Manage individual project priorities, deadlines and deliverables.\n\n## Ideal Answer Points for 'System Design Question'\n- Clarify requirements.\n- Define system interface.\n- Estimate scale.\n- Discuss trade-offs of different approaches.\n- Identify bottlenecks and propose solutions.")
            print(f"Created default {sample_kb_path_main} for local testing/examples.")
        except Exception as e:
            print(f"Could not create {sample_kb_path_main}: {e}")

    sample_audio_dir = "sample"
    if not os.path.exists(sample_audio_dir):
        try:
            os.makedirs(sample_audio_dir, exist_ok=True)
            print(f"Created '{sample_audio_dir}' directory for sample audio files.")
            print(f"Please add sample audio files (e.g., example_interview.wav) to the '{sample_audio_dir}' directory for Phase 1 examples to work if you uncomment them.")
        except Exception as e:
            print(f"Could not create sample directory '{sample_audio_dir}': {e}")

    print("\nLaunching Gradio Interface...")
    demo.launch(server_name="0.0.0.0", server_port=8000, share=False, debug=True, pwa=True)