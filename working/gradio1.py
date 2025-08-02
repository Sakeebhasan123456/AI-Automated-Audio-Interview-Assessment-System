from Phase1 import (
    transcribe_interview, initialize_models)
import gradio as gr
# === INITIALIZE MODELS ON STARTUP ===
models_loaded = initialize_models()


# === GRADIO INTERFACE ===
with gr.Blocks() as demo:
    gr.Markdown("# Audio Interview Assessment System (Phase 1 Demo)")
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
                ["sample/kate_andrews-trimmed.wav"],
                ["sample/kate_andrews-trimmed_longer.wav"]
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, pwa=True)