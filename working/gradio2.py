from Phase2 import ( run_pipeline_gradio, load_sample_transcript )
import gradio as gr
import os
# --- Define Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Audio Interview Assessment System") as demo:
    gr.Markdown("# Audio Interview Assessment System")
    gr.Markdown("Upload an interview transcript (JSON) and an optional Reference Knowledge Base (Markdown/Text) to generate an assessment.")

    with gr.Row():
        with gr.Column(scale=1):
            transcript_file = gr.File(label="Interview Transcript (JSON)", file_types=[".json"])
            kb_file = gr.File(label="Reference Knowledge Base (MD/Text - Optional)", file_types=[".md", ".txt"])
            submit_button = gr.Button("Generate Assessment Report", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("## Assessment Summary")
            output_summary_md = gr.Markdown(label="Report Summary")
            gr.Markdown("## Detailed JSON Report")
            output_report_json = gr.JSON(label="Full Report JSON")
            output_error_text = gr.Textbox(label="Errors/Status", interactive=False)

    submit_button.click(
        fn=run_pipeline_gradio,
        inputs=[transcript_file, kb_file],
        outputs=[output_summary_md, output_report_json, output_error_text]
    )

    gr.Examples(
        examples=[
            # To use examples, you'd need to have these files accessible by the Gradio app
            # Or provide a way for users to select pre-loaded examples.
            # For now, users will upload their own files.
            # ["sample_transcript.json", "sample_reference_kb.md"], # If these files are in the app's root
        ],
        inputs=[transcript_file, kb_file],
        outputs=[output_summary_md, output_report_json, output_error_text],
        fn=run_pipeline_gradio, # Cache examples can be problematic with file objects
        cache_examples=False
    )
    gr.Markdown("--- \n *Powered by LangGraph and AI.*")


if __name__ == "__main__":
    # Create dummy sample_transcript.json if it doesn't exist for local testing without upload
    if not os.path.exists("sample_transcript.json"):
        _ = load_sample_transcript("sample_transcript.json")
        print("Created default sample_transcript.json for local testing.")
    
    # Create dummy sample_reference_kb.md if it doesn't exist
    if not os.path.exists("sample_reference_kb.md"):
        print(f"Creating sample reference KB: sample_reference_kb.md")
        with open("sample_reference_kb.md", "w", encoding='utf-8') as f:
            f.write("# Job Role: Senior Python Developer\n\n")
            f.write("## Key Expectations & Responsibilities\n- Design, develop, and maintain robust backend APIs using Python and modern frameworks like FastAPI or Django.\n- Demonstrate strong understanding of asynchronous programming concepts.\n\n")
            f.write("## Ideal Answer Points for 'Problem Solving'\n- Clearly define the problem.\n- Discuss alternative solutions and trade-offs.\n- Explain the chosen solution logically.\n")
        print("Created default sample_reference_kb.md for local testing.")

    print("Starting Gradio App...")
    demo.launch()
