import os
import json
import re
import uuid
import html
import asyncio
import traceback
from typing import List, Dict, Optional, Any, TypedDict
import tempfile

import gradio as gr
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- Langchain Imports ---
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document # For text splitter if using Document objects

# --- Optional Text Splitter ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
    print("[INFO] langchain_text_splitters found.")
except ImportError:
    TEXT_SPLITTER_AVAILABLE = False
    print("[WARN] langchain_text_splitters not available. Using basic newline splitting for KB.")

# --- Load Environment Variables ---
load_dotenv() # Ensure your GROQ_API_KEY is in a .env file or environment

from data_models import (
    DialogueTurn, TopicSegment, RetrievedReferenceChunk, TopicLevelAssessment,
    TopicCriterionAssessment, OverallReportData, InterviewAssessmentReport, ASSESSMENT_CRITERIA
)

from data_models import ( parse_emotion_from_text, preprocess_transcript ,
basic_chunker, load_sample_transcript, cosine_similarity )

from data_models import LLMClient as llm_client_instance, EmbeddingClient as embedding_client_instance
# These are initialized once when the script starts.

# --- LangGraph State Definition ---
class InterviewGraphState(TypedDict):
    raw_transcript_path: str
    reference_kb_path: Optional[str]
    processed_transcript: Optional[List[DialogueTurn]]
    topic_segments: Optional[List[TopicSegment]]
    processed_reference_kb_chunks: Optional[List[Dict[str, Any]]]
    reference_kb_embeddings: Optional[Dict[str, List[float]]]
    assessment_criteria: List[Dict] # Retained as a guide
    current_topic_index: int
    current_topic_data: Optional[TopicSegment]
    candidate_speech_for_current_topic_str: Optional[str]
    retrieved_reference_chunks_for_current_topic: Optional[List[RetrievedReferenceChunk]]
    topic_level_assessments: List[TopicLevelAssessment]
    final_assessment_report: Optional[InterviewAssessmentReport]
    error_message: Optional[str]

# --- Instantiate clients for LangGraph nodes (using structured output binding) ---
# These specific instances are configured for structured output with Pydantic models
from pydantic import RootModel
llm_topic_segmenter = llm_client_instance.with_structured_output(RootModel[list[TopicSegment]])
llm_assessor = llm_client_instance.with_structured_output(TopicLevelAssessment)
llm_summarizer = llm_client_instance.with_structured_output(OverallReportData)
# embedding_client is already an instance, no further binding needed for its methods
embedding_client = embedding_client_instance # Use the global instance

# --- LangGraph Nodes ---
async def load_and_preprocess_transcript_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Load and Preprocess Transcript ---")
    try:
        raw_transcript_path = state["raw_transcript_path"]
        raw_data = load_sample_transcript(raw_transcript_path) # This function handles file reading
        processed_turns = preprocess_transcript(raw_data)
        print(f"Successfully processed {len(processed_turns)} dialogue turns.")
        return {
            "processed_transcript": processed_turns,
            "assessment_criteria": ASSESSMENT_CRITERIA, # Changed from "individual_assessments"
            "topic_level_assessments": [], # Initialize for topic-based assessments
            "error_message": None
        }
    except Exception as e:
        print(f"Error in load_and_preprocess_transcript_node: {e}\n{traceback.format_exc()}")
        return {"error_message": f"Failed to load/preprocess transcript: {e}"}

async def load_reference_kb_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Load Reference Knowledge Base ---")
    if state.get("error_message"): return {}
    kb_path = state.get("reference_kb_path")
    if not kb_path or not os.path.exists(kb_path):
        print(f"[WARN] Reference KB path '{kb_path}' not provided/found. Proceeding without KB.")
        return {"processed_reference_kb_chunks": [], "reference_kb_embeddings": {}, "error_message": state.get("error_message")}
    try:
        with open(kb_path, 'r', encoding='utf-8') as f: kb_text_content = f.read()
        chunks_text_list: List[str]
        if TEXT_SPLITTER_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, length_function=len, add_start_index=True)
            # If your splitter expects Document objects:
            docs = [Document(page_content=kb_text_content, metadata={"source": os.path.basename(kb_path)})]
            chunk_docs = text_splitter.split_documents(docs)
            chunks_text_list = [doc.page_content for doc in chunk_docs]
            # chunks_text_list = text_splitter.split_text(kb_text_content)
        else:
            chunks_text_list = basic_chunker(kb_text_content, chunk_size=300, chunk_overlap=30)

        if not chunks_text_list:
            print("[WARN] Reference KB content resulted in no text chunks.")
            return {"processed_reference_kb_chunks": [], "reference_kb_embeddings": {}, "error_message": "KB chunking failed."}
        print(f"  Reference KB chunked into {len(chunks_text_list)} pieces.")

        processed_chunks = [{"chunk_id": f"ref_chunk_{i}", "text": text, "source": os.path.basename(kb_path)} for i, text in enumerate(chunks_text_list)]
        
        # Use embed_documents for batch embedding (synchronous)
        # chunk_embeddings_list = embedding_client.embed_documents([chunk["text"] for chunk in processed_chunks])
        chunk_embeddings_list = [ embedding_client.embed_query(x) for x in [chunk["text"] for chunk in processed_chunks]]
        
        kb_embeddings_map = {chunk["chunk_id"]: emb for chunk, emb in zip(processed_chunks, chunk_embeddings_list)}
        print(f"  Successfully embedded {len(kb_embeddings_map)} KB chunks.")
        return {"processed_reference_kb_chunks": processed_chunks, "reference_kb_embeddings": kb_embeddings_map, "error_message": None}
    except Exception as e:
        print(f"Error in load_reference_kb_node: {e}\n{traceback.format_exc()}")
        return {"processed_reference_kb_chunks": [], "reference_kb_embeddings": {}, "error_message": f"Failed to load/process reference KB: {e}"}

async def segment_topics_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Segment Topics ---")
    if state.get("error_message"): return {}
    processed_transcript = state.get("processed_transcript", []) # Added default empty list
    if not processed_transcript: return {"error_message": "No processed transcript for topic segmentation."}

    transcript_for_llm = "\n".join([f"Turn {t.turn_id} ({t.speaker}): {t.clean_text if t.clean_text else t.raw_text}" for t in processed_transcript])
    system_prompt_str = "You are an expert in analyzing conversational flow. Respond ONLY with a valid JSON list of topic objects, where each object has 'topic_id', 'topic_label', 'start_turn_id', and 'end_turn_id'."
    prompt_str = f'''
    Segment the following interview dialogue into distinct topics.
    Ensure 'start_turn_id' and 'end_turn_id' are valid integer turn IDs from the transcript.
    'topic_id' should be a short unique string like "T1", "T2".
    Transcript:
    {transcript_for_llm}
    JSON Output:
    '''
    try:
        # llm_topic_segmenter is already bound to List[TopicSegment]
        # Langchain's structured output handles the conversion from model's string output to Pydantic list
        from google import genai
        client = genai.Client(api_key='AIzaSyBKQsap0OCWcqXR0GZfBDA1hJBouchFzA8')
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{system_prompt_str}, {prompt_str}",
            config={
                "response_mime_type": "application/json",
                "response_schema": RootModel[list[TopicSegment]],
                },
                )
        '''
        response_content = await llm_topic_segmenter.ainvoke([ # Use ainvoke for async
            SystemMessage(content=system_prompt_str),
            HumanMessage(content=prompt_str)
        ])
        # Assuming response_content is already List[TopicSegment] due to .with_structured_output
        topic_segments: List[TopicSegment] = response_content
        '''
        from pydantic import parse_obj_as
        topic_segments: List[TopicSegment] = parse_obj_as(List[TopicSegment], json.loads(response.text))        

        if not topic_segments or not isinstance(topic_segments, list) or not all(isinstance(ts, TopicSegment) for ts in topic_segments):
            print(f"[WARN] Topic segmentation did not return a valid list of TopicSegment. Received: {topic_segments}")
            topic_segments = [TopicSegment(topic_id="T_FALLBACK", topic_label="Entire Interview (Fallback)", start_turn_id=0, end_turn_id=len(processed_transcript) -1 if processed_transcript else 0)]
            print("[INFO] Using fallback single topic.")
        else:
            print(f"Successfully segmented into {len(topic_segments)} topics.")
            for topic in topic_segments: print(f"  - ID: {topic.topic_id}, Label: {topic.topic_label}, Turns: {topic.start_turn_id}-{topic.end_turn_id}")
        return {"topic_segments": topic_segments, "error_message": None}
    except Exception as e:
        print(f"Error in segment_topics_node: {e}\n{traceback.format_exc()}")
        # Fallback if LLM call fails
        fallback_topics = [TopicSegment(topic_id="T_FALLBACK_ERROR", topic_label="Entire Interview (Segmentation Error)", start_turn_id=0, end_turn_id=len(processed_transcript) -1 if processed_transcript else 0)]
        return {"error_message": f"Topic segmentation failed: {e}", "topic_segments": fallback_topics}

def initialize_assessment_loop_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Initialize Topic Assessment Loop ---")
    if state.get("error_message"): return {}
    if not state.get("topic_segments"):
        return {"error_message": "Cannot initialize assessment loop: Topic segments are missing."}
    return {
        "current_topic_index": 0,
        # "topic_level_assessments": [], # Already initialized in load_and_preprocess
        "error_message": None
    }

def select_next_topic_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Select Next Topic ---")
    if state.get("error_message"): return {}
    topic_segments = state.get("topic_segments", [])
    current_idx = state.get("current_topic_index", 0)
    if current_idx < len(topic_segments):
        current_topic = topic_segments[current_idx]
        print(f"Selected topic ({current_idx + 1}/{len(topic_segments)}): '{current_topic.topic_label}' (ID: {current_topic.topic_id})")
        return {"current_topic_data": current_topic, "error_message": None}
    else:
        print("All topics processed.")
        return {"current_topic_data": None, "error_message": None}

def should_continue_topic_assessment_node(state: InterviewGraphState) -> str:
    print("\n--- Condition: Should Continue Topic Assessment? ---")
    if state.get("error_message") and state.get("current_topic_data") is None and state.get("current_topic_index", 0) == 0:
        print(f"Critical error detected before topic loop: {state['error_message']}. Halting.")
        return "error_handler"
    if state.get("current_topic_data") is not None:
        print("Yes, more topics to assess.")
        return "retrieve_reference_for_topic"
    else:
        print("No more topics. Proceeding to final report aggregation.")
        return "aggregate_final_report"

async def retrieve_reference_for_topic_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Retrieve Reference Info for Current Topic ---")
    if state.get("error_message") and not state.get("current_topic_data"): return {}
    current_topic = state.get("current_topic_data")
    if not current_topic: return {"error_message": "No current topic selected for reference retrieval."}

    print(f"Retrieving reference KB info for topic: '{current_topic.topic_label}'")
    processed_reference_chunks = state.get("processed_reference_kb_chunks", [])
    reference_kb_embeddings = state.get("reference_kb_embeddings", {})
    processed_transcript = state.get("processed_transcript", []) # Added default

    if not processed_reference_chunks or not reference_kb_embeddings:
        print("[WARN] Reference KB not loaded/embedded. Cannot retrieve reference info for this topic.")
        return {"retrieved_reference_chunks_for_current_topic": [], "error_message": state.get("error_message")}

    candidate_speech_this_topic_turns = []
    if processed_transcript: # Check if transcript exists
        for turn in processed_transcript:
            if current_topic.start_turn_id <= turn.turn_id <= current_topic.end_turn_id and \
               turn.speaker.lower() == "candidate" and turn.clean_text:
                candidate_speech_this_topic_turns.append(turn)
    
    query_text_for_kb = current_topic.topic_label
    if candidate_speech_this_topic_turns:
        query_text_for_kb += ". Candidate mentioned: " + " ".join(t.clean_text for t in candidate_speech_this_topic_turns[:2])
    
    # Use embed_query (synchronous)
    query_embedding = embedding_client.embed_query(query_text_for_kb[:512])

    relevant_kb_chunks_with_scores = []
    for chunk_id, chunk_embedding in reference_kb_embeddings.items():
        sim = cosine_similarity(query_embedding, chunk_embedding)
        chunk_data = next((c for c in processed_reference_chunks if c["chunk_id"] == chunk_id), None)
        if chunk_data:
            relevant_kb_chunks_with_scores.append({
                "chunk_id": chunk_id, "text": chunk_data["text"],
                "source_document": chunk_data.get("source"), "similarity_score": sim
            })
    relevant_kb_chunks_with_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_k_ref = 3
    top_retrieved_chunks_data = relevant_kb_chunks_with_scores[:top_k_ref]
    retrieved_reference_chunks = [RetrievedReferenceChunk(**data) for data in top_retrieved_chunks_data]
    print(f"  Retrieved {len(retrieved_reference_chunks)} relevant chunks from KB for topic '{current_topic.topic_label}'.")
    return {"retrieved_reference_chunks_for_current_topic": retrieved_reference_chunks, "error_message": None}

async def assess_candidate_performance_on_topic_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Assess Candidate Performance on Topic ---")
    if state.get("error_message") and not state.get("current_topic_data"): return {}
    current_topic = state.get("current_topic_data")
    reference_chunks = state.get("retrieved_reference_chunks_for_current_topic", [])
    processed_transcript = state.get("processed_transcript", [])
    assessment_criteria_guide = state.get("assessment_criteria", [])

    if not current_topic: return {"error_message": "Current topic data not available for assessment."}
    if not processed_transcript: return {"error_message": "Processed transcript not available."}

    print(f"Assessing candidate performance on topic: '{current_topic.topic_label}' (ID: {current_topic.topic_id})")
    candidate_speech_this_topic_turns = []
    for turn in processed_transcript:
        if current_topic.start_turn_id <= turn.turn_id <= current_topic.end_turn_id and \
           turn.speaker.lower() == "candidate" and turn.clean_text:
            candidate_speech_this_topic_turns.append(turn)

    if not candidate_speech_this_topic_turns:
        print(f"  No candidate speech found for topic '{current_topic.topic_label}'. Skipping detailed assessment.")
        assessment_result = TopicLevelAssessment(
            topic_id=current_topic.topic_id, topic_label=current_topic.topic_label,
            candidate_contribution_summary="Candidate made no significant verbal contribution on this topic.",
            key_candidate_statements=[], reference_kb_alignment="N/A due to no candidate input.",
            overall_topic_performance_score=None, detailed_criteria_observations=[]
        )
        current_assessments = state.get("topic_level_assessments", [])
        current_assessments.append(assessment_result)
        return {"topic_level_assessments": current_assessments, "current_topic_index": state["current_topic_index"] + 1, "error_message": state.get("error_message")}

    candidate_speech_on_topic_str = "\n".join([f"Turn {t.turn_id} Candidate ({t.emotion if t.emotion else 'N/A'}): {t.clean_text}" for t in candidate_speech_this_topic_turns])

    reference_info_str = "\n".join([f"- {chunk.text} (Source: {chunk.source_document or 'KB'})" for chunk in reference_chunks]) if reference_chunks else "No specific reference information was retrieved for this topic."

    criteria_guidance_str = "\nWhen assessing, consider the following aspects (if applicable to the topic and candidate's speech):\n"
    for crit in assessment_criteria_guide: criteria_guidance_str += f"- {crit['criterion']}: {crit['description']} (Score on a 1-5 scale where relevant: {crit['scoring_guide']})\n"

    system_prompt_str = f'''
    You are an expert interview assessor. Your task is to evaluate a candidate's performance on a specific interview topic.
    You will be given the topic label, the candidate's statements related to this topic, and relevant reference material/ideal points from a knowledge base.
    Respond ONLY with a valid JSON object matching the 'TopicLevelAssessment' structure, including:
    - "topic_id": "{current_topic.topic_id}"
    - "topic_label": "{current_topic.topic_label}"
    - "candidate_contribution_summary": A brief summary of what the candidate discussed for this topic.
    - "key_candidate_statements": A list of 1-3 key verbatim quotes from the candidate on this topic.
    - "reference_kb_alignment": How well did the candidate's contribution align with the provided reference KB material for this topic? (e.g., "Strong alignment", "Partial alignment", "Misaligned", "KB not applicable/retrieved").
    - "overall_topic_performance_score": A single float score from 1.0 (Poor) to 5.0 (Excellent) for the candidate's overall performance *on this specific topic*.
    - "detailed_criteria_observations": An optional list of objects, where each object assesses a specific criterion for this topic:
    {{ "criterion_name": "Name of criterion", "score": float (1-5, optional), "observation": "Your specific observation for this criterion on this topic." }}
    Base these observations on the general assessment criteria provided below. Only include criteria relevant to the current topic and candidate's speech.
    Focus ONLY on the provided dialogue and reference material for this topic.
    '''
    prompt_str = f'''Interview Topic for Assessment: {current_topic.topic_label} (ID: {current_topic.topic_id})
    Reference Material / Ideal Points ...:
    ---
    {reference_info_str}
    ---
    Candidate's Speech on this Topic:
    ---
    {candidate_speech_on_topic_str}
    ---
    General Assessment Criteria to Guide Your Observations:
    {criteria_guidance_str}
    ---
    Please provide your assessment ... JSON Assessment:
    '''
    try:
        # llm_assessor is already bound to TopicLevelAssessment
        assessment_result = await llm_assessor.ainvoke([ # Use ainvoke
            SystemMessage(content=system_prompt_str),
            HumanMessage(content=prompt_str)
        ])
        if not assessment_result or not isinstance(assessment_result, TopicLevelAssessment):
            print(f"[WARN] Topic assessment LLM call invalid. Received: {assessment_result}")
            assessment_result = TopicLevelAssessment(topic_id=current_topic.topic_id, topic_label=current_topic.topic_label, candidate_contribution_summary="Assessment generation failed.", key_candidate_statements=[], reference_kb_alignment="Error.", overall_topic_performance_score=None)
        else:
            assessment_result.topic_id = current_topic.topic_id # Ensure correct IDs
            assessment_result.topic_label = current_topic.topic_label
            print(f"  Assessment successful for topic '{assessment_result.topic_label}': Score {assessment_result.overall_topic_performance_score}")
        current_assessments = state.get("topic_level_assessments", [])
        current_assessments.append(assessment_result)
        return {"topic_level_assessments": current_assessments, "current_topic_index": state["current_topic_index"] + 1, "error_message": state.get("error_message")}
    except Exception as e:
        print(f"Error in assess_candidate_performance_on_topic_node for topic '{current_topic.topic_label}': {e}\n{traceback.format_exc()}")
        error_assessment = TopicLevelAssessment(topic_id=current_topic.topic_id, topic_label=current_topic.topic_label, candidate_contribution_summary=f"Error: {str(e)}", key_candidate_statements=[], reference_kb_alignment="Error.", overall_topic_performance_score=None)
        current_assessments = state.get("topic_level_assessments", [])
        current_assessments.append(error_assessment)
        return {"topic_level_assessments": current_assessments, "current_topic_index": state["current_topic_index"] + 1, "error_message": f"Failed to assess topic '{current_topic.topic_label}': {e}"}

async def aggregate_final_report_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Aggregate Final Report ---")
    final_error_message = state.get("error_message")
    topic_assessments = state.get("topic_level_assessments", [])
    if not topic_assessments and not final_error_message: final_error_message = "No topic-level assessments to aggregate."
    print(f"Aggregating {len(topic_assessments)} topic-level assessments.")

    topic_assessments_summary_str = "\n\n".join([f"Topic: {a.topic_label} (Score: {a.overall_topic_performance_score if a.overall_topic_performance_score is not None else 'N/A'})\n"
        f"  Summary: {a.candidate_contribution_summary}\n"
        f"  KB Alignment: {a.reference_kb_alignment}\n"
        f"  Key Statements: {'; '.join(a.key_candidate_statements) if a.key_candidate_statements else 'None'}\n" +
        ("  Detailed Criteria Observations:\n" + "\n".join([f"    - {crit_obs.criterion_name} (Score: {crit_obs.score if crit_obs.score is not None else 'N/A'}): {crit_obs.observation}" for crit_obs in a.detailed_criteria_observations]) if a.detailed_criteria_observations else "") for a in topic_assessments]) if topic_assessments else "No topic assessments available to summarize." # Simplified for prompt

    system_prompt_str = f'''
    You are an expert HR analyst. Synthesize topic-level interview assessments into an overall summary, key strengths across topics, and key areas for improvement across topics. Respond ONLY with a valid JSON object for OverallReportData.
    '''
    prompt_str = f'''
    Topic-Level Assessments Summary:\n---\n{topic_assessments_summary_str}\n---\n    
    Based ONLY on these topic assessments (and being mindful of any errors if no assessments are present), provide the OverallReportData JSON (overall_summary, key_strengths_across_topics, key_areas_for_improvement_across_topics):
    If there are no topic assessments or critical errors prevented meaningful assessment, reflect this in the summary.
    '''
    try:
        # llm_summarizer is bound to OverallReportData
        overall_data = await llm_summarizer.ainvoke([ # Use ainvoke
            SystemMessage(content=system_prompt_str),
            HumanMessage(content=prompt_str)
        ])
        if not overall_data or not isinstance(overall_data, OverallReportData):
            print(f"[WARN] Final report aggregation LLM call invalid. Received: {overall_data}")
            overall_data = OverallReportData(overall_summary="Automated summary failed.", strengths=[], areas_for_improvement=[])
        else: print("  Overall summary generated.")
        final_report = InterviewAssessmentReport(
            topic_by_topic_assessments=topic_assessments, 
            overall_summary=overall_data.overall_summary, 
            key_strengths_across_topics=overall_data.strengths, key_areas_for_improvement_across_topics=overall_data.areas_for_improvement
            )
        return {"final_assessment_report": final_report, "error_message": final_error_message}
    except Exception as e:
        print(f"Error in aggregate_final_report_node: {e}\n{traceback.format_exc()}")
        final_report_err = InterviewAssessmentReport(topic_by_topic_assessments=topic_assessments, overall_summary=f"Error: {e}", strengths=[], key_areas_for_improvement_across_topics=[])
        return {"final_assessment_report": final_report_err, "error_message": final_error_message or f"Failed to aggregate final report: {e}"}

def error_handler_node(state: InterviewGraphState) -> Dict[str, Any]:
    err_msg = state.get('error_message', "Unknown error in pipeline.")
    print(f"\n--- Node: Error Handler ---\nPipeline Error: {err_msg}")
    topic_assessments = state.get("topic_level_assessments", [])
    if topic_assessments:
        report = InterviewAssessmentReport(topic_by_topic_assessments=topic_assessments, overall_summary=f"Pipeline error: {err_msg}. Partial assessments provided.", key_strengths_across_topics=[], key_areas_for_improvement_across_topics=[])
        return {"final_assessment_report": report, "error_message": err_msg}
    return {"error_message": err_msg}

# --- Graph Definition ---
def build_graph():
    workflow = StateGraph(InterviewGraphState)
    workflow.add_node("load_preprocess_transcript", load_and_preprocess_transcript_node)
    workflow.add_node("load_reference_kb", load_reference_kb_node)
    workflow.add_node("segment_topics", segment_topics_node)    
    workflow.add_node("init_topic_assessment_loop", initialize_assessment_loop_node)
    workflow.add_node("select_next_topic", select_next_topic_node)
    workflow.add_node("retrieve_reference_for_topic", retrieve_reference_for_topic_node)
    workflow.add_node("assess_topic_performance", assess_candidate_performance_on_topic_node)
    workflow.add_node("aggregate_final_report", aggregate_final_report_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("load_preprocess_transcript")
    workflow.add_edge("load_preprocess_transcript", "load_reference_kb")
    workflow.add_edge("load_reference_kb", "segment_topics")
    workflow.add_edge("segment_topics", "init_topic_assessment_loop")
    workflow.add_edge("init_topic_assessment_loop", "select_next_topic")
    workflow.add_conditional_edges("select_next_topic", should_continue_topic_assessment_node,
        {"retrieve_reference_for_topic": "retrieve_reference_for_topic", "aggregate_final_report": "aggregate_final_report", "error_handler": "error_handler"})
    workflow.add_edge("retrieve_reference_for_topic", "assess_topic_performance")
    workflow.add_edge("assess_topic_performance", "select_next_topic")
    workflow.add_edge("aggregate_final_report", END)
    workflow.add_edge("error_handler", END)
    app = workflow.compile()
    return app

# --- Gradio App ---
async def run_pipeline_gradio(transcript_file_obj, kb_file_obj, progress=gr.Progress(track_tqdm=True)):
    """
    Gradio handler function to run the assessment pipeline.
    Accepts Gradio File objects, saves them temporarily, and runs the pipeline.
    """
    if not transcript_file_obj:
        return "Error: Interview transcript file is required.", None, None

    transcript_path = transcript_file_obj.name # Gradio File object has .name attribute for temp path
    kb_path = None
    if kb_file_obj:
        kb_path = kb_file_obj.name
        print(f"Using uploaded KB file: {kb_path}")
    else:
        # Create a dummy KB file if none is uploaded, so the pipeline doesn't break
        # Or, make KB mandatory by returning an error if kb_file_obj is None
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md", encoding='utf-8') as tmp_kb:
            tmp_kb.write("# Sample Reference KB\n\nThis is a placeholder knowledge base.\n- Ideal communication is clear and concise.\n- Problem solving should be logical.\n")
            kb_path = tmp_kb.name
        print(f"No KB file uploaded, using a temporary dummy KB: {kb_path}")


    # The main pipeline execution logic (adapted from your if __name__ == "__main__":)
    print(f"Current Working Directory for Gradio run: {os.getcwd()}")
    print("Building RAG assessment graph for Gradio...")
    rag_app = build_graph() # Build graph on each call, or make it global (consider thread safety if global)
    print("Graph built for Gradio.")

    initial_state = InterviewGraphState(
        raw_transcript_path=transcript_path, reference_kb_path=kb_path,
        processed_transcript=None, topic_segments=None,
        processed_reference_kb_chunks=None, reference_kb_embeddings=None,
        assessment_criteria=ASSESSMENT_CRITERIA,
        current_topic_index=0, current_topic_data=None,
        candidate_speech_for_current_topic_str=None,
        retrieved_reference_chunks_for_current_topic=None,
        topic_level_assessments=[],
        final_assessment_report=None, error_message=None
    )
    config = {"recursion_limit": 50}
    print(f"\nStarting Gradio pipeline for transcript: '{transcript_path}' and KB: '{kb_path}'")
    
    summary_output = "Processing..."
    json_output = {"status": "Processing..."}
    error_output = ""

    try:
        # progress_callback = lambda x: progress(x['step'] / total_steps, desc=x['desc']) # If you can get total steps
        # For now, just general progress updates. LangGraph's astream_events can be used for finer progress.
        # Using astream_events to provide some progress feedback
        '''
        async for event in rag_app.astream(initial_state, config=config):
            # event is a dictionary, the last one will contain the final state in event['__end__']
            # or you can check event.keys() for node names to update progress
            # For simplicity, we'll just get the final state after the loop.
            # This is a simplified progress, actual progress tracking would require more detailed event handling.
            current_keys = list(event.keys())
            print(f"Current event keys: {current_keys}")
            if current_keys:
                last_node_processed = current_keys[-1]
                progress(0.1 + (hash(last_node_processed) % 81) / 100.0 , desc=f"Processing node: {last_node_processed}") # Crude progress

            if END in event: # Check if the graph has finished
                final_state_invoke = event[END]
                break
            else: # Should not happen if graph has an END state properly reached
                final_state_invoke = initial_state # Fallback
                error_output = "Pipeline finished without reaching END state."
        '''
        final_state_invoke = await rag_app.ainvoke(initial_state, config=config)
        if final_state_invoke:            
            print("Final state reached.")
            print(f"Final state keys: {list(final_state_invoke.keys())}")
            final_report_model = final_state_invoke.get("final_assessment_report")                        
            print(f"topic_level_assessments: {final_state_invoke.get("topic_level_assessments")}")
            err_msg = final_state_invoke.get("error_message")

            if final_report_model and isinstance(final_report_model, InterviewAssessmentReport):
                print("Final report model successfully generated.")
                # Generate Markdown summary
                md_summary = f"# Interview Assessment Report\n\n"
                md_summary += f"## Overall Summary\n{final_report_model.overall_summary}\n\n"
                if final_report_model.key_strengths_across_topics:
                    md_summary += "## Key Strengths Across Topics\n"
                    for strength in final_report_model.key_strengths_across_topics: md_summary += f"- {strength}\n"
                    md_summary += "\n"
                if final_report_model.key_areas_for_improvement_across_topics:
                    md_summary += "## Key Areas for Improvement Across Topics\n"
                    for area in final_report_model.key_areas_for_improvement_across_topics: md_summary += f"- {area}\n"
                    md_summary += "\n"
                
                md_summary += "## Topic-by-Topic Assessments\n"
                for topic_assess in final_report_model.topic_by_topic_assessments:
                    md_summary += f"\n### Topic: {topic_assess.topic_label} (ID: {topic_assess.topic_id})\n"
                    md_summary += f"- **Candidate Contribution Summary:** {topic_assess.candidate_contribution_summary}\n"
                    md_summary += f"- **KB Alignment:** {topic_assess.reference_kb_alignment}\n"
                    md_summary += f"- **Overall Topic Score:** {topic_assess.overall_topic_performance_score if topic_assess.overall_topic_performance_score is not None else 'N/A'}\n"
                    if topic_assess.key_candidate_statements:
                        md_summary += "- **Key Candidate Statements:**\n"
                        for stmt in topic_assess.key_candidate_statements: md_summary += f"  - \"{stmt}\"\n"
                    if topic_assess.detailed_criteria_observations:
                        md_summary += "- **Detailed Criteria Observations:**\n"
                        for crit_obs in topic_assess.detailed_criteria_observations:
                            md_summary += f"  - **{crit_obs.criterion_name}** (Score: {crit_obs.score if crit_obs.score is not None else 'N/A'}): {crit_obs.observation}\n"
                
                summary_output = md_summary
                json_output = final_report_model.model_dump(mode='json') # Get dict for JSON component

                '''
                from datetime import datetime # For unique filenames
                if not os.path.exists(OUTPUT_REPORTS_DIR):
                    os.makedirs(OUTPUT_REPORTS_DIR)

                # Create a unique filename from the input transcript name
                base_transcript_name = "assessment" # Default
                if transcript_file_obj and hasattr(transcript_file_obj, 'name'):
                    # Get the original uploaded filename if available, otherwise use the temp path name
                    original_uploaded_filename = getattr(transcript_file_obj, 'orig_name', os.path.basename(transcript_file_obj.name))
                    base_transcript_name = os.path.splitext(original_uploaded_filename)[0]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"Assessment_Report_{base_transcript_name}_{timestamp}.json"
                current_saved_report_path = os.path.join(OUTPUT_REPORTS_DIR, report_filename)
                
                with open(current_saved_report_path, 'w', encoding='utf-8') as f:
                    json.dump(json_output_dict, f, indent=2) # Save the dict version
                print(f"Successfully saved assessment report to: {current_saved_report_path}")
                # This path is server-side. Gradio's gr.File output can make it downloadable.
                download_file_path = current_saved_report_path 
                # Update error_output to include success message and path
                status_message = f"Report successfully generated and saved to server at: {current_saved_report_path}"
                error_output = f"{err_msg}\n{status_message}" if err_msg else status_message
                '''

            
            if err_msg:
                error_output = f"Pipeline completed with an error: {err_msg}"
                if not final_report_model: # If error prevented report generation
                    summary_output = "Error: Could not generate report."
                    json_output = {"error": err_msg}
            
            if not final_report_model and not err_msg:
                 error_output = "Pipeline completed, but no final report was generated and no explicit error message was found."
                 summary_output = "No report generated."
                 json_output = {"status": "No report generated."}

        else:
            error_output = "Pipeline did not return a final state."
            summary_output = "Error: Pipeline execution failed."
            json_output = {"error": "Pipeline execution failed."}

    except Exception as e:
        print(f"Error during Gradio pipeline execution: {e}\n{traceback.format_exc()}")
        error_output = f"An unexpected error occurred: {str(e)}"
        summary_output = f"Error: {str(e)}"
        json_output = {"error": str(e), "traceback": traceback.format_exc()}
    
    # Clean up temporary KB file if one was created
    if kb_file_obj is None and kb_path and os.path.exists(kb_path) and "dummy_kb_placeholder" in kb_path: # Be more specific if needed
        try:
            os.remove(kb_path)
            print(f"Removed temporary dummy KB: {kb_path}")
        except Exception as e_clean:
            print(f"Error removing temporary dummy KB: {e_clean}")

    return summary_output, json_output, error_output


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
