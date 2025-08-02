import os
import json
import re
import uuid
import html
import asyncio
import traceback
from typing import List, Dict, Optional, Any, TypedDict

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaverInMemory # For debugging state persistence

# --- Optional Text Splitter ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
    print("[INFO] langchain_text_splitters found.")
except ImportError:

    TEXT_SPLITTER_AVAILABLE = False
    print("[WARN] langchain_text_splitters not available. Using basic newline splitting for KB.")

from data_models import (
    DialogueTurn, TopicSegment, RetrievedReferenceChunk, TopicLevelAssessment,
    TopicCriterionAssessment, OverallReportData, InterviewAssessmentReport, ASSESSMENT_CRITERIA
)

from data_models import ( parse_emotion_from_text, preprocess_transcript , 
basic_chunker, load_sample_transcript, cosine_similarity )

from langchain_core.messages import HumanMessage, SystemMessage
# global instance
from data_models import LLMClient, EmbeddingClient


# --- LangGraph State Definition ---
# (main.py - InterviewGraphState)
class InterviewGraphState(TypedDict):
    raw_transcript_path: str
    reference_kb_path: Optional[str]

    processed_transcript: Optional[List[DialogueTurn]]
    # candidate_turn_embeddings: Optional[Dict[int, List[float]]] # Less critical if assessing whole topic speech
    topic_segments: Optional[List[TopicSegment]]

    processed_reference_kb_chunks: Optional[List[Dict[str, Any]]]
    reference_kb_embeddings: Optional[Dict[str, List[float]]]

    assessment_criteria: List[Dict] # Retained as a guide for the LLM

    # --- Loop control for topics (NEW/MODIFIED) ---
    current_topic_index: int
    current_topic_data: Optional[TopicSegment] # The TopicSegment object being processed

    # --- Context for the current topic (NEW/MODIFIED) ---
    candidate_speech_for_current_topic_str: Optional[str] # Formatted string of candidate's speech
    retrieved_reference_chunks_for_current_topic: Optional[List[RetrievedReferenceChunk]]

    topic_level_assessments: List[TopicLevelAssessment] # MODIFIED: Stores assessments per topic
    # final_report_input: Optional[Dict] # Can be removed if OverallReportData is used directly
    final_assessment_report: Optional[InterviewAssessmentReport]
    error_message: Optional[str]
    
# --- Instantiate clients ---
llm_topic_segmenter = LLMClient.with_structured_output(TopicSegment)
llm_assessor = LLMClient.with_structured_output(TopicLevelAssessment)
llm_summarizer = LLMClient.with_structured_output(OverallReportData)
embedding_client = EmbeddingClient


# --- LangGraph Nodes ---
async def load_and_preprocess_transcript_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Load and Preprocess Transcript ---")
    try:
        raw_transcript_path = state["raw_transcript_path"]
        raw_data = load_sample_transcript(raw_transcript_path)
        processed_turns = preprocess_transcript(raw_data)
        print(f"Successfully processed {len(processed_turns)} dialogue turns.")
        return {
            "processed_transcript": processed_turns,
            "assessment_criteria": ASSESSMENT_CRITERIA,
            "individual_assessments": [],
            "error_message": None
        }
    except Exception as e:
        print(f"Error in load_and_preprocess_transcript_node: {e}")
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
        chunks_text: List[str]
        if TEXT_SPLITTER_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, length_function=len, add_start_index=True) # Smaller chunks
            # Create documents for splitter if it expects them
            docs = [Document(page_content=kb_text_content, metadata={"source": os.path.basename(kb_path)})]
            chunks_text = text_splitter.split_documents(docs) # if splitting documents
            # chunks_text = text_splitter.split_text(kb_text_content)

        else:
            chunks_text = basic_chunker(kb_text_content, chunk_size=300, chunk_overlap=30) # Smaller chunks

        if not chunks_text:
            print("[WARN] Reference KB content resulted in no text chunks.")
            return {"processed_reference_kb_chunks": [], "reference_kb_embeddings": {}, "error_message": "KB chunking failed."}
        print(f"  Reference KB chunked into {len(chunks_text)} pieces.")

        processed_chunks = [{"chunk_id": f"ref_chunk_{i}", "text": text, "source": os.path.basename(kb_path)} for i, text in enumerate(chunks_text)]
        chunk_embeddings_list = await embedding_client.get_embeddings([chunk["text"] for chunk in processed_chunks])
        kb_embeddings_map = {chunk["chunk_id"]: emb for chunk, emb in zip(processed_chunks, chunk_embeddings_list)}
        print(f"  Successfully embedded {len(kb_embeddings_map)} KB chunks.")
        return {"processed_reference_kb_chunks": processed_chunks, "reference_kb_embeddings": kb_embeddings_map, "error_message": None}
    except Exception as e:
        print(f"Error in load_reference_kb_node: {e}\n{traceback.format_exc()}")
        return {"processed_reference_kb_chunks": [], "reference_kb_embeddings": {}, "error_message": f"Failed to load/process reference KB: {e}"}

async def segment_topics_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Segment Topics ---")
    if state.get("error_message"): return {}
    processed_transcript = state["processed_transcript"]
    if not processed_transcript: return {"error_message": "No processed transcript for topic segmentation."}

    transcript_for_llm = "\n".join([f"Turn {t.turn_id} ({t.speaker}): {t.clean_text if t.clean_text else t.raw_text}" for t in processed_transcript])

    system_prompt = "You are an expert in analyzing conversational flow. Respond ONLY with a valid JSON list of topic objects, each with 'topic_id', 'topic_label', 'start_turn_id', and 'end_turn_id'."

    prompt = f'''
    Segment the following interview dialogue into distinct topics. 
    Ensure 'start_turn_id' and 'end_turn_id' are valid integer turn IDs from the transcript. 
    'topic_id' should be a short unique string like \"T1\", \"T2\". 
    Transcript:
    {transcript_for_llm}
    JSON Output:
    '''

    try:
        topic_segments = await llm_topic_segmenter.invoke([
            SystemMessage(system_prompt),
            HumanMessage(prompt)            
        ])
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
        return {"error_message": f"Topic segmentation failed: {e}", "topic_segments": []}

async def embed_candidate_turns_node(state: InterviewGraphState) -> Dict[str, Any]:
    print("\n--- Node: Embed Candidate Turns ---")
    if state.get("error_message"): return {}
    processed_transcript = state["processed_transcript"]
    if not processed_transcript: return {"error_message": "No processed transcript for embedding."}
    candidate_texts_with_ids = [{"turn_id": t.turn_id, "text": t.clean_text} for t in processed_transcript if t.speaker.lower() == "candidate" and t.clean_text]
    if not candidate_texts_with_ids:
        print("No candidate text found to embed.")
        return {"candidate_turn_embeddings": {}, "error_message": None}
    texts_to_embed = [item["text"] for item in candidate_texts_with_ids]
    try:
        embeddings = await embedding_client.get_embeddings(texts_to_embed)
        candidate_embeddings_map = {item["turn_id"]: emb for item, emb in zip(candidate_texts_with_ids, embeddings)}
        print(f"Successfully embedded {len(candidate_embeddings_map)} candidate turns.")
        return {"candidate_turn_embeddings": candidate_embeddings_map, "error_message": None}
    except Exception as e:
        print(f"Error in embed_candidate_turns_node: {e}\n{traceback.format_exc()}")
        return {"error_message": f"Failed to embed candidate turns: {e}"}

def initialize_assessment_loop_node(state: InterviewGraphState) -> Dict[str, Any]: # MODIFIED
    print("\n--- Node: Initialize Topic Assessment Loop ---")
    if state.get("error_message"): return {}
    # Ensure topic_segments exist before trying to initialize loop
    if not state.get("topic_segments"):
        return {"error_message": "Cannot initialize assessment loop: Topic segments are missing."}
    return {
        "current_topic_index": 0,
        "topic_level_assessments": [], # Initialize list for new assessment type
        "error_message": None
    }

def select_next_topic_node(state: InterviewGraphState) -> Dict[str, Any]: # NEW/REPLACES select_next_criterion
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
        return {"current_topic_data": None, "error_message": None} # Signal loop completion

def should_continue_topic_assessment_node(state: InterviewGraphState) -> str: # NEW/REPLACES should_continue_assessment
    print("\n--- Condition: Should Continue Topic Assessment? ---")
    # If a critical error occurred that prevented even starting the loop (e.g., no topics)
    if state.get("error_message") and state.get("current_topic_data") is None and state.get("current_topic_index", 0) == 0:
        print(f"Critical error detected before topic loop: {state['error_message']}. Halting.")
        return "error_handler"

    if state.get("current_topic_data") is not None: # If a topic is selected
        print("Yes, more topics to assess.")
        return "retrieve_reference_for_topic"
    else: # No more topics
        print("No more topics. Proceeding to final report aggregation.")
        return "aggregate_final_report" # Ensure this matches the aggregate node name

async def retrieve_reference_for_topic_node(state: InterviewGraphState) -> Dict[str, Any]: # MODIFIED
    print("\n--- Node: Retrieve Reference Info for Current Topic ---")
    if state.get("error_message") and not state.get("current_topic_data"): return {} # Allow proceeding if error is from previous topic's assessment

    current_topic = state.get("current_topic_data")
    if not current_topic:
        return {"error_message": "No current topic selected for reference retrieval."}

    print(f"Retrieving reference KB info for topic: '{current_topic.topic_label}'")
    processed_reference_chunks = state.get("processed_reference_kb_chunks", [])
    reference_kb_embeddings = state.get("reference_kb_embeddings", {})

    if not processed_reference_chunks or not reference_kb_embeddings:
        print("[WARN] Reference KB not loaded/embedded. Cannot retrieve reference info for this topic.")
        return {"retrieved_reference_chunks_for_current_topic": [], "error_message": state.get("error_message")}

    # Use topic label + candidate speech summary from topic for embedding (more context)
    # First, get candidate speech for this topic
    candidate_speech_this_topic_turns = []
    processed_transcript = state.get("processed_transcript", [])
    for turn in processed_transcript:
        if current_topic.start_turn_id <= turn.turn_id <= current_topic.end_turn_id and \
           turn.speaker.lower() == "candidate" and turn.clean_text:
            candidate_speech_this_topic_turns.append(turn)
    
    # Create a query for retrieval from KB: topic label + first few candidate utterances
    query_text_for_kb = current_topic.topic_label
    if candidate_speech_this_topic_turns:
        query_text_for_kb += ". Candidate mentioned: " + " ".join(t.clean_text for t in candidate_speech_this_topic_turns[:2]) # First 2 utterances
    
    query_embedding = await embedding_client.embed_query(query_text_for_kb[:512]) # Limit query length

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


async def assess_candidate_performance_on_topic_node(state: InterviewGraphState) -> Dict[str, Any]: # NEW/REPLACES assess_candidate_response...
    print("\n--- Node: Assess Candidate Performance on Topic ---")
    # Allow proceeding if error is from previous topic's assessment, but not if current topic data is missing
    if state.get("error_message") and not state.get("current_topic_data"): return {}

    current_topic = state.get("current_topic_data")
    reference_chunks = state.get("retrieved_reference_chunks_for_current_topic", [])
    processed_transcript = state.get("processed_transcript", [])
    assessment_criteria_guide = state.get("assessment_criteria", []) # Guide for LLM

    if not current_topic: return {"error_message": "Current topic data not available for assessment."}
    if not processed_transcript: return {"error_message": "Processed transcript not available."}

    print(f"Assessing candidate performance on topic: '{current_topic.topic_label}' (ID: {current_topic.topic_id})")

    # Extract candidate's speech for this specific topic
    candidate_speech_this_topic_turns = []
    for turn in processed_transcript:
        if current_topic.start_turn_id <= turn.turn_id <= current_topic.end_turn_id and \
           turn.speaker.lower() == "candidate" and turn.clean_text:
            candidate_speech_this_topic_turns.append(turn)

    if not candidate_speech_this_topic_turns:
        print(f"  No candidate speech found for topic '{current_topic.topic_label}'. Skipping detailed assessment for this topic.")
        # Create a minimal assessment indicating no contribution or add to error
        assessment_result = TopicLevelAssessment(
            topic_id=current_topic.topic_id, topic_label=current_topic.topic_label,
            candidate_contribution_summary="Candidate made no significant verbal contribution on this topic.",
            key_candidate_statements=[], reference_kb_alignment="N/A due to no candidate input.",
            overall_topic_performance_score=None, # Or a low score like 1.0 or 2.0
            detailed_criteria_observations=[]
        )
        current_assessments = state.get("topic_level_assessments", [])
        current_assessments.append(assessment_result)
        return {
            "topic_level_assessments": current_assessments,
            "current_topic_index": state["current_topic_index"] + 1,
            "error_message": state.get("error_message") # Preserve previous non-critical error
        }

    candidate_speech_on_topic_str = "\n".join(
        [f"Turn {t.turn_id} Candidate ({t.emotion if t.emotion else 'N/A'}): {t.clean_text}"
         for t in candidate_speech_this_topic_turns]
    )

    reference_info_str = "\n".join([f"- {chunk.text} (Source: {chunk.source_document or 'KB'})" for chunk in reference_chunks]) \
        if reference_chunks else "No specific reference information was retrieved for this topic."

    # Construct criteria guidance for the LLM
    criteria_guidance_str = "\nWhen assessing, consider the following aspects (if applicable to the topic and candidate's speech):\n"
    for crit in assessment_criteria_guide:
        criteria_guidance_str += f"- {crit['criterion']}: {crit['description']} (Score on a 1-5 scale where relevant: {crit['scoring_guide']})\n"


    system_prompt = f'''
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

    prompt = f'''Interview Topic for Assessment: {current_topic.topic_label} (ID: {current_topic.topic_id})
    
    Reference Material / Ideal Points for this Topic (from Knowledge Base):
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
    Please provide your assessment for the candidate's performance *on this topic* in the specified JSON format for TopicLevelAssessment.
    JSON Assessment:
    '''
    try:
        assessment_result = await llm_assessor.invoke([
            SystemMessage(system_prompt),
            HumanMessage(prompt)            
        ])

        if not assessment_result or not isinstance(assessment_result, TopicLevelAssessment):
            print(f"[WARN] Topic assessment LLM call did not return valid TopicLevelAssessment. Received: {assessment_result}")
            assessment_result = TopicLevelAssessment(
                topic_id=current_topic.topic_id, topic_label=current_topic.topic_label,
                candidate_contribution_summary="Assessment generation for this topic failed or returned invalid format.",
                key_candidate_statements=[], reference_kb_alignment="Error in assessment generation.",
                overall_topic_performance_score=None
            )
        else:
            # Ensure topic_id and topic_label from the LLM match the current topic, or override
            assessment_result.topic_id = current_topic.topic_id
            assessment_result.topic_label = current_topic.topic_label
            print(f"  Assessment successful for topic '{assessment_result.topic_label}': Overall Score {assessment_result.overall_topic_performance_score}")

        current_assessments = state.get("topic_level_assessments", [])
        current_assessments.append(assessment_result)
        return {
            "topic_level_assessments": current_assessments,
            "current_topic_index": state["current_topic_index"] + 1,
            "error_message": state.get("error_message") # Preserve prior non-critical error
        }
    except Exception as e:
        print(f"Error in assess_candidate_performance_on_topic_node for topic '{current_topic.topic_label}': {e}\n{traceback.format_exc()}")
        error_assessment = TopicLevelAssessment(
            topic_id=current_topic.topic_id, topic_label=current_topic.topic_label,
            candidate_contribution_summary=f"Error during assessment generation for this topic: {str(e)}",
            key_candidate_statements=[], reference_kb_alignment="Error.",
            overall_topic_performance_score=None
        )
        current_assessments = state.get("topic_level_assessments", [])
        current_assessments.append(error_assessment)
        return {
            "topic_level_assessments": current_assessments,
            "current_topic_index": state["current_topic_index"] + 1, # Increment to avoid loop on one bad topic
            "error_message": f"Failed to assess topic '{current_topic.topic_label}': {e}"
        }

async def aggregate_final_report_node(state: InterviewGraphState) -> Dict[str, Any]: # MODIFIED: Name clarity
    print("\n--- Node: Aggregate Final Report ---")
    final_error_message = state.get("error_message")
    topic_assessments = state.get("topic_level_assessments", [])

    if not topic_assessments and not final_error_message:
        final_error_message = "No topic-level assessments were generated to create a final report."
    
    print(f"Aggregating {len(topic_assessments)} topic-level assessments into a final report.")

    topic_assessments_summary_str = "\n\n".join([
        f"Topic: {assess.topic_label} (Overall Score: {assess.overall_topic_performance_score if assess.overall_topic_performance_score is not None else 'N/A'})\n"
        f"  Summary: {assess.candidate_contribution_summary}\n"
        f"  KB Alignment: {assess.reference_kb_alignment}\n"
        f"  Key Statements: {'; '.join(assess.key_candidate_statements)}\n" +
        ("  Detailed Criteria Observations:\n" + "\n".join([f"    - {crit_obs.criterion_name} (Score: {crit_obs.score if crit_obs.score is not None else 'N/A'}): {crit_obs.observation}" for crit_obs in assess.detailed_criteria_observations]) if assess.detailed_criteria_observations else "")
        for assess in topic_assessments
    ])

    system_prompt = f'''
    You are an expert HR analyst. Synthesize topic-level interview assessments into an overall summary, key strengths across topics, and key areas for improvement across topics. Respond ONLY with a valid JSON object for OverallReportData.
    '''
    prompt = f'''
    Topic-Level Assessments Summary:\n---\n{topic_assessments_summary_str}\n---\nBased ONLY on these topic assessments, provide the OverallReportData JSON (overall_summary, key_strengths_across_topics, key_areas_for_improvement_across_topics):"
    '''

    try:
        overall_data = await llm_summarizer.invoke([
            SystemMessage(system_prompt),
            HumanMessage(prompt)
        ])
        if not overall_data or not isinstance(overall_data, OverallReportData):
            print(f"[WARN] Final report aggregation LLM call invalid. Received: {overall_data}")
            overall_data = OverallReportData(overall_summary="Automated overall summary generation failed. Review topic assessments.", strengths=[], areas_for_improvement=[])
        else:
            print("  Overall summary, strengths, and areas for improvement generated successfully.")

        final_report = InterviewAssessmentReport(
            topic_by_topic_assessments=topic_assessments, # Store the list of TopicLevelAssessment
            overall_summary=overall_data.overall_summary,
            key_strengths_across_topics=overall_data.strengths,
            key_areas_for_improvement_across_topics=overall_data.areas_for_improvement
        )
        return {"final_assessment_report": final_report, "error_message": final_error_message}
    except Exception as e:
        print(f"Error in aggregate_final_report_node: {e}\n{traceback.format_exc()}")
        final_report_err = InterviewAssessmentReport(
            topic_by_topic_assessments=topic_assessments,
            overall_summary=f"Error during final report aggregation: {e}",
            key_strengths_across_topics=[], key_areas_for_improvement_across_topics=[]
        )
        return {"final_assessment_report": final_report_err, "error_message": final_error_message or f"Failed to aggregate final report: {e}"}


# (error_handler_node can remain similar, but will now receive TopicLevelAssessment in the state)
def error_handler_node(state: InterviewGraphState) -> Dict[str, Any]:
    err_msg = state.get('error_message', "Unknown error occurred in pipeline.")
    print(f"\n--- Node: Error Handler ---")
    print(f"Pipeline Error: {err_msg}")
    # If partial assessments exist, we can still put them in the final report
    topic_assessments = state.get("topic_level_assessments", [])
    if topic_assessments: # Even if an error occurred, we might have some topic assessments
        report = InterviewAssessmentReport(
            topic_by_topic_assessments=topic_assessments,
            overall_summary=f"Pipeline terminated or encountered an error: {err_msg}. Partial topic assessments (if any) are provided below.",
            key_strengths_across_topics=[],
            key_areas_for_improvement_across_topics=[]
        )
        return {"final_assessment_report": report, "error_message": err_msg} # Keep error_message
    # If no topic assessments and an error, just pass the error
    return {"error_message": err_msg} # Ensure error message is propagated

# --- Graph Definition ---
def build_graph():
    workflow = StateGraph(InterviewGraphState)

    # --- Add Nodes ---
    workflow.add_node("load_preprocess_transcript", load_and_preprocess_transcript_node)
    workflow.add_node("load_reference_kb", load_reference_kb_node)
    workflow.add_node("segment_topics", segment_topics_node)    

    workflow.add_node("init_topic_assessment_loop", initialize_assessment_loop_node) # Renamed
    workflow.add_node("select_next_topic", select_next_topic_node) # Renamed
    workflow.add_node("retrieve_reference_for_topic", retrieve_reference_for_topic_node) # Renamed
    workflow.add_node("assess_topic_performance", assess_candidate_performance_on_topic_node) # Renamed

    workflow.add_node("aggregate_final_report", aggregate_final_report_node) # Renamed
    workflow.add_node("error_handler", error_handler_node) # Renamed for consistency

    # --- Define Edges ---
    workflow.set_entry_point("load_preprocess_transcript")
    workflow.add_edge("load_preprocess_transcript", "load_reference_kb")
    workflow.add_edge("load_reference_kb", "segment_topics")    
    workflow.add_edge("segment_topics", "init_topic_assessment_loop")


    workflow.add_edge("init_topic_assessment_loop", "select_next_topic")

    workflow.add_conditional_edges(
        "select_next_topic", # Source node
        should_continue_topic_assessment_node, # Condition function
        {
            "retrieve_reference_for_topic": "retrieve_reference_for_topic", # If true
            "aggregate_final_report": "aggregate_final_report",      # If false (all topics done)
            "error_handler": "error_handler"                         # If error detected by condition
        }
    )

    workflow.add_edge("retrieve_reference_for_topic", "assess_topic_performance")
    workflow.add_edge("assess_topic_performance", "select_next_topic") # Loop back

    workflow.add_edge("aggregate_final_report", END)
    workflow.add_edge("error_handler", END)

    app = workflow.compile()
    return app

# --- Main Execution ---
# (main.py - run_pipeline and if __name__ == "__main__":)

async def run_pipeline(transcript_path: str, ref_kb_path: Optional[str] = None):    
    print(f"Current Working Directory: {os.getcwd()}") # For debugging paths
    print("Building RAG assessment graph with TOPIC-DRIVEN loop...")
    rag_app = build_graph()
    print("Graph built.")

    initial_state = InterviewGraphState(
        raw_transcript_path=transcript_path, reference_kb_path=ref_kb_path,
        processed_transcript=None, topic_segments=None,
        processed_reference_kb_chunks=None, reference_kb_embeddings=None,
        assessment_criteria=ASSESSMENT_CRITERIA, # Still passed as a guide
        current_topic_index=0, current_topic_data=None, # Topic loop init
        candidate_speech_for_current_topic_str=None,
        retrieved_reference_chunks_for_current_topic=None,
        topic_level_assessments=[], # Initialize for new assessment type
        final_assessment_report=None, error_message=None
    )    

    config = {"recursion_limit": 50} # Increased for potentially more steps with topics
    print(f"\nStarting TOPIC-DRIVEN pipeline for transcript: '{transcript_path}' and KB: '{ref_kb_path}'")
    final_state_invoke = await rag_app.ainvoke(initial_state, config=config)

    print("\n\n" + "="*15 + " FINAL PIPELINE STATE & REPORT (TOPIC-DRIVEN) " + "="*15)
    if final_state_invoke:
        final_report = final_state_invoke.get("final_assessment_report")
        err_msg = final_state_invoke.get("error_message")

        if final_report:
            print("\n--- Final Assessment Report ---")
            print(f"Overall Summary: {final_report.overall_summary}")
            print("\nKey Strengths Across Topics:")
            for strength in final_report.key_strengths_across_topics if final_report.key_strengths_across_topics else []: print(f"  - {strength}")
            print("\nKey Areas for Improvement Across Topics:")
            for area in final_report.key_areas_for_improvement_across_topics if final_report.key_areas_for_improvement_across_topics else []: print(f"  - {area}")

            print(f"\nTopic-by-Topic Assessments ({len(final_report.topic_by_topic_assessments)}):")
            for topic_assess in final_report.topic_by_topic_assessments:
                print(f"\n  Topic: {topic_assess.topic_label} (ID: {topic_assess.topic_id})")
                print(f"    Candidate Contribution Summary: {topic_assess.candidate_contribution_summary}")
                print(f"    KB Alignment: {topic_assess.reference_kb_alignment}")
                print(f"    Overall Topic Score: {topic_assess.overall_topic_performance_score if topic_assess.overall_topic_performance_score is not None else 'N/A'}")
                if topic_assess.key_candidate_statements:
                    print(f"    Key Candidate Statements:")
                    for stmt in topic_assess.key_candidate_statements: print(f"      - \"{stmt}\"")
                if topic_assess.detailed_criteria_observations:
                    print(f"    Detailed Criteria Observations:")
                    for crit_obs in topic_assess.detailed_criteria_observations:
                        print(f"      - {crit_obs.criterion_name} (Score: {crit_obs.score if crit_obs.score is not None else 'N/A'}): {crit_obs.observation}")
                print("-" * 15)
        elif err_msg :
             print(f"\nPipeline completed with an error: {err_msg}")
             print("No final report generated, or report generation itself failed.")
        else:
            print("\nPipeline completed, but no final report was generated and no explicit error message was found.")

        # Debug output for state if things go wrong
        if err_msg and not final_report:
            print(f"\nDEBUG: Error Message in State: {err_msg}")
            print(f"DEBUG: Number of topic assessments made: {len(final_state_invoke.get('topic_level_assessments', []))}")
            if final_state_invoke.get("topic_level_assessments"):
                print(final_state_invoke.get("topic_level_assessments")[0] if final_state_invoke.get("topic_level_assessments") else "No topic assessments")


    else:
        print("Pipeline did not return a final state.")
    print("="*60)
    return final_state_invoke

if __name__ == "__main__": 
    _ = load_sample_transcript()
    print("Core data structures, clients, and LangGraph nodes defined.")
    print("Running the RAG assessment pipeline...")
    asyncio.run(run_pipeline(transcript_path="sample_transcript.json"))