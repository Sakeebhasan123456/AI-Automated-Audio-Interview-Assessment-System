import json
import re
import uuid
from typing import List, Dict, Optional, Any, TypedDict
from pydantic import BaseModel, Field, validator

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For persisting state if needed

# (Keep existing imports like os, tempfile, traceback, html, asyncio)
# ... (previous Pydantic models: DialogueTurn, TopicSegment, etc. remain the same) ...
from data_models import (DialogueTurn, TopicSegment, RetrievedContext, IndividualAssessment, InterviewAssessmentReport, ASSESSMENT_CRITERIA)

# --- Data Models (Ensure they are defined as in the previous step) ---
# DialogueTurn, TopicSegment, RetrievedContext, IndividualAssessment, InterviewAssessmentReport
# ASSESSMENT_CRITERIA list

# --- Preprocessing Functions (parse_emotion_from_text, preprocess_transcript) ---
# ... (Keep from previous step) ...
from data_models import (parse_emotion_from_text, preprocess_transcript, load_sample_transcript)


from data_models import (LLMClient, EmbeddingClient)
from langchain_core.messages import HumanMessage, SystemMessage


# --- LangGraph State Definition (from previous step) ---
class InterviewGraphState(TypedDict):
    raw_transcript_path: str
    processed_transcript: Optional[List[DialogueTurn]]
    # Storing embeddings for candidate turns only for simplicity in retrieval
    candidate_turn_embeddings: Optional[Dict[int, List[float]]] # turn_id (of candidate) -> embedding

    assessment_criteria: List[Dict]
    current_criterion_index: int
    current_criterion_data: Optional[Dict] # The criterion dict being processed

    topic_segments: Optional[List[TopicSegment]]

    # Stores context specifically retrieved for the current_criterion_data
    retrieved_context_for_current_criterion: Optional[RetrievedContext]

    individual_assessments: List[IndividualAssessment] # Accumulates assessments
    final_report_input: Optional[Dict] # Data for final summarization LLM
    final_assessment_report: Optional[InterviewAssessmentReport]

    error_message: Optional[str]


# --- Instantiate clients (global for graph nodes to access) ---
# These will be used by the nodes.
# For a real application, you might pass these into the node functions or use a class structure for nodes.
llm_topic_segmenter = LLMClient.with_structured_output(TopicSegment)
llm_assessor = LLMClient
llm_summarizer = LLMClient
embedding_client = EmbeddingClient


# --- LangGraph Nodes ---

async def load_and_preprocess_transcript_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Loads transcript from path, preprocesses it."""
    print("\n--- Node: Load and Preprocess Transcript ---")
    try:
        raw_transcript_path = state["raw_transcript_path"]
        raw_data = load_sample_transcript(raw_transcript_path) # Using our helper
        processed_turns = preprocess_transcript(raw_data)
        print(f"Successfully processed {len(processed_turns)} dialogue turns.")
        return {
            "processed_transcript": processed_turns,
            "assessment_criteria": ASSESSMENT_CRITERIA, # Initialize criteria
            "individual_assessments": [], # Initialize empty list for assessments
            "error_message": None
        }
    except Exception as e:
        print(f"Error in load_and_preprocess_transcript_node: {e}")
        return {"error_message": f"Failed to load/preprocess transcript: {e}"}


async def segment_topics_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Segments the processed transcript into topics using an LLM."""
    print("\n--- Node: Segment Topics ---")
    if state.get("error_message"): return {} # Skip if previous error

    processed_transcript = state["processed_transcript"]
    if not processed_transcript:
        return {"error_message": "No processed transcript available for topic segmentation."}

    transcript_for_llm = "\n".join(
        [f"Turn {t.turn_id} ({t.speaker}): {t.clean_text}" for t in processed_transcript]
    )

    system_prompt = "You are an expert in analyzing conversational flow. Your task is to segment an interview transcript into distinct topics. Respond ONLY with a valid JSON list of topic objects, where each object has 'topic_id', 'topic_label', 'start_turn_id', and 'end_turn_id'."
    prompt = f"""Based on the following interview transcript, identify distinct topics. Ensure 'start_turn_id' and 'end_turn_id' are valid integer turn IDs from the transcript. 'topic_id' should be a short unique string like "T1", "T2", etc.
    Transcript:
    {transcript_for_llm}
    JSON Output:
    """
    try:
        # Using generate_structured to directly get List[TopicSegment]        
        # For a real LLM, you might use a library like 'instructor' or OpenAI's tool calling.
        topic_segments_data_list = await llm_topic_segmenter.invoke(
            [
                SystemMessage(system_prompt),
                HumanMessage(prompt)
            ]
        )
        # The mock logic for generate_structured needs careful implementation to return List[PydanticModel]
        # Assuming the mock for list of topics now returns a list of dicts parsed by PydanticModel inside generate_structured
        if topic_segments_data_list is None or not isinstance(topic_segments_data_list, list):
            # This check is important because the mock's generate_structured might return None on error
            # and for list types, the calling code expects a list.
            # A real generate_structured should ideally raise an error or return an empty list on failure.
            print(f"[WARN] Topic segmentation LLM call did not return a valid list. Received: {topic_segments_data_list}")
            # Fallback or error handling
            # For simplicity, if the mock fails to return a list, we create a single default topic.
            if not topic_segments_data_list: # if None or empty list that wasn't intended
                topic_segments = [
                    TopicSegment(
                        topic_id="T_FALLBACK",
                        topic_label="Entire Interview (Fallback)",
                        start_turn_id=0,
                        end_turn_id=len(processed_transcript) - 1 if processed_transcript else 0
                    )
                ]
                print("[INFO] Using fallback single topic for entire interview.")
            else: # if it was some other non-list type, treat as error
                 return {"error_message": "Topic segmentation failed to produce a list of topics.", "topic_segments": []}
        else:
            # Ensure the type is correct if generate_structured handles it
            topic_segments = [TopicSegment(**ts_data.model_dump()) if isinstance(ts_data, BaseModel) else TopicSegment(**ts_data) for ts_data in topic_segments_data_list]


        print(f"Successfully segmented into {len(topic_segments)} topics.")
        for topic in topic_segments: print(f"  - {topic.topic_label} (Turns {topic.start_turn_id}-{topic.end_turn_id})")
        return {"topic_segments": topic_segments, "error_message": None}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from topic segmentation LLM: {e}")
        return {"error_message": f"Failed to parse topic segments: {e}"}
    except Exception as e:
        print(f"Error in segment_topics_node: {e}")
        return {"error_message": f"Topic segmentation failed: {e}"}


async def embed_candidate_turns_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Embeds the clean_text of candidate's turns."""
    print("\n--- Node: Embed Candidate Turns ---")
    if state.get("error_message"): return {}

    processed_transcript = state["processed_transcript"]
    if not processed_transcript:
        return {"error_message": "No processed transcript for embedding."}

    candidate_texts_with_ids = []
    for turn in processed_transcript:
        if turn.speaker.lower() == "candidate" and turn.clean_text:
            candidate_texts_with_ids.append({"turn_id": turn.turn_id, "text": turn.clean_text})

    if not candidate_texts_with_ids:
        print("No candidate text found to embed.")
        return {"candidate_turn_embeddings": {}, "error_message": None}

    texts_to_embed = [item["text"] for item in candidate_texts_with_ids]
    try:
        embeddings = [await EmbeddingClient.embed_query(text) for text in texts_to_embed]        
        candidate_embeddings_map = {
            item["turn_id"]: emb
            for item, emb in zip(candidate_texts_with_ids, embeddings)
        }
        print(f"Successfully embedded {len(candidate_embeddings_map)} candidate turns.")
        return {"candidate_turn_embeddings": candidate_embeddings_map, "error_message": None}
    except Exception as e:
        print(f"Error in embed_candidate_turns_node: {e}")
        return {"error_message": f"Failed to embed candidate turns: {e}"}


def initialize_assessment_loop_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Initializes variables for iterating through assessment criteria."""
    print("\n--- Node: Initialize Assessment Loop ---")
    if state.get("error_message"): return {}
    return {"current_criterion_index": 0, "error_message": None}

import numpy as np

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    if not vec1 or not vec2:
        return 0.0
    vec1_arr = np.array(vec1)
    vec2_arr = np.array(vec2)
    # Check for zero vectors to prevent division by zero or NaN
    if np.all(vec1_arr == 0) or np.all(vec2_arr == 0):
        return 0.0
    similarity = np.dot(vec1_arr, vec2_arr) / (np.linalg.norm(vec1_arr) * np.linalg.norm(vec2_arr))
    return float(similarity)



# NEW NODES for the assessment loop:

def select_next_criterion_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Selects the next criterion to assess from the list."""
    print("\n--- Node: Select Next Criterion ---")
    if state.get("error_message"): return {}

    criteria_list = state["assessment_criteria"]
    current_idx = state["current_criterion_index"]

    if current_idx < len(criteria_list):
        current_criterion = criteria_list[current_idx]
        print(f"Selected criterion ({current_idx + 1}/{len(criteria_list)}): {current_criterion['criterion']}")
        return {"current_criterion_data": current_criterion, "error_message": None}
    else:
        print("All criteria assessed.")
        return {"current_criterion_data": None, "error_message": None} # Signal completion


def should_continue_assessment_node(state: InterviewGraphState) -> str:
    """Determines if there are more criteria to assess."""
    print("\n--- Condition: Should Continue Assessment? ---")
    if state.get("error_message"):
        print(f"Error detected: {state['error_message']}. Halting assessment.")
        return "error_handler" # Or directly to END if no specific error handler

    if state["current_criterion_data"] is not None:
        print("Yes, more criteria to assess.")
        return "retrieve_context" # Name of the next node in the loop
    else:
        print("No more criteria. Proceeding to aggregation.")
        return "aggregate_report" # Name of the aggregation node (to be defined)


async def retrieve_context_for_criterion_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Retrieves relevant dialogue context for the current assessment criterion."""
    print("\n--- Node: Retrieve Context for Criterion ---")
    if state.get("error_message"): return {}

    criterion_data = state["current_criterion_data"]
    if not criterion_data:
        return {"error_message": "No current criterion selected for context retrieval."}

    print(f"Retrieving context for: {criterion_data['criterion']}")
    processed_transcript = state["processed_transcript"]
    candidate_embeddings = state["candidate_turn_embeddings"] # turn_id -> embedding
    topic_segments = state["topic_segments"]

    if not processed_transcript or candidate_embeddings is None: # Check for None explicitly
        return {"error_message": "Transcript or embeddings not available for context retrieval."}

    criterion_description = criterion_data["description"]
    criterion_embedding = await embedding_client.get_embedding(criterion_description)

    # --- Topic Focusing (Simple V1: find best matching topic if any) ---
    best_matching_topic_label = "General" # Default
    if topic_segments:
        topic_label_embeddings = await embedding_client.get_embeddings([ts.topic_label for ts in topic_segments])
        similarities = [cosine_similarity(criterion_embedding, tl_emb) for tl_emb in topic_label_embeddings]
        if similarities:
            best_topic_idx = np.argmax(similarities)
            # Could add a threshold, if similarity is too low, don't focus on a topic
            if similarities[best_topic_idx] > 0.3: # Arbitrary threshold
                best_matching_topic_label = topic_segments[best_topic_idx].topic_label
                # For V1.1, we could filter candidate_turn_ids by this topic_segment's start/end turn_ids
                print(f"  Criterion best matches topic: {best_matching_topic_label} (Similarity: {similarities[best_topic_idx]:.2f})")


    # --- Semantic Search for Candidate Turns ---
    relevant_candidate_turns_with_scores = []
    for turn_id, turn_embedding in candidate_embeddings.items():
        sim = cosine_similarity(criterion_embedding, turn_embedding)
        relevant_candidate_turns_with_scores.append({"turn_id": turn_id, "similarity": sim})

    # Sort by similarity and get top K (e.g., top 3)
    relevant_candidate_turns_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    top_k = 3
    top_candidate_turn_ids = [item["turn_id"] for item in relevant_candidate_turns_with_scores[:top_k]]
    print(f"  Top {top_k} relevant candidate turn IDs: {top_candidate_turn_ids} (based on similarity)")

    # --- Build Context Window (Simplified: 1 interviewer question + top candidate answer) ---
    # This needs to be more robust. For now, just pick the top one and its preceding turn.
    # A more robust version would iterate top_candidate_turn_ids and build richer contexts.
    context_dialogue_turns = []
    if top_candidate_turn_ids:
        primary_candidate_turn_id = top_candidate_turn_ids[0] # Focus on the most relevant one for now
        
        # Find the turn in the full transcript
        candidate_turn_index_in_transcript = -1
        for idx, turn in enumerate(processed_transcript):
            if turn.turn_id == primary_candidate_turn_id:
                candidate_turn_index_in_transcript = idx
                break
        
        if candidate_turn_index_in_transcript != -1:
            # Try to get 1 preceding turn (likely interviewer's question)
            if candidate_turn_index_in_transcript > 0:
                preceding_turn = processed_transcript[candidate_turn_index_in_transcript - 1]
                # Basic check if it's the interviewer
                if preceding_turn.speaker.lower() == "interviewer":
                    context_dialogue_turns.append(preceding_turn)
                else: # if preceding is also candidate, maybe include it and go one further back?
                    context_dialogue_turns.append(preceding_turn) # For now, just add it
                    if candidate_turn_index_in_transcript > 1:
                         even_earlier_turn = processed_transcript[candidate_turn_index_in_transcript - 2]
                         if even_earlier_turn.speaker.lower() == "interviewer":
                             context_dialogue_turns.insert(0, even_earlier_turn)


            context_dialogue_turns.append(processed_transcript[candidate_turn_index_in_transcript]) # The candidate's turn

            # Optional: Add 1-2 succeeding turns from the same candidate if they are part of the same thought
            current_speaker = processed_transcript[candidate_turn_index_in_transcript].speaker
            for i in range(1, 3): # Look for up to 2 more turns
                if candidate_turn_index_in_transcript + i < len(processed_transcript):
                    next_turn = processed_transcript[candidate_turn_index_in_transcript + i]
                    if next_turn.speaker == current_speaker:
                        context_dialogue_turns.append(next_turn)
                    else:
                        break # Different speaker, stop context window here
                else:
                    break # End of transcript
        else:
             print(f"  [WARN] Could not find candidate turn_id {primary_candidate_turn_id} in processed transcript.")

    if not context_dialogue_turns:
        print("  [WARN] No context could be retrieved. Using full candidate transcript as fallback (simplified).")
        # Fallback: use all candidate turns (not ideal for RAG, but prevents error for mock)
        context_dialogue_turns = [t for t in processed_transcript if t.speaker.lower() == "candidate"][:5] # Limit for mock

    retrieved_context = RetrievedContext(
        criterion_assessed=criterion_data["criterion"],
        topic_label_context=best_matching_topic_label,
        dialogue_turns=context_dialogue_turns
    )
    print(f"  Retrieved context with {len(retrieved_context.dialogue_turns)} turns for assessment.")
    return {"retrieved_context_for_current_criterion": retrieved_context, "error_message": None}


async def assess_retrieved_context_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Assesses the retrieved context against the current criterion using an LLM."""
    print("\n--- Node: Assess Retrieved Context ---")
    if state.get("error_message"): return {}

    criterion_data = state["current_criterion_data"]
    retrieved_context = state["retrieved_context_for_current_criterion"]

    if not criterion_data or not retrieved_context:
        return {"error_message": "Criterion data or retrieved context not available for assessment."}

    print(f"Assessing criterion: {criterion_data['criterion']} (Topic Context: {retrieved_context.topic_label_context})")

    # Format the context for the LLM
    context_str = "\n".join(
        [f"{turn.speaker} ({turn.emotion if turn.emotion else 'N/A'}): {turn.clean_text}"
         for turn in retrieved_context.dialogue_turns]
    )

    system_prompt = f"""You are an expert interview assessor. Your task is to assess a candidate's performance based on a specific criterion and dialogue snippets from an interview. Respond ONLY with a valid JSON object matching the following structure:
    {{
      "criterion": "{criterion_data['criterion']}",
      "topic_assessed": "{retrieved_context.topic_label_context or 'General'}",
      "score": float (e.g., 1.0 to 5.0, based on: {criterion_data['scoring_guide']}),
      "reasoning": "Your detailed explanation for the score.",
      "evidence": ["List of direct quotes from the candidate's speech in the provided dialogue snippets that support your reasoning."]
    }}
    Focus ONLY on the provided dialogue snippets. Be objective and stick to the criterion.
    """
    prompt = f"""Assessment Criterion: {criterion_data['criterion']}
    Scoring Guide: {criterion_data['scoring_guide']}
    Relevant Topic Context: {retrieved_context.topic_label_context or 'N/A'}

    Dialogue Snippets:
    ---
    {context_str}
    ---
    Based *only* on the dialogue snippets above, provide your assessment for the stated criterion in the specified JSON format.
    Ensure 'evidence' quotes are verbatim from the candidate's speech in the snippets.
    If the snippets are insufficient to assess the criterion, reflect this in your reasoning and assign a neutral or N/A score if appropriate (e.g. score 2.5 or null).
    JSON Assessment:
    """

    try:
        assessment_result_model = await llm_assessor.generate_structured(
            prompt,
            IndividualAssessment,
            system_prompt=system_prompt,
            temperature=0.2
        )
        if not assessment_result_model or not isinstance(assessment_result_model, IndividualAssessment):
            print(f"[WARN] Assessment LLM call did not return a valid IndividualAssessment object. Received: {assessment_result_model}")
            # Create a fallback/error assessment
            assessment_result_model = IndividualAssessment(
                criterion=criterion_data['criterion'],
                topic_assessed=retrieved_context.topic_label_context or 'General',
                reasoning="Assessment generation failed or returned invalid format.",
                evidence=[]
            )
        else:
            print(f"  Assessment successful for '{assessment_result_model.criterion}': Score {assessment_result_model.score}")

        current_assessments = state.get("individual_assessments", [])
        current_assessments.append(assessment_result_model)

        return {
            "individual_assessments": current_assessments,
            "current_criterion_index": state["current_criterion_index"] + 1, # Increment for next iteration
            "error_message": None
        }
    except Exception as e:
        print(f"Error in assess_retrieved_context_node: {e}")
        error_assessment = IndividualAssessment(
            criterion=criterion_data['criterion'],
            topic_assessed=retrieved_context.topic_label_context or 'General',
            reasoning=f"Error during assessment generation: {str(e)}",
            evidence=[]
        )
        current_assessments = state.get("individual_assessments", [])
        current_assessments.append(error_assessment)
        return {
            "individual_assessments": current_assessments,
            # Don't increment index on error, or handle retry logic, for now, just record error and move on
            "current_criterion_index": state["current_criterion_index"] + 1,
            "error_message": f"Failed to assess criterion '{criterion_data['criterion']}': {e}"
        }

# (Existing node functions: load_and_preprocess_transcript_node, ..., assess_retrieved_context_node)

async def aggregate_report_node(state: InterviewGraphState) -> Dict[str, Any]:
    """Generates an overall summary, strengths, and areas for improvement based on individual assessments."""
    print("\n--- Node: Aggregate Report ---")
    if state.get("error_message") and not state.get("individual_assessments"): # Critical error before any assessment
        return {} # Propagate error or end

    individual_assessments = state.get("individual_assessments", [])
    if not individual_assessments:
        print("[WARN] No individual assessments available to generate a final report.")
        # Create a minimal report indicating no assessments were done
        final_report = InterviewAssessmentReport(
            detailed_assessments=[],
            overall_summary="No individual assessments were completed to generate an overall summary.",
            strengths=[],
            areas_for_improvement=[]
        )
        return {"final_assessment_report": final_report, "error_message": state.get("error_message") or "No assessments to aggregate."}

    print(f"Aggregating {len(individual_assessments)} individual assessments into a final report.")

    # Format individual assessments for the summarizer LLM
    assessments_summary_str = "\n\n".join([
        f"Criterion: {assess.criterion}\nTopic Assessed: {assess.topic_assessed or 'General'}\nScore: {assess.score if assess.score is not None else 'N/A'}\nReasoning: {assess.reasoning}\nEvidence: {', '.join(assess.evidence) if assess.evidence else 'N/A'}"
        for assess in individual_assessments
    ])

    system_prompt = f"""You are an expert HR analyst and report writer. Your task is to synthesize a series of individual assessment points from an interview into a cohesive overall summary. Respond ONLY with a valid JSON object matching the following structure:
    {{
  "overall_summary": "A concise overall summary of the candidate's performance.",
  "strengths": ["A list of key strengths observed, based on the assessments."],
  "areas_for_improvement": ["A list of key areas where the candidate could improve, based on the assessments."]
  }}
  Focus ONLY on the provided assessment summaries. Do not add external information or make up new assessment points.
  Be objective and provide a balanced view.
  """

    prompt = f"""Please generate an overall interview assessment report based on the following individual assessment details:
    Individual Assessments:
    ---
    {assessments_summary_str}
    ---
    Based *only* on these assessments, provide the overall summary, key strengths, and key areas for improvement in the specified JSON format.
    Overall Report JSON:
    """
    try:
        overall_report_data = await llm_summarizer.generate_structured(
            prompt,
            OverallReportData, # Expecting this Pydantic model as output
            system_prompt=system_prompt,
            temperature=0.3
        )

        if not overall_report_data or not isinstance(overall_report_data, OverallReportData):
            print(f"[WARN] Report aggregation LLM call did not return a valid OverallReportData object. Received: {overall_report_data}")
            # Fallback if structured generation fails
            overall_report_data = OverallReportData(
                overall_summary="Automated overall summary generation failed or returned invalid format. Please review individual assessments.",
                strengths=["Review individual assessments"],
                areas_for_improvement=["Review individual assessments"]
            )
        else:
            print("  Overall summary generated successfully.")

        final_report = InterviewAssessmentReport(
            detailed_assessments=individual_assessments,
            overall_summary=overall_report_data.overall_summary,
            strengths=overall_report_data.strengths,
            areas_for_improvement=overall_report_data.areas_for_improvement
        )
        return {"final_assessment_report": final_report, "error_message": state.get("error_message")} # Carry over any non-critical error message

    except Exception as e:
        print(f"Error in aggregate_report_node: {e}")
        # Create a report indicating the aggregation error
        final_report_with_error = InterviewAssessmentReport(
            detailed_assessments=individual_assessments,
            overall_summary=f"Error during report aggregation: {str(e)}. Please review individual assessments.",
            strengths=[],
            areas_for_improvement=[]
        )
        return {
            "final_assessment_report": final_report_with_error,
            "error_message": f"Failed to aggregate report: {e}"
        }

# --- Graph Definition (Updated) ---
# --- Graph Definition (Updated) ---
def build_graph():
    workflow = StateGraph(InterviewGraphState)

    workflow.add_node("load_preprocess", load_and_preprocess_transcript_node)
    workflow.add_node("segment_topics", segment_topics_node)
    workflow.add_node("embed_candidate_turns", embed_candidate_turns_node)
    workflow.add_node("init_assessment_loop", initialize_assessment_loop_node)
    workflow.add_node("select_criterion", select_next_criterion_node)
    workflow.add_node("retrieve_context", retrieve_context_for_criterion_node)
    workflow.add_node("assess_context", assess_retrieved_context_node)

    # Replace placeholder with the new node
    workflow.add_node("aggregate_report", aggregate_report_node) # New node added
    workflow.add_node("error_handler_placeholder", lambda state: print(f"\n--- Node: Error Handler (Placeholder) --- \nError: {state.get('error_message')}") or {})


    workflow.set_entry_point("load_preprocess")
    workflow.add_edge("load_preprocess", "segment_topics")
    workflow.add_edge("segment_topics", "embed_candidate_turns")
    workflow.add_edge("embed_candidate_turns", "init_assessment_loop")
    workflow.add_edge("init_assessment_loop", "select_criterion")

    workflow.add_conditional_edges(
        "select_criterion",
        should_continue_assessment_node,
        {
            "retrieve_context": "retrieve_context",
            "aggregate_report": "aggregate_report", # Route to the new node
            "error_handler": "error_handler_placeholder"
        }
    )

    workflow.add_edge("retrieve_context", "assess_context")
    workflow.add_edge("assess_context", "select_criterion") # Loop back

    workflow.add_edge("aggregate_report", END) # End after aggregation
    workflow.add_edge("error_handler_placeholder", END)

    app = workflow.compile()
    return app

# --- Main Execution (Updated) ---
# --- Main Execution (Updated `run_pipeline`'s print section) ---
async def run_pipeline(transcript_path: str):
    # ... (graph building and invocation logic remains the same) ...
    # print(f"\nStarting pipeline for: {transcript_path}")
    # final_state_invoke = await rag_app.ainvoke(initial_state, config=config)
    # ... (rest of the invocation logic)

    print("\n\n--- 최종 파이프라인 상태 (Final Pipeline State) ---")
    if final_state_invoke:
        if final_state_invoke.get("error_message") and not final_state_invoke.get("final_assessment_report") and not final_state_invoke.get("individual_assessments"):
            print(f"Pipeline Error before assessments: {final_state_invoke['error_message']}")
        else:
            # ... (printing of processed transcript and topic segments) ...

            print(f"\nIndividual Assessments ({len(final_state_invoke.get('individual_assessments', []))} criteria assessed):")
            for assessment in final_state_invoke.get("individual_assessments", []):
                print(f"  Criterion: {assessment.criterion}")
                print(f"    Topic Focus: {assessment.topic_assessed}")
                print(f"    Score: {assessment.score}")
                print(f"    Reasoning: {assessment.reasoning}")
                # print(f"    Evidence: {assessment.evidence}") # Can be verbose
                print("-" * 10)

            final_report = final_state_invoke.get("final_assessment_report")
            if final_report:
                print("\n--- Final Assessment Report ---")
                print(f"Overall Summary: {final_report.overall_summary}")
                print("\nStrengths:")
                for strength in final_report.strengths:
                    print(f"  - {strength}")
                print("\nAreas for Improvement:")
                for area in final_report.areas_for_improvement:
                    print(f"  - {area}")
            elif final_state_invoke.get("error_message"):
                 print(f"\nPipeline completed with an error, final report might be incomplete: {final_state_invoke['error_message']}")
            else:
                print("\nFinal report was not generated (check pipeline flow).")
    else:
        print("Pipeline did not return a final state.")
    return final_state_invoke

if __name__ == "__main__":
    # Ensure sample_transcript.json exists
    _ = load_sample_transcript()
    print("Core data structures, clients, and LangGraph nodes defined.")
    print("Running the RAG assessment pipeline...")
    asyncio.run(run_pipeline(transcript_path="sample_transcript.json"))