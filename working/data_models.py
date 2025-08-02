# main.py or data_models.py
import os
from typing import List, Dict, Optional, Any, TypedDict
from pydantic import BaseModel, Field
import json
import re
# --- Helper for similarity ---
import numpy as np

# --- Utility Functions ---
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2: return 0.0
    vec1_arr, vec2_arr = np.array(vec1), np.array(vec2)
    if np.all(vec1_arr == 0) or np.all(vec2_arr == 0): return 0.0
    norm_vec1 = np.linalg.norm(vec1_arr)
    norm_vec2 = np.linalg.norm(vec2_arr)
    if norm_vec1 == 0 or norm_vec2 == 0 : return 0.0
    return float(np.dot(vec1_arr, vec2_arr) / (norm_vec1 * norm_vec2)) # dot product/ product of magnitudes

def basic_chunker(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = min(current_pos + chunk_size, len(words))
        chunks.append(" ".join(words[current_pos:end_pos]))
        if end_pos == len(words): break
        current_pos = max(0, end_pos - chunk_overlap) # Ensure overlap doesn't go negative
        if current_pos <= (min(current_pos + chunk_size, len(words)) - chunk_size + chunk_overlap) and chunks: # Avoid infinite loop on small texts
            break # Safety break if overlap logic causes issues
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# --- Data Models ---
class DialogueTurn(BaseModel):
    turn_id: int
    speaker: str
    start_timestamp: str = Field(alias="start")
    end_timestamp: str = Field(alias="end")
    raw_text: str = Field(alias="text")
    clean_text: Optional[str] = None
    emotion: Optional[str] = None

class TopicSegment(BaseModel):
    topic_id: str
    topic_label: str
    start_turn_id: int
    end_turn_id: int

class RetrievedReferenceChunk(BaseModel):
    chunk_id: str
    text: str
    source_document: Optional[str] = None
    similarity_score: Optional[float] = None

class TopicCriterionAssessment(BaseModel): # NEW: For detailed breakdown within a topic
    criterion_name: str # e.g., "Clarity of Communication"
    score: Optional[float] = None # Score for this criterion *within this topic*
    observation: str # Specific observation for this criterion in this topic

class TopicLevelAssessment(BaseModel): # REFINES IndividualAssessment
    topic_id: str
    topic_label: str
    candidate_contribution_summary: str # What the candidate mainly said on this topic
    key_candidate_statements: List[str] # Verbatim quotes from candidate on this topic
    reference_kb_alignment: str # How did candidate's contribution align with relevant KB points?
    overall_topic_performance_score: Optional[float] = None # Holistic score for this topic
    detailed_criteria_observations: Optional[List[TopicCriterionAssessment]] = None # Optional breakdown

# (OverallReportData for summarizer LLM output)
class OverallReportData(BaseModel):
    overall_summary: str
    strengths: List[str] # Now across topics
    areas_for_improvement: List[str] # Now across topics

class InterviewAssessmentReport(BaseModel): # MODIFIED
    overall_summary: Optional[str] = None
    key_strengths_across_topics: Optional[List[str]] = None
    key_areas_for_improvement_across_topics: Optional[List[str]] = None
    topic_by_topic_assessments: List[TopicLevelAssessment] # Stores the detailed topic assessments

ASSESSMENT_CRITERIA = [
    {
        "id": "clarity_communication", # Changed id to be more specific for mock matching
        "criterion": "Clarity of Communication",
        "description": "Assesses how clearly and concisely the candidate expresses their thoughts and answers questions, considering the structure and coherence of their responses.",
        "scoring_guide": "1 (Very Unclear, rambling) to 5 (Exceptionally Clear, structured, and concise)"
    },
    {
        "id": "engagement_enthusiasm",
        "criterion": "Engagement and Enthusiasm",
        "description": "Assesses the candidate's level of interest, energy, and enthusiasm displayed during the interview, considering verbal cues, emotional expressions, and proactiveness.",
        "scoring_guide": "1 (Disengaged, passive) to 5 (Highly Engaged, enthusiastic, proactive)"
    },
    {
        "id": "problem_solving_explanation", # Changed id
        "criterion": "Problem-Solving Approach Explanation",
        "description": "Assesses how well the candidate explains their approach to solving a relevant problem discussed, focusing on logic, structure, consideration of alternatives, and clarity of their explanation, compared to ideal problem-solving methodologies if applicable from reference material.",
        "scoring_guide": "1 (Poor, illogical explanation) to 5 (Excellent, clear, logical, and well-structured explanation)"
    },
    {
        "id": "active_listening_comprehension", # Changed id
        "criterion": "Active Listening and Comprehension",
        "description": "Assesses if the candidate listens attentively, understands the interviewer's questions (including nuances), addresses all parts of multi-faceted questions, and references previous points where appropriate.",
        "scoring_guide": "1 (Poor listener, misunderstands questions) to 5 (Excellent listener, fully comprehends and responds aptly)"
    }
]

# --- Preprocessing Functions ---
def parse_emotion_from_text(text: str) -> tuple[Optional[str], str]:
    match = re.match(r"\[Emotion: ([\w\s]+)\]\s*(.*)", text, re.IGNORECASE) # Added IGNORECASE
    if match:
        emotion = match.group(1).strip().upper()
        clean_text = match.group(2).strip()
        return emotion, clean_text
    return None, text.strip()

def preprocess_transcript(raw_transcript_data: List[Dict]) -> List[DialogueTurn]:
    processed_turns = []
    for i, raw_turn in enumerate(raw_transcript_data):
        emotion, clean_text = parse_emotion_from_text(raw_turn["text"])
        turn_data_for_model = {
            "turn_id": i, "speaker": raw_turn["speaker"],
            "start": raw_turn["start"], "end": raw_turn["end"],
            "text": raw_turn["text"], # This will be mapped to raw_text by Pydantic
            "clean_text": clean_text, "emotion": emotion
        }
        turn = DialogueTurn(**turn_data_for_model)
        processed_turns.append(turn)
    return processed_turns

def load_sample_transcript(file_path="sample_transcript.json") -> List[Dict]:
    default_transcript = [
        {"speaker": "Interviewer", "start": "0:00:02", "end": "0:00:03", "text": "Welcome to the interview."},
        {"speaker": "Candidate", "start": "0:00:04", "end": "0:00:06", "text": "[Emotion: HAPPY] Thank you for having me!"},
        {"speaker": "Interviewer", "start": "0:00:07", "end": "0:00:10", "text": "Can you tell me about your experience with Project Alpha, particularly any challenges?"},
        {"speaker": "Candidate", "start": "0:00:11", "end": "0:00:20", "text": "[Emotion: NEUTRAL] Certainly. In Project Alpha, my main role was developing the backend API. It involved Python, FastAPI, and PostgreSQL. One of the key challenges was optimizing database queries for high traffic, which we initially struggled with due to complex joins."},
        {"speaker": "Candidate", "start": "0:00:21", "end": "0:00:25", "text": "We managed to reduce latency by 30% by implementing better indexing strategies and denormalizing some data after analyzing query patterns."},
        {"speaker": "Interviewer", "start": "0:00:26", "end": "0:00:28", "text": "That sounds impressive. What about teamwork and collaboration in that project?"},
        {"speaker": "Candidate", "start": "0:00:29", "end": "0:00:35", "text": "[Emotion: NEUTRAL] Teamwork was crucial. We had daily stand-ups and used Jira for task management. I believe in open communication and pair programming for tough problems."},
        {"speaker": "Interviewer", "start": "0:00:36", "end": "0:00:38", "text": "Okay. Do you have any questions for me regarding the role or the team?"},
        {"speaker": "Candidate", "start": "0:00:39", "end": "0:00:42", "text": "Yes, what does the typical career progression look like for this role, and what are the key performance indicators?"},
        {"speaker": "Interviewer", "start": "0:00:43", "end": "0:00:45", "text": "That's a good two-part question... (explains)"}
    ] # 10 turns, IDs 0-9
    try:
        if not os.path.exists(file_path):
            print(f"Warning: Sample transcript file '{file_path}' not found. Creating and using default transcript.")
            with open(file_path, 'w', encoding='utf-8') as f_out:
                json.dump(default_transcript, f_out, indent=2)
            return default_transcript
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Using default transcript.")
        return default_transcript


from dotenv import load_dotenv  
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()
LLMClient = init_chat_model("llama3-8b-8192", model_provider="groq")

from langchain_huggingface import HuggingFaceEmbeddings
EmbeddingClient = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
