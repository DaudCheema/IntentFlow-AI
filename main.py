from datetime import datetime
import os
import json
import re
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from models import UserQueryRequest, FinalAgentResponse
from logger_config import setup_logger
from web_search import perform_web_search
from llm_client import call_groq  # Imported from your dedicated LLM module

#############################################
# Session management and history tracking
#############################################
SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}

def get_or_create_session(session_id: str) -> List[Dict[str, str]]:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []
    return SESSION_STORE[session_id]

def append_to_session(session_id: str, user_query: str, agent_response: str, max_turns: int = 6):
    history = get_or_create_session(session_id)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": agent_response})
    if len(history) > max_turns:
        SESSION_STORE[session_id] = history[-max_turns:]

#############################################

logger = setup_logger()
app = FastAPI(title="Intent Clarifying AI Agent (Groq Cloud)")

# Dynamically inject current date
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")
CURRENT_YEAR = str(datetime.now().year)

SYSTEM_INTENT_PROMPT = f"""
You are an intent classification and context resolution assistant. Today's date is {CURRENT_DATE}.

You are given a CONVERSATION HISTORY and a NEW USER QUERY.

YOUR TASKS:
1. "is_ambiguous": boolean.
   - Set to TRUE if the query consists of a single name or entity with NO specific context (e.g., "Who is Tom?", "Tell me about John").
   - Set to TRUE if vital parameters (location, entity, currency, or scope) are missing.
   - CRITICAL: Never assume or guess which entity the user means if multiple famous entities share the same name. You MUST mark "is_ambiguous": true if there are 2 or more plausible interpretations.
   - Set to FALSE if the query is conceptual ("What is ML?") or has clear, specific context ("Who is Tom Cruise?").

2. "needs_web_search": boolean.
   - Set to FALSE if "is_ambiguous" is TRUE (do NOT search the web for vague, single-word entities).
   - Set to TRUE ONLY if the query has specific context AND requires real-time/live data, news, or current facts.
   - Set to FALSE for conceptual topics, meta-chat questions ("What did we last talk about?"), or historical facts.

3. "resolved_query": string.
   - Rephrase the NEW USER QUERY into a clear, single standalone question using CONVERSATION HISTORY if pronouns are present.
   - DO NOT attempt to answer the question or invent details not present in the history or query.

4. "search_query": string.
   - A clean, keyword-focused search query derived from "resolved_query".
   - Include {CURRENT_YEAR} ONLY if real-time/latest data is explicitly requested.

Respond strictly in valid JSON matching the schema.
"""

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_ambiguous": {"type": "boolean"},
        "needs_web_search": {"type": "boolean"},
        "resolved_query": {"type": "string"},
        "search_query": {"type": "string"}
    },
    "required": ["is_ambiguous", "needs_web_search", "resolved_query", "search_query"]
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "clarifying_questions": {"type": "array", "items": {"type": "string"}},
        "suggested_questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["answer", "clarifying_questions", "suggested_questions"]
}

SYSTEM_RESPONSE_PROMPT = f"""
You are an expert information synthesizer and research assistant.
CRITICAL BASELINE: Today's date is {CURRENT_DATE}. The current year is strictly {CURRENT_YEAR}.

CRITICAL INSTRUCTIONS:
1. IF WEB CONTEXT IS PROVIDED:
   - Compare and combine key details from ALL provided sources using bold headers and bullet points.
2. IF NO WEB CONTEXT IS PROVIDED (OR QUERY IS AMBIGUOUS):
   - Directly provide a helpful, structured response using your knowledge.
   - If the user query is ambiguous, explain that the query lacks specifics and ask 2-3 clarifying questions.
3. TEMPORAL ACCURACY: Interpret terms like "this year" or "latest" as happening right now in {CURRENT_YEAR}.

Respond strictly in valid JSON matching the schema.
"""


@app.post("/query", response_model=FinalAgentResponse)
async def process_query(request: UserQueryRequest):
    session_id = getattr(request, "session_id", "default_session") or "default_session"
    history = get_or_create_session(session_id)
    
    logger.info(f"Incoming Query: '{request.query}' | Session: '{session_id}'")
    history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history])
    
    # Step 1: Intent Analysis via Groq Cloud
    logger.info("Step 1: Evaluating Intent via Groq Cloud...")
    step1_prompt = f"CONVERSATION HISTORY:\n{history_str if history_str else 'None'}\n\nNEW USER QUERY: {request.query}"
    
    try:
        intent_data = call_groq(SYSTEM_INTENT_PROMPT, step1_prompt, json_schema=INTENT_SCHEMA)
        is_ambiguous = intent_data.get("is_ambiguous", False)
        needs_search = intent_data.get("needs_web_search", False)
        resolved_query = intent_data.get("resolved_query") or request.query
        search_query = intent_data.get("search_query") or resolved_query
    except Exception as e:
        logger.error(f"Step 1 failed, defaulting: {e}")
        is_ambiguous = False
        needs_search = True
        resolved_query = request.query
        search_query = request.query

    if is_ambiguous:
        needs_search = False

    if needs_search:
        search_query = re.sub(r'\b(this year|this years|latest|current|now|today)\b', '', search_query, flags=re.IGNORECASE).strip()
        search_query = re.sub(r'[^\w\s-]', '', search_query)
        search_query = " ".join(search_query.split())

    logger.info(f"Analysis Results -> Ambiguous: {is_ambiguous} | Trigger Web Search: {needs_search}")

    # Step 2: Web Search Execution
    web_context = ""
    sources = []
    if needs_search:
        logger.info("Step 2: Executing real-time web retrieval...")
        web_context, sources = perform_web_search(search_query)

    # Step 3: Response Synthesis & Session Saving
    logger.info("Step 3: Processing retrieved data and generating final response payload...")

    if is_ambiguous:
        prompt_payload = (
            f"CONVERSATION HISTORY:\n{history_str if history_str else 'None'}\n\n"
            f"RESOLVED USER QUERY: {resolved_query}\n\n"
            f"NOTE: The user's query is highly ambiguous or lacks context. "
            f"Please state in 'answer' that the request is ambiguous, and provide 2-3 specific clarifying options."
        )
    elif needs_search and web_context:
        MAX_CONTEXT_CHARS = 3500
        truncated_context = web_context[:MAX_CONTEXT_CHARS]
        prompt_payload = f"CONVERSATION HISTORY:\n{history_str if history_str else 'None'}\n\nRESOLVED USER QUERY: {resolved_query}\n\nRETRIEVED WEB CONTEXT:\n{truncated_context}"
    elif needs_search and not web_context:
        prompt_payload = f"CONVERSATION HISTORY:\n{history_str if history_str else 'None'}\n\nRESOLVED USER QUERY: {resolved_query}\n\nSEARCH_STATUS: FAILED - live web search was attempted but returned no usable data."
    else:
        prompt_payload = f"CONVERSATION HISTORY:\n{history_str if history_str else 'None'}\n\nRESOLVED USER QUERY: {resolved_query}"

    try:
        generation_data = call_groq(
            system_prompt=SYSTEM_RESPONSE_PROMPT,
            user_prompt=prompt_payload,
            json_schema=RESPONSE_SCHEMA
        )
    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        generation_data = {
            "answer": "I noticed your question could refer to a few different things. Could you clarify?",
            "clarifying_questions": ["Who specifically are you asking about?"],
            "suggested_questions": []
        }

    final_answer = generation_data.get("answer") or "I wasn't able to generate an answer."
    append_to_session(session_id, request.query, final_answer)

    clarifying = generation_data.get("clarifying_questions", []) if is_ambiguous else []

    return FinalAgentResponse(
        answer=final_answer,
        clarifying_questions=clarifying,
        suggested_questions=generation_data.get("suggested_questions", []),
        sources=sources
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)