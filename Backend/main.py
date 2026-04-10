"""
ECHO — Main FastAPI Server
--------------------------
Orchestrates the full 5-layer pipeline:
  1. Signal Decoder
  2. Narrative Engine (update)
  3. Retrieval Core (knowledge grounding)
  4. Memory Store (session + semantic memory)
  5. Response Synthesizer → Safety Gate → Final Response

Run with: uvicorn main:app --reload --port 8000
"""
from openai import OpenAI
import os
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Local modules
from signal_decoder import decode_signals
from narrative_engine import update_narrative, get_recent_conversation
from memory_store import get_or_create_session, save_session, store_turn_in_longterm, retrieve_similar_moments
from retrieval_core import seed_knowledge_base, retrieve_grounding_context
from response_synthesizer import build_response, anti_generic_check, SYSTEM_PROMPT
from safety_gate import check_crisis, apply_safety_layer

load_dotenv(override=True)

# ─── Global model references ───────────────────────────────────────────────────
gemini_model = None
embedding_model = None

# --- OPENROUTER WRAPPER ---
class OpenRouterWrapper:
    def __init__(self, system_instruction=""):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        # Using the free OpenRouter Gemini 2.0 Flash endpoint
        self.model = "google/gemini-2.5-flash-lite" 
        self.system_prompt = system_instruction

    def generate_content(self, prompt, **kwargs):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1
        )
        class MockResponse: text = res.choices[0].message.content
        return MockResponse()

    def start_chat(self, history=None):
        class MockChat:
            def __init__(self, parent, hist):
                self.parent = parent
                self.messages = []
                if self.parent.system_prompt:
                    self.messages.append({"role": "system", "content": self.parent.system_prompt})
                if hist:
                    for h in hist:
                        # Translate Google's history format to OpenAI format
                        role = "assistant" if h.get("role") == "model" else "user"
                        content = h["parts"][0] if "parts" in h else h.get("content", "")
                        self.messages.append({"role": role, "content": content})

            def send_message(self, prompt, **kwargs):
                self.messages.append({"role": "user", "content": prompt})
                res = self.parent.client.chat.completions.create(
                    model=self.parent.model,
                    messages=self.messages,
                    temperature=0.75
                )
                reply = res.choices[0].message.content
                self.messages.append({"role": "assistant", "content": reply})
                class MockResponse: text = reply
                return MockResponse()
                
        return MockChat(self, history)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global gemini_model, embedding_model
    
    print("🚀 Initializing ECHO via OpenRouter...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ WARNING: OPENROUTER_API_KEY not set in .env")
    else:
        # Initialize our wrapper instead of genai
        gemini_model = OpenRouterWrapper(system_instruction=SYSTEM_PROMPT)
        print("✅ OpenRouter (Gemini 2.0 Flash) loaded.")
    
    print("⏳ Loading sentence transformer...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")
    
    seed_knowledge_base(embedding_model)
    print("✅ Knowledge base ready.")
    print("🎯 ECHO is ready!\n")
    
    yield
    print("Shutting down ECHO...")


app = FastAPI(title="ECHO Mental Health AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    signals: dict
    turn_number: int
    is_crisis: bool


class SessionResetRequest(BaseModel):
    session_id: str


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "ECHO API running. Frontend not found at ../frontend/index.html"}


@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "gemini": gemini_model is not None,
        "embeddings": embedding_model is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — runs the full 5-layer ECHO pipeline.
    """
    if not gemini_model or not embedding_model:
        raise HTTPException(status_code=503, detail="Models not initialized. Check GEMINI_API_KEY.")
    
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # Session management
    session_id = request.session_id or str(uuid.uuid4())
    narrative = get_or_create_session(session_id)
    
    # ── LAYER 1: Safety pre-check (fast keyword scan) ──
    pre_signals = {"distress_severity": 5, "is_crisis": False}  # Default for pre-check
    pre_crisis = check_crisis(message, pre_signals)
    
    # ── LAYER 2: Signal Decoder ──
    context_summary = f"Themes: {', '.join(narrative.recurring_themes[:3])}" if narrative.recurring_themes else ""
    signals = decode_signals(message, gemini_model, context=context_summary)
    
    # ── LAYER 3: Crisis check with full signals ──
    crisis_check = check_crisis(message, signals)
    
    if crisis_check["is_crisis"]:
        signals["response_tone"] = "grounding"
        signals["distress_severity"] = max(signals.get("distress_severity", 0), 8)
    
    # ── LAYER 4: Retrieval ──
    # 4a. Knowledge base grounding
    grounding_docs = retrieve_grounding_context(message, signals, embedding_model, n_results=3)
    
    # 4b. Similar past moments (semantic memory)
    similar_moments = retrieve_similar_moments(message, session_id, embedding_model, n_results=2)
    
    # ── LAYER 5: Response Synthesis ──
    response = build_response(
        user_message=message,
        signals=signals,
        narrative=narrative,
        retrieved_knowledge=grounding_docs,
        similar_moments=similar_moments,
        model=gemini_model
    )
    
    # ── Anti-generic check + retry once if failed ──
    if not anti_generic_check(response):
        signals["_retry"] = True
        response = build_response(
            user_message=message,
            signals=signals,
            narrative=narrative,
            retrieved_knowledge=grounding_docs,
            similar_moments=similar_moments,
            model=gemini_model
        )
    
    # ── Apply safety layer (append crisis resources if needed) ──
    final_response = apply_safety_layer(response, crisis_check)
    
    # ── Update narrative ──
    updated_narrative = update_narrative(narrative, message, signals, final_response)
    save_session(updated_narrative)
    
    # ── Store in long-term memory ──
    store_turn_in_longterm(session_id, message, final_response, signals, embedding_model)
    
    return ChatResponse(
        response=final_response,
        session_id=session_id,
        signals=signals,
        turn_number=updated_narrative.turns,
        is_crisis=crisis_check["is_crisis"]
    )


@app.post("/reset")
async def reset_session(request: SessionResetRequest):
    """Reset a conversation session."""
    from memory_store import _sessions
    if request.session_id in _sessions:
        del _sessions[request.session_id]
    return {"status": "reset", "session_id": request.session_id}


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get current session narrative info (for debugging/demo)."""
    from memory_store import _sessions
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    narrative = _sessions[session_id]
    return {
        "session_id": session_id,
        "turns": narrative.turns,
        "themes": narrative.recurring_themes,
        "entities": narrative.known_entities,
        "cognitive_patterns": narrative.cognitive_patterns,
        "emotional_vocabulary": narrative.emotional_vocabulary,
        "trajectory": narrative.emotional_trajectory,
        "has_opened_up": narrative.has_opened_up
    }