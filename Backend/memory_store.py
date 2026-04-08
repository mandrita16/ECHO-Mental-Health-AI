"""
MEMORY STORE
------------
Two-tier memory system:
  1. Short-term: In-memory session store (full narrative per session)
  2. Long-term: ChromaDB vector store (semantic memory, retrievable by similarity)

This enables:
  - Perfect recall within a session (entities, what was said)
  - Semantic retrieval of similar past moments for grounding responses
"""

import chromadb
from chromadb.config import Settings
import json
import uuid
from narrative_engine import UserNarrative


# In-memory session store: session_id -> UserNarrative
_sessions: dict[str, UserNarrative] = {}

# ChromaDB client (persistent)
_chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
_memory_collection = _chroma_client.get_or_create_collection(
    name="echo_memory",
    metadata={"hnsw:space": "cosine"}
)


def get_or_create_session(session_id: str) -> UserNarrative:
    """Retrieve existing session or create new narrative."""
    if session_id not in _sessions:
        _sessions[session_id] = UserNarrative(session_id=session_id)
    return _sessions[session_id]


def save_session(narrative: UserNarrative):
    """Persist updated narrative back to session store."""
    _sessions[narrative.session_id] = narrative


def store_turn_in_longterm(session_id: str, user_message: str, ai_response: str, signals: dict, embedding_model):
    """
    Store a conversation turn in ChromaDB for long-term semantic retrieval.
    Uses the combined emotional context as the document.
    """
    try:
        doc_text = f"User: {user_message}\nEmotion: {signals.get('primary_emotion', '')}\nNeed: {signals.get('underlying_need', '')}"
        
        # Generate embedding
        embedding = embedding_model.encode(doc_text).tolist()
        
        doc_id = f"{session_id}_{str(uuid.uuid4())[:8]}"
        
        _memory_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "session_id": session_id,
                "distress_severity": str(signals.get("distress_severity", 0)),
                "emotion": signals.get("primary_emotion", ""),
                "response": ai_response[:500]
            }]
        )
    except Exception as e:
        pass  # Non-critical — don't fail the main flow


def retrieve_similar_moments(query: str, session_id: str, embedding_model, n_results: int = 3) -> list:
    """
    Retrieve semantically similar past conversation moments.
    Used to provide continuity: 'Earlier you mentioned X — is this related?'
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()
        
        results = _memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, _memory_collection.count()),
            where={"session_id": session_id} if _memory_collection.count() > 0 else None
        )
        
        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []
    except Exception:
        return []