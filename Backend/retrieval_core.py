"""
RETRIEVAL CORE
--------------
Hybrid retrieval pipeline combining:
  1. Semantic search (ChromaDB + sentence-transformers)
  2. Emotional state matching (retrieve docs matching similar distress patterns)
  3. Knowledge base grounding (mental health resource corpus)

The retrieved context is used to ground responses in real therapeutic
frameworks and validated coping strategies — not hallucinated advice.
"""

import chromadb
from chromadb.config import Settings
import uuid


# Separate collection for the mental health knowledge base
_chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
_kb_collection = _chroma_client.get_or_create_collection(
    name="echo_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

_kb_seeded = False


MENTAL_HEALTH_KNOWLEDGE = [
    # Validation techniques
    {"text": "When someone expresses feeling like a burden, reflect their worth without dismissing their feelings. Say: 'You reaching out matters. The fact that you're still trying to understand this says something.'", "category": "validation"},
    {"text": "Emotional numbness often follows a period of intense stress or unprocessed grief. It is a protective response, not laziness or indifference.", "category": "psychoeducation"},
    {"text": "Self-blame after failure is a cognitive distortion. People rarely hold others to the same standard they apply to themselves.", "category": "cognitive-reframe"},
    
    # Implicit distress patterns
    {"text": "Minimizing language like 'it's fine', 'whatever', 'it doesn't matter' often signals the opposite — the speaker cares deeply but feels vulnerable about showing it.", "category": "signal-pattern"},
    {"text": "When someone stops participating in activities they used to enjoy (anhedonia), this is a significant clinical signal that warrants gentle, direct inquiry.", "category": "signal-pattern"},
    {"text": "Statements about not seeing the point, things feeling pointless, or questioning one's purpose may indicate hopelessness — a key risk factor worth exploring carefully.", "category": "risk-signal"},
    
    # Response techniques
    {"text": "Socratic questioning in emotional support: Instead of giving advice, ask questions that help the person arrive at their own insight. 'What do you think would happen if...' or 'When you imagine the best possible outcome...'", "category": "technique"},
    {"text": "Normalizing and naming: Giving language to vague distress ('that sounds like the kind of exhaustion that builds when nothing feels safe') helps people feel seen and understood.", "category": "technique"},
    {"text": "Avoid giving unsolicited solutions. Most people in distress want to feel heard first. Solutions feel dismissive when offered too early.", "category": "technique"},
    {"text": "Bridging technique: Connect current distress to prior mentions. 'Earlier you mentioned your parents — is any of what you're feeling now connected to that?'", "category": "technique"},
    
    # CBT-adjacent frameworks
    {"text": "Cognitive distortion: Catastrophizing — assuming the worst possible outcome is inevitable. Response: Gently explore evidence for and against the feared outcome.", "category": "cbt"},
    {"text": "Cognitive distortion: Personalization — taking responsibility for things outside one's control. Response: Explore what was actually in the person's control versus external factors.", "category": "cbt"},
    {"text": "Cognitive distortion: All-or-nothing thinking — seeing situations in black and white, no middle ground. Response: Introduce the idea of a spectrum.", "category": "cbt"},
    
    # Safety and escalation
    {"text": "If a person expresses thoughts of suicide or self-harm, prioritize safety. Acknowledge their pain, ask directly if they are safe, and provide crisis resources: iCall India: 9152987821, Vandrevala Foundation: 1860-2662-345 (24/7).", "category": "safety"},
    {"text": "Warning signs requiring escalation: giving away possessions, saying goodbye, expressing hopelessness about the future, specific plans for self-harm.", "category": "safety"},
    
    # Student-specific (relevant for target demographic)
    {"text": "Academic pressure combined with family stress creates compound burden. Students often feel they must perform to stabilize both their own future and their family's emotions.", "category": "student-context"},
    {"text": "Imposter syndrome in academic settings: the feeling that everyone else belongs except you. This is extremely common but rarely talked about, which makes it feel more isolating.", "category": "student-context"},
    {"text": "Social comparison on campus amplifies feelings of inadequacy. People curate what they show publicly, so comparisons are always asymmetric.", "category": "student-context"},
    
    # Grief and loss
    {"text": "Grief is not only about death. People grieve lost relationships, lost opportunities, lost versions of themselves. These losses are valid and often unacknowledged.", "category": "grief"},
    {"text": "Complicated grief can present as anger, numbness, or even relief — not just sadness. Validating the full spectrum of grief responses helps the person feel less abnormal.", "category": "grief"},
]


def seed_knowledge_base(embedding_model):
    """Seed the ChromaDB knowledge base with mental health corpus."""
    global _kb_seeded
    
    if _kb_seeded or _kb_collection.count() > 0:
        _kb_seeded = True
        return
    
    print("Seeding mental health knowledge base...")
    
    texts = [item["text"] for item in MENTAL_HEALTH_KNOWLEDGE]
    categories = [item["category"] for item in MENTAL_HEALTH_KNOWLEDGE]
    ids = [f"kb_{i}" for i in range(len(texts))]
    
    embeddings = embedding_model.encode(texts).tolist()
    
    _kb_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"category": c} for c in categories]
    )
    
    _kb_seeded = True
    print(f"Knowledge base seeded with {len(texts)} documents.")


def retrieve_grounding_context(query: str, signals: dict, embedding_model, n_results: int = 3) -> list:
    """
    Retrieve relevant therapeutic knowledge to ground the response.
    Combines semantic similarity + distress-type filtering.
    """
    if _kb_collection.count() == 0:
        return []
    
    try:
        # Build a rich query from the emotional signals
        enriched_query = f"{query} {signals.get('primary_emotion', '')} {signals.get('underlying_need', '')} {signals.get('hidden_distress', '')}"
        
        query_embedding = embedding_model.encode(enriched_query).tolist()
        
        results = _kb_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, _kb_collection.count())
        )
        
        if results and results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []
        
    except Exception:
        return []