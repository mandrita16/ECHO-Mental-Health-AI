"""
RESPONSE SYNTHESIZER
--------------------
Builds the final, hyper-personalized response prompt.

This is where the magic happens: all signals, narrative context, retrieved
knowledge, and conversation history are woven into a single, carefully
engineered prompt that produces responses matching the quality bar set
by the competition examples.

Key principle: The prompt doesn't just tell the model WHAT to say —
it tells the model WHO this specific person is and what they need right now.
"""

import google.generativeai as genai
from narrative_engine import UserNarrative, build_narrative_context_string


SYSTEM_PROMPT = """You are ECHO — an emotionally intelligent conversational AI designed for mental health support.

Your defining quality: You never respond to the message. You respond to the PERSON behind the message.

Core principles:
1. ZERO GENERIC RESPONSES. Every word must reflect what THIS person actually said.
2. READ BETWEEN THE LINES. What they said and what they mean are rarely the same thing.
3. ASK, DON'T TELL. One precise question beats three unsolicited suggestions.
4. MIRROR THEIR LANGUAGE. Use their vocabulary, not clinical terminology.
5. HOLD THE THREAD. Reference what they said earlier. Show you were listening.
6. SHORT AND SHARP. 3-5 sentences maximum unless they've opened up significantly.

Response structure (never make it obvious — let it feel natural):
  - Reflect what you actually heard (the subtext, not just the text)
  - One precise observation or gentle challenge
  - One question that opens the right door

Tone calibration:
  - warm-curious: Gentle, open, exploratory
  - gentle-challenging: Warm but willing to name something difficult
  - validating: Pure acknowledgment, no push
  - grounding: Practical, anchoring, here-and-now
  - redirecting: Carefully shifting focus when avoidance is happening

NEVER say: "I understand how you feel", "That sounds really hard", "Have you tried...", 
"You should...", "It's normal to feel...", "I'm here for you" as standalone phrases.
These are the responses of someone who isn't really listening.

If crisis signals are present: Acknowledge pain directly, ask if they're safe, 
provide: iCall India helpline 9152987821 or Vandrevala Foundation 1860-2662-345 (24/7 free)."""


def build_response(
    user_message: str,
    signals: dict,
    narrative: UserNarrative,
    retrieved_knowledge: list,
    similar_moments: list,
    model
) -> str:
    """
    Build the final personalized response using all available context.
    """
    
    # Build narrative context
    narrative_ctx = build_narrative_context_string(narrative)
    
    # Format retrieved knowledge (use sparingly — don't let it override personalization)
    knowledge_ctx = ""
    if retrieved_knowledge:
        knowledge_ctx = "Relevant therapeutic grounding (use implicitly, never quote directly):\n"
        knowledge_ctx += "\n".join([f"- {k}" for k in retrieved_knowledge[:2]])
    
    # Format similar past moments for continuity
    memory_ctx = ""
    if similar_moments and narrative.turns > 2:
        memory_ctx = "Semantically similar moments from this conversation:\n"
        memory_ctx += "\n".join([f"- {m}" for m in similar_moments[:2]])
    
    # Determine if we should surface a memory reference
    should_bridge = (
        narrative.turns > 2 and 
        len(narrative.known_entities) > 1 and
        signals.get("distress_severity", 0) >= 4
    )
    
    # Build the personalized context block
    context_block = f"""
=== USER PSYCHOLOGICAL PROFILE (this session) ===
{narrative_ctx}

=== CURRENT MESSAGE ANALYSIS ===
Surface emotion: {signals.get('primary_emotion', 'unknown')}
Underlying need: {signals.get('underlying_need', 'support')}
What they're NOT saying: {signals.get('hidden_distress', 'unclear')}
Distress severity: {signals.get('distress_severity', 0)}/10
Distress type: {signals.get('distress_type', 'implicit')}
Linguistic markers: {', '.join(signals.get('linguistic_markers', [])) or 'none'}
Recommended tone: {signals.get('response_tone', narrative.preferred_engagement_style)}
Is crisis: {signals.get('is_crisis', False)}

{"BRIDGE TO EARLIER: Consider referencing earlier context naturally." if should_bridge else ""}

{knowledge_ctx}
{memory_ctx}
=== END CONTEXT ===

Now respond to this user message: "{user_message}"

Remember: Respond to the PERSON, not the message. 3-5 sentences. Zero generic phrases."""

    # Build conversation history for multi-turn context
    messages = []
    for turn in narrative.turn_history[-8:]:  # Last 4 exchanges
        messages.append({
            "role": turn["role"],
            "parts": [turn["content"]]
        })
    
    # Add current context + message as the new user turn
    messages.append({
        "role": "user",
        "parts": [context_block]
    })
    
    try:
        chat = model.start_chat(history=messages[:-1] if len(messages) > 1 else [])
        response = chat.send_message(
            context_block,
            generation_config=genai.types.GenerationConfig(
                temperature=0.75,
                max_output_tokens=300,
                top_p=0.9,
            )
        )
        return response.text.strip()
        
    except Exception as e:
        # Fallback with minimal context
        try:
            response = model.generate_content(
                f"{SYSTEM_PROMPT}\n\nUser said: {user_message}\n\nRespond with emotional intelligence.",
                generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=200)
            )
            return response.text.strip()
        except Exception as e2:
            print(f"❌ GEMINI API ERROR (Fallback): {e2}")
            return "Something's weighing on you. Can you tell me a bit more about what's going on?"


def anti_generic_check(response: str) -> bool:
    """
    Lightweight check: flag responses that contain generic fallback phrases.
    Returns True if response passes (is specific enough).
    """
    generic_phrases = [
        "i understand how you feel",
        "that sounds really hard",
        "you're not alone",
        "it's normal to feel",
        "have you tried",
        "you should consider",
        "i'm here for you",
        "things will get better",
        "hang in there",
        "everything will be okay",
        "i hear you",
    ]
    
    response_lower = response.lower()
    for phrase in generic_phrases:
        if phrase in response_lower:
            return False  # Failed — too generic
    
    return True  # Passed