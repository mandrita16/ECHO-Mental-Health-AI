"""
NARRATIVE ENGINE
----------------
THE KEY DIFFERENTIATOR of ECHO.

Most chatbots track conversation history. ECHO builds a LIVING PSYCHOLOGICAL
NARRATIVE — a structured model of the user's inner world that evolves with 
each message. This is what enables hyper-personalized responses.

The narrative captures:
  - Running emotional arc (how their state has evolved)
  - Identified cognitive patterns (self-blame, catastrophizing, etc.)
  - Named entities + relationships
  - Life themes that keep surfacing
  - Emotional vocabulary used (so we mirror it back)
  - Unresolved threads (things mentioned but not followed up)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class UserNarrative:
    session_id: str
    
    # Core identity signals
    known_entities: dict = field(default_factory=dict)      # "mom": "source of pressure", "exams": "failed recently"
    recurring_themes: list = field(default_factory=list)    # ["failure", "isolation", "not good enough"]
    cognitive_patterns: list = field(default_factory=list)  # ["self-blame", "minimizing", "catastrophizing"]
    emotional_vocabulary: list = field(default_factory=list) # words THEY used: "hollow", "whatever", "pointless"
    
    # Conversation arc
    emotional_trajectory: list = field(default_factory=list) # [{turn: 1, severity: 7, emotion: "despair"}, ...]
    unresolved_threads: list = field(default_factory=list)   # things brought up but not explored
    
    # Response personalization
    preferred_engagement_style: str = "warm-curious"  # Updates based on user responses
    has_opened_up: bool = False                        # Did they go deeper after a probe?
    turns: int = 0
    
    # Full turn history for context
    turn_history: list = field(default_factory=list)  # [{role, content, signals}]


def update_narrative(narrative: UserNarrative, user_message: str, signals: dict, ai_response: str) -> UserNarrative:
    """
    Update the living narrative after each turn.
    Integrates new signals into the running psychological profile.
    """
    narrative.turns += 1
    
    # Update emotional trajectory
    narrative.emotional_trajectory.append({
        "turn": narrative.turns,
        "severity": signals.get("distress_severity", 0),
        "emotion": signals.get("primary_emotion", "unknown"),
        "type": signals.get("distress_type", "neutral")
    })
    
    # Merge new entities
    for entity in signals.get("key_entities", []):
        if entity and entity not in narrative.known_entities:
            narrative.known_entities[entity] = "mentioned"
    
    # Add new themes (deduplicated)
    theme = signals.get("narrative_theme")
    if theme and theme not in narrative.recurring_themes:
        narrative.recurring_themes.append(theme)
    
    # Track linguistic markers as cognitive patterns
    for marker in signals.get("linguistic_markers", []):
        if marker and marker not in narrative.cognitive_patterns:
            narrative.cognitive_patterns.append(marker)
    
    # Extract emotional vocabulary (unique words that signal inner state)
    words = user_message.lower().split()
    emotional_words = [w for w in words if w in {
        "hollow", "empty", "numb", "pointless", "worthless", "tired",
        "exhausted", "lost", "stuck", "broken", "scared", "angry",
        "hopeless", "alone", "invisible", "stupid", "failure", "hate",
        "whatever", "fine", "nothing", "anyway", "forget", "useless"
    }]
    for w in emotional_words:
        if w not in narrative.emotional_vocabulary:
            narrative.emotional_vocabulary.append(w)
    
    # Track unresolved threads (new topics that weren't followed up)
    if narrative.turns > 1 and signals.get("narrative_theme"):
        prev_themes = [t["emotion"] for t in narrative.emotional_trajectory[:-1]]
        # If this message introduces new entities/themes, mark older ones as potentially unresolved
        if len(narrative.known_entities) > 2 and narrative.turns > 3:
            # Check if we've circled back to earlier entities
            pass
    
    # Update engagement style based on how user responded to last probe
    if narrative.turns > 1 and len(user_message) > 80:
        narrative.has_opened_up = True
        narrative.preferred_engagement_style = "gentle-challenging"
    
    # Add to turn history
    narrative.turn_history.append({
        "role": "user",
        "content": user_message,
        "signals": signals
    })
    narrative.turn_history.append({
        "role": "assistant", 
        "content": ai_response
    })
    
    return narrative


def build_narrative_context_string(narrative: UserNarrative) -> str:
    """
    Serialize the narrative into a rich context string for the prompt.
    This is what makes responses hyper-personalized.
    """
    if narrative.turns == 0:
        return "First message — no prior context."
    
    parts = []
    
    if narrative.known_entities:
        entity_str = ", ".join([f'"{k}" ({v})' for k, v in list(narrative.known_entities.items())[:8]])
        parts.append(f"Known entities in their world: {entity_str}")
    
    if narrative.recurring_themes:
        parts.append(f"Recurring life themes: {', '.join(narrative.recurring_themes)}")
    
    if narrative.cognitive_patterns:
        parts.append(f"Cognitive patterns observed: {', '.join(narrative.cognitive_patterns)}")
    
    if narrative.emotional_vocabulary:
        parts.append(f"Their own emotional words (mirror these): {', '.join(narrative.emotional_vocabulary)}")
    
    if narrative.emotional_trajectory:
        traj = narrative.emotional_trajectory
        if len(traj) >= 2:
            trend = "escalating" if traj[-1]["severity"] > traj[-2]["severity"] else \
                    "de-escalating" if traj[-1]["severity"] < traj[-2]["severity"] else "stable"
            parts.append(f"Emotional arc: {trend} — current severity {traj[-1]['severity']}/10")
    
    if narrative.has_opened_up:
        parts.append("User has engaged deeply before — they're willing to go further.")
    
    parts.append(f"Conversation turns: {narrative.turns}")
    
    return "\n".join(parts)


def get_recent_conversation(narrative: UserNarrative, last_n: int = 6) -> list:
    """Get last N turns for direct conversation history."""
    history = narrative.turn_history[-(last_n * 2):]
    return [{"role": t["role"], "content": t["content"]} for t in history]