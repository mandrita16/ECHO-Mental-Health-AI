"""
SIGNAL DECODER
--------------
Extracts multi-dimensional emotional signals from user messages.
Goes far beyond basic sentiment — detects:
  - Primary emotion (surface)
  - Underlying need (hidden driver)
  - Linguistic markers (dismissive language, minimizing, etc.)
  - Distress severity (0-10)
  - Implicit vs explicit distress flag
"""

import json
import google.generativeai as genai
from typing import Optional


SIGNAL_EXTRACTION_PROMPT = """You are an expert clinical psychologist performing rapid emotional triage.

Analyze this message from a user in a mental health support conversation.
Return ONLY a valid JSON object — no markdown, no explanation.

Message: "{message}"

Prior context summary (if any): "{context}"

Extract:
{{
  "primary_emotion": "<the surface emotion shown>",
  "underlying_need": "<what they actually need/want — validation, connection, clarity, relief, etc.>",
  "hidden_distress": "<what they are NOT saying but implies — be specific>",
  "linguistic_markers": ["<list of notable language patterns: minimizing words like 'whatever/fine', self-blame, helplessness, etc.>"],
  "distress_severity": <0-10 integer, 0=none, 10=crisis>,
  "distress_type": "<implicit|explicit|masked|neutral>",
  "key_entities": ["<people, places, events mentioned>"],
  "narrative_theme": "<the overarching life story theme: abandonment, failure, isolation, loss of identity, etc. or null>",
  "response_tone": "<warm-curious|gentle-challenging|validating|grounding|redirecting>",
  "is_crisis": <true|false — true only if suicidal ideation, self-harm, or immediate danger>
}}"""


def decode_signals(message: str, model, context: str = "") -> dict:
    """
    Run multi-signal emotional decoding on a user message.
    Returns structured signal dict.
    """
    prompt = SIGNAL_EXTRACTION_PROMPT.format(
        message=message,
        context=context if context else "None"
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temp for consistent extraction
                max_output_tokens=512,
            )
        )
        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        signals = json.loads(raw)
        return signals

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: return minimal signal dict
        return {
            "primary_emotion": "unknown",
            "underlying_need": "support",
            "hidden_distress": "unclear",
            "linguistic_markers": [],
            "distress_severity": 3,
            "distress_type": "implicit",
            "key_entities": [],
            "narrative_theme": None,
            "response_tone": "warm-curious",
            "is_crisis": False
        }