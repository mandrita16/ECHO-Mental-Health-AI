"""
SAFETY GATE
-----------
Lightweight risk detection layer.
Runs BEFORE and AFTER response generation.

Detects:
  - Suicidal ideation (explicit and implicit)
  - Self-harm signals
  - Severe hopelessness
  - Immediate danger signals

When triggered: Appends crisis resources and shifts response tone to grounding.
"""

CRISIS_KEYWORDS = [
    "kill myself", "end my life", "don't want to be here", "want to die",
    "not worth living", "better off dead", "no point in living",
    "cut myself", "hurt myself", "self harm", "self-harm",
    "suicide", "suicidal", "ending it", "end it all",
    "goodbye forever", "won't be around", "last message",
]

HOPELESSNESS_SIGNALS = [
    "never going to get better", "no way out", "trapped forever",
    "pointless to even try", "nothing will ever change",
    "everyone would be better without me", "no one would miss me",
]

INDIA_CRISIS_RESOURCES = """

---
 If you're in crisis or having thoughts of harming yourself, please reach out:
• **iCall (India):** 9152987821 (Mon–Sat, 8am–10pm)
• **Vandrevala Foundation:** 1860-2662-345 (24/7, free, confidential)
• **iCall WhatsApp:** wa.me/919152987821

You don't have to figure this out alone right now."""


def check_crisis(message: str, signals: dict) -> dict:
    """
    Check for crisis signals in the message and decoded signals.
    Returns: {is_crisis: bool, severity: str, append_resources: bool}
    """
    message_lower = message.lower()
    
    # Check explicit keywords
    for keyword in CRISIS_KEYWORDS:
        if keyword in message_lower:
            return {
                "is_crisis": True,
                "severity": "high",
                "append_resources": True,
                "reason": f"keyword: {keyword}"
            }
    
    # Check hopelessness signals
    for signal in HOPELESSNESS_SIGNALS:
        if signal in message_lower:
            return {
                "is_crisis": True,
                "severity": "medium",
                "append_resources": True,
                "reason": f"hopelessness: {signal}"
            }
    
    # Check from decoded signals
    if signals.get("is_crisis", False):
        return {
            "is_crisis": True,
            "severity": "high",
            "append_resources": True,
            "reason": "signal_decoder_flagged"
        }
    
    # High distress severity (8+) warrants soft mention
    if signals.get("distress_severity", 0) >= 8:
        return {
            "is_crisis": False,
            "severity": "elevated",
            "append_resources": False,
            "reason": "high_distress"
        }
    
    return {
        "is_crisis": False,
        "severity": "normal",
        "append_resources": False,
        "reason": None
    }


def apply_safety_layer(response: str, crisis_check: dict) -> str:
    """
    Post-process the response to add crisis resources if needed.
    """
    if crisis_check.get("append_resources"):
        return response + INDIA_CRISIS_RESOURCES
    return response