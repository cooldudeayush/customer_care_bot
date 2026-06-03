"""Emotional adaptation — the tone policy (Phase 5).

The PERCEIVE call already returns an emotion read {state, intensity}. This module
turns that into a TONE DIRECTIVE injected into the RESPOND prompt, so the bot's
voice adapts to the customer's mood — calm and apologetic for someone angry,
slow and step-by-step for someone confused. This is adaptive (smarter), NOT naive
always-cheerfulness.
"""

from __future__ import annotations

# Per-state guidance appended to the RESPOND system prompt.
TONE_POLICY: dict[str, str] = {
    "angry": (
        "Tone: the customer is ANGRY. Acknowledge their frustration sincerely and "
        "apologize FIRST, before anything else. Stay calm, warm, and concise. Lead "
        "with the fix, not pleasantries. Use NO cheerfulness, exclamation marks, or "
        "upbeat filler. Never be defensive or blame the customer."
    ),
    "frustrated": (
        "Tone: the customer is FRUSTRATED. Briefly acknowledge the frustration, then "
        "get straight to resolving it. Be calm, efficient, and reassuring. Skip small "
        "talk and keep cheerfulness minimal."
    ),
    "confused": (
        "Tone: the customer is CONFUSED. Be patient and go slowly. Explain step by "
        "step in plain language, define any jargon, and offer to do things for them. "
        "Reassure them that it's easy and you'll guide them through it."
    ),
    "neutral": (
        "Tone: be warm, friendly, and efficient. Get to the point without being curt."
    ),
    "happy": (
        "Tone: the customer is in a good mood. Match their positive energy, keep it "
        "brief and upbeat, and don't over-explain."
    ),
}

_HIGH_INTENSITY = (
    " They appear VERY upset (high intensity) — apologize sincerely, prioritize "
    "fixing this right now, avoid anything that could read as dismissive, and be "
    "ready to escalate to a specialist if you can't fully resolve it."
)


def tone_directive(emotion) -> str:
    """Build the tone directive for a perceived emotion {state, intensity}."""
    state = (getattr(emotion, "state", None) or "neutral").lower()
    try:
        intensity = int(getattr(emotion, "intensity", 1) or 1)
    except (TypeError, ValueError):
        intensity = 1

    directive = TONE_POLICY.get(state, TONE_POLICY["neutral"])
    if state in ("angry", "frustrated") and intensity >= 4:
        directive += _HIGH_INTENSITY
    return f"The customer's current emotional state reads as {state} " \
           f"(intensity {intensity}/5).\n{directive}"
