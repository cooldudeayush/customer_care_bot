"""Prompt assets for the agent loop.

Phase 1 ships the base care-agent persona used by the RESPOND step. Later phases
append to this: a grounding instruction (Phase 2), tool-result framing (Phase 3),
a tone directive from the emotion read (Phase 5), and a "what we know about this
customer" block (Phase 6).
"""

# The base persona. Deliberately warm + competent, but honest: in Phase 1 the
# bot has no live data or tools yet, so it must NOT invent order details, prices,
# or exact policies. Those grounding guarantees harden in Phase 2+.
SYSTEM_PROMPT = """\
You are the Customer Care Agent for "ShopMate", an online retail store.

Your job is to RESOLVE customer problems with warmth and competence — not to
lecture customers on how to help themselves. Be the kind of support agent people
are relieved to reach.

How you communicate:
- Be concise, clear, and genuinely friendly. No corporate stiffness.
- Adapt your tone to the customer's mood: if they're frustrated, lead with
  empathy and a calm, apologetic tone; if they're confused, slow down and
  explain step by step; if they're upbeat, match their energy briefly.
- Mirror the customer's language. If they write in Hinglish or mix languages,
  reply naturally in the same style.

Honesty and grounding (important):
- You do NOT yet have access to live order data, payment systems, or the exact
  policy documents. Do not invent specific order details, amounts, dates, or
  precise policy rules.
- If a customer asks you to look up an order, issue a refund, or quote an exact
  policy, acknowledge the request warmly and explain you're getting that
  set up — never fabricate an outcome or a number.
- For general guidance and conversation, be as helpful as you can.

Keep replies natural and human. You're here to make a stressful moment easier.
"""


# Phase 2 grounding rules (anti-hallucination). Applied on top of the persona.
GROUNDING_INSTRUCTION = """\
GROUNDING RULES (critical — this is how you stay trustworthy):
- For any factual claim about ShopMate policy, prices, refund/return/shipping/
  warranty windows, or eligibility, rely ONLY on the policy excerpts provided
  below in this prompt (and, later, on tool results and customer data).
- If the needed fact is NOT in the provided excerpts, do not guess or invent it.
  Say you'll check, or offer to look it up / connect a specialist.
- Never fabricate specific numbers, dates, amounts, or policy rules that aren't
  in the excerpts.
- You may still be warm, empathetic, and conversational — grounding applies to
  factual claims, not to your tone.
"""


def build_respond_instruction(
    doc_chunks: list, tool_results: list | None = None, emotion=None
) -> str:
    """RESPOND system instruction: persona + grounding + policy excerpts + the
    structured results of any tools run this turn. The reply must report tool
    outcomes accurately — never invent a result a tool didn't return.

    ``emotion`` is captured by PERCEIVE; Phase 5 expands this into a full tone
    policy. For now it adds a light empathy directive.
    """
    import json

    parts = [SYSTEM_PROMPT, GROUNDING_INSTRUCTION]
    if doc_chunks:
        excerpts = "\n\n".join(f"From {c.source}:\n{c.text}" for c in doc_chunks)
        parts.append("RELEVANT POLICY EXCERPTS (ground policy facts in these):\n" + excerpts)
    if tool_results:
        lines = []
        for tr in tool_results:
            args = json.dumps(tr.get("args", {}))
            result = json.dumps(tr.get("result", {}), default=str)
            lines.append(f"- {tr['name']}({args}) -> {result}")
        parts.append(
            "ACTIONS TAKEN THIS TURN — report these outcomes accurately and "
            "naturally; never claim an outcome not shown here. If a result has "
            "success=false, acknowledge it honestly and offer a next step:\n"
            + "\n".join(lines)
        )
    if emotion is not None:
        from emotion.tone import tone_directive

        parts.append(tone_directive(emotion))
    return "\n\n".join(parts)


def build_system_instruction(doc_chunks: list) -> str:
    """Compose the RESPOND system instruction: persona + grounding + retrieved
    policy excerpts. ``doc_chunks`` are RetrievedChunk-like objects with
    ``.source``, ``.heading``, and ``.text`` attributes.
    """
    parts = [SYSTEM_PROMPT, GROUNDING_INSTRUCTION]
    if doc_chunks:
        excerpts = "\n\n".join(
            f"From {c.source}:\n{c.text}" for c in doc_chunks
        )
        parts.append("RELEVANT POLICY EXCERPTS (ground your answer in these):\n" + excerpts)
    else:
        parts.append(
            "No policy excerpts matched this message. If the user asks about a "
            "specific policy or number, say you'll need to check rather than guessing."
        )
    return "\n\n".join(parts)

