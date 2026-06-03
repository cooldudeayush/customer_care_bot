"""PERCEIVE / DECIDE — the single structured Gemini call (Phase 3).

One call returns {emotion, intents, action plan} together, instead of separate
classify-emotion / classify-intent / pick-tool calls. This is the free-tier
discipline (Section 3): ~2 LLM calls per turn (this PERCEIVE + a RESPOND).

The output drives routing: ANSWER | ACT | CLARIFY | CONFIRM | ESCALATE, plus the
concrete tool(s) + args to run. Money/irreversible tools are proposed via CONFIRM
(never executed before the user says yes).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm import GeminiClient, gemini_client
from memory.store import Message
from tools.actions import customer_snapshot
from tools.registry import catalog_text

logger = logging.getLogger("ccb.perceive")

ActionType = Literal["ANSWER", "ACT", "CLARIFY", "CONFIRM", "ESCALATE"]


class Emotion(BaseModel):
    state: Literal["angry", "frustrated", "confused", "neutral", "happy"] = "neutral"
    intensity: int = Field(default=1, description="1 (mild) to 5 (intense)")


class ToolCall(BaseModel):
    name: str = Field(description="A tool name from the catalog.")
    args_json: str = Field(default="{}", description="A JSON object of the tool's arguments.")


class Perception(BaseModel):
    emotion: Emotion
    intents: list[str] = Field(description="Short labels for what the user wants.")
    action_type: ActionType
    tools: list[ToolCall] = Field(
        default_factory=list, description="Tools to run (for ACT/CONFIRM), in order."
    )
    needs_clarification: bool = False
    clarification_question: str | None = None
    confirm_message: str | None = Field(
        default=None,
        description="For CONFIRM: a warm message stating exactly what will be done, ending in a yes/no ask.",
    )
    pending_decision: Literal["confirm", "deny", "none"] = "none"
    reasoning: str = ""


def _history_to_contents(history: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in history:
        role = "user" if m.role == "user" else "model"
        out.append({"role": role, "parts": [{"text": m.content}]})
    return out


def _snapshot_text(snapshot: dict) -> str:
    if not snapshot.get("known"):
        return "No customer record is available for this session."
    lines = [
        f"Customer: {snapshot['name']} (id={snapshot['customer_id']}, tier={snapshot['tier']}).",
        f"Saved address: {snapshot.get('address')}",
        "Orders:",
    ]
    for o in snapshot.get("orders", []):
        dup = (
            f" — DUPLICATE charge of Rs.{o['duplicate_charge']:.0f} detected"
            if o.get("duplicate_charge")
            else ""
        )
        lines.append(
            f"  - #{o['order_id']}: {o['items']} | status={o['status']} | total Rs.{o['total']:.0f}{dup}"
        )
    return "\n".join(lines)


def build_perception_system_instruction(
    snapshot: dict,
    pending_action: dict | None,
    customer_id: str | None,
    memory_text: str = "",
) -> str:
    pending_block = ""
    if pending_action:
        pending_block = (
            "\nA PROPOSED ACTION IS AWAITING THE USER'S CONFIRMATION:\n"
            f"  {pending_action.get('summary', pending_action)}\n"
            "Set pending_decision='confirm' ONLY if the latest user message clearly "
            "agrees to THIS EXACT action (e.g. 'yes', 'go ahead', 'do it', 'haan'). "
            "Set 'deny' if they decline (no/stop/cancel/wait). If they instead ask for "
            "something different (a different order or a new request), set 'none' and "
            "handle that new request — never treat an unrelated message as confirmation.\n"
        )
    return f"""\
You are the PERCEIVE/DECIDE stage of a customer care agent. Read the conversation
and analyze ONLY the latest user message (in context). Return the structured object.

WHAT TO PRODUCE:
1) emotion: state (angry|frustrated|confused|neutral|happy) + intensity 1–5.
2) intents: short labels (there may be several — multi-intent is fine).
3) action_type, and for ACT/CONFIRM the exact tools to run.

When you write confirm_message or clarification_question, phrase them to MATCH the
detected emotion: calm and apologetic if the customer is angry/frustrated, simple
and reassuring if they're confused, warm and brief otherwise.

ACTION TYPES:
- ANSWER: a question answerable from policy/knowledge or small talk. No tools.
- ACT: the request needs one or more SAFE tools (reads, or writes that are NOT
  money/irreversible: check_order_status, get_refund_eligibility, track_shipment,
  update_address, reschedule_delivery, create_ticket). Put the tool calls in `tools`.
- CONFIRM: the request needs a MONEY or IRREVERSIBLE tool (issue_refund,
  cancel_order). Put the planned tool call(s) in `tools` AND write `confirm_message`
  — a warm, specific message stating exactly what you'll do (amounts, order id) and
  ending by asking the user to confirm. Do NOT mark these as ACT; they must be
  confirmed first.
- CLARIFY: information is missing or ambiguous. Set needs_clarification=true and
  write clarification_question.
- ESCALATE: out of scope, legal disputes, repeated failures, or an explicit request
  for a human.

TOOL CALLS:
- Use ONLY these tools. args_json must be a JSON object string.
- Use exact order ids from the customer's orders below.
- The current customer_id is "{customer_id}". Use it for tools needing customer_id.
- For a duplicate charge, use issue_refund with reason="duplicate" and the duplicate amount.
- Compound / multi-intent requests (e.g. "refund this AND change my address"):
  set action_type=ACT and list ALL the needed tools in `tools`, in a sensible
  order. The system automatically runs the safe ones and asks the customer to
  confirm any money/irreversible ones — you don't need to split them yourself.

AVAILABLE TOOLS:
{catalog_text()}

CUSTOMER CONTEXT:
{_snapshot_text(snapshot)}

{memory_text}
{pending_block}
Be accurate: never invent order ids or amounts not present above.
"""


async def perceive(
    *,
    history: list[Message],
    customer_id: str | None,
    pending_action: dict | None,
    memory_text: str = "",
    llm: GeminiClient | None = None,
) -> Perception:
    """Run the single structured PERCEIVE/DECIDE call."""
    client = llm or gemini_client
    snapshot = (
        await asyncio.to_thread(customer_snapshot, customer_id)
        if customer_id
        else {"known": False}
    )
    system_instruction = build_perception_system_instruction(
        snapshot, pending_action, customer_id, memory_text
    )
    contents = _history_to_contents(history)
    perception = await client.generate_structured(
        contents,
        response_schema=Perception,
        system_instruction=system_instruction,
        temperature=0.2,
    )
    return perception


def parse_args(tool_call: ToolCall) -> dict:
    """Parse a ToolCall's args_json into a dict (tolerant of bad JSON)."""
    try:
        data = json.loads(tool_call.args_json or "{}")
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}
