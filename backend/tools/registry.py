"""Tool registry — the single source of truth for the agent's actions.

Drives two things:
  * the PERCEIVE prompt (catalog_text describes every tool + its args), and
  * ACT execution (execute() validates args and calls the handler).

Each tool maps to a function in actions.py. ``requires_confirmation`` marks the
money/irreversible tools that MUST pass the CONFIRM gate before running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import actions


@dataclass
class Param:
    name: str
    type: str  # "str" | "number"
    required: bool = True
    desc: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    params: list[Param]
    handler: Callable[..., dict]
    requires_confirmation: bool = False
    is_write: bool = False
    extra: dict = field(default_factory=dict)


TOOLS: dict[str, ToolSpec] = {
    "check_order_status": ToolSpec(
        name="check_order_status",
        description="Look up an order's status, items, and payments (detects duplicate charges).",
        params=[Param("order_id", "str", True, "The order id, e.g. '1234'.")],
        handler=actions.check_order_status,
    ),
    "get_refund_eligibility": ToolSpec(
        name="get_refund_eligibility",
        description="Check whether an order qualifies for a refund (applies the policy windows).",
        params=[Param("order_id", "str", True, "The order id.")],
        handler=actions.get_refund_eligibility,
    ),
    "track_shipment": ToolSpec(
        name="track_shipment",
        description="Get the shipping/delivery stage of an order.",
        params=[Param("order_id", "str", True, "The order id.")],
        handler=actions.track_shipment,
    ),
    "issue_refund": ToolSpec(
        name="issue_refund",
        description=(
            "Issue a refund for an order. For a duplicate charge, pass reason='duplicate' "
            "to reverse only the extra charge."
        ),
        params=[
            Param("order_id", "str", True, "The order id."),
            Param("amount", "number", False, "Amount to refund; omit for full order total."),
            Param("reason", "str", False, "Why (e.g. 'duplicate charge', 'defective')."),
        ],
        handler=actions.issue_refund,
        requires_confirmation=True,
        is_write=True,
    ),
    "cancel_order": ToolSpec(
        name="cancel_order",
        description="Cancel an order (only before it ships) and refund it.",
        params=[Param("order_id", "str", True, "The order id.")],
        handler=actions.cancel_order,
        requires_confirmation=True,
        is_write=True,
    ),
    "update_address": ToolSpec(
        name="update_address",
        description="Update the customer's delivery address (and any not-yet-shipped orders).",
        params=[
            Param("customer_id", "str", True, "The customer id."),
            Param("new_address", "str", True, "The full new address."),
        ],
        handler=actions.update_address,
        is_write=True,
    ),
    "reschedule_delivery": ToolSpec(
        name="reschedule_delivery",
        description="Reschedule delivery of an in-transit or preparing order to a new date.",
        params=[
            Param("order_id", "str", True, "The order id."),
            Param("new_date", "str", True, "The requested new delivery date."),
        ],
        handler=actions.reschedule_delivery,
        is_write=True,
    ),
    "create_ticket": ToolSpec(
        name="create_ticket",
        description="Open a support ticket (also used as a fallback when another tool fails).",
        params=[
            Param("customer_id", "str", True, "The customer id."),
            Param("subject", "str", True, "Short subject."),
            Param("body", "str", False, "Details."),
            Param("priority", "str", False, "low | normal | high | urgent."),
            Param("order_id", "str", False, "Related order id, if any."),
        ],
        handler=actions.create_ticket,
        is_write=True,
    ),
}


def requires_confirmation(name: str) -> bool:
    spec = TOOLS.get(name)
    return bool(spec and spec.requires_confirmation)


def catalog_text() -> str:
    """Human-readable catalog injected into the PERCEIVE prompt."""
    lines: list[str] = []
    for spec in TOOLS.values():
        args = ", ".join(
            f"{p.name}{'' if p.required else '?'}: {p.type}" for p in spec.params
        )
        tags = []
        if spec.is_write:
            tags.append("WRITE")
        if spec.requires_confirmation:
            tags.append("needs confirmation")
        tag_str = f" [{', '.join(tags)}]" if tags else " [read]"
        lines.append(f"- {spec.name}({args}){tag_str}: {spec.description}")
        for p in spec.params:
            if p.desc:
                lines.append(f"    • {p.name}: {p.desc}")
    return "\n".join(lines)


def _coerce(value: Any, type_: str) -> Any:
    if type_ == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return value


def execute(name: str, args: dict[str, Any]) -> dict:
    """Validate args and run the tool. Returns the structured {success,data,error}."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"success": False, "data": None, "error": f"Unknown tool '{name}'."}

    args = args or {}
    call_args: dict[str, Any] = {}
    for p in spec.params:
        if p.name in args and args[p.name] is not None:
            coerced = _coerce(args[p.name], p.type)
            if coerced is None and p.type == "number":
                return {"success": False, "data": None, "error": f"Invalid number for '{p.name}'."}
            call_args[p.name] = coerced
        elif p.required:
            return {
                "success": False,
                "data": None,
                "error": f"Missing required argument '{p.name}' for {name}.",
            }
    try:
        return spec.handler(**call_args)
    except Exception as exc:  # noqa: BLE001 - tools must return errors, not raise
        return {"success": False, "data": None, "error": f"{name} failed: {exc}"}
