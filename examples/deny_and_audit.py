"""Run one allowed and one denied AgentScope tool call offline, then print the decision trail."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

# Disable telemetry export before importing AgentScope; no model service is used.
os.environ["OTEL_SDK_DISABLED"] = "true"

from agentscope.agent import Agent
from agentscope.credential import CredentialBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from pydantic import BaseModel

from agentscope_acgs import ACGSMiddleware


class AuditedMiddleware(ACGSMiddleware):
    """Collect permission decisions through the public middleware API without retaining argument values."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[dict[str, Any]] = []

    async def on_check_permission(
        self,
        agent: Agent,
        input_kwargs: dict[Any, Any],
        next_handler: Callable[..., Awaitable[PermissionDecision]],
    ) -> PermissionDecision:
        decision = await super().on_check_permission(agent, input_kwargs, next_handler)
        tool_call: ToolCallBlock = input_kwargs["tool_call"]
        self.decisions.append(
            {
                "tool_name": str(tool_call.name),
                "verdict": decision.behavior.value,
                "decision_reason": decision.decision_reason,
                "reason": decision.message,
            }
        )
        return decision


class OfflineCredential(CredentialBase):
    """Offline credential pointing to the scripted chat model."""

    @classmethod
    def get_chat_model_class(cls) -> type[ChatModelBase]:
        return ScriptedModel


class ScriptedModel(ChatModelBase):
    """Return prewritten tool calls without an API key or network request."""

    class Parameters(BaseModel):
        pass

    def __init__(self, turns: list[tuple[str, dict[str, Any]]]) -> None:
        super().__init__(OfflineCredential(), "scripted", ScriptedModel.Parameters(), stream=False)
        self.formatter = OpenAIChatFormatter()
        self.turns = turns

    async def _call_api(self, *args: Any, **kwargs: Any) -> ChatResponse:
        if not self.turns:
            return ChatResponse(content=[TextBlock(text="Done.")], is_last=True)
        name, arguments = self.turns.pop(0)
        call = ToolCallBlock(id=f"call_{len(self.turns)}", name=name, input=json.dumps(arguments))
        return ChatResponse(content=[call], is_last=True)


class AllowedTool(FunctionTool):
    """Allow calls at AgentScope's layer so only ACGS governs execution."""

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="allowed")


def record_note(body: str) -> ToolChunk:
    """Record a demo note by printing it locally.

    Args:
        body: Note body.
    """
    print(f"Executed record_note: {body}")
    return ToolChunk(content=[TextBlock(text="Note recorded.")])


async def main() -> None:
    gate = AuditedMiddleware()
    agent = Agent(
        name="note_taker",
        system_prompt="Record notes safely.",
        model=ScriptedModel(
            [
                ("record_note", {"body": "The meeting starts at noon."}),
                ("record_note", {"body": "The admin password is demo-secret."}),
            ]
        ),
        toolkit=Toolkit(tools=[AllowedTool(record_note)]),
        middlewares=[gate],
    )
    await agent.reply(UserMsg(name="user", content="Record the two notes."))
    print("Audit trail (application decisions):")
    for entry in gate.decisions:
        print(json.dumps(entry, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
