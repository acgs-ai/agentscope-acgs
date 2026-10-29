# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance for AgentScope agents: a middleware that checks each tool call before the tool runs."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from acgs_lite import Constitution
from agentscope.message import TextBlock, ToolCallBlock, ToolCallState, ToolResultState
from agentscope.middleware import MiddlewareBase
from agentscope.permission import PermissionBehavior, PermissionDecision, PermissionMode
from agentscope.tool import ToolChunk, ToolResponse

from agentscope_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

if TYPE_CHECKING:
    from agentscope.agent import Agent

logger = logging.getLogger(__name__)

_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."
_REVIEW_DISABLED = "Review is turned off (on_review='deny'), so it was denied."
_REVIEW_NOT_ASKED = "The agent's permission mode is DONT_ASK, so nobody can review it and it was denied."
_NOT_CHECKED = (
    "ACGS governance did not approve this exact tool call before it reached execution, so it was not executed."
)
_MAX_PENDING_APPROVALS = 4096

PermissionHandler = Callable[..., Awaitable[PermissionDecision]]
ActingHandler = Callable[..., AsyncGenerator[ToolChunk | ToolResponse, None]]


def _not_executed(tool_name: str, reason: str) -> str:
    return f"Tool call '{tool_name}' was not executed. {reason}"


class ACGSMiddleware(MiddlewareBase):
    """Check every AgentScope tool call against an ACGS constitution before the tool runs.

    The middleware implements two AgentScope hooks:

    - ``on_check_permission`` evaluates the validated tool input. A blocking rule returns ``DENY``; a call that
      matches only review rules returns ``ASK`` (or ``DENY`` with ``on_review="deny"``); an allowed call is passed on
      to the next middleware and AgentScope's own permission engine, whose decision is kept. Any error denies.
    - ``on_acting`` runs a tool only if ``on_check_permission`` returned ``ALLOW`` for that same call ID, tool name
      and raw input. Anything else gets a ``DENIED`` tool result instead of running.

    Register it first in ``Agent(middlewares=[...])`` so it sees the call before any other middleware.
    """

    def __init__(
        self,
        constitution: Constitution | None = None,
        *,
        agent_id: str = "agentscope",
        on_review: Literal["ask", "deny"] = "ask",
        tool_call_formatter: ToolCallFormatter | None = None,
    ) -> None:
        """Create the middleware.

        Args:
            constitution: The rules to apply. Defaults to the acgs-lite default constitution.
            agent_id: The agent ID passed to the ACGS engine.
            on_review: ``"ask"`` turns a review-only match into AgentScope user confirmation (``ASK``);
                ``"deny"`` denies it.
            tool_call_formatter: Optional custom formatter for the text the rules are matched against.

        Raises:
            ValueError: If ``on_review`` is not ``"ask"`` or ``"deny"``.
        """
        if on_review not in ("ask", "deny"):
            msg = f"on_review must be 'ask' or 'deny', not {on_review!r}"
            raise ValueError(msg)
        self.agent_id = agent_id
        self.on_review = on_review
        self.governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)
        self._approved: OrderedDict[tuple[int, str], tuple[str, str]] = OrderedDict()

    def evaluate(self, tool_name: str, arguments: Any) -> GovernanceDecision:
        """Evaluate one tool call without running it. Never raises; errors give ``DENY``."""
        return self.governor.evaluate(tool_name, arguments)

    async def on_check_permission(
        self,
        agent: Agent,
        input_kwargs: dict[Any, Any],
        next_handler: PermissionHandler,
    ) -> PermissionDecision:
        """Decide whether a tool call may run."""
        try:
            tool_call: ToolCallBlock = input_kwargs["tool_call"]
            name = str(tool_call.name)
            decision = self.governor.evaluate(name, input_kwargs["tool_input"])
            confirmed = tool_call.state == ToolCallState.ALLOWED
            dont_ask = agent.state.permission_context.mode == PermissionMode.DONT_ASK
        except Exception:
            logger.warning("ACGS permission check failed; denying the tool call", exc_info=True)
            return PermissionDecision(
                behavior=PermissionBehavior.DENY,
                message=_EVALUATION_FAILED,
                decision_reason="acgs: evaluation failed",
            )

        if decision.verdict is not Verdict.ALLOW:
            logger.info("ACGS %s for tool %s: %s", decision.verdict.value, name, decision.rule_ids)
        if decision.verdict is Verdict.DENY:
            return self._deny(name, decision.reason)
        if decision.verdict is Verdict.REVIEW and not confirmed:
            return await self._review(name, decision, input_kwargs, next_handler, dont_ask=dont_ask)

        # ALLOW, or REVIEW that the user already confirmed: AgentScope's own checks still apply.
        inner = await next_handler(**input_kwargs)
        if inner.behavior == PermissionBehavior.ALLOW:
            self._remember(agent, tool_call)
        return inner

    async def on_acting(
        self,
        agent: Agent,
        input_kwargs: dict[Any, Any],
        next_handler: ActingHandler,
    ) -> AsyncGenerator[ToolChunk | ToolResponse, None]:
        """Run the tool only if this exact call was approved by ``on_check_permission``."""
        tool_call: ToolCallBlock = input_kwargs["tool_call"]
        expected = self._approved.pop((id(agent), tool_call.id), None)
        if expected != (tool_call.name, tool_call.input):
            logger.warning("ACGS refused to run tool %s: it was not approved in this exact form", tool_call.name)
            yield ToolResponse(
                content=[TextBlock(text=_not_executed(str(tool_call.name), _NOT_CHECKED))],
                state=ToolResultState.DENIED,
            )
            return
        async for chunk in next_handler(**input_kwargs):
            yield chunk

    async def _review(
        self,
        name: str,
        decision: GovernanceDecision,
        input_kwargs: dict[Any, Any],
        next_handler: PermissionHandler,
        *,
        dont_ask: bool,
    ) -> PermissionDecision:
        if self.on_review == "deny":
            return self._deny(name, f"{decision.reason} {_REVIEW_DISABLED}")
        if dont_ask:
            return self._deny(name, f"{decision.reason} {_REVIEW_NOT_ASKED}")
        inner = await next_handler(**input_kwargs)
        if inner.behavior == PermissionBehavior.DENY:
            return inner
        # bypass_immune: no allow rule or batch decision may turn this into an ALLOW without a per-call confirmation.
        return PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message=f"ACGS: the agent wants to call the tool '{name}'. {decision.reason} Approve this call?",
            decision_reason="acgs: review required",
            bypass_immune=True,
        )

    def _remember(self, agent: Agent, tool_call: ToolCallBlock) -> None:
        self._approved[(id(agent), tool_call.id)] = (tool_call.name, tool_call.input)
        while len(self._approved) > _MAX_PENDING_APPROVALS:
            self._approved.popitem(last=False)

    @staticmethod
    def _deny(name: str, reason: str) -> PermissionDecision:
        return PermissionDecision(
            behavior=PermissionBehavior.DENY,
            message=_not_executed(name, reason),
            decision_reason="acgs: denied",
        )
