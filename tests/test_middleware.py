# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests with a real AgentScope ``Agent``, real tools and a scripted offline model."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import pytest
from agentscope.agent import Agent
from agentscope.message import ToolCallBlock, ToolResultState
from agentscope.middleware import MiddlewareBase
from agentscope.permission import PermissionBehavior, PermissionDecision, PermissionMode
from agentscope.tool import ToolChunk, ToolResponse

from agentscope_acgs import ACGSMiddleware, Verdict
from agentscope_acgs import middleware as module
from tests.fakes import (
    BLOCKED,
    HELLO,
    LEAK,
    REFUND,
    REFUND_REVIEW,
    Recorder,
    answer,
    ask,
    confirmations,
    make_agent,
    results,
    review_rules,
    text,
    tool_call,
)

NOT_RUN = "Tool call 'issue_refund' was not executed. "


def _refund_agent(recorder: Recorder, guard: ACGSMiddleware | None = None, **kwargs: Any) -> Agent:
    guard = guard or ACGSMiddleware(review_rules())
    return make_agent([tool_call("issue_refund", REFUND), text("Done.")], recorder, [guard], **kwargs)


async def test_allowed_call_runs() -> None:
    recorder = Recorder()
    agent = make_agent([tool_call("send_email", HELLO), text("Sent.")], recorder)
    events = await ask(agent)
    assert recorder.ran == [("send_email", HELLO)]
    assert results(agent) == {"call_1": ("success", "sent to friend@example.com")}
    assert confirmations(events) == []


async def test_blocked_call_is_denied_and_the_model_sees_why(caplog: pytest.LogCaptureFixture) -> None:
    recorder = Recorder()
    agent = make_agent([tool_call("send_email", LEAK), text("I could not send it.")], recorder)
    with caplog.at_level(logging.INFO, logger="agentscope_acgs.middleware"):
        await ask(agent)
    assert recorder.ran == []
    state, message = results(agent)["call_1"]
    assert state == "denied"
    assert message.startswith(BLOCKED)
    assert "hunter2" not in message
    assert agent.model.calls == 2  # type: ignore[attr-defined]
    assert "ACGS deny for tool send_email: ('ACGS-006',)" in caplog.text


async def test_review_asks_the_user_and_runs_after_confirmation() -> None:
    recorder = Recorder()
    agent = _refund_agent(recorder)
    events = await ask(agent)
    [pending] = confirmations(events)
    assert (pending.name, recorder.ran) == ("issue_refund", [])
    await answer(agent, confirmed=True)
    assert recorder.ran == [("issue_refund", REFUND)]
    assert results(agent)["call_1"] == ("success", "refunded 20.0 on A-17")


async def test_review_rejected_by_the_user_does_not_run() -> None:
    recorder = Recorder()
    agent = _refund_agent(recorder)
    assert len(confirmations(await ask(agent))) == 1
    await answer(agent, confirmed=False)
    assert recorder.ran == []
    assert results(agent)["call_1"][0] == "denied"


async def test_a_confirmed_call_edited_into_a_blocked_call_is_still_denied() -> None:
    recorder = Recorder()
    agent = _refund_agent(recorder)
    await ask(agent)
    await answer(agent, confirmed=True, name="send_email", arguments=LEAK)
    assert recorder.ran == []
    state, message = results(agent)["call_1"]
    assert state == "denied"
    assert message.startswith(BLOCKED)


async def test_review_turned_off_is_denied_without_asking() -> None:
    recorder = Recorder()
    agent = _refund_agent(recorder, ACGSMiddleware(review_rules(), on_review="deny"))
    assert confirmations(await ask(agent)) == []
    assert recorder.ran == []
    assert results(agent)["call_1"] == (
        "denied",
        f"{NOT_RUN}{REFUND_REVIEW} Review is turned off (on_review='deny'), so it was denied.",
    )


async def test_review_in_dont_ask_mode_is_denied() -> None:
    recorder = Recorder()
    agent = _refund_agent(recorder)
    agent.state.permission_context.mode = PermissionMode.DONT_ASK
    assert confirmations(await ask(agent)) == []
    assert recorder.ran == []
    assert results(agent)["call_1"] == (
        "denied",
        f"{NOT_RUN}{REFUND_REVIEW} The agent's permission mode is DONT_ASK, so nobody can review it and it was denied.",
    )


async def test_bypass_mode_does_not_skip_review_or_blocks() -> None:
    recorder = Recorder()
    agent = _refund_agent(recorder)
    agent.state.permission_context.mode = PermissionMode.BYPASS
    assert len(confirmations(await ask(agent))) == 1
    assert recorder.ran == []

    blocked = make_agent([tool_call("send_email", LEAK), text("ok")], recorder)
    blocked.state.permission_context.mode = PermissionMode.BYPASS
    await ask(blocked)
    assert recorder.ran == []
    assert results(blocked)["call_1"][0] == "denied"


async def test_a_deny_from_agentscope_wins_over_a_review() -> None:
    recorder = Recorder()
    tool_deny = PermissionDecision(behavior=PermissionBehavior.DENY, message="the tool refuses")
    agent = make_agent(
        [tool_call("issue_refund", REFUND), text("ok")],
        recorder,
        [ACGSMiddleware(review_rules())],
        permission=tool_deny,
    )
    assert confirmations(await ask(agent)) == []
    assert recorder.ran == []
    assert results(agent)["call_1"] == ("denied", "the tool refuses")


async def test_agentscope_permission_engine_still_applies_to_allowed_calls() -> None:
    recorder = Recorder()
    # Without a tool-level ALLOW, AgentScope's own engine asks in DEFAULT mode; ACGS keeps that decision.
    agent = make_agent([tool_call("send_email", HELLO), text("Sent.")], recorder, permission=None)
    assert len(confirmations(await ask(agent))) == 1
    assert recorder.ran == []
    await answer(agent, confirmed=True, name="send_email", arguments=HELLO)
    assert recorder.ran == [("send_email", HELLO)]


class _AllowEverything(MiddlewareBase):
    """A middleware that returns ALLOW without calling the rest of the chain."""

    async def on_check_permission(
        self,
        agent: Agent,
        input_kwargs: dict[Any, Any],
        next_handler: Callable[..., Awaitable[PermissionDecision]],
    ) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="allowed by an earlier middleware")


async def test_a_call_that_skipped_the_check_is_not_executed() -> None:
    recorder = Recorder()
    agent = make_agent([tool_call("send_email", LEAK), text("ok")], recorder, [_AllowEverything(), ACGSMiddleware()])
    await ask(agent)
    assert recorder.ran == []
    assert results(agent)["call_1"] == (
        "denied",
        "Tool call 'send_email' was not executed. ACGS governance did not approve this exact tool call before it "
        "reached execution, so it was not executed.",
    )


async def _no_permission(**kwargs: Any) -> PermissionDecision:
    raise AssertionError


async def test_an_evaluation_error_denies(caplog: pytest.LogCaptureFixture) -> None:
    agent = make_agent([], Recorder())
    call = ToolCallBlock(id="c", name="send_email", input="{}")
    with caplog.at_level(logging.WARNING, logger="agentscope_acgs.middleware"):
        decision = await ACGSMiddleware().on_check_permission(agent, {"tool_call": call}, _no_permission)
    assert decision.behavior == PermissionBehavior.DENY
    assert decision.message == "ACGS governance could not evaluate this tool call, so it was denied."
    assert "ACGS permission check failed" in caplog.text


async def _run_tool(**kwargs: Any) -> AsyncGenerator[ToolChunk | ToolResponse, None]:
    yield ToolResponse()


async def test_acting_refuses_a_call_changed_after_approval() -> None:
    agent = make_agent([], Recorder())
    guard = ACGSMiddleware()

    async def allow(**kwargs: Any) -> PermissionDecision:
        return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="ok")

    approved = ToolCallBlock(id="c", name="send_email", input='{"to": "a@example.com", "body": "hi"}')
    decision = await guard.on_check_permission(
        agent, {"tool_call": approved, "tool": None, "tool_input": {"to": "a@example.com", "body": "hi"}}, allow
    )
    assert decision.behavior == PermissionBehavior.ALLOW
    changed = approved.model_copy(update={"input": '{"to": "b@example.com", "body": "hi"}'})
    [response] = [chunk async for chunk in guard.on_acting(agent, {"tool_call": changed}, _run_tool)]
    assert isinstance(response, ToolResponse)
    assert response.state == ToolResultState.DENIED
    # The approval is used up, so the original call cannot run either.
    [again] = [chunk async for chunk in guard.on_acting(agent, {"tool_call": approved}, _run_tool)]
    assert isinstance(again, ToolResponse)
    assert again.state == ToolResultState.DENIED


async def test_pending_approvals_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_MAX_PENDING_APPROVALS", 1)
    agent = make_agent([], Recorder())
    guard = ACGSMiddleware()
    first = ToolCallBlock(id="1", name="t", input="{}")
    second = ToolCallBlock(id="2", name="t", input="{}")
    guard._remember(agent, first)
    guard._remember(agent, second)
    assert list(guard._approved) == [(id(agent), "2")]


def test_evaluate_and_options() -> None:
    guard = ACGSMiddleware(agent_id="payments")
    assert guard.evaluate("send_email", LEAK).verdict is Verdict.DENY
    assert guard.evaluate("send_email", HELLO).verdict is Verdict.ALLOW
    with pytest.raises(ValueError, match="on_review must be 'ask' or 'deny'"):
        ACGSMiddleware(on_review="maybe")  # type: ignore[arg-type]


async def test_default_engine_keeps_no_record_of_checked_calls(caplog: pytest.LogCaptureFixture) -> None:
    """The README's data-handling section: argument values are not stored, echoed or logged."""
    guard = ACGSMiddleware(agent_id="payments-app")
    recorder = Recorder()
    agent = make_agent([tool_call("send_email", LEAK), text("ok")], recorder, [guard])
    with caplog.at_level(logging.DEBUG, logger="agentscope_acgs"):
        await ask(agent)
    engine = guard.governor._engine
    assert engine.stats["audit_mode"] == "fast"
    assert engine.stats["audit_entry_count"] == 0
    assert engine.audit_log.entries == []
    assert "hunter2" not in results(agent)["call_1"][1]
    assert "hunter2" not in caplog.text
