# agentscope-acgs

[![CI](https://github.com/acgs-ai/agentscope-acgs/actions/workflows/ci.yml/badge.svg)](https://github.com/acgs-ai/agentscope-acgs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentscope-acgs.svg)](https://pypi.org/project/agentscope-acgs/)
[![Python](https://img.shields.io/pypi/pyversions/agentscope-acgs.svg)](https://pypi.org/project/agentscope-acgs/)

`agentscope-acgs` checks [AgentScope](https://github.com/agentscope-ai/agentscope) 2.x tool calls against
[ACGS](https://pypi.org/project/acgs-lite/) constitution rules before the tool runs. `ACGSMiddleware` is an
AgentScope agent middleware: pass it in `Agent(middlewares=[...])` and it answers AgentScope's permission check
for every tool call.

What happens to a call:

- Calls that match no blocking rule go on to AgentScope's own permission engine, whose decision is kept.
- Calls that match a blocking rule are denied. The model gets the reason as a `denied` tool result and can continue.
- Calls that only match review rules pause the reply with AgentScope's user confirmation
  (`RequireUserConfirmEvent`). They run only if the user confirms that call.
- A blocking rule always wins, also over a user confirmation and over AgentScope's `BYPASS` mode.
- Any error while evaluating a call denies it (fail-closed).
- A tool runs only if the middleware allowed that exact call (same ID, tool name and input). A call that reaches
  execution any other way gets a `denied` result instead of running.

The package is a thin adapter over the public `acgs-lite` API.

> [!NOTE]
> This is a community package. It is not part of AgentScope and is not maintained, reviewed or endorsed by the
> AgentScope maintainers.

## Installation

```bash
pip install agentscope-acgs
```

Requires Python 3.11 to 3.14, `agentscope>=2.0.6,<2.1` and `acgs-lite>=3.0.1,<4`. AgentScope 2.0.6 is the first
release with the `on_check_permission` middleware hook.

## Usage

```python
from agentscope.agent import Agent

from agentscope_acgs import ACGSMiddleware

# model is any AgentScope chat model and toolkit any agentscope.tool.Toolkit.
agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    model=model,
    toolkit=toolkit,
    middlewares=[ACGSMiddleware()],  # register it first, see "How it hooks in"
)
```

## Quickstart

`examples/quickstart.py` runs offline. A small scripted model stands in for a real one:

```python
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
```

Output:

```text
review: issue_refund {"order_id": "A-17", "amount": 20.0}
success: send_email -> sent to friend@example.com
denied: send_email -> Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses).
success: issue_refund -> refunded 20.0 for A-17
answer: Done.
```

## How a call is evaluated

1. The tool name and the call's input are rendered as text by `format_tool_call`:

   ```python
   from agentscope_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

   The input is the `tool_input` AgentScope passes to the permission check: the model's arguments parsed and
   validated against the tool's `input_schema`. Calls whose input fails validation never reach the permission
   check; AgentScope returns an error result for them.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | AgentScope's own permission check decides |
   | Only `require_human_review` / `escalate_to_senior` rules matched | `ASK` (see [Review](#review)) |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | `DENY` |
   | A rule ID not found in the constitution, or an invalid result without a rule | `DENY` |
   | Any exception while reading or evaluating the call | `DENY` (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.

A denied call is not executed. The message is `"Tool call '<name>' was not executed. <reason>"`, which AgentScope
stores as the tool result, so the model reads the reason. The reason names the matched rules but never repeats
argument values. `ACGSMiddleware.evaluate(tool_name, args)` returns the `GovernanceDecision` without running
anything.

## How it hooks in

AgentScope 2.x runs every tool call through a permission check before it executes it. Agent middlewares that
implement `on_check_permission` wrap that check in registration order, around AgentScope's `PermissionEngine`
(permission mode, rules and the tool's own `check_permissions`). The returned `PermissionDecision` decides:
`ALLOW` runs the tool, `ASK` pauses the reply for user confirmation, `DENY` stores a `denied` result.

`ACGSMiddleware` implements two hooks:

1. `on_check_permission` evaluates the call. `DENY` is returned without consulting the rest of the chain. For an
   allowed call it calls the next handler and returns its decision unchanged, so ACGS never loosens AgentScope's
   own checks. For a review call it also calls the next handler, keeps a `DENY` from it, and otherwise returns
   `ASK`.
2. `on_acting` wraps tool execution. It runs the tool only if `on_check_permission` returned `ALLOW` for the same
   agent, call ID, tool name and raw input; each approval is used once. This catches a middleware registered
   before this one that returns `ALLOW` without calling the rest of the chain.

AgentScope hands middlewares a copy of the call, and a middleware can pass a changed copy down the chain.
Register `ACGSMiddleware` first in `middlewares=[...]` so it sees the call exactly as AgentScope parsed it.

## Review

A call that matches only review rules gets `ASK` with `bypass_immune=True`:

- The reply pauses with a `RequireUserConfirmEvent`. Answer it with a `UserConfirmResultEvent`, as for any
  AgentScope tool that needs confirmation. The call runs only if the user confirms it.
- `bypass_immune=True` means AgentScope does not merge it into a batch decision; every such call is asked
  about. The middleware also does not suggest permission rules, and it asks again for the next review call even
  if the user added an allow rule.
- When the user confirms, AgentScope runs the permission check again with the (possibly edited) call. ACGS
  evaluates it again: a confirmed call that now matches a blocking rule is still denied.
- In `PermissionMode.DONT_ASK` nobody can answer, so review calls are denied. `PermissionMode.BYPASS` does not
  skip ACGS review or blocks.

```python
from agentscope.event import ConfirmResult, RequireUserConfirmEvent, UserConfirmResultEvent
from agentscope.message import UserMsg


async def run_with_review(agent, text):
    pending = []
    async for event in agent.reply_stream(UserMsg(name="user", content=text)):
        if isinstance(event, RequireUserConfirmEvent):
            pending.extend(event.tool_calls)
    # Show each pending call to a person; confirmed=False denies it.
    answers = [ConfirmResult(confirmed=True, tool_call=call) for call in pending]
    return await agent.reply(UserConfirmResultEvent(reply_id=agent.state.reply_id, confirm_results=answers))
```

`on_review="deny"` denies every call that needs review without asking.

## What is covered

| Tool | Checked |
|---|---|
| `FunctionTool`, `ToolBase` subclasses, AgentScope's built-in tools, MCP tools | Yes |
| Several tool calls in one model turn | Yes, each call |
| External tools (`is_external_tool`, executed by your code) | The permission check only; `on_acting` does not run for them |
| AgentScope's internal `CompressContext` tool | Yes (it has no arguments, so the default rules allow it) |
| Tools of other agents (teams, A2A) | Only if those agents have the middleware too |
| `toolkit.call_tool(...)` called directly, outside an agent | No |

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from agentscope_acgs import ACGSMiddleware

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
middleware = ACGSMiddleware(constitution=constitution, agent_id="payments-app")
```

`Constitution.from_yaml("rules.yaml")` works as well.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"agentscope"` | Agent identifier recorded by ACGS |
| `on_review` | `"ask"` | `"ask"` returns `ASK` (user confirmation) for calls that need review; `"deny"` denies them |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |

## Data handling

- Checks run in the same process. The middleware makes no network calls and needs no account or API key.
- The text that rules are checked against contains the tool name and argument values. In its default audit mode,
  acgs-lite 3.0.1 keeps only a count of checks and stores neither that text nor `agent_id`
  (`tests/test_middleware.py::test_default_engine_keeps_no_record_of_checked_calls` checks this). An
  `acgs_lite.AuditLog` in full audit mode would store the checked text, but the middleware does not create one.
- Denial and review reasons name the matched rules and their text, never argument values. The middleware logs the
  tool name and rule IDs (`INFO`), and logs a traceback when a check fails (`WARNING`).
- To check a call at execution time, the middleware keeps the call ID, tool name and raw input of each allowed
  call in memory until the tool starts (at most 4096 entries; the oldest are dropped). Nothing is written to disk.
- A review is an AgentScope user confirmation, so AgentScope keeps the pending call, including its arguments, in
  the agent state like any other confirmation.

## Limitations

- Only tool calls are checked. User messages, model requests and responses, and tool results are not.
- A middleware registered before `ACGSMiddleware` that passes a changed `tool_input` down the chain changes what
  ACGS evaluates; register `ACGSMiddleware` first.
- Tested with a real `Agent`, `Toolkit` and `FunctionTool`s, a scripted offline model, the user-confirmation
  pause and resume (confirm, reject, edit), and the `DEFAULT`, `BYPASS` and `DONT_ASK` permission modes, against
  agentscope 2.0.6 and the latest 2.0.x release. Real providers, MCP servers, teams, A2A and the AgentScope
  app/server are not tested.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied under
  the default constitution whenever it is called.

## Development

```bash
uv sync --group dev        # or use hatch: hatch run prepare
hatch run format           # ruff format + ruff check --fix
hatch run lint             # ruff check + ruff format --check
hatch run typecheck        # mypy --strict
hatch run cov              # pytest with a 100% branch-coverage gate, network disabled
hatch run min:cov          # the same tests against agentscope 2.0.6 and acgs-lite 3.0.1
pre-commit install         # format, lint, type check and tests on commit; commitizen on commit messages
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/); `cz bump` updates the version
and the changelog.

## License

Apache-2.0. See [LICENSE](LICENSE).
