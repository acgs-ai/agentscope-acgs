# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The README only shows code that runs, and the quickstart output it shows is the real output."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from agentscope.agent import Agent

from agentscope_acgs import ACGSMiddleware
from tests.fakes import REFUND, Recorder, ScriptedModel, answer, ask, confirmations, review_rules, text, tool_call

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUICKSTART = ROOT / "examples" / "quickstart.py"


def _blocks(language: str) -> list[str]:
    return [textwrap.dedent(block) for block in re.findall(rf"```{language}\n(.*?)```", README, re.DOTALL)]


def test_quickstart_output_matches_readme() -> None:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error", str(QUICKSTART)],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        timeout=300,
    )
    [expected] = [block for block in _blocks("text") if block.startswith("review:")]
    assert completed.stdout == expected
    assert completed.stderr == ""


def test_readme_embeds_the_quickstart_verbatim() -> None:
    assert QUICKSTART.read_text(encoding="utf-8") in _blocks("python")


def test_readme_states_the_supported_versions() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("agentscope", "acgs-lite"):
        [requirement] = re.findall(rf'"({package}>=[^"]+)"', pyproject)
        assert f"`{requirement}`" in README
        [floor] = re.findall(rf'"{package}==([^"]+)"', pyproject)
        assert requirement.startswith(f"{package}>={floor},")
    assert README.count("agentscope 2.0.6") == 2
    assert "Requires Python 3.11 to 3.14" in README
    assert 'requires-python = ">=3.11,<3.15"' in pyproject


async def test_every_readme_python_block_runs(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks("python")
    assert len(blocks) == 5
    namespaces: list[dict[str, Any]] = []
    for block in blocks:
        namespace: dict[str, Any] = {
            "__name__": "readme",
            "model": ScriptedModel([text("Done.")]),
            "toolkit": Recorder().toolkit(),
        }
        exec(compile(block, "<README>", "exec", dont_inherit=True), namespace)  # noqa: S102
        namespaces.append(namespace)
    assert "normalized: update config | dry run" in capsys.readouterr().out
    usage = namespaces[0]["agent"]
    assert isinstance(usage, Agent)
    assert [type(m) for m in usage._check_permission_middlewares] == [ACGSMiddleware]
    assert [type(m) for m in usage._acting_middlewares] == [ACGSMiddleware]
    assert namespaces[-1]["middleware"].agent_id == "payments-app"

    # The review helper from the README confirms the pending refund, which then runs.
    recorder = Recorder()
    agent = Agent(
        name="support",
        system_prompt="test",
        model=ScriptedModel([tool_call("issue_refund", REFUND), text("Refunded.")]),
        toolkit=recorder.toolkit(),
        middlewares=[ACGSMiddleware(review_rules())],
    )
    reply = await namespaces[3]["run_with_review"](agent, "refund A-17")
    assert recorder.ran == [("issue_refund", REFUND)]
    assert reply.get_text_content() == "Refunded."


async def test_helpers_used_by_the_readme_review_flow() -> None:
    recorder = Recorder()
    agent = Agent(
        name="support",
        system_prompt="test",
        model=ScriptedModel([tool_call("issue_refund", REFUND), text("Refunded.")]),
        toolkit=recorder.toolkit(),
        middlewares=[ACGSMiddleware(review_rules())],
    )
    assert len(confirmations(await ask(agent))) == 1
    await answer(agent, confirmed=True)
    assert recorder.ran == [("issue_refund", REFUND)]
