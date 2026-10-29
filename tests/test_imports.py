# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
import agentscope_acgs


def test_public_api() -> None:
    assert sorted(agentscope_acgs.__all__) == sorted(
        ["ACGSMiddleware", "GovernanceDecision", "ToolCallFormatter", "Verdict", "format_tool_call"]
    )
    for name in agentscope_acgs.__all__:
        assert getattr(agentscope_acgs, name) is not None
