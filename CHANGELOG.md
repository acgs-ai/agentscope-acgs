# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.2.0 (unreleased)

### Added

- Offline `examples/deny_and_audit.py`: one allowed and one denied AgentScope tool call, with an application
  decision trail.
- Regression coverage for nested JSON-native arguments and bytes/bytearray formatting against acgs-lite 4.0.0.

### Changed

- Runtime support now allows `acgs-lite>=3.0.1,<5`. The locked environment resolves acgs-lite 4.0.0; the CI
  `test-min` job keeps testing the 3.0.1 floor, and default compatibility behavior is unchanged.
- Document the compatibility matrix and acgs-lite 4.x production signing and JSON-only payload requirements.

## 0.1.0 (2026-10-28)

### Added

- `ACGSMiddleware`: an AgentScope 2.x agent middleware. Its `on_check_permission` hook checks each tool call's
  name and validated input against an ACGS constitution and returns `DENY` for blocking rules, `ASK` (AgentScope
  user confirmation, `bypass_immune=True`) for review-only matches, and AgentScope's own decision otherwise. Any
  error denies the call.
- `on_review="deny"` denies review calls without asking; review calls are also denied in `PermissionMode.DONT_ASK`.
- An execution check in `on_acting`: a tool runs only if the middleware allowed that exact call (call ID, tool
  name and raw input); anything else gets a `denied` tool result.
- `ACGSMiddleware.evaluate`; the `tool_call_formatter=` option.
