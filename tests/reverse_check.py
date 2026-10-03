#!/usr/bin/env python3
"""Reverse verification (mutation check) for the reboot plugin.

Each entry below reverts ONE design decision of the 2.0 rewrite and asserts that a
specific test turns red. A green suite on mutated code would mean the test does not
actually guard the decision it claims to guard.

Usage:
    KIRA_CORE_PATH=/path/to/KiraAI python3 tests/reverse_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (id, file, edits, substring of the test that must fail)
# `edits` is either a single (old, new) pair or a list of them, all applied to the
# same file in order.
MUTATIONS = [
    (
        "clear-mode-default",
        "schema.json",
        '"default": "memory_only"',
        '"default": "delete_session"',
        "test_reset_keeps_title_description_timestamp_and_capabilities",
    ),
    (
        "summary-writes-a-message-instead-of-a-chunk",
        "main.py",
        "_write_memory(sid, [build_summary_chunk(final)])",
        "_write_memory(sid, build_summary_chunk(final))",
        "test_sync_summary_is_written_to_the_memory_head",
    ),
    (
        "summary-marker-drifts-from-ads",
        "summarizer.py",
        'SUMMARY_MARKER = "[前情摘要|系统注入]"',
        'SUMMARY_MARKER = "[上下文重置摘要|系统注入]"',
        "test_summary_marker_is_byte_identical_to_ads",
    ),
    (
        "resume-keyword-collides-with-ads",
        "schema.json",
        '"default": ["/resume"]',
        '"default": ["/resum"]',
        "test_default_keywords_do_not_collide",
    ),
    (
        "llm-tool-gate-removed",
        "main.py",
        "                req.tool_set.remove(TOOL_NAME)",
        "                pass",
        "test_tool_hidden_from_the_model_when_disabled",
    ),
    (
        "async-write-gains-a-suspension-point",
        "main.py",
        "        chunks = sm.read_memory(sid) or []\n        chunks = [list(chunk) for chunk in chunks]",
        "        chunks = sm.read_memory(sid) or []\n        await asyncio.sleep(0)  # MUTATION\n        chunks = [list(chunk) for chunk in chunks]",
        "test_async_summary_does_not_lose_a_turn_written_meanwhile",
    ),
    (
        "summary-turns-on-by-default",
        "schema.json",
        '"name": "重开时生成摘要",\n        "default": false',
        '"name": "重开时生成摘要",\n        "default": true',
        "test_reboot_keeps_summary_off_when_the_switch_is_off",
    ),
    (
        "whitelist-fails-open",
        "main.py",
        "        if not self.allowed_users:\n            return False",
        "        if not self.allowed_users:\n            return True",
        "test_permission_on_with_empty_whitelist_denies_everyone",
    ),
    (
        "buffer-dropping-disabled",
        "main.py",
        "        if self.drop_pending_buffer:\n            self._drop_buffer(sid)",
        "        if False:\n            self._drop_buffer(sid)  # MUTATION",
        "test_pending_buffer_is_dropped_before_clearing",
    ),
    (
        "buffer-dropped-too-early",
        "main.py",
        [
            # drop the buffer before the awaited summary call ...
            (
                "        want_summary = bool(force_summary or self.enable_summary)",
                "        if self.drop_pending_buffer:\n            self._drop_buffer(sid)  # MUTATION\n"
                "        want_summary = bool(force_summary or self.enable_summary)",
            ),
            # ... and not atomically with the write
            (
                "        if self.drop_pending_buffer:\n            self._drop_buffer(sid)\n"
                "        if self._event_bus is not None:",
                "        if False:\n            self._drop_buffer(sid)  # MUTATION\n"
                "        if self._event_bus is not None:",
            ),
        ],
        "test_buffer_is_dropped_at_write_time_not_before_the_summary_call",
    ),
    (
        "store-cap-removed",
        "summarizer.py",
        ("        self._ensure_loaded()\n        self.prune()\n        try:",
         "        self._ensure_loaded()\n        try:"),
        "test_store_is_capped_and_evicts_the_oldest",
    ),
    (
        "short-plugin-id-false-positive",
        "compat.py",
        ("matched = target in norm or (len(norm) >= 4 and norm in target)",
         "matched = target in norm or norm in target"),
        "test_short_plugin_ids_do_not_produce_bogus_conflict_reports",
    ),
    (
        "notice-guard-removed",
        "main.py",
        ("        if getattr(event, \"is_notice\", False):", "        if False:"),
        "test_notice_messages_cannot_trigger_a_reset",
    ),
    (
        "config-hint-removed",
        "schema.json",
        ('"name": "\u91cd\u5f00\u65f6\u751f\u6210\u6458\u8981",\n        "default": false,\n        "hint":',
         '"name": "\u91cd\u5f00\u65f6\u751f\u6210\u6458\u8981",\n        "default": false,\n        "hint_disabled_for_test":'),
        "test_every_config_item_has_a_hint",
    ),
    (
        "legacy-keyword-not-migrated",
        "main.py",
        '                cmd["reboot_commands"] = legacy_commands\n',
        '                pass  # MUTATION\n',
        "test_legacy_command_prefix_is_migrated",
    ),
    (
        "legacy-command-not-re-enabled-on-upgrade",
        "main.py",
        '            if not cmd.get("enable_reboot_command", False):\n'
        '                cmd["enable_reboot_command"] = True',
        '            if not cmd.get("enable_reboot_command", False):\n'
        '                cmd["enable_reboot_command"] = False  # MUTATION',
        "test_legacy_command_prefix_is_migrated",
    ),
    (
        "fresh-install-command-auto-enabled",
        "main.py",
        "        if not self.enable_reboot_command and legacy_commands:",
        "        if not self.enable_reboot_command:  # MUTATION",
        "test_a_fresh_install_keeps_the_command_off",
    ),
    (
        "runtime-legacy-fallback-removed",
        "main.py",
        "        if not self.enable_reboot_command and legacy_commands:",
        "        if False:  # MUTATION",
        "test_legacy_values_are_honoured_without_a_writable_config",
    ),
    (
        "migration-not-persisted",
        "main.py",
        "                persisted = self._persist_config(cfg)",
        "                persisted = False  # MUTATION",
        "test_migration_is_written_back_to_the_config_file",
    ),
    (
        "legacy-keys-not-deleted",
        "main.py",
        '        legacy_commands = _split_commands(cfg.pop("command_prefix"))',
        '        legacy_commands = _split_commands(cfg.get("command_prefix"))  # MUTATION',
        "test_migration_is_idempotent",
    ),
    (
        "folded-section-not-removed",
        "main.py",
        "        node = cfg.pop(legacy_section, None)",
        "        node = cfg.get(legacy_section)  # MUTATION",
        "test_a_short_lived_debug_section_is_folded_into_compat",
    ),
]


def run_suite(root: Path, core_path: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["KIRA_CORE_PATH"] = core_path
    env["REBOOT_PLUGIN_ROOT"] = str(root)
    proc = subprocess.run(
        [sys.executable, str(HERE / "run_tests.py")],
        cwd=str(HERE),
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    core_path = os.environ.get("KIRA_CORE_PATH")
    if not core_path:
        sys.path.insert(0, str(HERE))
        import harness

        core_path = harness._detect_core_path()

    ok, failures = 0, []
    for entry in MUTATIONS:
        if len(entry) == 5:
            name, filename, old, new, expected = entry
            edits = [(old, new)]
        else:
            name, filename, edits, expected = entry
            if isinstance(edits, tuple):
                edits = [edits]
        tmp = Path(tempfile.mkdtemp(prefix=f"mut_{name}_"))
        root = tmp / "reboot_plugin"
        try:
            shutil.copytree(REPO, root, ignore=shutil.ignore_patterns(".git", "__pycache__", "tests"))
            shutil.copytree(REPO / "tests", root / "tests",
                            ignore=shutil.ignore_patterns("__pycache__"))
            target = root / filename
            text = target.read_text(encoding="utf-8")
            missing = [old for old, _ in edits if old not in text]
            if missing:
                failures.append((name, "mutation snippet not found"))
                print(f"SKIP  {name}: snippet not found in {filename}")
                continue
            for old, new in edits:
                text = text.replace(old, new, 1)
            target.write_text(text, encoding="utf-8")

            code, output = run_suite(root, core_path)
            if code == 0:
                failures.append((name, "suite stayed GREEN on mutated code"))
                print(f"BAD   {name}: suite stayed green - the test does not guard this")
            elif expected not in output:
                failures.append((name, f"{expected} did not fail"))
                print(f"BAD   {name}: expected {expected} to fail, it did not")
            else:
                ok += 1
                print(f"ok    {name} -> {expected} went red as expected")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 68)
    print(f"mutations verified: {ok}/{len(MUTATIONS)}")
    for name, reason in failures:
        print(f"  FAILED {name}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
