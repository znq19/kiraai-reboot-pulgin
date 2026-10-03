from __future__ import annotations

"""
Context reset (reboot) plugin.

Two independent entry points:

* commands - ``/reboot`` (clear the context) and ``/resume`` (clear + force a
  cumulative summary). Both are off by default.
* LLM tool - ``reset_context``, gated by ``allow_llm_reboot``.

Clearing never deletes the session. ``SessionManager.delete_session()`` pops the
whole session entry, which also drops the title, the description and the
per-session capability overrides; ``write_memory()`` only replaces the ``memory``
field, so all of that survives.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.plugin import BasePlugin, logger, on, Priority, register
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.chat import MessageChain
from core.chat.message_elements import Text
from core.agent.message import OpenAIMessage
from core.provider import LLMRequest

from .compat import (
    find_ads_plugin,
    is_plugin_active,
    read_ads_reset_commands,
    report_memory_plugins,
    verify_ads_marker,
    yield_overlapping_commands,
)
from .summarizer import (
    DEFAULT_SUMMARIZE_PROMPT,
    SUMMARY_MARKER,
    CumulativeSummaryStore,
    _safe_format,
    build_summary_chunk,
    extract_summary_text,
    is_summary_chunk,
    merge_summaries,
    self_compress_summary,
    summarize_history,
)

TOOL_NAME = "reset_context"

# A reset issued by the LLM tool is ignored when another one happened this recently
# (a model can emit several tool calls in a single step). Explicit commands are
# never rate limited - the user asked for them.
TOOL_COOLDOWN_SECONDS = 5.0

DEFAULT_SUCCESS_MESSAGE = "✅ 已重置本会话上下文{summary}，我们可以重新开始对话了！"
DEFAULT_DENIED_MESSAGE = "❌ 权限不足：您没有重置上下文的权限"
DEFAULT_ERROR_MESSAGE = "❌ 重置上下文失败: {error}"
DEFAULT_EMPTY_MESSAGE = "📭 当前会话没有可重置的历史"


def _msg_get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _as_command_list(raw: Any, fallback: List[str]) -> List[str]:
    """Normalize a command list config value (list, or comma separated string)."""
    if isinstance(raw, str):
        raw = raw.split(",")
    if isinstance(raw, (list, tuple, set)):
        out: List[str] = []
        for item in raw:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        if out:
            return out
    return list(fallback)


def _as_str_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        raw = raw.replace("\n", ",").split(",")
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _slice(value: Any, default: Any) -> Any:
    """0 is a legal value for the char limits, so ``or default`` must not be used."""
    return default if value is None else value


def _as_int(value: Any, default: int) -> int:
    try:
        return int(_slice(value, default))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(_slice(value, default))
    except (TypeError, ValueError):
        return default


def _legacy(new_value: Any, default: Any, legacy_value: Any) -> Any:
    """Honor a pre-2.0 flat setting only while the new key still holds its default.

    The framework merges schema defaults into the config file before the plugin is
    constructed, so "new key missing" can never be used to detect an upgrade. The
    old keys are still present though, so: if the user has not touched the new key
    (it still equals the default) and an old key exists, take the old value.
    """
    if legacy_value is None:
        return new_value
    if new_value == default:
        return legacy_value
    return new_value


# ── configuration migration (1.0/1.1 -> 2.x) ─────────────────────────────────
#
# The host framework never migrates plugin configuration.
# ``PluginManager._ensure_plugin_config()`` reads the existing file, adds the
# schema default for every key that is *missing*, and writes the file back. Keys
# that already exist are left alone, and keys that disappeared from the schema are
# never mentioned again. So when 2.0 replaced the flat 1.x layout
# (``command_prefix``, ``allowed_users``, ...) with prefixed section keys, every
# upgraded installation ended up with:
#
#   * the user's real values still on disk, under the now-unknown legacy keys;
#   * the *new* keys filled with schema defaults by the framework;
#   * the WebUI rendering those defaults, because it is driven by the schema.
#
# That is what an upgrade looked like as "all my settings are gone" - and because
# 1.x had no command switch at all (a configured prefix was always live), the
# command additionally stopped firing. The fix is a one-shot *persisted* migration:
# move each legacy value into its new key and delete the legacy key. Deleting is
# what makes it idempotent, and the presence of a legacy key is exactly the
# "this installation came from 1.x" marker - so a fresh 2.x install, which never
# has those keys, is never touched (in particular ``enable_reboot_command`` stays
# at its documented default of ``false`` for new users).

LEGACY_FLAT_KEYS = (
    "command_prefix",
    "enable_permission",
    "allowed_users",
    "success_message",
    "permission_denied_message",
    "error_message",
    "verbose_log",
)

# Sections that existed in an earlier 2.x build and have since been folded into
# another one. Kept as an explicit list so "does this config need work?" stays a
# single question with a single answer, and so the test suite can freeze it.
LEGACY_SECTIONS = ("section_debug",)

# Legacy key -> the new setting it feeds. Doubles as the machine readable
# migration map used by the test suite to prove nothing is lost silently.
MIGRATION_TARGETS = {
    "command_prefix": "section_command.reboot_commands",
    "enable_permission": "section_command.reboot_enable_permission",
    "allowed_users": "section_command.reboot_allowed_users",
    "success_message": "section_command.reboot_success_message",
    "permission_denied_message": "section_command.reboot_permission_denied_message",
    "error_message": "section_command.reboot_error_message",
    "verbose_log": "section_compat.verbose_log",
}


def _split_commands(raw: Any) -> List[str]:
    """Split a legacy command value (string / list) into keywords.

    Blank input yields ``[]``: 1.x compared the message against the configured
    prefix, so an empty prefix meant the command never matched. Migrating it into
    a bogus ``/reboot`` would resurrect a command the user had effectively turned
    off.
    """
    if isinstance(raw, str):
        raw = raw.replace("\n", ",").split(",")
    elif isinstance(raw, (list, tuple, set)):
        raw = list(raw)
    else:
        return []
    out: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def migrate_config(cfg: dict) -> List[str]:
    """Move 1.0/1.1 flat settings into their 2.x keys, in place.

    Returns a human readable list of the changes made (empty when there is nothing
    to migrate). A config with no legacy key and no folded section is left
    completely untouched.
    """
    if not isinstance(cfg, dict):
        return []
    if not any(key in cfg for key in LEGACY_FLAT_KEYS) and not any(
        isinstance(cfg.get(name), dict) for name in LEGACY_SECTIONS
    ):
        return []

    changed: List[str] = []
    cmd = cfg.get("section_command")
    if not isinstance(cmd, dict):
        cmd = {}
        cfg["section_command"] = cmd
    compat = cfg.get("section_compat")
    if not isinstance(compat, dict):
        compat = {}
        cfg["section_compat"] = compat

    # --- command_prefix -> reboot_commands ---------------------------------
    # 1.x had no enable switch: a configured prefix was always live, so a migrated
    # user keeps a working command. This branch can only run when the legacy key
    # exists, so a fresh install never gets the command switched on.
    if "command_prefix" in cfg:
        legacy_commands = _split_commands(cfg.pop("command_prefix"))
        if legacy_commands:
            if _as_command_list(cmd.get("reboot_commands"), ["/reboot"]) == ["/reboot"]:
                cmd["reboot_commands"] = legacy_commands
                changed.append("command_prefix -> section_command.reboot_commands")
            if not cmd.get("enable_reboot_command", False):
                cmd["enable_reboot_command"] = True
                changed.append("section_command.enable_reboot_command = true (1.x had no switch)")

    # --- permission ---------------------------------------------------------
    if "enable_permission" in cfg:
        raw = cfg.pop("enable_permission")
        if not bool(cmd.get("reboot_enable_permission", False)) and bool(raw):
            cmd["reboot_enable_permission"] = True
            changed.append("enable_permission -> section_command.reboot_enable_permission")

    if "allowed_users" in cfg:
        raw = cfg.pop("allowed_users")
        if not _as_str_list(cmd.get("reboot_allowed_users")):
            value = _as_str_list(raw)
            if value:
                cmd["reboot_allowed_users"] = value
                changed.append("allowed_users -> section_command.reboot_allowed_users")

    # --- message templates --------------------------------------------------
    for legacy_key, new_key, default in (
        ("success_message", "reboot_success_message", DEFAULT_SUCCESS_MESSAGE),
        ("permission_denied_message", "reboot_permission_denied_message", DEFAULT_DENIED_MESSAGE),
        ("error_message", "reboot_error_message", DEFAULT_ERROR_MESSAGE),
    ):
        if legacy_key not in cfg:
            continue
        raw = cfg.pop(legacy_key)
        current = str(cmd.get(new_key) or default)
        value = "" if raw is None else str(raw)
        if current == default and value and value != default:
            cmd[new_key] = value
            changed.append(f"{legacy_key} -> section_command.{new_key}")

    # --- verbose_log --------------------------------------------------------
    # 1.x kept it as a flat key; a pre-release 2.1 build kept it in its own
    # "调试" section before the two sections were merged into "兼容与调试".
    if "verbose_log" in cfg:
        raw = cfg.pop("verbose_log")
        if not bool(compat.get("verbose_log", False)) and bool(raw):
            compat["verbose_log"] = True
            changed.append("verbose_log -> section_compat.verbose_log")

    for legacy_section in LEGACY_SECTIONS:
        node = cfg.pop(legacy_section, None)
        if not isinstance(node, dict):
            continue
        if node.get("verbose_log") and not bool(compat.get("verbose_log", False)):
            compat["verbose_log"] = True
            changed.append(f"{legacy_section}.verbose_log -> section_compat.verbose_log")
        else:
            # still report it: the removal itself must be persisted, otherwise the
            # dead section lingers on disk forever
            changed.append(f"{legacy_section} removed (folded into section_compat)")

    return changed


class RebootPlugin(BasePlugin):
    """Reset the current session context, optionally carrying a summary over."""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        # 1.x -> 2.x: move the old flat settings into their new keys and persist.
        # ``cfg`` is the dict the plugin manager keeps in ``plugin_configs`` (it
        # hands the object over rather than a copy), so editing it in place also
        # refreshes what the WebUI reads back on its next GET - which is the whole
        # point: the settings page must show the values the plugin actually uses.
        self._migration: List[str] = []
        try:
            self._migration = migrate_config(cfg)
            if self._migration:
                persisted = self._persist_config(cfg)
                self._write_migration_audit(self._migration, persisted)
                logger.info(
                    "[reboot] 已迁移 %d 项 1.x 配置：%s"
                    % (len(self._migration), "；".join(self._migration))
                )
                if not persisted:
                    logger.warning(
                        "[reboot] 迁移结果未能写回配置文件，本次运行仍然正确；"
                        "请检查 data/config/plugins 是否可写"
                    )
        except Exception as e:
            logger.warning(f"[reboot] configuration migration skipped: {e}")
        self._read_config(cfg or {})

        self.session_mgr = None
        self._store: Optional[CumulativeSummaryStore] = None
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_tool_reset: Dict[str, float] = {}
        self._async_tasks: Dict[str, asyncio.Task] = {}
        # sid -> [memory_fingerprint, last_seen_monotonic]: identifies memory_written
        # events that WE produced. Those must not be mistaken for an external clear
        # (which would forget the store and cancel the summary task we are about to
        # schedule). Matching is by content, not by count: a pending event from an
        # earlier write can arrive after ours and would otherwise consume the marker.
        self._own_writes: Dict[str, list] = {}
        self._event_bus = None
        self._command_map: Dict[str, bool] = {}
        self._ads_plugin_id: Optional[str] = None
        self._ads_marker_ok = True

    # ── configuration ────────────────────────────────────────────────────────

    def _read_config(self, cfg: dict) -> None:
        reset = cfg.get("section_reset") or {}
        cmd = cfg.get("section_command") or {}
        tool = cfg.get("section_llm_tool") or {}
        summ = cfg.get("section_summary") or {}
        compat = cfg.get("section_compat") or {}
        debug = cfg.get("section_debug") or {}

        # --- legacy flat keys (pre-2.0 layout) ---
        # migrate_config() normally removes these before _read_config runs; they are
        # still honoured here so a config that could not be written back (read-only
        # data dir, unmappable plugin id) behaves correctly for the current run
        # instead of silently falling back to defaults.
        legacy_prefix = cfg.get("command_prefix")
        legacy_permission = cfg.get("enable_permission")
        legacy_allowed = cfg.get("allowed_users")
        legacy_success = cfg.get("success_message")
        legacy_denied = cfg.get("permission_denied_message")
        legacy_error = cfg.get("error_message")
        legacy_commands = _split_commands(legacy_prefix)
        # verbose_log moved twice: 1.x flat key -> the short lived "调试" section of
        # an unreleased 2.1 build -> 兼容与调试. Honour both.
        legacy_verbose = cfg.get("verbose_log")
        if legacy_verbose is None and isinstance(debug, dict):
            legacy_verbose = debug.get("verbose_log")

        self.verbose_log = bool(
            _legacy(bool(compat.get("verbose_log", False)), False, legacy_verbose)
        )

        # --- section_reset ---
        clear_mode = str(reset.get("clear_mode", "memory_only") or "memory_only").lower()
        self.clear_mode = clear_mode if clear_mode in ("memory_only", "delete_session") else "memory_only"
        self.drop_pending_buffer = bool(reset.get("drop_pending_buffer", True))

        # --- section_command ---
        self.enable_reboot_command = bool(cmd.get("enable_reboot_command", False))
        if not self.enable_reboot_command and legacy_commands:
            # 1.x had no switch at all: a configured prefix was always live. This
            # can only trigger while a legacy key is still present.
            self.enable_reboot_command = True
        reboot_commands = _as_command_list(cmd.get("reboot_commands"), ["/reboot"])
        if legacy_commands:
            reboot_commands = _legacy(reboot_commands, ["/reboot"], legacy_commands)
        self.reboot_commands = reboot_commands

        self.enable_resum_command = bool(cmd.get("enable_resum_command", False))
        self.resum_commands = _as_command_list(cmd.get("resum_commands"), ["/resume"])

        self.enable_permission = bool(
            _legacy(
                bool(cmd.get("reboot_enable_permission", False)),
                False,
                None if legacy_permission is None else bool(legacy_permission),
            )
        )
        allowed = _as_str_list(cmd.get("reboot_allowed_users") or [])
        if not allowed and legacy_allowed:
            allowed = _as_str_list(legacy_allowed)
        self.allowed_users = allowed

        self.success_message = str(
            _legacy(
                str(cmd.get("reboot_success_message") or DEFAULT_SUCCESS_MESSAGE),
                DEFAULT_SUCCESS_MESSAGE,
                legacy_success,
            )
            or DEFAULT_SUCCESS_MESSAGE
        )
        self.permission_denied_message = str(
            _legacy(
                str(cmd.get("reboot_permission_denied_message") or DEFAULT_DENIED_MESSAGE),
                DEFAULT_DENIED_MESSAGE,
                legacy_denied,
            )
            or DEFAULT_DENIED_MESSAGE
        )
        self.error_message = str(
            _legacy(
                str(cmd.get("reboot_error_message") or DEFAULT_ERROR_MESSAGE),
                DEFAULT_ERROR_MESSAGE,
                legacy_error,
            )
            or DEFAULT_ERROR_MESSAGE
        )
        self.empty_message = str(
            cmd.get("reboot_empty_message") or DEFAULT_EMPTY_MESSAGE
        ) or DEFAULT_EMPTY_MESSAGE

        # --- section_llm_tool ---
        self.allow_llm_reboot = bool(tool.get("allow_llm_reboot", True))

        # --- section_summary ---
        self.enable_summary = bool(summ.get("enable_summary", False))
        mode = str(summ.get("summarize_mode", "async") or "async").lower()
        self.summarize_mode = mode if mode in ("async", "sync") else "async"
        self.summarize_model = str(summ.get("summarize_model", "") or "")
        self.summarize_timeout_sec = max(
            0.5, _as_float(summ.get("summarize_timeout_sec"), 60.0)
        )
        self.summarize_max_input_chars = _as_int(summ.get("summarize_max_input_chars"), 10000)
        self.summarize_max_output_chars = _as_int(summ.get("summarize_max_output_chars"), 5000)
        self.summarize_prompt_template = (
            str(summ.get("summarize_prompt_template", "") or "") or DEFAULT_SUMMARIZE_PROMPT
        )
        self.cumulative_summary = bool(summ.get("cumulative_summary", True))
        self.merge_prompt_template = str(summ.get("merge_prompt_template", "") or "")
        self.self_compress_prompt_template = str(
            summ.get("self_compress_prompt_template", "") or ""
        )
        self.merge_timeout_sec = max(1.0, _as_float(summ.get("merge_timeout_sec"), 120.0))
        self.keep_summary_alive = bool(summ.get("keep_summary_alive", True))
        self.enable_summary_logging = bool(summ.get("enable_summary_logging", False))

        # --- section_compat ---
        self.ads_interop = bool(compat.get("ads_interop", True))
        self.warn_conflict = bool(compat.get("warn_conflict", True))

    # ── migration persistence ────────────────────────────────────────────────

    def _config_file(self) -> Optional[Path]:
        """Locate this plugin's config file, or None when it cannot be resolved.

        Three fallbacks, because failing to find the file must never be fatal:
        the plugin manager (authoritative), then our own manifest, then give up.
        """
        plugin_id = None
        plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
        if plugin_mgr is not None:
            try:
                plugin_id = plugin_mgr.get_plugin_id_for_module(__name__)
            except Exception:
                plugin_id = None
        if not plugin_id:
            try:
                manifest = json.loads(
                    (Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8")
                )
                plugin_id = str(manifest.get("plugin_id") or "").strip() or None
            except Exception:
                plugin_id = None
        if not plugin_id:
            return None
        try:
            from core.utils.path_utils import get_config_path

            return Path(get_config_path()) / "plugins" / f"{plugin_id}.json"
        except Exception:
            return None

    def _persist_config(self, cfg: dict) -> bool:
        """Write the migrated config back so the settings page shows the truth.

        Deliberately does NOT go through ``plugin_mgr.update_plugin_config()``: that
        one re-initializes the plugin (``await init_plugin()``), which would re-enter
        ``__init__`` from inside ``__init__``. Writing the file ourselves is safe
        because the framework re-reads it on the next start, and the in-memory cache
        is already correct - ``cfg`` *is* the object in ``plugin_configs``.
        """
        path = self._config_file()
        if path is None:
            logger.warning("[reboot] 无法定位插件配置文件，本次迁移仅作用于内存")
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(cfg, indent=4, ensure_ascii=False)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)  # atomic: never leave a half-written config behind
            return True
        except Exception as e:
            logger.warning(f"[reboot] 配置文件写入失败（{e}），旧键保持原样，下次启动会重试")
            return False

    def _write_migration_audit(self, changed: List[str], persisted: bool) -> None:
        """Append one line to data/plugin_data/<id>/migration.log for later auditing."""
        try:
            data_dir = self.ctx.get_plugin_data_dir()
            if data_dir is None:
                return
            line = "%s | migrated=%d | persisted=%s | %s\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                len(changed),
                persisted,
                "; ".join(changed),
            )
            with (Path(data_dir) / "migration.log").open("a", encoding="utf-8") as handle:
                handle.write(line)
        except Exception:
            pass

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self):
        self.session_mgr = getattr(self.ctx, "session_mgr", None) or self._find_session_manager()
        if self.session_mgr is None:
            logger.error("[reboot] SessionManager unavailable, plugin disabled")
            return

        required = (
            "fetch_memory",
            "read_memory",
            "write_memory",
            "delete_session",
            "get_session_info",
            "get_memory_count",
        )
        missing = [name for name in required if not hasattr(self.session_mgr, name)]
        if missing:
            logger.error(f"[reboot] SessionManager is missing methods: {missing}")
            self.session_mgr = None
            return

        try:
            data_dir = self.ctx.get_plugin_data_dir()
            if data_dir is None:
                # PluginContext returns None when it cannot map the calling module to
                # a plugin id; the plugin still works, only the cumulative summary
                # store is unavailable.
                raise RuntimeError("plugin data directory is unavailable")
            self._store = CumulativeSummaryStore(Path(data_dir) / "cumulative_summaries.json")
            self._store.load()
        except Exception as e:
            logger.warning(f"[reboot] cumulative summary store unavailable, degrading: {e}")
            self._store = None

        self._setup_ads_interop()
        self._command_map = self._build_command_map()
        self._subscribe_session_events()

        logger.info(
            "[reboot] ready | clear_mode=%s | reboot_cmd=%s%s | resum_cmd=%s%s | "
            "llm_tool=%s | summary=%s/%s cumulative=%s | drop_buffer=%s%s"
            % (
                self.clear_mode,
                "on" if self.enable_reboot_command else "off",
                self.reboot_commands,
                "on" if self.enable_resum_command else "off",
                self.resum_commands,
                self.allow_llm_reboot,
                self.enable_summary,
                self.summarize_mode,
                self.cumulative_summary,
                self.drop_pending_buffer,
                "" if self._ads_marker_ok else " | ADS marker mismatch",
            )
        )

    async def terminate(self):
        bus = self._event_bus
        if bus is not None:
            try:
                bus.unsubscribe("session_memory_written", self._on_memory_written)
                bus.unsubscribe("session_deleted", self._on_session_deleted)
            except Exception:
                pass
            self._event_bus = None
        for task in list(self._async_tasks.values()):
            if task is not None and not task.done():
                task.cancel()
        self._async_tasks.clear()
        self._locks.clear()
        self._last_tool_reset.clear()
        self._own_writes.clear()
        self._command_map.clear()
        if self._store is not None:
            try:
                self._store.save()
            except Exception:
                pass
        logger.info("[reboot] terminated")

    def _find_session_manager(self):
        for name in ("session_mgr", "session_manager", "mem_mgr", "memory_manager"):
            obj = getattr(self.ctx, name, None)
            if obj is not None and hasattr(obj, "write_memory"):
                return obj
        return None

    def _setup_ads_interop(self) -> None:
        plugin_mgr = getattr(self.ctx, "plugin_mgr", None)
        if self.ads_interop:
            self._ads_plugin_id = find_ads_plugin(plugin_mgr)
            if self._ads_plugin_id and is_plugin_active(plugin_mgr, self._ads_plugin_id):
                self._ads_marker_ok = verify_ads_marker(
                    plugin_mgr, self._ads_plugin_id, SUMMARY_MARKER, logger
                )
                ads_commands = read_ads_reset_commands(plugin_mgr, self._ads_plugin_id)
                if ads_commands:
                    if self.enable_reboot_command:
                        self.reboot_commands = yield_overlapping_commands(
                            self.reboot_commands, ads_commands, logger
                        )
                    if self.enable_resum_command:
                        self.resum_commands = yield_overlapping_commands(
                            self.resum_commands, ads_commands, logger
                        )
        report_memory_plugins(plugin_mgr, self.warn_conflict, logger)

    def _build_command_map(self) -> Dict[str, bool]:
        mapping: Dict[str, bool] = {}
        for command in self.reboot_commands if self.enable_reboot_command else []:
            mapping[command.strip().lower()] = False
        for command in self.resum_commands if self.enable_resum_command else []:
            mapping[command.strip().lower()] = True  # resum wins on duplicates
        reboot_set = {c.strip().lower() for c in self.reboot_commands}
        resum_set = {c.strip().lower() for c in self.resum_commands}
        overlap = reboot_set & resum_set
        if overlap:
            logger.warning(
                f"[reboot] 关键词 {sorted(overlap)} 同时出现在两条指令中，将按 /resume（强制摘要）处理"
            )
        return mapping

    def _subscribe_session_events(self) -> None:
        bus = getattr(self.ctx, "event_bus", None)
        if bus is None:
            return
        try:
            bus.subscribe("session_memory_written", self._on_memory_written)
            bus.subscribe("session_deleted", self._on_session_deleted)
            self._event_bus = bus
        except Exception as e:
            logger.warning(f"[reboot] session event subscription failed: {e}")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self.verbose_log:
            logger.debug(f"[reboot] {message}")

    def _get_lock(self, sid: str) -> asyncio.Lock:
        lock = self._locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[sid] = lock
        return lock

    def _extract_text(self, event) -> str:
        try:
            return "".join(
                elem.text for elem in event.message.chain if isinstance(elem, Text)
            ).strip()
        except Exception:
            return ""

    def _user_id(self, event) -> str:
        try:
            msg = getattr(event, "message", None)
            if msg is None:
                messages = getattr(event, "messages", None) or []
                msg = messages[-1] if messages else None
            sender = getattr(msg, "sender", None) if msg is not None else None
            user_id = getattr(sender, "user_id", None) if sender is not None else None
            if user_id is not None:
                return str(user_id)
        except Exception:
            pass
        return "unknown"

    def _allowed(self, event) -> bool:
        """Permission check. Fail-closed: enabling permission with an empty
        allow-list denies everybody (an empty list must not mean "everyone")."""
        if not self.enable_permission:
            return True
        if not self.allowed_users:
            return False
        return self._user_id(event) in self.allowed_users

    async def _reply(self, sid: str, text: str) -> None:
        if not text:
            return
        try:
            await self.ctx.message_processor.send_message_chain(
                session=sid, chain=MessageChain([Text(text)])
            )
        except Exception:
            logger.exception(f"[reboot] failed to send reply to {sid}")

    def _summary_suffix(self, want_summary: bool, summary_text: Optional[str]) -> str:
        if summary_text:
            return "，已注入累计摘要"
        if not want_summary:
            return ""
        if self.summarize_mode == "async":
            return "，摘要将在后台生成后补写"
        return "（无可摘要历史或摘要生成失败）"

    def _drop_buffer(self, sid: str) -> None:
        """Discard messages that are buffered but not yet in the context.

        Without this, a debounce flush right after the reset would push them into
        the freshly cleared context.
        """
        try:
            buf = self.ctx.get_buffer(sid)
            count = buf.get_length()
            if count:
                buf.pop(count=count)
                self._log(f"dropped {count} buffered message(s) for {sid}")
        except Exception as e:
            self._log(f"buffer drop skipped: {e}")

    # ── commands ─────────────────────────────────────────────────────────────

    @on.im_message(priority=Priority.HIGH)
    async def handle_command(self, event: KiraMessageEvent, *_):
        if not self._command_map or self.session_mgr is None:
            return
        if getattr(event, "is_notice", False):
            # Notices are synthesized (cross-session sends, publish_notice). Text that
            # merely happens to equal a keyword must not be able to reset a session
            # from the outside.
            return
        text = self._extract_text(event)
        if not text:
            return
        force_summary = self._command_map.get(text.lower())
        if force_summary is None:
            return

        sid = event.session.sid
        # The command itself must neither reach the LLM nor enter the memory.
        event.discard(force=True)
        event.stop()

        user_id = self._user_id(event)
        self._log(f"command matched: {text!r} force_summary={force_summary} user={user_id} sid={sid}")

        if not self._allowed(event):
            await self._reply(sid, self.permission_denied_message)
            return

        want_summary = bool(force_summary or self.enable_summary)
        try:
            status, summary_text = await self._reset(
                sid, force_summary=force_summary, reason=f"command {text}"
            )
        except Exception as e:
            logger.exception(f"[reboot] reset failed sid={sid}")
            await self._reply(sid, _safe_format(self.error_message, {"error": str(e)}))
            return

        if status == "empty":
            await self._reply(sid, self.empty_message)
            return
        if status == "error":
            await self._reply(
                sid,
                _safe_format(self.error_message, {"error": summary_text or "internal error"}),
            )
            return
        await self._reply(
            sid,
            _safe_format(
                self.success_message, {"summary": self._summary_suffix(want_summary, summary_text)}
            ),
        )

    # ── LLM tool ─────────────────────────────────────────────────────────────

    @register.tool(
        name=TOOL_NAME,
        description=(
            "重置当前会话的上下文记忆。当用户想要重新开始对话、忘掉之前聊过的内容、"
            "清除记忆时调用。调用后本会话之前的聊天记录会被清空。"
        ),
        params={"type": "object", "properties": {}, "required": []},
    )
    async def reset_context(self, event: KiraMessageBatchEvent, **kwargs) -> str:
        if not self.allow_llm_reboot:
            return "管理员已关闭自动重置上下文"
        if self.session_mgr is None:
            return _safe_format(self.error_message, {"error": "session manager unavailable"})

        sid = getattr(event, "sid", None) or getattr(
            getattr(event, "session", None), "sid", None
        )
        if not sid:
            return _safe_format(self.error_message, {"error": "unknown session"})
        if not self._allowed(event):
            return self.permission_denied_message

        want_summary = bool(self.enable_summary)
        try:
            status, summary_text = await self._reset(
                sid, force_summary=False, reason="llm tool", apply_cooldown=True
            )
        except Exception as e:
            logger.exception(f"[reboot] tool reset failed sid={sid}")
            return _safe_format(self.error_message, {"error": str(e)})

        if status == "empty":
            return self.empty_message
        if status == "cooldown":
            return "刚刚已经重置过本会话上下文，无需重复操作"
        if status == "error":
            return _safe_format(self.error_message, {"error": summary_text or "internal error"})
        suffix = self._summary_suffix(want_summary, summary_text)
        return _safe_format(
            self.success_message, {"summary": suffix} if suffix else {"summary": ""}
        )

    @on.loaded()
    async def drop_llm_tool_if_disabled(self, *_):
        """Remove the tool from the global registry when the switch is off.

        Note: tools are registered *after* ``initialize()`` runs, so this cannot be
        done there. ON_LOADED is not replayed on a single-plugin hot reload either,
        which is why the request hook below is the authoritative gate.
        """
        if self.allow_llm_reboot:
            return
        try:
            self.ctx.tool_mgr.unregister_tool(TOOL_NAME)
            self._log("LLM tool unregistered (allow_llm_reboot=false)")
        except Exception as e:
            self._log(f"tool unregister skipped: {e}")

    # ── request hook: tool gate + summary bridge ──────────────────────────────

    @on.llm_request(priority=Priority.LOW)
    async def on_llm_request(self, event, req: LLMRequest, tag_set, *_):
        if not self.allow_llm_reboot:
            try:
                # Runs before the framework serialises the tool list
                # (message_manager.py: ON_LLM_REQUEST -> request.tool_set.to_list()).
                req.tool_set.remove(TOOL_NAME)
            except Exception:
                pass
        if self.session_mgr is None:
            return
        sid = getattr(event, "sid", None) or getattr(
            getattr(event, "session", None), "sid", None
        )
        if sid:
            self._bridge_summary(sid, req)

    def _bridge_summary(self, sid: str, req: LLMRequest) -> None:
        """Re-inject the cumulative summary for this request only.

        The summary lives in the first memory chunk, which the framework evicts
        first once the session reaches ``max_memory_length`` turns. When that
        happens the store still holds it; we prepend it to this request (not to
        the memory) so the session keeps its long-term context.
        """
        if not self.keep_summary_alive or not self.cumulative_summary or self._store is None:
            return
        try:
            messages = req.messages
            if not messages:
                return
            content = _msg_get(messages[0], "content")
            if isinstance(content, str) and content.startswith(SUMMARY_MARKER):
                return
            stored = (self._store.get(sid) or "").strip()
            if not stored:
                return
            req.messages[:0] = [OpenAIMessage(**build_summary_chunk(stored)[0])]
            self._log(f"summary head missing, bridged back for this request ({len(stored)} chars)")
        except Exception:
            pass

    # ── session lifecycle events ─────────────────────────────────────────────

    async def _on_memory_written(self, event) -> None:
        """Session memory written.

        Two very different cases share this event:
        * our own write (marker consumes it) - the reset flow manages the store
          explicitly, so nothing to do;
        * somebody else wrote - an empty write means the session was cleared, and
          then the stored summary must go too.
        """
        payload = getattr(event, "payload", None) or {}
        sid = payload.get("session")
        if not sid:
            return
        now = time.monotonic()
        for key in [k for k, v in self._own_writes.items() if now - v[1] > 30.0]:
            self._own_writes.pop(key, None)
        entry = self._own_writes.get(sid)
        if entry is not None:
            if entry[0] == self._mem_fingerprint(payload.get("new_memory") or []):
                self._own_writes.pop(sid, None)
                return
        if not payload.get("new_memory"):
            self._forget_session(sid)

    @staticmethod
    def _mem_fingerprint(memory) -> str:
        try:
            blob = json.dumps(memory, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            blob = repr(memory)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    async def _on_session_deleted(self, event) -> None:
        payload = getattr(event, "payload", None) or {}
        sid = payload.get("session")
        if sid:
            self._forget_session(sid)

    def _forget_session(self, sid: str) -> None:
        self._store_forget(sid)
        task = self._async_tasks.pop(sid, None)
        if task is not None and not task.done():
            task.cancel()
        self._last_tool_reset.pop(sid, None)
        self._log(f"session {sid} cleared/deleted, cumulative summary dropped")

    # ── store helpers ────────────────────────────────────────────────────────

    def _store_set(self, sid: str, summary: str) -> None:
        if self._store is None:
            return
        try:
            self._store.set(sid, summary)
            self._store.save()
        except Exception:
            pass

    def _store_forget(self, sid: str) -> None:
        if self._store is None:
            return
        try:
            self._store.pop(sid)
            self._store.save()
        except Exception:
            pass

    # ── reset core ───────────────────────────────────────────────────────────

    @staticmethod
    def _split_head_summary(flat: List[Any]):
        """Return (summary text, remaining messages) for the first memory chunk."""
        if flat:
            first = flat[0]
            content = _msg_get(first, "content")
            if isinstance(content, str) and content.startswith(SUMMARY_MARKER):
                return extract_summary_text(content), list(flat[1:])
        return "", list(flat)

    def _write_memory(self, sid: str, chunks: List[list]) -> None:
        """Replace the session memory, keeping every other session field.

        Buffered-but-unwritten messages are discarded here (and not earlier): this
        function is synchronous, so no message can slip into the buffer between the
        drop and the write. Dropping the buffer before an awaited summary call
        would leave a window in which a debounce flush re-fills it.
        """
        sm = self.session_mgr
        if self.drop_pending_buffer:
            self._drop_buffer(sid)
        if self._event_bus is not None:
            self._own_writes[sid] = [self._mem_fingerprint(chunks), time.monotonic()]
        if self.clear_mode == "delete_session":
            sm.delete_session(sid)
            sm.get_session_info(sid)  # recreate an empty shell
            sm.write_memory(sid, chunks)
            self._log(f"session {sid} deleted and recreated (delete_session mode)")
            return
        sm.write_memory(sid, chunks)

    async def _reset(
        self, sid: str, *, force_summary: bool, reason: str, apply_cooldown: bool = False
    ):
        """Serialize resets per session, then perform one.

        Returns (status, summary_text) with status in ok/empty/error/cooldown.
        """
        if self.session_mgr is None:
            return ("error", "session manager unavailable")
        async with self._get_lock(sid):
            if apply_cooldown:
                now = time.time()
                last = self._last_tool_reset.get(sid, 0.0)
                if last and (now - last) < TOOL_COOLDOWN_SECONDS:
                    self._log(f"tool reset ignored (cooldown) sid={sid}")
                    return ("cooldown", None)
            result = await self._reset_locked(sid, force_summary=force_summary, reason=reason)
            if apply_cooldown and result[0] == "ok":
                # Only a reset that actually did something starts the window, so a
                # legitimate retry after "empty" is not punished.
                self._last_tool_reset[sid] = time.time()
            return result

    async def _reset_locked(self, sid: str, *, force_summary: bool, reason: str):
        sm = self.session_mgr
        old_flat = list(sm.fetch_memory(sid) or [])
        if not old_flat:
            logger.info(f"[reboot] no history to reset sid={sid} ({reason})")
            return ("empty", None)

        head_text, rest = self._split_head_summary(old_flat)
        if self.cumulative_summary and self._store is not None:
            base = self._store.sync_with_head(
                sid, head_text, session_has_messages=bool(rest)
            )
        else:
            base = head_text
        if self.enable_summary_logging:
            logger.info(
                f"[reboot][summary] {reason} sid={sid} history={len(old_flat)} msgs, "
                f"base={len(base)} chars, mode={self.summarize_mode}"
            )

        want_summary = bool(force_summary or self.enable_summary)

        if not want_summary:
            self._write_memory(sid, [])
            self._store_forget(sid)
            logger.info(f"[reboot] context cleared sid={sid} ({reason})")
            return ("ok", None)

        if self.summarize_mode == "sync":
            delta = await self._summarize(sid, rest)
            final = await self._merge(sid, base, delta)
            if final:
                self._write_memory(sid, [build_summary_chunk(final)])
                self._store_set(sid, final)
            else:
                self._write_memory(sid, [])
                self._store_forget(sid)
            logger.info(
                f"[reboot] context cleared{' with summary' if final else ' (summary failed)'} "
                f"sid={sid} ({reason})"
            )
            return ("ok", final or None)

        # async (default): carry the previous cumulative summary over immediately so
        # the session is never fully amnesiac while the new one is generated.
        if base:
            self._write_memory(sid, [build_summary_chunk(base)])
            self._store_set(sid, base)
        else:
            self._write_memory(sid, [])
            self._store_forget(sid)
        if rest:
            self._schedule_async_summary(sid, rest, base)
        logger.info(f"[reboot] context cleared sid={sid} ({reason}), summary scheduled")
        return ("ok", None)

    async def _summarize(self, sid: str, messages: List[Any]) -> Optional[str]:
        if not messages:
            return None
        return await summarize_history(
            self.ctx,
            sid,
            messages,
            model_id=self.summarize_model,
            prompt_template=self.summarize_prompt_template,
            timeout_sec=self.summarize_timeout_sec,
            max_input_chars=self.summarize_max_input_chars,
            max_output_chars=self.summarize_max_output_chars,
            logger=logger,
            enable_detail_log=self.enable_summary_logging,
        )

    async def _merge(self, sid: str, base: str, delta: Optional[str]) -> str:
        base = (base or "").strip()
        delta = (delta or "").strip()
        if not base and not delta:
            return ""
        if not base:
            return await self._cap_summary(sid, delta)
        if not delta:
            return base
        merged = await merge_summaries(
            self.ctx,
            sid,
            base,
            delta,
            model_id=self.summarize_model,
            prompt_template=self.merge_prompt_template,
            timeout_sec=self.merge_timeout_sec,
            logger=logger,
            enable_detail_log=self.enable_summary_logging,
        )
        if not merged:
            merged = f"{base}\n{delta}"
            if self.enable_summary_logging:
                logger.info("[reboot][summary] merge failed, fell back to concatenation")
        return await self._cap_summary(sid, merged)

    async def _cap_summary(self, sid: str, text: str) -> str:
        cap = self.summarize_max_output_chars
        text = (text or "").strip()
        if not text or cap <= 0 or len(text) <= cap:
            return text
        compressed = await self_compress_summary(
            self.ctx,
            sid,
            text,
            model_id=self.summarize_model,
            prompt_template=self.self_compress_prompt_template,
            timeout_sec=self.merge_timeout_sec,
            logger=logger,
            enable_detail_log=self.enable_summary_logging,
        )
        if compressed and compressed.strip():
            text = compressed.strip()
        if len(text) > cap:
            text = text[:cap] + "…"
        return text

    # ── async summary ────────────────────────────────────────────────────────

    def _schedule_async_summary(self, sid: str, messages: List[Any], base: str) -> None:
        old = self._async_tasks.pop(sid, None)
        if old is not None and not old.done():
            old.cancel()

        async def _run():
            try:
                # Every await happens HERE, outside the read/modify/write section.
                delta = await self._summarize(sid, messages)
                final = await self._merge(sid, base, delta)
                if not final:
                    return
                current_base = base
                for _ in range(2):
                    applied, head_now = self._apply_summary(sid, current_base, final)
                    if applied:
                        self._store_set(sid, final)
                        logger.info(
                            f"[reboot] cumulative summary updated sid={sid} ({len(final)} chars)"
                        )
                        return
                    if not head_now or head_now == current_base:
                        return
                    # Another writer (e.g. ADS) replaced the head meanwhile: adopt it
                    # as the new base and merge once more.
                    if self.enable_summary_logging:
                        logger.info("[reboot][summary] summary head changed, re-merging")
                    current_base = head_now
                    final = await self._merge(sid, head_now, final)
                    if not final:
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"[reboot] async summary failed sid={sid}")
            finally:
                # Only clear our own slot: a newer task may already have replaced it.
                if self._async_tasks.get(sid) is asyncio.current_task():
                    self._async_tasks.pop(sid, None)

        self._async_tasks[sid] = asyncio.create_task(_run())

    def _apply_summary(self, sid: str, expected_base: str, final: str):
        """Replace the summary head. Deliberately synchronous.

        There must be NO await between ``read_memory`` and ``write_memory``: the
        framework appends a turn with ``update_memory`` at the end of every round,
        and a suspension point here would let that append be overwritten by our
        stale snapshot. With no await (single-threaded event loop) the race is
        impossible by construction.

        Returns (applied, current_head).
        """
        sm = self.session_mgr
        if sm is None:
            return (False, "")
        chunks = sm.read_memory(sid) or []
        chunks = [list(chunk) for chunk in chunks]
        head_now = ""
        if chunks and is_summary_chunk(chunks[0]):
            head_now = extract_summary_text(str(_msg_get(chunks[0][0], "content", "") or ""))
        if head_now != (expected_base or "").strip():
            return (False, head_now)
        message = build_summary_chunk(final)[0]
        if chunks and is_summary_chunk(chunks[0]):
            chunks[0] = [message] + list(chunks[0][1:])
        elif chunks:
            chunks[0] = [message] + list(chunks[0])
        else:
            chunks = [[message]]
        sm.write_memory(sid, chunks)
        return (True, head_now)
