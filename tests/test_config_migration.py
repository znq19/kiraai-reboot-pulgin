"""Config surface: schema validity and migration from the 1.0/1.1 flat layout."""

import json
from pathlib import Path

from harness import bootstrap, check, check_eq, default_cfg, make_fixture, make_message_event

SID = "qq:dm:10001"


def build_plugin(cfg, data_dir=None):
    """Construct the plugin the way the plugin manager does: with the config
    *object* (not a copy), so in-place migration is observable from the outside."""
    from harness import FakeCtx, FakeEventBus, make_session_manager, plugin_class

    bus = FakeEventBus()
    sm = make_session_manager(event_bus=bus)
    ctx = FakeCtx(sm, event_bus=bus, data_dir=data_dir)
    return plugin_class()(ctx, cfg)


# ── schema ───────────────────────────────────────────────────────────────────


async def test_schema_parses_with_the_framework_parser():
    """Runs our schema.json through core.config.config_field, exactly like the
    plugin manager does on load."""
    bootstrap()
    import json
    from harness import _STATE
    from core.config.config_field import build_fields, SectionField

    raw = json.loads((_STATE["plugin_dir"] / "schema.json").read_text(encoding="utf-8"))
    fields = build_fields(raw)
    check(len(fields) >= 5, f"expected the 5 sections, got {len(fields)}")

    total = 0
    for field in fields:
        if isinstance(field, SectionField):
            check(bool(field.fields), f"section {field.key} must not be empty")
            total += len(field.fields)
    check_eq(total, 29, "the documented number of configuration items")


async def test_every_config_item_has_a_display_name_and_a_type():
    bootstrap()
    import json
    from harness import _STATE

    raw = json.loads((_STATE["plugin_dir"] / "schema.json").read_text(encoding="utf-8"))
    for section_key, section in raw.items():
        check_eq(section.get("type"), "section", f"{section_key} must be a section")
        check(bool(section.get("name")), f"{section_key} needs a name")
        for key, item in (section.get("fields") or {}).items():
            check(bool(item.get("type")), f"{section_key}.{key} needs a type")
            check(bool(item.get("name")), f"{section_key}.{key} needs a name")
            check("default" in item, f"{section_key}.{key} should declare a default")
            if item["type"] == "enum":
                check(item.get("default") in item.get("options", []),
                      f"{section_key}.{key}: default must be one of the options")


async def test_security_and_behaviour_defaults_are_as_designed():
    cfg = default_cfg()
    check_eq(cfg["section_reset"]["clear_mode"], "memory_only", "keep the session by default")
    check_eq(cfg["section_command"]["enable_reboot_command"], False, "commands off by default")
    check_eq(cfg["section_command"]["enable_resum_command"], False, "commands off by default")
    check_eq(cfg["section_summary"]["enable_summary"], False, "summary off by default")
    check_eq(cfg["section_summary"]["summarize_mode"], "async", "non blocking by default")
    check_eq(cfg["section_llm_tool"]["allow_llm_reboot"], True, "the tool stays on by default")
    check_eq(cfg["section_reset"]["drop_pending_buffer"], True, "drop the buffer by default")
    check_eq(cfg["section_compat"]["ads_interop"], True, "interop on by default")
    check_eq(cfg["section_command"]["reboot_commands"], ["/reboot"], "default keyword")
    check_eq(cfg["section_command"]["resum_commands"], ["/resume"], "default keyword")
    check_eq(cfg["section_command"]["reboot_allowed_users"], [], "empty whitelist")


# ── 1.0 -> 2.0 migration ─────────────────────────────────────────────────────


async def test_legacy_command_prefix_is_migrated():
    """The framework fills in section defaults before the plugin is constructed, so
    an upgrade must honour the old flat keys explicitly.

    1.x had no command switch at all - a configured prefix was always live - so the
    migrated command must come back enabled. Asserting an empty command map here
    would lock in the regression this release fixes.
    """
    f = await make_fixture({"command_prefix": "/重开"})
    check_eq(f.plugin.reboot_commands, ["/重开"], "legacy prefix must win while the new key is default")
    check_eq(f.plugin.enable_reboot_command, True, "1.x commands were always on")
    check_eq(f.plugin._command_map, {"/重开": False}, "and the keyword is live")


async def test_a_fresh_install_keeps_the_command_off():
    """The auto-enable above is reserved for migrated installations: a brand new
    config has no legacy key, so the documented default (off) still applies."""
    f = await make_fixture({})
    check_eq(f.plugin.enable_reboot_command, False, "new installs opt in")
    check_eq(f.plugin._command_map, {}, "no keyword is claimed")


async def test_legacy_blank_prefix_does_not_resurrect_the_command():
    """In 1.x an empty prefix never matched, so it must not become '/reboot'."""
    cfg = default_cfg()
    cfg["command_prefix"] = "   "
    from reboot_plugin.main import migrate_config

    migrate_config(cfg)
    check_eq(cfg["section_command"]["reboot_commands"], ["/reboot"], "the new key keeps its default")
    check_eq(cfg["section_command"]["enable_reboot_command"], False, "nothing to enable")


async def test_explicit_new_value_beats_the_legacy_one():
    f = await make_fixture({"command_prefix": "/legacy",
                            "section_command.reboot_commands": ["/new"],
                            "section_command.enable_reboot_command": True})
    check_eq(f.plugin.reboot_commands, ["/new"], "a user-set new value is authoritative")
    check_eq(f.plugin._command_map, {"/new": False}, "and it is the live keyword")


async def test_legacy_permission_settings_are_migrated():
    f = await make_fixture({"enable_permission": True, "allowed_users": ["1", "2"]})
    check_eq(f.plugin.enable_permission, True, "legacy switch honoured")
    check_eq(f.plugin.allowed_users, ["1", "2"], "legacy list honoured")


async def test_legacy_message_templates_are_migrated():
    f = await make_fixture({"success_message": "OLD-OK{summary}",
                            "permission_denied_message": "OLD-DENIED",
                            "error_message": "OLD-ERR {error}"})
    check_eq(f.plugin.success_message, "OLD-OK{summary}", "legacy success text")
    check_eq(f.plugin.permission_denied_message, "OLD-DENIED", "legacy denied text")
    check_eq(f.plugin.error_message, "OLD-ERR {error}", "legacy error text")


async def test_legacy_settings_are_actually_used_at_runtime():
    f = await make_fixture({"command_prefix": "/old",
                            "success_message": "LEGACY-OK",
                            "section_command.enable_reboot_command": True})
    f.seed_session(SID, turns=1)
    await f.plugin.handle_command(make_message_event("/old"))
    await f.settle()
    check_eq(f.memory(SID), [], "the legacy keyword works in the live command map")
    check_eq(f.last_reply(), "LEGACY-OK", "the legacy template is used")


async def test_legacy_allowed_users_accepts_multiline_strings():
    f = await make_fixture({"enable_permission": True, "allowed_users": "1\n2\n3"})
    check_eq(f.plugin.allowed_users, ["1", "2", "3"], "newline separated ids")


# ── the persisted migration (2.1) ────────────────────────────────────────────
#
# 2.0 renamed every 1.x setting. The framework keeps the orphan keys on disk and
# fills the new ones with defaults, so the WebUI showed defaults while the plugin
# (thanks to a runtime shim) used the real values - the settings page and the
# behaviour disagreed. These tests pin the fix: migrate once, persist, delete the
# legacy keys, and keep the two views identical from then on.


async def test_the_migration_map_covers_every_legacy_key():
    """Release guard. Frozen on purpose: a future rename must add a migration entry
    (and this table must be updated), never drop a user's value silently."""
    bootstrap()
    from harness import _STATE
    from reboot_plugin.main import LEGACY_FLAT_KEYS, LEGACY_SECTIONS, MIGRATION_TARGETS

    check_eq(
        set(LEGACY_FLAT_KEYS),
        {"command_prefix", "enable_permission", "allowed_users", "success_message",
         "permission_denied_message", "error_message", "verbose_log"},
        "the 1.x flat layout is frozen - add a new entry instead of editing this one",
    )
    check_eq(set(MIGRATION_TARGETS), set(LEGACY_FLAT_KEYS), "every legacy key needs a target")
    check_eq(LEGACY_SECTIONS, ("section_debug",),
             "the folded section list is frozen - update this when a section moves")

    schema = json.loads((Path(_STATE["plugin_dir"]) / "schema.json").read_text(encoding="utf-8"))
    for legacy, target in MIGRATION_TARGETS.items():
        section, _, key = target.partition(".")
        check(section in schema, f"{legacy} points at a missing section: {target}")
        check(key in schema[section]["fields"], f"{legacy} points at a missing field: {target}")
    for folded in LEGACY_SECTIONS:
        check(folded not in schema, f"{folded} was folded away and must not come back")


async def test_migration_moves_every_legacy_key():
    bootstrap()
    from reboot_plugin.main import LEGACY_FLAT_KEYS, migrate_config

    cfg = default_cfg()
    cfg.update({"command_prefix": "/重开", "enable_permission": True,
                "allowed_users": ["1", "2"], "success_message": "OK-LEGACY",
                "permission_denied_message": "DENY-LEGACY", "error_message": "ERR-LEGACY {error}",
                "verbose_log": True})
    changed = migrate_config(cfg)
    check(len(changed) >= 7, f"expected a report entry per setting, got {changed}")

    cmd = cfg["section_command"]
    check_eq(cmd["reboot_commands"], ["/重开"], "keyword")
    check_eq(cmd["enable_reboot_command"], True, "1.x command was always on")
    check_eq(cmd["reboot_enable_permission"], True, "permission switch")
    check_eq(cmd["reboot_allowed_users"], ["1", "2"], "whitelist")
    check_eq(cmd["reboot_success_message"], "OK-LEGACY", "success template")
    check_eq(cmd["reboot_permission_denied_message"], "DENY-LEGACY", "denied template")
    check_eq(cmd["reboot_error_message"], "ERR-LEGACY {error}", "error template")
    check_eq(cfg["section_compat"]["verbose_log"], True, "verbose switch")
    for key in LEGACY_FLAT_KEYS:
        check(key not in cfg, f"{key} must be deleted, otherwise the migration is not idempotent")


async def test_migration_is_idempotent():
    bootstrap()
    from reboot_plugin.main import migrate_config

    cfg = default_cfg()
    cfg.update({"command_prefix": "/重开", "allowed_users": ["1"], "verbose_log": True})
    check(bool(migrate_config(cfg)), "the first pass migrates")
    snapshot = json.dumps(cfg, sort_keys=True)
    check_eq(migrate_config(cfg), [], "the second pass has nothing left to do")
    check_eq(json.dumps(cfg, sort_keys=True), snapshot, "and changes nothing at all")


async def test_migration_never_overrides_a_value_the_user_already_changed():
    """The new key is authoritative when it no longer holds the default."""
    bootstrap()
    from reboot_plugin.main import migrate_config

    cfg = default_cfg()
    cfg["section_command"]["reboot_commands"] = ["/already-new"]
    cfg["section_command"]["reboot_allowed_users"] = ["999"]
    cfg.update({"command_prefix": "/重开", "allowed_users": ["111"]})
    migrate_config(cfg)
    check_eq(cfg["section_command"]["reboot_commands"], ["/already-new"], "new value wins")
    check_eq(cfg["section_command"]["reboot_allowed_users"], ["999"], "new value wins")
    check("command_prefix" not in cfg, "the orphan key is still cleaned up")
    check("allowed_users" not in cfg, "the orphan key is still cleaned up")


async def test_a_short_lived_debug_section_is_folded_into_compat():
    """verbose_log has now moved twice (1.x flat key -> "调试" section -> 兼容与调试).
    Anyone who ran the intermediate build must keep their switch instead of losing
    it a second time."""
    bootstrap()
    from reboot_plugin.main import migrate_config

    cfg = default_cfg()
    cfg["section_debug"] = {"verbose_log": True}
    changed = migrate_config(cfg)
    check(bool(changed), f"the fold must be reported, got {changed}")
    check("section_debug" not in cfg, "the folded section is removed")
    check_eq(cfg["section_compat"]["verbose_log"], True, "the switch survives the move")

    plugin = build_plugin(cfg)
    check_eq(plugin.verbose_log, True, "and the plugin reads it")
    check_eq(cfg["section_compat"]["verbose_log"], plugin.verbose_log, "display == behaviour")


async def test_the_debug_fold_follows_the_same_default_rule():
    """Same rule as the flat keys: the legacy value wins only while the target still
    holds its default (False). Either way the dead section is removed and the removal
    is reported, so it gets persisted instead of lingering on disk."""
    bootstrap()
    from reboot_plugin.main import migrate_config

    bumped = default_cfg()
    bumped["section_debug"] = {"verbose_log": True}
    migrate_config(bumped)
    check_eq(bumped["section_compat"]["verbose_log"], True, "target was default -> legacy wins")
    check("section_debug" not in bumped, "cleaned up")

    already_on = default_cfg()
    already_on["section_debug"] = {"verbose_log": True}
    already_on["section_compat"]["verbose_log"] = True
    check(bool(migrate_config(already_on)), "the removal must still be reported")
    check("section_debug" not in already_on, "cleaned up")
    check_eq(already_on["section_compat"]["verbose_log"], True, "value untouched")


async def test_migration_leaves_a_fresh_config_alone():
    """A brand new installation has no legacy keys and no folded section, so
    migration must be a no-op - otherwise a future bug could switch the command on
    for new users."""
    bootstrap()
    from reboot_plugin.main import migrate_config

    cfg = default_cfg()
    snapshot = json.dumps(cfg, sort_keys=True)
    check_eq(migrate_config(cfg), [], "nothing to migrate")
    check_eq(json.dumps(cfg, sort_keys=True), snapshot, "byte for byte untouched")
    check_eq(cfg["section_command"]["enable_reboot_command"], False, "new users stay opt-in")


async def test_migration_is_written_back_to_the_config_file():
    """The settings page reads the file through the plugin manager, so a migration
    that only lived in memory would keep showing the old defaults."""
    bootstrap()
    from harness import _STATE
    from reboot_plugin.main import LEGACY_FLAT_KEYS

    cfg_dir = Path(_STATE["sandbox"]) / "data" / "config" / "plugins"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "reboot_plugin.json"
    path.write_text("{}", encoding="utf-8")

    cfg = default_cfg()
    cfg.update({"command_prefix": "/重开", "allowed_users": ["10001"],
                "enable_permission": True, "success_message": "OLD-OK",
                "verbose_log": True})
    build_plugin(cfg)

    disk = json.loads(path.read_text(encoding="utf-8"))
    check_eq(disk["section_command"]["reboot_commands"], ["/重开"], "keyword written back")
    check_eq(disk["section_command"]["enable_reboot_command"], True, "command enabled on disk")
    check_eq(disk["section_command"]["reboot_allowed_users"], ["10001"], "whitelist written back")
    check_eq(disk["section_command"]["reboot_enable_permission"], True, "permission written back")
    check_eq(disk["section_command"]["reboot_success_message"], "OLD-OK", "template written back")
    check_eq(disk["section_compat"]["verbose_log"], True, "verbose flag written back")
    for key in LEGACY_FLAT_KEYS:
        check(key not in disk, f"{key} must be gone from the file")


async def test_settings_page_and_runtime_agree_after_migration():
    """★ The invariant this release exists for: whatever GET /plugins/<id>/config
    renders must be exactly what the plugin uses. No orphan key may shadow a new
    one, in either direction."""
    bootstrap()
    cfg = default_cfg()
    cfg.update({"command_prefix": "/重开", "allowed_users": ["10001"],
                "enable_permission": True, "success_message": "OLD-OK",
                "permission_denied_message": "OLD-DENY",
                "error_message": "OLD-ERR {error}", "verbose_log": True})
    plugin = build_plugin(cfg)

    # `cfg` is the very dict the plugin manager keeps in plugin_configs, i.e. what
    # the settings form is populated from.
    cmd = cfg["section_command"]
    check_eq(plugin.reboot_commands, cmd["reboot_commands"], "keyword")
    check_eq(plugin.enable_reboot_command, cmd["enable_reboot_command"], "command switch")
    check_eq(plugin.allowed_users, cmd["reboot_allowed_users"], "whitelist")
    check_eq(plugin.enable_permission, cmd["reboot_enable_permission"], "permission switch")
    check_eq(plugin.success_message, cmd["reboot_success_message"], "success text")
    check_eq(plugin.permission_denied_message, cmd["reboot_permission_denied_message"], "denied text")
    check_eq(plugin.error_message, cmd["reboot_error_message"], "error text")
    check_eq(plugin.verbose_log, cfg["section_compat"]["verbose_log"], "verbose switch")
    check_eq(plugin._build_command_map(), {"/重开": False}, "and the keyword is live")
    for key in ("command_prefix", "allowed_users", "enable_permission", "success_message",
                "permission_denied_message", "error_message", "verbose_log"):
        check(key not in cfg, f"{key} must not linger and shadow the new key")


async def test_user_can_restore_the_default_keyword_after_migration():
    """The nastiest pre-2.1 symptom: setting the keyword back to the shipped default
    ('/reboot') lost against the orphan 'command_prefix' forever, because the shim
    compared the new value against that same default. With the legacy key deleted
    the user's choice is authoritative again."""
    bootstrap()
    cfg = default_cfg()
    cfg.update({"command_prefix": "/重开"})
    build_plugin(cfg)
    check("command_prefix" not in cfg, "migrated")

    # the settings form posts the whole section back, keyword set to the default
    cfg2 = default_cfg()
    cfg2["section_command"]["reboot_commands"] = ["/reboot"]
    cfg2["section_command"]["enable_reboot_command"] = True
    plugin2 = build_plugin(cfg2)
    check_eq(plugin2.reboot_commands, ["/reboot"], "the user's explicit choice")
    check_eq(plugin2._build_command_map(), {"/reboot": False}, "and it is what fires")
    check("/重开" not in plugin2._build_command_map(), "the orphan keyword is gone")


async def test_read_config_reads_every_section_declared_in_the_schema():
    """Future-proofing: adding a section to schema.json without wiring it into
    _read_config would silently expose settings that do nothing."""
    bootstrap()
    import re
    from harness import _STATE

    schema = json.loads((Path(_STATE["plugin_dir"]) / "schema.json").read_text(encoding="utf-8"))
    source = (_STATE["plugin_dir"] / "main.py").read_text(encoding="utf-8")
    read_sections = set(re.findall(r'cfg\.get\("(section_\w+)"\)', source))
    check_eq(sorted(set(schema) - read_sections), [],
             "every schema section must be pulled out of cfg in _read_config")


async def test_legacy_values_are_honoured_without_a_writable_config():
    """Degradation path: when the file cannot be located or written, the orphan keys
    must still drive the current run instead of silently reverting to defaults."""
    f = await make_fixture({})
    cfg = default_cfg()
    cfg.update({"command_prefix": "/重开", "allowed_users": ["7"],
                "enable_permission": True, "verbose_log": True,
                "success_message": "OLD-OK"})
    f.plugin._read_config(cfg)
    check_eq(f.plugin.reboot_commands, ["/重开"], "keyword")
    check_eq(f.plugin.enable_reboot_command, True, "1.x command was live")
    check_eq(f.plugin.allowed_users, ["7"], "whitelist")
    check_eq(f.plugin.enable_permission, True, "permission")
    check_eq(f.plugin.verbose_log, True, "verbose flag")
    check_eq(f.plugin.success_message, "OLD-OK", "template")
    check_eq(f.plugin._build_command_map(), {"/重开": False}, "the command still fires")


# ── value handling ───────────────────────────────────────────────────────────


async def test_zero_is_a_valid_limit_value():
    """0 means 'unlimited' for the char limits, so `or default` must not be used."""
    f = await make_fixture({"section_summary.summarize_max_input_chars": 0,
                            "section_summary.summarize_max_output_chars": 0})
    check_eq(f.plugin.summarize_max_input_chars, 0, "0 must survive")
    check_eq(f.plugin.summarize_max_output_chars, 0, "0 must survive")


async def test_invalid_enum_values_fall_back_to_the_safe_default():
    f = await make_fixture({"section_reset.clear_mode": "wat",
                            "section_summary.summarize_mode": "wat"})
    check_eq(f.plugin.clear_mode, "memory_only", "unknown clear mode is not destructive")
    check_eq(f.plugin.summarize_mode, "async", "unknown mode is non blocking")


async def test_command_lists_accept_strings_and_deduplicate():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_commands": "/a, /b, /a"})
    check_eq(f.plugin.reboot_commands, ["/a", "/b"], "comma separated string + dedup")
    check_eq(sorted(f.plugin._command_map), ["/a", "/b"], "both keywords live")


async def test_blank_command_list_falls_back_to_the_default():
    f = await make_fixture({"section_command.enable_reboot_command": True,
                            "section_command.reboot_commands": []})
    check_eq(f.plugin.reboot_commands, ["/reboot"], "an empty list is not a valid config")




async def test_plugin_survives_without_a_data_directory():
    """PluginContext returns None when it cannot map the module to a plugin id;
    the reset must still work, only the cumulative store degrades."""
    from harness import FakeEventBus, FakeCtx, default_cfg, make_session_manager, plugin_class

    bus = FakeEventBus()
    sm = make_session_manager(event_bus=bus)
    ctx = FakeCtx(sm, data_dir=None, event_bus=bus)
    plugin = plugin_class()(ctx, default_cfg())
    ctx.get_plugin_data_dir = lambda: None  # simulate the unmappable module
    await plugin.initialize()
    check(plugin.session_mgr is not None, "the plugin must stay enabled")
    check_eq(plugin._store, None, "store degraded")

    plugin.session_mgr.get_session_info(SID)
    plugin.session_mgr.write_memory(SID, [[{"role": "user", "content": "hi"},
                                           {"role": "assistant", "content": "yo"}]])
    status, _ = await plugin._reset(SID, force_summary=False, reason="test")
    check_eq(status, "ok", "a reset still works without the store")
    check_eq(plugin.session_mgr.read_memory(SID), [], "context cleared")


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]


TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]
