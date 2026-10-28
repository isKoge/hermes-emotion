"""Score trajectory: per-turn snapshots, the /emotion history curve, and amplitude.

The engine keeps a single upserted row, so these tests pin the two things that were
missing: a visible history, and enough per-turn travel for the grid to actually move.
"""

from __future__ import annotations

from engine import emotion_engine as EE
from engine import store
from test_emotion_grid import FakeCtx, FakeLLM, load_plugin


def _db(tmp_path):
    path = tmp_path / "grid.db"
    store.ensure_schema(str(path))
    return str(path)


def _ctx(tmp_path, **cfg):
    """FakeCtx whose engine db is the file the test writes to."""
    config = {
        "state_path": str(tmp_path / "state.json"),
        "state_source": "internal",
        "engine_db_path": _db(tmp_path),
    }
    config.update(cfg)
    return FakeCtx(config, llm=FakeLLM())


# ── history storage ──────────────────────────────────────────────────
def test_score_history_roundtrip_and_order(tmp_path):
    db = _db(tmp_path)
    for i, v in enumerate((10.0, 20.0, 30.0)):
        store.append_score_history(
            {"aff": v, "sec": -v, "aro": 0.0, "dom": 5.0},
            session_id="s1", event_type="praise", coords=("近", "稳", "热", "进"), db_path=db,
        )
    rows = store.load_score_history(10, db_path=db)
    assert [r["aff"] for r in rows] == [10.0, 20.0, 30.0]
    assert rows[-1]["sec"] == -30.0
    assert rows[-1]["event_type"] == "praise"


def test_score_history_trim_keeps_newest(tmp_path):
    db = _db(tmp_path)
    for i in range(12):
        store.append_score_history(
            {"aff": float(i), "sec": 0, "aro": 0, "dom": 0}, db_path=db, keep=5
        )
    rows = store.load_score_history(50, db_path=db)
    assert len(rows) == 5
    assert [r["aff"] for r in rows] == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_score_history_never_raises_on_bad_scores(tmp_path):
    db = _db(tmp_path)
    # a non-numeric score must be swallowed, not kill the turn
    store.append_score_history({"aff": None, "sec": 1, "aro": 1, "dom": 1}, db_path=db)
    assert store.load_score_history(5, db_path=db) == []


# ── amplitude ────────────────────────────────────────────────────────
def _step(prev, event_type, amplitude, intensity=1.0):
    event = {"type": event_type, "intensity": intensity, "sincerity": 1.0}
    return EE.emotion_step(event, {}, dict(prev), amplitude=amplitude)


def test_amplitude_scales_movement():
    # mid intensity keeps both runs below the per-turn clamp, so the ratio is visible
    prev = {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0}
    quiet = _step(prev, "praise", 1.0, intensity=0.3)
    loud = _step(prev, "praise", 2.5, intensity=0.3)
    assert abs(loud["aff"]) > abs(quiet["aff"]) * 1.8


def test_saturated_axis_still_moves():
    """At 100 the old cap scale (0.1) froze the axis; the floor keeps it responsive."""
    prev = {"aff": 100.0, "sec": 100.0, "aro": 0.0, "dom": 0.0}
    out = _step(prev, "cold", EE.EMOTION_AMPLITUDE)
    assert out["aff"] < 100.0, "aff pinned at the ceiling"
    assert prev["aff"] - out["aff"] > 1.0


def _aff_after(tmp_path, amplitude, message):
    mod = load_plugin()
    ctx = _ctx(tmp_path, score_amplitude=amplitude)
    mod.register(ctx)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    ctx.hooks["pre_llm_call"](session_id="s-amp", user_message=message)
    rows = store.load_score_history(5, db_path=db)
    assert len(rows) == 1, "hook did not record a snapshot"
    assert ctx.commands, "command not registered"
    return rows[0]


def test_score_amplitude_config_changes_real_movement(tmp_path):
    quiet_dir = tmp_path / "quiet"
    loud_dir = tmp_path / "loud"
    quiet_dir.mkdir()
    loud_dir.mkdir()
    _db(quiet_dir)
    _db(loud_dir)

    quiet = _aff_after(quiet_dir, 1.0, "你真棒，太厉害了！")
    loud = _aff_after(loud_dir, 4.0, "你真棒，太厉害了！")
    assert loud["aff"] - quiet["aff"] > 2.0, (
        f"amplitude config had no effect: quiet={quiet['aff']} loud={loud['aff']}"
    )


def test_short_brushoffs_are_not_plain_chat():
    """'呵呵'/'随便吧' used to land on casual_chat, so nothing ever moved."""
    assert EE.interpret_event("呵呵")["type"] == "cold"
    assert EE.interpret_event("随便吧")["type"] == "cold"
    assert EE.interpret_event("你根本不懂我在说什么")["type"] == "criticism"
    assert EE.interpret_event("今天中午吃什么")["type"] == "casual_chat"


def test_criticism_moves_scores_down():
    prev = {"aff": 40.0, "sec": 20.0, "aro": 10.0, "dom": 0.0}
    event = EE.interpret_event("你根本不懂我在说什么")
    out = EE.emotion_step(event, {}, dict(prev))
    assert out["aff"] < prev["aff"]
    assert out["sec"] < prev["sec"]


# ── /emotion history curve ───────────────────────────────────────────
def test_history_command_renders_sparklines(tmp_path):
    mod = load_plugin()
    ctx = _ctx(tmp_path)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    for aff in (10.0, -20.0, 60.0):
        store.append_score_history(
            {"aff": aff, "sec": 1.0, "aro": 2.0, "dom": 3.0}, event_type="praise", db_path=db
        )
    mod.register(ctx)

    out = ctx.commands["emotion"]("history 20")
    assert "好感" in out and "安全感" in out and "唤醒" in out and "支配" in out
    assert any(ch in out for ch in "▁▂▃▄▅▆▇█")
    assert "praise" in out


def test_history_command_empty_db(tmp_path):
    mod = load_plugin()
    ctx = _ctx(tmp_path)
    store.set_db_path(ctx.config["engine_db_path"])
    mod.register(ctx)
    assert "还没有分数历史" in ctx.commands["emotion"]("history")


# ── review regressions ───────────────────────────────────────────────
def test_clamp_holds_after_atmosphere_modulation():
    """Warm atmosphere used to multiply after the clamp and breach the ceiling."""
    prev = {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0}
    event = {"type": "vulnerable", "intensity": 1.0, "sincerity": 0.8}
    out = EE.emotion_step(event, {"atmosphere": "warm"}, dict(prev))
    assert abs(out["aff"] - prev["aff"]) <= EE.SINGLE_TURN_CLAMP + 1e-6
    assert abs(out["sec"] - prev["sec"]) <= EE.SINGLE_TURN_CLAMP + 1e-6


def test_non_finite_amplitude_falls_back():
    prev = {"aff": 20.0, "sec": 10.0, "aro": 5.0, "dom": 0.0}
    event = {"type": "casual_chat", "intensity": 1.0, "sincerity": 1.0}
    default_out = EE.emotion_step(event, {}, dict(prev))
    for bad in (float("inf"), float("nan"), -3.0, "junk"):
        out = EE.emotion_step(event, {}, dict(prev), amplitude=bad)
        for k in ("aff", "sec", "aro", "dom"):
            assert out[k] == default_out[k], f"amplitude={bad!r} diverged on {k}"
    # dom has no stimulus in casual_chat: it must stay put, not jump to the clamp
    assert default_out["dom"] == 0.0


def test_history_skips_out_of_range_rows(tmp_path):
    import sqlite3

    db = _db(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO score_history (session_id, event_type, aff, sec, aro, dom) "
        "VALUES ('s', 'praise', 1000, 1, 2, 3)"
    )
    conn.execute(
        "INSERT INTO score_history (session_id, event_type, aff, sec, aro, dom) "
        "VALUES ('s', 'praise', 100, -100, 2, 3)"
    )
    conn.commit()
    conn.close()

    rows = store.load_score_history(10, db_path=db)
    assert len(rows) == 1, "out-of-range row was rewritten instead of rejected"
    assert rows[0]["aff"] == 100.0 and rows[0]["sec"] == -100.0


def test_history_skips_unreadable_rows(tmp_path):
    import sqlite3

    db = _db(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO score_history (session_id, event_type, aff, sec, aro, dom) "
        "VALUES ('s', 'praise', 'bad', 1, 2, 3)"
    )
    conn.execute(
        "INSERT INTO score_history (session_id, event_type, aff, sec, aro, dom) "
        "VALUES ('s', 'praise', 4, 5, 6, 7)"
    )
    conn.commit()
    conn.close()

    rows = store.load_score_history(10, db_path=db)
    assert len(rows) == 1
    assert rows[0]["aff"] == 4.0


def test_append_rejects_non_finite(tmp_path):
    db = _db(tmp_path)
    store.append_score_history(
        {"aff": float("inf"), "sec": 1, "aro": 1, "dom": 1}, db_path=db
    )
    assert store.load_score_history(5, db_path=db) == []


def test_broken_history_table_does_not_break_schema(tmp_path):
    import sqlite3

    db = tmp_path / "broken.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE VIEW score_history AS SELECT 1 AS aff")
    conn.commit()
    conn.close()

    store.ensure_schema(str(db))  # must not raise
    assert store.load_score_history(5, db_path=str(db)) == []

    # the core tables must still exist after the history failure
    import sqlite3

    names = {
        r[0]
        for r in sqlite3.connect(str(db)).execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"emotion_engine", "relationship_state", "event_history"} <= names

    # and a reader asking strictly must get an error rather than a silent empty list
    try:
        store.load_score_history(5, db_path=str(db), strict=True)
    except sqlite3.Error:
        pass
    else:  # pragma: no cover
        raise AssertionError("strict read swallowed a broken history table")


def test_snapshot_is_idempotent_per_turn(tmp_path):
    """Two hook calls with the same turn_id must produce one snapshot."""
    from test_emotion_grid import FakeCtx, FakeLLM, load_plugin

    mod = load_plugin()
    ctx = FakeCtx(
        {
            "state_path": str(tmp_path / "state.json"),
            "state_source": "internal",
            "engine_db_path": _db(tmp_path),
        },
        llm=FakeLLM(),
    )
    mod.register(ctx)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    hook = ctx.hooks["pre_llm_call"]

    coords = ("近", "稳", "静", "进")
    scores = {"aff": 1.0, "sec": 1.0, "aro": 1.0, "dom": 1.0}

    for _ in range(2):
        mod._record_score_snapshot(
            sess="s-idem", turn_id="t1", event_type="praise", scores=scores,
            coords=coords, db_path=db,
        )
    assert len(store.load_score_history(10, db_path=db)) == 1, "same turn duplicated"

    # a real next turn produces a different score tuple (decay + new noise)
    mod._record_score_snapshot(
        sess="s-idem", turn_id="t2", event_type="praise",
        scores={"aff": 1.4, "sec": 1.1, "aro": 1.0, "dom": 1.0},
        coords=coords, db_path=db,
    )
    assert len(store.load_score_history(10, db_path=db)) == 2, "new turn was swallowed"

    # no turn_id: identity is unknowable, so every call is recorded (never swallowed)
    for _ in range(2):
        mod._record_score_snapshot(
            sess="s-out", turn_id="", event_type="casual_chat",
            scores={"aff": 2.0, "sec": 2.0, "aro": 2.0, "dom": 2.0},
            coords=coords, db_path=db,
        )
    assert len(store.load_score_history(10, db_path=db)) == 4, (
        "a snapshot without turn_id was dropped"
    )


def test_history_command_reports_endpoints_and_delta(tmp_path):
    mod = load_plugin()
    ctx = _ctx(tmp_path)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    for aff in (10.0, 35.0, 60.0):
        store.append_score_history(
            {"aff": aff, "sec": 0.0, "aro": 0.0, "dom": 0.0}, event_type="praise", db_path=db
        )
    mod.register(ctx)

    out = ctx.commands["emotion"]("history 10")
    assert "10.0 → 60.0 (+50.0)" in out
    assert "最近 3 轮" in out


def test_sparkline_is_flat_for_constant_series(tmp_path):
    mod = load_plugin()
    ctx = _ctx(tmp_path)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    for _ in range(4):
        store.append_score_history(
            {"aff": 5.0, "sec": 5.0, "aro": 5.0, "dom": 5.0}, event_type="plain", db_path=db
        )
    mod.register(ctx)

    line = [ln for ln in ctx.commands["emotion"]("history 10").splitlines() if ln.startswith("好感")][0]
    assert "5.0 → 5.0 (+0.0)" in line
    glyphs = [ch for ch in line if ch in "▁▂▃▄▅▆▇█"]
    assert len(set(glyphs)) == 1, f"constant series rendered {set(glyphs)}"


def test_clamp_holds_with_adult_mode_reversal():
    prev = {"aff": 0.0, "sec": 0.0, "aro": 0.0, "dom": 0.0}
    event = {"type": "hurtful", "intensity": 1.0, "sincerity": 1.0}
    out = EE.emotion_step(event, {}, dict(prev), adult_mode=True, sensitivity=0)
    for k in ("aff", "sec", "aro", "dom"):
        assert abs(out[k] - prev[k]) <= EE.SINGLE_TURN_CLAMP + 1e-6, k


# ── round-3 regressions ──────────────────────────────────────────────
def _snap(mod, db, *, sess="s-cmp", history_len=12, msg="嗯", history=None, turn_id=""):
    mod._record_score_snapshot(
        sess=sess, turn_id=turn_id, history_len=history_len, user_message=msg,
        conversation_history=history, event_type="praise",
        scores={"aff": 1.0, "sec": 1.0, "aro": 1.0, "dom": 1.0},
        coords=("近", "稳", "静", "进"), db_path=db,
    )


def test_hook_records_one_row_per_real_turn(tmp_path):
    """Real hook path: repeated calls inside a turn collapse, the next turn lands."""
    mod = load_plugin()
    ctx = _ctx(tmp_path)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    # production shape: the host passes turn_id (agent/turn_context.py pre_llm_call payload)
    hook(session_id="s-turn", user_message="你真棒，太厉害了！", turn_id="turn-1")
    hook(session_id="s-turn", user_message="你真棒，太厉害了！", turn_id="turn-1")
    assert len(store.load_score_history(10, db_path=db)) == 1, "duplicate call not deduped"

    hook(session_id="s-turn", user_message="在呢", turn_id="turn-2")
    assert len(store.load_score_history(10, db_path=db)) == 2, "new turn was swallowed"


def test_turn_id_is_what_makes_a_turn_repeatable(tmp_path):
    """Hermes always passes turn_id (agent/turn_context.py); that is the dedupe contract."""
    mod = load_plugin()
    ctx = _ctx(tmp_path)
    db = ctx.config["engine_db_path"]
    store.set_db_path(db)
    mod.register(ctx)
    hook = ctx.hooks["pre_llm_call"]

    first = hook(session_id="s-nodup", user_message="你真棒，太厉害了！", turn_id="t-1")
    second = hook(session_id="s-nodup", user_message="你真棒，太厉害了！", turn_id="t-1")
    assert first == second, "same turn_id produced a second injection"
    assert len(store.load_score_history(10, db_path=db)) == 1


def test_final_change_never_exceeds_turn_clamp():
    """Decay + noise run after the stimulus clamp, so the final movement must be clamped."""
    prev = {"aff": -20.0, "sec": 10.0, "aro": 0.0, "dom": 0.0}
    event = {"type": "adult_submissive", "intensity": 1.0, "sincerity": 1.0}
    out = EE.emotion_step(event, {}, dict(prev), adult_mode=True, sensitivity=0)
    for k in ("aff", "sec", "aro", "dom"):
        assert abs(out[k] - prev[k]) <= EE.SINGLE_TURN_CLAMP + 1e-6, k


def test_final_clamp_catches_decay_plus_stimulus():
    """Discriminating case: decay AND a saturated stimulus both push the same way.

    A saturated negative stimulus (-10) plus decay pulling +50 down by 1.0 gives -11, so
    deleting the final clamp makes this assertion fail.
    """
    prev = {"aff": 50.0, "sec": 20.0, "aro": 0.0, "dom": 0.0}
    event = {"type": "hurtful", "intensity": 1.0, "sincerity": 1.0}
    out = EE.emotion_step(event, {}, dict(prev), amplitude=6.0)
    change = out["aff"] - prev["aff"]
    assert change == -EE.SINGLE_TURN_CLAMP, f"expected the clamp to bite, got {change}"
    for k in ("aff", "sec", "aro", "dom"):
        assert abs(out[k] - prev[k]) <= EE.SINGLE_TURN_CLAMP + 1e-6, k


def test_final_clamp_holds_where_noise_is_active():
    """|score| > NOISE_THRESHOLD_ABS adds noise after the stimulus clamp: still bounded."""
    prev = {"aff": 80.0, "sec": 70.0, "aro": 60.0, "dom": 0.0}
    event = {"type": "praise", "intensity": 1.0, "sincerity": 1.0}
    for turn_index in range(6):
        out = EE.emotion_step(
            event, {"atmosphere": "warm"}, dict(prev),
            session_id="noise", turn_index=turn_index,
        )
        for k in ("aff", "sec", "aro", "dom"):
            assert abs(out[k] - prev[k]) <= EE.SINGLE_TURN_CLAMP + 1e-6, (k, turn_index)


def test_normal_short_replies_are_not_cold():
    for msg in ("哈哈", "没事", "都行"):
        assert EE.interpret_event(msg)["type"] != "cold", msg


def _adult_state_after(tmp_path, message):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mod = load_plugin()
    # use the plugin's own module instance: importing engine.adult_mode separately would
    # hand the assertion a different state machine than the one the hook mutates
    adult_mode = mod.adult_mode
    adult_mode.reset_adult_mode()
    ctx = _ctx(tmp_path, adult_enabled=True)
    store.set_db_path(ctx.config["engine_db_path"])
    mod.register(ctx)
    adult_mode.get_adult_state().state = "FLIRTING"
    ctx.hooks["pre_llm_call"](session_id="s-adult", user_message=message, turn_id="a1")
    return adult_mode.get_adult_state().state


def test_criticism_knocks_adult_state_back_to_normal(tmp_path):
    """Driven through the real hook: criticism must reach the state machine as negative."""
    assert _adult_state_after(tmp_path / "c", "你根本不懂我在说什么") == "NORMAL"
    assert _adult_state_after(tmp_path / "p", "你真棒，太厉害了！") != "NORMAL"
