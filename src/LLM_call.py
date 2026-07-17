#!/usr/bin/env python3
"""LLM driver: builds a compact prompt from context_v5.toon + run_history.json,
calls `claude -p --output-format json` with that prompt, and writes sanitized
Tcl ECO commands to llm_eco.tcl.

All per-iteration artifacts (prompt, raw reply + thinking, token counts, tool
calls) are logged under prompts/dynamic/claude_logs/.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # SDK is optional; only required when call_claude_api() is used

ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_GUIDE_FILE = ROOT_DIR / "AGENTS.md"
FEW_SHOT_FILE = ROOT_DIR / "prompts" / "static" / "few_shot" / "few_shot.toon"
CELL_DELAY_REFERENCE_FILE = ROOT_DIR / "prompts" / "static" / "cell_delay_reference.toon"
BACKTRACE_NOTICE_FILE = ROOT_DIR / "prompts" / "dynamic" / "backtrace_notice.json"
LLM_SESSION_LOG = ROOT_DIR / "prompts" / "dynamic" / "llm_session.log"
DYNAMIC_CONTEXT_FILE = ROOT_DIR / "prompts" / "dynamic" / "context.json"
DYNAMIC_TOON_FILE = ROOT_DIR / "prompts" / "dynamic" / "context.toon"
HIST_FILE = ROOT_DIR / "prompts" / "history.json"
ECO_OUT = ROOT_DIR / "prompts" / "dynamic" / "llm_eco.tcl"
ANALYSIS_OUT = ROOT_DIR / "prompts" / "dynamic" / "llm_analysis.md"
PREV_TARGET_PATH_FILE = ROOT_DIR / "prompts" / "dynamic" / "prev_target_path_status.json"
PROBE_RESPONSES_FILE = ROOT_DIR / "prompts" / "dynamic" / "probe_responses.txt"
DRT_STATE_FILE = ROOT_DIR / "prompts" / "dynamic" / "drt_state.json"
RPT_PATH = ROOT_DIR / "prompts" / "dynamic" / "dynamic_timing_rpt.txt"
PROMPT_DUMP = ROOT_DIR / "prompts" / "dynamic" / "last_prompt.txt"
DEFAULT_CLAUDE_LOG_DIR = ROOT_DIR / "prompts" / "dynamic" / "claude_logs"

# run_history.json is the single LLM-facing iteration log. eco_history.json
# and qor_history.json are no longer written.
RUN_HISTORY_FILE = ROOT_DIR / "prompts" / "run_history.json"

MAX_COMMANDS = 5

ALLOWED_PREFIXES = {
    "make_net", "make_instance", "connect_pin", "disconnect_pin",
    "replace_cell", "insert_buffer", "remove_buffers", "set",
    # ECO buffer procs (sourced into OpenROAD session)
    "eco_insert_buffer", "eco_insert_buffer_midpoint",
    "eco_insert_buffer_optimal_alpha", "eco_remove_buffer",
    "eco_buffer_driver_fanout", "eco_buffer_sink_cluster",
    # Gate resize / clone / fanout analysis
    "eco_resize_gate", "eco_clone_gate", "eco_clone_gate_worst_half",
    "eco_rank_fanout_by_slack", "eco_dump_path_siblings",
    "eco_dump_fanout_ranks", "eco_top_paths_through",
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate used before Claude returns real usage numbers."""
    words = len(text.split())
    chars = len(text)
    return max(int(words / 0.75), chars // 4)


def load_json(path: Path):
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"  [WARN] load_json: {path.name} is not valid JSON ({e}); returning {{}}", flush=True)
        return {}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def load_history():
    if not HIST_FILE.exists():
        return []
    try:
        raw = HIST_FILE.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return []


def save_history(hist):
    HIST_FILE.write_text(json.dumps(hist, indent=2), encoding="utf-8")


def _ts_for_path() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def call_claude(system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Call `claude -p --output-format json` with system + user split.

    `system_prompt` is passed via `--append-system-prompt` so it lands in the
    real system slot (better for caching across fresh sessions and harder for
    in-band content to override). `user_prompt` is sent on stdin.

    Returns (reply_text, meta) where meta contains the raw Claude JSON
    (usage tokens, num_turns, duration, cost) plus local timing.
    """
    sys_tokens_est = estimate_tokens(system_prompt)
    usr_tokens_est = estimate_tokens(user_prompt)
    print(
        f"  [LLM] Sending: system={len(system_prompt)}c (~{sys_tokens_est}t) "
        f"user={len(user_prompt)}c (~{usr_tokens_est}t)",
        flush=True,
    )

    #cmd = ["claude", "-p", "--output-format", "json"]
    cmd = ["claude", "-p", "--output-format", "json", "--model", "claude-sonnet-4-6"]
    if system_prompt.strip():
        cmd += ["--append-system-prompt", system_prompt]
    ts = _ts_for_path()
    meta: Dict[str, Any] = {
        "timestamp": ts,
        "cmd": [c if c is not system_prompt else f"<system_prompt:{len(system_prompt)}c>" for c in cmd],
        "system_chars": len(system_prompt),
        "user_chars": len(user_prompt),
        "prompt_chars": len(system_prompt) + len(user_prompt),
        "prompt_tokens_est": sys_tokens_est + usr_tokens_est,
    }

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, input=user_prompt, text=True, capture_output=True, check=True,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        meta.update({"elapsed_s": elapsed, "returncode": e.returncode,
                     "stderr_tail": (e.stderr or "").splitlines()[-5:]})
        print(f"  [LLM] ERROR: claude returned {e.returncode} after {elapsed:.1f}s", flush=True)
        raise

    elapsed = time.time() - t0
    meta.update({"elapsed_s": elapsed, "returncode": returncode})
    if stderr_text:
        for line in stderr_text.strip().splitlines()[:5]:
            print(f"  [LLM stderr] {line}", flush=True)

    # Parse the JSON envelope. Strip any non-JSON leading warning lines that
    # Claude prints (e.g. "⚠ Sandbox disabled: ...") so the last "{" starts a
    # valid JSON block.
    envelope: Dict[str, Any] = {}
    stdout_stripped = stdout_text.strip()
    first_brace = stdout_stripped.find("{")
    if first_brace >= 0:
        try:
            envelope = json.loads(stdout_stripped[first_brace:])
        except json.JSONDecodeError:
            envelope = {}
    if not envelope:
        print("  [LLM] WARNING: could not parse claude JSON envelope, treating stdout as text", flush=True)
        reply = stdout_stripped
        meta["raw_stdout"] = stdout_text
    else:
        reply = envelope.get("result", "") or ""
        meta["claude_envelope"] = envelope
        usage = envelope.get("usage", {}) or {}
        meta["usage_reported"] = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        }
        meta["num_turns"] = envelope.get("num_turns")
        meta["duration_ms"] = envelope.get("duration_ms")
        meta["total_cost_usd"] = envelope.get("total_cost_usd")
        meta["session_id"] = envelope.get("session_id")

    print(f"  [LLM] Reply: {len(reply)} chars, {elapsed:.1f}s "
          f"[in={meta.get('usage_reported',{}).get('input_tokens')}, "
          f"out={meta.get('usage_reported',{}).get('output_tokens')}]", flush=True)
    meta["reply_chars"] = len(reply)
    return reply, meta




def append_llm_session_log(iteration, system_prompt, user_prompt, reply_raw, meta, label="call"):
    """Append a fully-unfiltered record of one Claude call (system + user
    prompts, raw reply, token counts) to LLM_SESSION_LOG. This is the
    single chronological view of what the LLM saw and said across the run."""
    try:
        LLM_SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
        usage = meta.get("usage_reported", {}) if isinstance(meta, dict) else {}
        with LLM_SESSION_LOG.open("a", encoding="utf-8") as fp:
            fp.write("\n" + "=" * 72 + "\n")
            fp.write(f"=== ITER {iteration} | {label} | {_ts_for_path()}\n")
            fp.write(f"=== prompt_chars={meta.get('prompt_chars')} "
                     f"reply_chars={meta.get('reply_chars')} "
                     f"elapsed_s={meta.get('elapsed_s')}\n")
            fp.write(f"=== usage={usage}\n")
            fp.write("=" * 72 + "\n")
            fp.write("\n>>> SYSTEM PROMPT (full)\n")
            fp.write(system_prompt or "")
            fp.write("\n\n>>> USER PROMPT (full)\n")
            fp.write(user_prompt or "")
            fp.write("\n\n>>> RAW REPLY (UNFILTERED, no sanitization)\n")
            fp.write(reply_raw or "")
            fp.write("\n")
            fp.flush()
    except OSError as e:
        print(f"  [WARN] llm_session.log write failed: {e}", flush=True)

def _load_run_history(max_iters: int = 8) -> list:
    """Last N iterations with WNS, score, outcome, and commands — lets the
    LLM attribute WNS changes to specific commands from prior iterations."""
    try:
        data = json.loads(RUN_HISTORY_FILE.read_text())
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    trimmed = []
    for rec in data[-max_iters:]:
        trimmed.append({
            "iteration": rec.get("iteration"),
            "wns": rec.get("wns"),
            "tns": rec.get("tns"),
            "score": rec.get("score"),
            "outcome": rec.get("outcome"),
            "eco_error": rec.get("eco_error"),
            "commands": rec.get("commands", []),
            "analysis": rec.get("analysis", "") or "",
            "predicted_delta_ps": rec.get("predicted_delta_ps"),
        })
    return trimmed


def sanitize_tcl_output(text: str) -> str:
    """Strip markdown fences, blank lines, and prose. A line is kept only if
    it starts with `#` or whose first token is in ALLOWED_PREFIXES."""
    kept: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("#"):
            kept.append(line)
            continue
        first_tok = line.split(None, 1)[0]
        if first_tok in ALLOWED_PREFIXES:
            kept.append(line)
    return "\n".join(kept)


def _parse_predicted_delta_ps_from_analysis(analysis_text: str):
    """Backfill helper: extract the first Move plan row's predicted
    Δslack (in ps) from the model's analysis. Mirrors main_orch's
    parse_predicted_delta_ps — duplicated here so LLM_call.py can render
    <prediction_calibration> for older run_history records that pre-date
    the predicted_delta_ps field. Returns None if not parseable."""
    if not analysis_text:
        return None
    import re as _re
    in_move_plan = False
    for raw in analysis_text.splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("### move plan") or low.startswith("### ranked"):
            in_move_plan = True
            continue
        if line.startswith("### ") and in_move_plan:
            break
        if not in_move_plan or not line.startswith("|"):
            continue
        if "---" in line or "δslack" in low or ("delta" in low and "est" in low):
            if "ps" not in low:
                continue
        m = _re.search(r"[~+\-]?(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*ps", line)
        if m:
            try:
                return (float(m.group(1)) + float(m.group(2))) / 2.0
            except ValueError:
                pass
        m = _re.search(r"([~+\-]?)(\d+(?:\.\d+)?)\s*ps", line)
        if m:
            sign = -1.0 if m.group(1) == "-" else 1.0
            try:
                v = sign * float(m.group(2))
                if -100 < v < 100:
                    return v
            except ValueError:
                pass
    return None


def extract_analysis(reply_raw: str) -> str:
    """Return the structured-analysis portion of the reply: everything
    before the first ```tcl fence. Empty if the reply has no Analysis
    marker (the model fell back to bare tcl)."""
    if not reply_raw:
        return ""
    txt = reply_raw
    idx = txt.find("```tcl")
    if idx != -1:
        txt = txt[:idx]
    txt = txt.strip()
    low = txt.lower()
    if "### analysis" not in low and "### move plan" not in low:
        return ""
    return txt


def parse_commands(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def extract_recent_commands(hist, n_turns=3) -> List[str]:
    cmds = []
    for t in hist[-n_turns:]:
        cmds.extend(parse_commands(t.get("assistant", "")))
    return cmds[-30:]


def validate_reply(
    reply: str,
    recent_cmds: List[str],
    allow_insts: set[str],
    buf_inv_masters: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """Sanity check the Tcl block before it reaches OpenROAD."""
    cmds = parse_commands(reply)
    if not cmds:
        return False, "empty output — no Tcl commands found"

    known = ALLOWED_PREFIXES | {"remove_buffers"}
    buf_inv_masters = buf_inv_masters or {}

    warnings = []
    for cmd in cmds:
        code_part = cmd.split("#", 1)[0].strip() if "#" in cmd else cmd
        toks = code_part.split()
        if not toks:
            continue
        head = toks[0]
        if head not in known:
            warnings.append(f"BLOCKED: unknown command '{head}' — not in allowed proc set")
            continue
        if head == "eco_buffer_driver_fanout" and len(toks) >= 2:
            net_name = toks[1]
            if "_eco_" in net_name:
                warnings.append(
                    f"BLOCKED: chaining buffer on eco net '{net_name}' — use eco_remove_buffer instead"
                )
        # Arg-count checks
        if head == "eco_insert_buffer" and len(toks) < 5:
            warnings.append(
                f"BLOCKED: eco_insert_buffer needs >= 4 positional args "
                f"(net driver_pin buf_cell buf_name) plus optional "
                f"-sinks {{...}} / -at driver|centroid|{{x y}}; got {len(toks)-1}"
            )
        if head == "eco_buffer_driver_fanout" and len(toks) != 5:
            warnings.append(f"BLOCKED: eco_buffer_driver_fanout expects 4 args; got {len(toks)-1}")
        if head == "eco_insert_buffer_midpoint" and len(toks) != 6:
            warnings.append(f"BLOCKED: eco_insert_buffer_midpoint expects 5 args; got {len(toks)-1}")
        if head == "eco_remove_buffer" and len(toks) != 2:
            warnings.append(f"BLOCKED: eco_remove_buffer expects 1 arg; got {len(toks)-1}")
        if head == "remove_buffers":
            for tgt in toks[1:]:
                master = buf_inv_masters.get(tgt, "")
                if master.startswith("INV"):
                    warnings.append(
                        f"BLOCKED: remove_buffers on INV cell '{tgt}' (master={master}) "
                        f"— inverters cannot be removed; use replace_cell to VT-swap instead"
                    )

    if warnings:
        return False, "; ".join(warnings)
    return True, ""


def build_prompt(static_guide: str, recent_cmds: List[str]) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) using XML-tagged sections.

    system_prompt = AGENTS.md (durable strategy guide — caches across fresh
    sessions when sent via --append-system-prompt or the API system slot).
    user_prompt   = per-iteration TOON + trajectory + recent cmds + final
    instruction, each wrapped in an XML tag for unambiguous boundaries.

    Inner content formats are unchanged: TOON stays TOON, JSON stays JSON,
    Markdown stays Markdown — XML is only the outer skeleton.
    """
    # The agent guide is the bulk of the system prompt; wrap it so the
    # model sees it as a single tagged block. System prompt =
    #   AGENTS.md (strategy)
    # + <cell_delay_reference> (Liberty NLDM tables for the technology)
    # + <few_shot_examples> (worked engineering walkthrough).
    # ALL of this is cached after the first iter (prefix cache, ~24-40k tokens).
    few_shot = load_text(FEW_SHOT_FILE).strip()
    cell_ref = load_text(CELL_DELAY_REFERENCE_FILE).strip()
    parts: list[str] = ["<system_prompt>", static_guide]
    if cell_ref:
        parts.append("<cell_delay_reference>\n" + cell_ref + "\n</cell_delay_reference>")
    if few_shot:
        parts.append("<few_shot_examples>\n" + few_shot + "\n</few_shot_examples>")
    parts.append("</system_prompt>")
    system_prompt = "\n".join(parts)

    sections: List[str] = []

    # 1. Per-iteration TOON (top-3 paths, siblings, fanout ranks, recent ECOs).
    toon_text = load_text(DYNAMIC_TOON_FILE)
    if toon_text.strip():
        sections.append(
            "<iteration_context format=\"toon\">\n"
            f"{toon_text.strip()}\n"
            "</iteration_context>"
        )

    # 1a. Parasitic context — tells the LLM whether the timing numbers above
    #     came from actual extracted SPEF (just after DRT) or from GRT estimates
    #     (within the 8-iter cycle). Includes GRT calibration from last DRT so
    #     the model can derate its predictions accordingly.
    if DRT_STATE_FILE.exists():
        try:
            drt_state = json.loads(DRT_STATE_FILE.read_text(encoding="utf-8"))
            src = drt_state.get("parasitic_source", "unknown")
            cyc = drt_state.get("cycle_num", 0)
            iic = drt_state.get("iter_in_cycle", 0)
            until_drt = drt_state.get("iters_until_drt", "?")
            cur_iter = drt_state.get("current_iter", "?")
            last_drt = drt_state.get("last_drt") or {}

            ldw = last_drt.get("post_drt_wns_ps")
            ldt = last_drt.get("post_drt_tns_ps")
            pgw = last_drt.get("pre_drt_grt_wns_ps")
            pgt = last_drt.get("pre_drt_grt_tns_ps")
            ew  = last_drt.get("grt_error_wns_ps")
            et  = last_drt.get("grt_error_tns_ps")
            lat = last_drt.get("at_iter", "?")
            lac = last_drt.get("cycle", "?")

            # source note is DATA-DRIVEN from the last DRT's measured grt_error
            # sign — NOT hardcoded "optimistic". grt_error = pre_DRT_grt - post_DRT;
            # negative => GRT pessimistic (DRT lands BETTER), positive => optimistic.
            if src == "SPEF_extracted":
                src_note = (
                    "SPEF_extracted — actual post-route RC from the last DRT "
                    "or baseline 6_final.odb load. Delays are accurate (~2-5% "
                    "error). Trust the numbers; they will correlate well with "
                    "final DRC-clean timing."
                )
            elif ew is None:
                src_note = (
                    "GRT_estimate — global-routing wire RC estimates applied "
                    "after the last ECO iteration; no DRT calibration measured "
                    "yet. Treat wire-dominated Δslack as +/-15% uncertain "
                    "until the first DRT lands."
                )
            elif ew < 0:
                src_note = (
                    "GRT_estimate — global-routing wire RC estimates. On THIS "
                    f"design the last DRT measured GRT to be {abs(ew):.0f} ps "
                    "PESSIMISTIC on WNS (detailed routing came out BETTER than "
                    f"GRT). Expect your true post-route WNS to land ~{abs(ew):.0f} "
                    "ps LESS negative than the GRT numbers above. Do NOT "
                    "over-derate gains — your real headroom is LARGER than GRT shows."
                )
            elif ew > 0:
                src_note = (
                    "GRT_estimate — global-routing wire RC estimates. On THIS "
                    f"design the last DRT measured GRT to be {ew:.0f} ps "
                    "OPTIMISTIC on WNS (detailed routing came out WORSE than "
                    f"GRT). Expect your true post-route WNS to land ~{ew:.0f} ps "
                    "MORE negative than the GRT numbers above. Derate "
                    "wire-dominated Δslack gains accordingly."
                )
            else:
                src_note = (
                    "GRT_estimate — GRT matched DRT on the last cycle "
                    "(near-zero error); treat GRT numbers as reliable."
                )

            lines = [
                "<parasitic_context>",
                f"  source         : {src_note}",
                f"  cycle_num      : {cyc}  "
                f"(DRT cycles completed so far)",
                f"  iter_in_cycle  : {iic + 1}/8  "
                f"(this is ECO iter {cur_iter}; first of cycle = SPEF, "
                f"rest = GRT estimate)",
                f"  iters_until_drt: {until_drt}  "
                f"(next DRT refresh fires after {until_drt} more ECO iters)",
            ]

            if last_drt:
                # grt_accuracy = |SPEF / GRT| using the REAL pre-DRT GRT estimate
                # (pgw/pgt) as denominator. The previous code used (ldw - ew) =
                # 2*ldw - pgw, which inverted the ratio (e.g. printed 107.8% when
                # the true |SPEF/GRT| was 93.3%).
                wns_acc = (
                    f"{abs(ldw / pgw) * 100:.1f}%"
                    if pgw not in (None, 0, 0.0) and ldw not in (None, 0, 0.0) else "n/a"
                )
                tns_acc = (
                    f"{abs(ldt / pgt) * 100:.1f}%"
                    if pgt not in (None, 0, 0.0) and ldt not in (None, 0, 0.0) else "n/a"
                )
                ew_sign = "optimistic (DRT worse)" if (ew or 0) > 0 else "pessimistic (DRT better)"
                et_sign = "optimistic (DRT worse)" if (et or 0) > 0 else "pessimistic (DRT better)"

                lines += [
                    f"  last_drt_calibration (cycle={lac}, after ECO iter {lat}):",
                    f"    post_DRT_actual : WNS={ldw} ps   TNS={ldt} ps  "
                    f"[SPEF extracted — ground truth]",
                    f"    pre_DRT_grt_est : WNS={pgw} ps   TNS={pgt} ps  "
                    f"[GRT estimate — iter {lat}]",
                    (f"    grt_error       : WNS={ew:+.1f} ps {ew_sign}   "
                     f"TNS={et:+.1f} ps {et_sign}"
                     if ew is not None and et is not None
                     else "    grt_error       : not yet available"),
                    f"    grt_accuracy    : WNS≈{wns_acc}   TNS≈{tns_acc}  (|SPEF/GRT|)",
                    "    → Use grt_error as an ADDITIVE offset, NOT a multiplier on Δslack:",
                    (f"      expected post-route WNS ≈ (GRT WNS shown) − ({ew:+.1f}) ps"
                     if ew is not None
                     else "      expected post-route WNS ≈ GRT WNS (no offset yet)"),
                    "      Budget your stop criterion against this DRT-corrected "
                    "value, not raw GRT. Cell-dominated stages track GRT well "
                    "(±5%); wire-dominated stages carry most of the offset.",
                ]
            else:
                lines.append(
                    "  last_drt_calibration: none yet — this is the first cycle. "
                    "Baseline SPEF from 6_final.odb is in use. Apply standard "
                    "ECO derating (cell-delay Liberty ±5%, wire-delay GRT ±15%)."
                )

            lines.append("</parasitic_context>")
            sections.append("\n".join(lines))
        except Exception as _e:
            print(f"  [WARN] parasitic_context build failed: {_e}", flush=True)

    # 1b. Backtrace notice (one-shot): if the orchestrator just rolled
    #     back to best snapshot, surface that to the LLM with last-10 moves.
    if BACKTRACE_NOTICE_FILE.exists():
        try:
            notice = json.loads(BACKTRACE_NOTICE_FILE.read_text())
            rolled_back = notice.get("last_iterations", []) or []
            # Histogram the rolled-back move types so the model sees its
            # past pattern explicitly (e.g. "10/10 iters were resize-only").
            type_counter = {}
            for it in rolled_back:
                for cmd in (it.get("commands") or []):
                    head = (cmd or "").split()[:1]
                    if head:
                        type_counter[head[0]] = type_counter.get(head[0], 0) + 1
            type_summary = ", ".join(
                f"{k}={v}" for k, v in sorted(type_counter.items(),
                                              key=lambda kv: -kv[1])
            ) or "(no commands recorded)"
            sections.append(
                "<backtrace_notice>\n"
                f"You were just BACKTRACED to iteration {notice.get('best_iter')} "
                f"(WNS={notice.get('best_wns')}). The last 10 moves degraded WNS, "
                f"so the orchestrator restored the best-ever .odb snapshot. The "
                f"path/cell state above reflects that restored snapshot — NOT "
                f"the state your last move produced.\n\n"
                f"Move-type histogram of the {len(rolled_back)} rolled-back iters:\n"
                f"  {type_summary}\n\n"
                "MANDATORY NEXT-ITER CONSTRAINT (this iter only):\n"
                "  If the histogram above is dominated by eco_resize_gate (>=80% of\n"
                "  rolled-back commands), your ### Tcl recipe THIS iter MUST contain\n"
                "  AT LEAST ONE of:\n"
                "    - eco_clone_gate_worst_half  (on a shared-prefix high-fanout driver)\n"
                "    - eco_insert_buffer          (on a wire-dominated stage)\n"
                "    - eco_remove_buffer          (to undo a bad earlier insertion from\n"
                "                                  <recent_eco_actions>, if any)\n"
                "  Repeating resize-only after a backtrace will re-enter the same\n"
                "  losing trajectory. Diversify the move type, not just the targets.\n\n"
                "Rolled-back iterations (for attribution):\n"
                f"{json.dumps(rolled_back, indent=2)}\n"
                "</backtrace_notice>"
            )
            BACKTRACE_NOTICE_FILE.unlink()  # consume once
        except Exception as e:
            print(f"  [WARN] backtrace_notice unreadable: {e}", flush=True)

    # 2. Per-iteration trajectory — for attribution + revert logic.
    traj = _load_run_history(max_iters=8)
    if traj:
        sections.append(
            "<iteration_trajectory format=\"json\">\n"
            "<note>Per-iteration WNS/TNS/outcome + the exact commands that iteration applied. "
            "If outcome=degraded, those commands CAUSED the drop — do NOT re-emit the same "
            "targets. If outcome=reverted_bad_eco, that iteration was rolled back. "
            "Analysis text is shown separately in <recent_analysis> for the last 2 iters.</note>\n"
            f"{json.dumps([{k: v for k, v in t.items() if k != 'analysis'} for t in traj], indent=2)}\n"
            "</iteration_trajectory>"
        )

    # 3. Commands to avoid repeating this turn.
    sections.append(
        "<recent_commands format=\"json\">\n"
        f"{json.dumps(recent_cmds, indent=2)}\n"
        "</recent_commands>"
    )

    # 3b. Recent analyses (the model's own ### Analysis / ### Move plan text
    #     from the last 2 iterations). Lets the model build on its prior
    #     reasoning rather than rediscover the same bottlenecks each turn.
    if traj:
        recent_anal = [
            {"iteration": t.get("iteration"), "analysis": t.get("analysis", "")}
            for t in traj[-2:] if (t.get("analysis") or "").strip()
        ]
        if recent_anal:
            blocks = []
            for t in recent_anal:
                blocks.append(
                    f"# iter {t['iteration']} analysis\n{t['analysis']}"
                )
            sections.append(
                "<recent_analysis>\n"
                "<note>Your own ### Analysis / ### Move plan from the last 2 iterations. "
                "Reference these directly: name which bottlenecks you addressed and which "
                "remain; do NOT restate; build on them.</note>\n"
                + "\n\n".join(blocks) + "\n"
                "</recent_analysis>"
            )

    # 3c. Prediction calibration — show the model its own predicted Δslack
    #     vs actual ΔWNS for the last up to 8 iters that have both.
    if traj:
        cal_rows = []
        for i, t in enumerate(traj):
            pred = t.get("predicted_delta_ps")
            if pred is None:
                # Backfill from analysis text (handles older run_history)
                pred = _parse_predicted_delta_ps_from_analysis(t.get("analysis", ""))
            if pred is None:
                continue
            # actual ΔWNS = curr.wns - prev.wns (prev is traj[i-1] or absent)
            if i == 0:
                continue
            prev = traj[i - 1]
            if t.get("wns") is None or prev.get("wns") is None:
                continue
            actual = round(t["wns"] - prev["wns"], 3)
            cmds = t.get("commands") or []
            cmd_summary = cmds[0][:80] if cmds else "(no commands)"
            cal_rows.append({
                "iter": t.get("iteration"),
                "cmd": cmd_summary,
                "predicted_ps": pred,
                "actual_ps": actual,
                "error_ps": round(actual - float(pred), 3),
            })
        if cal_rows:
            # Per-row classification: extract the dominant move type AND the
            # batch size. Bucketed ratios surface batch-interaction effects
            # the model can act on (e.g. "single-move resize is well calibrated,
            # but 4-move same-cone batches predict 25× over due to WNS-saturation").
            type_buckets: Dict[str, List[float]] = {}
            cumulative_pred = 0.0
            cumulative_actual = 0.0
            for r in cal_rows:
                cmd = r.get("cmd") or ""
                head = cmd.split()[0] if cmd.split() else ""
                # Categorize: structural vs sizing; record batch-size group
                cmds_full = (traj[next((i for i, t in enumerate(traj)
                                        if t.get("iteration") == r.get("iter")), 0)]
                             .get("commands") or [])
                n = len(cmds_full)
                # Skip `set` / `eco_rank_fanout_by_slack` (housekeeping; no slack impact)
                # when computing the bucket key, but keep the row in the dump
                if head in ("set", "eco_rank_fanout_by_slack",
                            "eco_top_paths_through", "eco_net_sink_report"):
                    bucket = f"{head} (probe)"
                else:
                    batch_tag = "single" if n <= 1 else ("small_batch_2-3" if n <= 3
                                                          else "batch_4+")
                    bucket = f"{head} ({batch_tag})"
                if r["predicted_ps"] and abs(r["predicted_ps"]) > 0.5:
                    ratio = float(r["actual_ps"]) / float(r["predicted_ps"])
                    type_buckets.setdefault(bucket, []).append(ratio)
                cumulative_pred += float(r.get("predicted_ps") or 0)
                cumulative_actual += float(r.get("actual_ps") or 0)

            # Emit the buckets sorted by sample size (more data → more trusted)
            by_type_lines: List[str] = []
            for bucket, ratios in sorted(type_buckets.items(),
                                          key=lambda kv: (-len(kv[1]), kv[0])):
                mean_r = sum(ratios) / len(ratios)
                spread = max(ratios) - min(ratios) if len(ratios) > 1 else 0.0
                trust = "trusted" if len(ratios) >= 3 else (
                    "thin (1-2 samples)" if len(ratios) >= 1 else "no data")
                by_type_lines.append(
                    f"  {bucket}:  n={len(ratios)}  mean_ratio={mean_r:+.3f}  "
                    f"spread={spread:.2f}  [{trust}]"
                )

            # Build the cumulative-vs-actual headline (the single most useful
            # number for the model to see — "you predicted +733, got +0.28").
            cum_ratio = (cumulative_actual / cumulative_pred) if abs(cumulative_pred) > 0.5 else None
            cum_line = (f"cumulative_predicted={cumulative_pred:+.1f} ps   "
                        f"cumulative_actual={cumulative_actual:+.2f} ps   "
                        f"overall_ratio={cum_ratio:+.3f}" if cum_ratio is not None
                        else "")

            # Interaction insight: surface a one-paragraph diagnosis the model
            # should apply to NEW predictions. Heuristics based on observed bucket.
            insights: List[str] = []
            for bucket, ratios in type_buckets.items():
                mean_r = sum(ratios) / len(ratios)
                if mean_r > 2.0 and len(ratios) >= 1:
                    insights.append(
                        f"  - {bucket}: actual ΔWNS ran ~{mean_r:.0f}x your prediction\n"
                        f"    (you UNDER-predicted). Before treating this as 'be less\n"
                        f"    conservative', check whether it was a SHARED-PREFIX / series\n"
                        f"    move: if so the miss is a SATURATION-MODEL error (the parallel-\n"
                        f"    path gap cap was wrongly applied to a move that lifts many tied\n"
                        f"    paths) — fix the model (shared-prefix exemption), do NOT inflate\n"
                        f"    predictions globally. Keep this correction in THIS bucket only.")
                if "batch_4+" in bucket and mean_r < 0.2 and len(ratios) >= 3:
                    insights.append(
                        f"  - {bucket}: predictions sum per-move deltas as if independent, but\n"
                        f"    actual WNS shift is ~{mean_r*100:.0f}% of that sum. WNS is\n"
                        f"    bounded by the worst path — once that path is fixed, subsequent\n"
                        f"    moves help only TNS, not WNS. Either (a) derate batch-4+\n"
                        f"    predictions by {1/max(mean_r,0.01):.0f}× for WNS estimation, OR\n"
                        f"    (b) spread moves across multiple cones so each fix targets a\n"
                        f"    different WNS-candidate endpoint.")
                if "single" in bucket and mean_r < 0 and len(ratios) >= 1:
                    insights.append(
                        f"  - {bucket}: single-move attempt regressed (ratio {mean_r:+.2f}).\n"
                        f"    THIS move type needs a what-if check before another attempt,\n"
                        f"    not blind re-try. If no query tool available, ESCALATE to a\n"
                        f"    different move type (or different target cone).")
                if "single" in bucket and mean_r > 0.3 and len(ratios) >= 2:
                    insights.append(
                        f"  - {bucket}: single-move attempts are well-calibrated\n"
                        f"    (ratio {mean_r:+.2f}). Predictions in this bucket can be trusted\n"
                        f"    near face value; only mild derating needed.")

            sections.append(
                "<prediction_calibration>\n"
                "<note>Per-bucket actual/predicted ratios. Buckets split by move type\n"
                "and batch size because they have systematically different accuracy:\n"
                "single-move predictions tend to be well-calibrated; large batches on\n"
                "the same cone over-predict due to WNS-saturation; structural moves\n"
                "(clone/insert/remove) need separate calibration because their physics\n"
                "is non-local.</note>\n"
                "\n"
                "<by_move_type_and_batch_size>\n"
                + "\n".join(by_type_lines) + "\n"
                "</by_move_type_and_batch_size>\n"
                "\n"
                + (f"<cumulative>\n{cum_line}\n</cumulative>\n\n"
                   if cum_line else "")
                + ("<interaction_insights>\n"
                   + "\n".join(insights) + "\n"
                   "</interaction_insights>\n\n" if insights else "")
                + "<full_data>\n"
                + json.dumps(cal_rows, indent=2) + "\n"
                + "</full_data>\n"
                + "</prediction_calibration>"
            )

    # 3d. Previous-target-path status: after last iter's ECO, the orchestrator
    #     re-ran report_checks on the path the model was targeting.
    #     Lets the model see EXACTLY how its move changed the targeted path
    #     (stage-level Δdelay, including upstream cascade if any).
    if PREV_TARGET_PATH_FILE.exists():
        try:
            ptp = json.loads(PREV_TARGET_PATH_FILE.read_text(encoding="utf-8"))
            sections.append(
                "<previous_target_path>\n"
                "<note>The path your last iter targeted (path 1 from the prior context), "
                "re-reported AFTER your ECO commands were applied. Compare 'before' and "
                "'after' rows per stage to see exactly which stage's delay changed, and "
                "whether the upstream driver's load (and delay) increased as a side "
                "effect of your move (cloning a driver doubles the upstream load).</note>\n"
                + json.dumps(ptp, indent=2) + "\n"
                "</previous_target_path>"
            )
        except Exception as e:
            print(f"  [WARN] prev_target_path unreadable: {e}", flush=True)

    # 3d.5 Stall diagnostic — if last iter's ECO commands produced |ΔWNS| < 0.05 ps
    #      on the targeted path, force the model into a probe-or-pivot mode
    #      before letting it propose another move on the same target.
    #      Catches the resize-plateau pattern where the model keeps emitting
    #      the same move type on the same cone without progress.
    if PREV_TARGET_PATH_FILE.exists():
        try:
            _ptp = json.loads(PREV_TARGET_PATH_FILE.read_text(encoding="utf-8"))
            _delta = _ptp.get('delta_ps')
            _cmds = _ptp.get('commands_applied') or []
            # Filter out housekeeping / probe commands so we only count
            # commands that SHOULD have moved slack.
            _real = [c for c in _cmds
                     if c.strip() and not c.strip().startswith('#')
                     and (c.split() or [''])[0] not in (
                         'set', 'eco_rank_fanout_by_slack',
                         'eco_top_paths_through', 'eco_net_sink_report')]
            if _delta is not None and abs(_delta) < 0.05 and len(_real) >= 1:
                sections.append(
                    "<stall_diagnostic>\n"
                    f"<note>Your last iter applied {len(_real)} ECO command(s) on the "
                    f"worst path and produced ΔWNS = {_delta:+.3f} ps on that targeted "
                    "path (see <previous_target_path>). This is a stall — the moves "
                    "either did not apply, hit a sub-resolution limit, or were "
                    "cancelled by an upstream cascade.</note>\n"
                    "\n"
                    "MANDATORY DIAGNOSTIC (this iter only) — your ### Analysis MUST answer:\n"
                    "  Q1. Which path[] row should have changed (cite stage X.Y, cell name,\n"
                    "      expected Δcell_delay_ps from Liberty)?\n"
                    "  Q2. Does the after_path TOON in <previous_target_path> show that\n"
                    "      row with the NEW cell master?\n"
                    "       - If YES but Δ=0 → cell was already at sub-resolution limit,\n"
                    "         OR upstream cascade (Δic on the upstream driver) exactly\n"
                    "         cancelled the cell delay reduction. Diagnose which.\n"
                    "       - If NO  → the command did not apply (likely a tooling /\n"
                    "         pin-compat issue). Report it; do NOT retry the same target.\n"
                    "  Q3. What is the FIRST stage in path[] where cell_delay_ps > 25 AND\n"
                    "      upsize=[] AND vt_swap=[] (the unfixable ECO floor for this\n"
                    "      path)? Sum the unfixable cell_delay_ps in path[] and compare\n"
                    "      to current slack — is the path even solvable by ECO?\n"
                    "\n"
                    "MANDATORY MOVE-TYPE CONSTRAINT (this iter only):\n"
                    "  Your ### Tcl recipe MUST be one of:\n"
                    "    (a) A PROBE — `eco_net_sink_report <net>` on the highest\n"
                    "        wire_delay_ps net in path[], OR `eco_rank_fanout_by_slack\n"
                    "        <driver>` on the highest-fanout shared-prefix driver. The\n"
                    "        probe enriches NEXT iter's context so you can pick a real\n"
                    "        move with data.\n"
                    "    (b) A move on a DIFFERENT path/cone than last iter — target an\n"
                    "        instance that does NOT appear in <recent_commands>. Pick from\n"
                    "        path[2..5] if path[1] is stuck.\n"
                    "    (c) EMPTY (0 commands) IF Q3 shows the unfixable floor exceeds\n"
                    "        the current violation — i.e. the path is unsolvable by ECO.\n"
                    "        Justify in Analysis with the floor arithmetic.\n"
                    "\n"
                    "  DO NOT re-emit the same move type on the same target. A zero-delta\n"
                    "  move repeated wastes an iter and corrupts <prediction_calibration>.\n"
                    "</stall_diagnostic>"
                )
        except Exception as _e:
            print(f"  [WARN] stall_diagnostic build failed: {_e}", flush=True)


    # 3e. Probe responses: stdout of read-only query commands the LLM emitted
    #     last iter (eco_rank_fanout_by_slack, eco_top_paths_through,
    #     eco_net_sink_report). Without this block the LLM never sees what
    #     its probes returned — the tool calls look like no-ops to it.
    if PROBE_RESPONSES_FILE.exists():
        try:
            probe_txt = PROBE_RESPONSES_FILE.read_text(encoding="utf-8", errors="ignore").strip()
            # Skip if only the header line is present (no actual probe ran).
            if probe_txt and "=== probe:" in probe_txt:
                sections.append(
                    "<probe_responses>\n"
                    "<note>Stdout from probe commands you issued last iter. Each "
                    "block is headed by '=== probe: <command line> ==='. Use these "
                    "to inform this iter's move — your probes were not no-ops.</note>\n"
                    + probe_txt + "\n"
                    "</probe_responses>"
                )
        except Exception as e:
            print(f"  [WARN] probe_responses unreadable: {e}", flush=True)

    # 4. Final output instruction. The model MUST produce the three sections
    #    from the system prompt's "Output format". sanitize_tcl_output extracts
    #    only the ```tcl block for OpenROAD; extract_analysis captures the
    #    Analysis + Move plan text and persists it for future <recent_analysis>.
    sections.append(
        "<instructions>\n"
        "OUTPUT NOW: Produce ALL THREE sections from the system prompt's \"Output\n"
        "format\". The orchestrator extracts only the ```tcl block for execution,\n"
        "but ### Analysis and ### Move plan are captured and surfaced back to you\n"
        "in the NEXT iteration's <recent_analysis> and <prediction_calibration>.\n"
        "Write them so future-you can audit your reasoning.\n"
        "\n"
        "VERBOSITY CAP (hard limits — exceeding wastes tokens and risks API timeout):\n"
        "  - ### Analysis: ≤ 1500 words. Cite specific instance names, deltas, and\n"
        "    calibration ratios. Skip restating the few-shot template.\n"
        f"  - ### Move plan: ≤ {MAX_COMMANDS} rows. One row per Tcl command. No\n"
        "    speculative \"future moves\" rows.\n"
        f"  - ### Tcl recipe: ≤ {MAX_COMMANDS} commands. Same count as Move plan.\n"
        "  Reasoning that doesn't change a move decision is wasted output.\n"
        "  If a 5-iter pattern fits the same diagnosis, summarize it in one line.\n"
        "\n"
        "  ### Analysis    — 3-5 concrete bullets naming bottleneck instances,\n"
        "                    whether each is cell-delay vs wire-delay dominated\n"
        "                    (cell_delay_ps vs wire_delay_ps when present; else\n"
        "                    load_cap_ff > 5ff with fanout >= 4 OR consecutive\n"
        "                    coords > 15um apart as wire-dominated proxies),\n"
        "                    shared-prefix cells (on_other_violating_paths>=1),\n"
        "                    sibling-risk stages (next sibling within 5ps), AND\n"
        "                    explicit cross-references to <previous_target_path>\n"
        "                    and <prediction_calibration> if those blocks exist:\n"
        "                    name which past predictions were wrong and how that\n"
        "                    changes your current decision.\n"
        "  ### Move plan   — table | # | Move | Target | Why-vs-alternatives |\n"
        "                    Δslack est. (ps, signed number) | Sibling risk |.\n"
        "                    The Δslack est. column MUST be a single numeric value\n"
        "                    (e.g. \"+3.5\" or \"-1.0\") — the orchestrator parses\n"
        "                    this and compares to actual ΔWNS next iter. Be honest;\n"
        "                    over-optimistic predictions surface in calibration.\n"
        f"  ### Tcl recipe  — fenced ```tcl block, max {MAX_COMMANDS} lines, only\n"
        "                    instances from <iteration_context> path[] rows or\n"
        "                    fanout_rank.worst_half_sinks. No repeats from\n"
        "                    <recent_commands>.\n"
        "\n"
        "<decision_rules>\n"
        "Do NOT follow fixed thresholds. For every candidate move, REASON about\n"
        "expected ΔWNS using the data you have, then COMPARE alternatives and\n"
        "pick the highest expected gain after calibration derating.\n"
        "\n"
        "For an eco_resize_gate (upsize or VT swap):\n"
        "  Expected gain ≈ Δcell_delay on the target stage.\n"
        "  - VT swap (_L/_R → _SL): typically 2-5 ps, zero input_cap change\n"
        "    → upstream is unaffected. Most reliable move type.\n"
        "  - Upsize (e.g. AO21x1 → AO21x2): typically 1-3 ps on target stage,\n"
        "    but upstream load increases by Δinput_cap × K2. If upstream is on\n"
        "    the critical path, the net gain can be near zero or negative.\n"
        "  - DRIVE-STRENGTH JUMP RULE: prefer 1-step upsizes; large jumps\n"
        "    are HIGH RISK but allowed when justified.\n"
        "      PREFERRED: BUFx6f → BUFx8 (1.33×)    BUFx8 → BUFx12 (1.5×)\n"
        "                 AO21x1 → AO21x2 (2×)     INVx2 → INVx4 (2×)\n"
        "      HIGH RISK: BUFx6f → BUFx24 (4×)    BUFx8 → BUFx24 (3×)\n"
        "                 INVx2 → INVx8 (4×)\n"
        "    Reason large jumps are risky: they cascade load into paths NOT\n"
        "    shown in path[] (only top-5 violating are shown). Previously-safe\n"
        "    paths can tip into worst-violator status. Both prior runs had\n"
        "    disasters from BUFx24 upsizes on shared-prefix fanout drivers.\n"
        "\n"
        "    You MAY propose a > 2× jump ONLY if AT LEAST ONE of the following\n"
        "    is true AND you state it explicitly in Why-vs-alternatives:\n"
        "      (a) The cell is on shared prefix (on_other_violating_paths >= 4)\n"
        "          AND its current cell_delay_ps >= 15 AND a 1-step upsize\n"
        "          on this same cell in a prior iter delivered < 30% of the\n"
        "          predicted gain (cite the iter via <previous_target_path>).\n"
        "      (b) The cell is a buffer driving fanout >= 8 with load >= 15ff\n"
        "          AND the bigger size is CAP-DECREASING (next size has LOWER\n"
        "          input_cap_ff, which actually frees upstream load).\n"
        "      (c) <previous_target_path> for this same path shows >= 3\n"
        "          consecutive iters with total ΔWNS < 1 ps — incremental\n"
        "          steps are not making progress, larger move warranted.\n"
        "    Without one of (a/b/c), keep jumps at 2× or less.\n"
        "\n"
        "For an eco_insert_buffer:\n"
        "  Adds a NEW cell to the path. For sinks moved behind the buffer,\n"
        "  net Δdelay = -Δwire_delay + Δbuf_intrinsic + Δbuf_load_delay.\n"
        "  Usually only positive when wire_delay_ps > buf_intrinsic (~10-15 ps).\n"
        "  This means the wire run must be LONG (>20um) or the original load\n"
        "  large enough that the wire delay component dominates.\n"
        "\n"
        "For an eco_clone_gate_worst_half:\n"
        "  Splitting load reduces target stage delay by roughly K2 × Cload/2,\n"
        "  NOT half of total delay (the intrinsic K1 does not change).\n"
        "  CRITICAL: the upstream driver of the cloned cell sees DOUBLED input\n"
        "  load (orig input_cap + clone input_cap). If upstream is on the\n"
        "  critical path, its delay increases — frequently wiping out the\n"
        "  cloned-stage gain. ALWAYS read the upstream stage row before\n"
        "  emitting a clone and account for this cascade in your Δslack est.\n"
        "\n"
        "Hard exclusions (NOT thresholds — physical impossibilities):\n"
        "  - DO NOT clone any instance whose name starts with `eco_clone_`.\n"
        "    Cloning a clone recursively adds buffers to the chain.\n"
        "  - DO NOT clone any instance whose master cell starts with BUFx,\n"
        "    BUFx, INVx, or any buffer/inverter. A buffer's job is to drive\n"
        "    load; cloning it adds a new buffer instance to the chain rather\n"
        "    than reducing path delay. Upsize the buffer instead.\n"
        "  - DO NOT touch flops, latches, clock cells, macros, IOs, or any\n"
        "    cell flagged dont_touch.\n"
        "\n"
        "Cost-benefit gate: emit a move ONLY if (your Δslack est.) > (alternative\n"
        "best Δslack est. + 1 ps). If your <prediction_calibration> shows your\n"
        "predictions for this move type have been wrong by >2x, derate your\n"
        "current prediction by that calibration ratio before applying the gate.\n"
        "\n"
        "BATCH-INTERACTION RULE (critical for multi-move iters):\n"
        "When you propose >=2 moves in one iter, your individual-move predictions\n"
        "from Liberty are NOT additive at the WNS level. WNS is the slack of the\n"
        "WORST-VIOLATING endpoint (the most-negative signed slack). Three mechanisms\n"
        "make actual ΔWNS much less than the per-move sum:\n"
        "\n"
        "  (A) WNS-saturation: the rank-1 worst path is the only one whose slack\n"
        "      sets WNS. If 4 swaps all target path 1, fixing it by +5 ps only\n"
        "      improves WNS by +5 ps IF path 2 is at least 5 ps better than path 1.\n"
        "      Once the rank-1 path improved-slack reaches the rank-2 path slack,\n"
        "      path 2 becomes the new WNS-holder and further gains on path 1 only\n"
        "      help TNS, NOT WNS. Compute the head-room from path[]:\n"
        "        headroom = |path[2].slack_ps - path[1].slack_ps| ps\n"
        "      (path[2] is the rank-2 worst path, also in path[]; nearby_endpoints[]\n"
        "       starts at rank 6 — do NOT confuse the two.)\n"
        "      Your max ΔWNS from same-cone moves on path 1 is at most that headroom.\n"
        "\n"
        "  (B) Cascade compounding (occasional bonus): two adjacent VT swaps on the\n"
        "      same path may help each other through slew chain — swap A outputs\n"
        "      cleaner slew, swap B sees better in_slew, B actual Δdelay exceeds\n"
        "      its solo prediction. Positive bonus, ~10-30% on top of saturated estimate.\n"
        "\n"
        "  (C) Load cascade (penalty for cap-INCREASING upsize batches): N upsizes\n"
        "      on the same upstream chain each add Δinput_cap. Cumulative upstream\n"
        "      load delta = sum of Δic across all upsizes sharing an upstream driver.\n"
        "      Upstream cell_delay penalty ≈ K2 × cumulative_Δic. The bigger N, the\n"
        "      more this dominates → batch upsizes can NET NEGATIVE on WNS even when\n"
        "      each looks positive solo.\n"
        "\n"
        "Operational rules for multi-move iters:\n"
        "  1. Predicted ΔWNS for a multi-move batch ≠ sum of per-move Liberty deltas.\n"
        "     Compute a SATURATED estimate for same-cone batches:\n"
        "        same_cone_batch_ΔWNS = min(sum_of_deltas, |path[2].slack - path[1].slack|)\n"
        "     This caps your gain at the head-room to rank-2.\n"
        "\n"
        "  *** SHARED-PREFIX SATURATION EXEMPTION (series vs parallel) ***\n"
        "  The min(Σ, |path[2].slack - path[1].slack|) cap above applies ONLY\n"
        "  across PARALLEL endpoint paths. It does NOT apply to:\n"
        "    (a) SERIES stages on the SAME path — improving K stages in series on\n"
        "        ONE path ACCUMULATES: predict Σ(per-stage Δ) minus the upstream-\n"
        "        load cascade, NOT min(Σ, gap). The gap cap is a parallel-path\n"
        "        effect only.\n"
        "    (b) a SHARED-PREFIX stage (on_other_violating_paths = R): a single\n"
        "        move there lifts rank-1 AND rank-2 ... AND rank-R TOGETHER, so it\n"
        "        is NOT capped by the path1<->path2 head-room — the tied paths\n"
        "        rise WITH it. Predict its ΔWNS as the FULL per-stage Δ.\n"
        "  This is the #1 cause of under-predicting high-oovp moves: a structural\n"
        "  move that actually delivers +101ps reads as +9 under the wrong cap.\n"
        "\n"
        "  *** SHARED-PREFIX PRIORITY (where to spend the move) ***\n"
        "  When the top-N violating paths share a long common prefix (many stages\n"
        "  with oovp ≈ N), the WNS bottleneck IS the shared prefix, not the tails:\n"
        "    1. Rank candidate stages by on_other_violating_paths DESCENDING.\n"
        "    2. Prefer the highest-oovp stages NOT already at max drive + fastest\n"
        "       VT (SL) — one swap there moves all oovp paths at once.\n"
        "    3. An oovp=1 TAIL move helps exactly ONE endpoint; with a sub-1ps gap\n"
        "       to rank-2 it cannot move a bunched WNS — treat as last resort.\n"
        "    4. If the shared prefix is ENTIRELY maxed (all SL + max drive) AND\n"
        "       deep (>30 logic levels), declare the ECO floor (floor arithmetic)\n"
        "       rather than churning tail resizes.\n"
        "\n"
        "  2. Spread moves across DIFFERENT violating paths for additive WNS gain.\n"
        "     If you can pick one cell per path[1..5] that improves each path,\n"
        "     the WNS gain ≈ the smallest per-move delta (the worst of your N moves).\n"
        "     This is usually much better for WNS than 4 swaps on one cone.\n"
        "\n"
        "  3. For cap-INCREASING upsize batches, sum the upstream load delta and\n"
        "     compute the upstream-cascade penalty BEFORE finalizing. If the\n"
        "     cumulative Δic on a shared upstream driver > 1 ff, expect net negative.\n"
        "\n"
        "  4. Apply the bucket-specific calibration ratio from <prediction_calibration>.\n"
        "     e.g. if `eco_resize_gate (batch_4+)` shows mean_ratio=0.04, derate\n"
        "     your raw batch prediction by 25× BEFORE applying the cost-benefit gate.\n"
        "\n"
        "  5. Single-move iters with `trusted` calibration: trust raw prediction.\n"
        "     Mixed-bucket iters: use the bucket with the LARGEST sample.\n"
        "  6. Bucket with `no data`: trust your raw prediction at face value ONLY if\n"
        "     the move type passed prior calibration on similar designs; otherwise\n"
        "     apply a 5x pessimism factor.\n"
        "\n"
        "</decision_rules>\n"
        "\n"
        "<exploratory_mentality>\n"
        "You are an explorer with a safety net, not a recipe-matcher.\n"
        "\n"
        "  1. ENUMERATE before SKIP. When path[1] worst stage is unfixable by\n"
        "     resize (upsize=[] AND vt_swap=[]), list ALL structural candidates\n"
        "     on neighboring stages BEFORE declaring the path stuck:\n"
        "       - buffer-insert upstream of the stage (clean its in_slew),\n"
        "       - buffer-insert downstream of the stage (split its load AND\n"
        "         regenerate slew for the next sinks),\n"
        "       - clone upstream driver (halve load on the upstream side),\n"
        "       - remove a redundant downstream buffer (free upstream cap).\n"
        "     Compute Liberty-derived raw ΔWNS for each, derate by the\n"
        "     calibration ratio, and pick the best. A skipped iter has zero\n"
        "     information value; an honest attempt teaches the calibration.\n"
        "\n"
        "  2. TRY MENTALITY. The orchestrator BACKTRACES (reverts to the best\n"
        "     .odb snapshot) if your moves degrade WNS over a stagnation window.\n"
        "     A move with positive expected value AFTER honest derating is worth\n"
        "     trying even at moderate sibling risk — the worst case is one\n"
        "     wasted iter, recovered by the backtrace mechanism. Do not let\n"
        "     fear of a small regression keep you from any move.\n"
        "\n"
        "  3. PATH ROTATION. If path[1] has no positive-ROI move after the\n"
        "     enumeration above, AND no candidate on path[1] can plausibly\n"
        "     close >5 ps of slack, target the worst stage of path[2..5]\n"
        "     instead. Many WNS plateaus break only when you stop hammering\n"
        "     the rank-1 path and start pulling rank-2/3 paths upward — they\n"
        "     are close enough that a single VT swap or buffer-insert on a\n"
        "     different cone can become the new WNS-holder fix.\n"
        "\n"
        "  4. OUTLIER SCAN. Before move-selection, scan every path[] row for\n"
        "     outliers: cell_delay_ps anomalously high vs Liberty at this\n"
        "     (load, slew); out_slew_ps anomalously high relative to the cell\n"
        "     family. Outliers are where the leverage is — the recipe-matched\n"
        "     average-case move is rarely the right one.\n"
        "</exploratory_mentality>\n"
        "\n"
                "\n"
        "<structural_isolation>\n"
        "In any iter where the recipe contains eco_insert_buffer, eco_clone_gate,\n"
        "eco_clone_gate_worst_half, or eco_remove_buffer: emit ONLY structural\n"
        "commands (one, or two if they target the same stage). NO concurrent\n"
        "eco_resize_gate. Mixing structural + sizing in one batch makes per-\n"
        "command attribution impossible — your <prediction_calibration> becomes\n"
        "noisy and you cannot learn from it.\n"
        "Pure-resize iters can still batch up to MAX_COMMANDS resizes.\n"
        "</structural_isolation>\n"
        "\n"
        "<sibling_safety>\n"
        "For every change, check siblings[] for the stage\'s inst. If any sibling\n"
        "slack is within 10ps of wns_slack_ps, prefer moves that help the sibling\n"
        "too (driver upsize > sink upsize, VT swap > upsize, clone of shared-\n"
        "prefix > per-path resize). A move that fixes path 1 by 5ps but pushes\n"
        "path 2 down by 5ps is a net zero.\n"
        "</sibling_safety>\n"
        "\n"
        "<sibling_tip_prevention>\n"
        "The path[] block shows only the 5 worst paths. The design has ~50\n"
        "more violating endpoints just outside that window, surfaced in two\n"
        "blocks below path[]:\n"
        "  - <nearby_endpoints[]>:        ranks 6..50 with their slacks and\n"
        "    how many cells they share with the top-5 paths.\n"
        "  - <shared_cells_to_nearby[]>:  for each top-5 instance that ALSO\n"
        "    appears on any nearby path, the list of nearby ranks affected\n"
        "    and the worst nearby slack.\n"
        "\n"
        "Before committing a move on instance X (especially a cap-changing\n"
        "upsize or any structural move), look X up in shared_cells_to_nearby.\n"
        "If X has n_nearby_paths >= 1 and worst_nearby_slack_ps is within 3ps\n"
        "of the current WNS, your move risks tipping that nearby path into\n"
        "worst-violator status — even if it helps your targeted path. Two\n"
        "rules:\n"
        "  1. If your predicted ΔWNS_target is +Y and your move would push\n"
        "     the worst nearby path down by >Y, the move is a net loss.\n"
        "     SKIP it.\n"
        "  2. State the sibling-tip check explicitly in Why-vs-alternatives,\n"
        "     e.g. 'X is on nearby paths [6,8,12]; worst nearby slack -76.5;\n"
        "     my move shifts that by ~-0.4 ps → still safe (gap is 2.0 ps).'\n"
        "\n"
        "Iter-3 and iter-10 disasters on prior runs were this pattern: model\n"
        "fixed top-5 path slack by +1.7 ps but tipped a path at -76 ps down\n"
        "by ~9 ps. WNS got worse by ~7 ps. shared_cells_to_nearby would\n"
        "have flagged the risk.\n"
        "</sibling_tip_prevention>\n"
        "\n"
        "<physical_locality>\n"
        "x_um/y_um and neighbors_5um are ADVISORY: high density means the\n"
        "legalizer may shift a new buffer/clone 1-3 um from optimum, slightly\n"
        "increasing wire delay vs your estimate. Account for this in your\n"
        "Δslack est.: at neighbors_5um > 600, add ~1 ps of pessimism;\n"
        "at >800, add ~2 ps. Long-wire stages (consecutive coords >15 um apart)\n"
        "are buffer-insertion targets regardless of neighbors_5um — wire delay\n"
        "dominates.\n"
        "</physical_locality>\n"
        "</instructions>"
    )

    user_prompt = "\n\n".join(sections)
    return system_prompt, user_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--claude-log-dir", type=Path, default=DEFAULT_CLAUDE_LOG_DIR)
    # Legacy flag name kept for CLI compatibility with auto_runme_loop_v5.py.
    ap.add_argument("--codex-log-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    log_dir = args.codex_log_dir or args.claude_log_dir

    static_guide = load_text(STATIC_GUIDE_FILE)
    hist = load_history()
    iterations = 0
    last_seen_mtime = None

    print(f"[INIT] Static guide: {len(static_guide)} chars, ~{estimate_tokens(static_guide)} tokens", flush=True)

    while True:
        if args.max_iter is not None and iterations >= args.max_iter:
            break

        user_msg = None
        if RPT_PATH.exists() and RPT_PATH.stat().st_size > 0:
            mtime = RPT_PATH.stat().st_mtime
            if last_seen_mtime is None or mtime > last_seen_mtime:
                user_msg = "Updated Timing Report:\n" + RPT_PATH.read_text(encoding="utf-8", errors="ignore")
                last_seen_mtime = mtime
            else:
                time.sleep(0.2)
                continue
        else:
            time.sleep(0.2)
            continue

        if user_msg is None:
            break
        user_msg = user_msg.strip()
        if not user_msg:
            continue

        iterations += 1
        hist.append({"user": f"iter_{iterations}", "assistant": ""})
        print(f"\n{'='*60}", flush=True)
        print(f"=== LLM Iteration {iterations} ===", flush=True)
        print(f"{'='*60}", flush=True)

        # Context (still loaded so we can build an allowed-inst set for
        # reply validation — but the LLM prompt itself no longer contains
        # the raw JSON dump).
        context = load_json(DYNAMIC_CONTEXT_FILE)
        dynamic = context.get("dynamic", context)
        # V5 context shape has no local_physical block; derive the set of
        # allowed insts from TOON path + fanout_rank drivers.
        allow_insts: set[str] = set()
        for p in (dynamic.get("paths") or [dynamic]):
            for row in p.get("path", []) or []:
                drv = row.get("driver_pin") or ""
                inst = drv.split("/", 1)[0] if "/" in drv else drv
                if inst:
                    allow_insts.add(inst)
        for drv in (dynamic.get("fanout_rank") or {}).keys():
            allow_insts.add(drv)
        # Back-compat: older context_v5.json layout used local_physical.
        allow_insts |= set(dynamic.get("local_physical", {}).get("allowed_instances", []))
        buf_inv_masters = dynamic.get("buf_inv_instances", {}) or {}

        recent_cmds = extract_recent_commands(hist, n_turns=3)
        print(f"  [CTX] Allowed instances: {len(allow_insts)} total", flush=True)
        print(f"  [CTX] Recent commands to avoid: {len(recent_cmds)}", flush=True)

        system_prompt, user_prompt = build_prompt(static_guide, recent_cmds)
        MAX_USER_CHARS = 160000
        if len(user_prompt) > MAX_USER_CHARS:
            print(f"  [WARN] User prompt too long ({len(user_prompt)} chars), truncating", flush=True)
            user_prompt = user_prompt[:MAX_USER_CHARS] + "\n...[truncated]"

        PROMPT_DUMP.write_text(
            "=== SYSTEM PROMPT ===\n" + system_prompt
            + "\n\n=== USER PROMPT ===\n" + user_prompt,
            encoding="utf-8",
        )
        print(f"  [DBG] Prompt saved to {PROMPT_DUMP}", flush=True)

        # Switch transport here: call_claude (CLI) or call_claude_api (SDK).
        reply_raw, claude_meta = call_claude(system_prompt, user_prompt)
        append_llm_session_log(iterations, system_prompt, user_prompt,
                               reply_raw, claude_meta, label="primary")
        reply = sanitize_tcl_output(reply_raw)

        print(f"\n  [REPLY] LLM output:", flush=True)
        print(f"  {'─'*50}", flush=True)
        for line in reply.splitlines():
            print(f"  │ {line}", flush=True)
        print(f"  {'─'*50}", flush=True)

        ok, reason = validate_reply(reply, recent_cmds, allow_insts, buf_inv_masters)
        if not ok:
            print(f"  [VALIDATE] FAILED: {reason}", flush=True)
            retry_user = (
                user_prompt
                + "\n\nYour last output was invalid: "
                + reason
                + "\nPrint ONLY valid Tcl ECO commands. Use replace_cell VT swaps where available. "
                "Do NOT buffer _eco_* nets. Use eco_remove_buffer to undo bad buffers."
            )
            retry_raw, retry_meta = call_claude(system_prompt, retry_user)
            append_llm_session_log(iterations, system_prompt, retry_user,
                                   retry_raw, retry_meta, label="retry")
            claude_meta = {"first": claude_meta, "retry": retry_meta}
            reply = sanitize_tcl_output(retry_raw)
            print(f"\n  [RETRY REPLY]:", flush=True)
            for line in reply.splitlines():
                print(f"  │ {line}", flush=True)
            ok, reason = validate_reply(reply, recent_cmds, allow_insts, buf_inv_masters)
            if not ok:
                print(f"  [VALIDATE] Still failed: {reason} — passing through anyway", flush=True)
        else:
            print(f"  [VALIDATE] OK — {len(parse_commands(reply))} commands", flush=True)

        filtered = reply.strip()
        hist[-1]["assistant"] = filtered
        ECO_OUT.write_text(filtered + "\n", encoding="utf-8")
        try:
            analysis_text = extract_analysis(reply_raw or "")
            ANALYSIS_OUT.write_text(
                (analysis_text + "\n") if analysis_text else "",
                encoding="utf-8",
            )
        except Exception as e:
            print(f"  [WARN] analysis sidecar write failed: {e}", flush=True)
        hist = hist[-5:]
        save_history(hist)

        # --- Persist per-iteration claude log: thinking + token counts ---
        try:
            _safe_mkdir(log_dir)
            ts_raw = _ts_for_path()
            reply_path = log_dir / f"reply.iter{iterations}.{ts_raw}.md"

            # Gather usage (handles retry-nested shape).
            def _usage_of(m):
                if not isinstance(m, dict):
                    return {}
                if "usage_reported" in m:
                    return m["usage_reported"] or {}
                return (m.get("first") or {}).get("usage_reported", {}) or \
                       (m.get("retry") or {}).get("usage_reported", {}) or {}
            usage = _usage_of(claude_meta)
            envelope = (claude_meta.get("claude_envelope")
                        if isinstance(claude_meta, dict) and "claude_envelope" in claude_meta
                        else (claude_meta.get("retry") or {}).get("claude_envelope")
                             or (claude_meta.get("first") or {}).get("claude_envelope")
                             or {}) if isinstance(claude_meta, dict) else {}

            # Full per-iteration markdown log: tokens + sanitized Tcl + full
            # raw reply (which contains any model thinking/reasoning text
            # that preceded the Tcl block).
            header = (
                f"# Claude ECO reply — iter {iterations} — {ts_raw}\n\n"
                f"## Token usage (from claude JSON envelope)\n\n"
                f"- input_tokens: {usage.get('input_tokens')}\n"
                f"- output_tokens: {usage.get('output_tokens')}\n"
                f"- cache_read_input_tokens: {usage.get('cache_read_input_tokens')}\n"
                f"- cache_creation_input_tokens: {usage.get('cache_creation_input_tokens')}\n"
                f"- num_turns: {claude_meta.get('num_turns') if isinstance(claude_meta, dict) else None}\n"
                f"- duration_ms: {claude_meta.get('duration_ms') if isinstance(claude_meta, dict) else None}\n"
                f"- total_cost_usd: {claude_meta.get('total_cost_usd') if isinstance(claude_meta, dict) else None}\n"
                f"- system_chars: {len(system_prompt)} | user_chars: {len(user_prompt)} "
                f"(est total tokens: {estimate_tokens(system_prompt) + estimate_tokens(user_prompt)})\n"
                f"- reply_chars: {len(reply_raw or '')}\n\n"
                f"## Sanitized Tcl (what was sourced into OpenROAD)\n\n"
                f"```tcl\n{filtered}\n```\n\n"
                f"## Full raw reply (reasoning + Tcl, as returned by Claude)\n\n"
            )
            reply_path.write_text(header + (reply_raw or ""), encoding="utf-8")

            # If the API path returned an extended-thinking trace, persist
            # it to its own file. The CLI path will never populate this —
            # claude -p does not return thinking text in its JSON envelope.
            def _thinking_of(m):
                if not isinstance(m, dict):
                    return ""
                if "thinking_text" in m:
                    return m.get("thinking_text") or ""
                return ((m.get("retry") or {}).get("thinking_text") or
                        (m.get("first") or {}).get("thinking_text") or "")
            thinking_text = _thinking_of(claude_meta)
            if thinking_text:
                thinking_path = log_dir / f"thinking.iter{iterations}.{ts_raw}.md"
                thinking_path.write_text(
                    f"# Claude extended-thinking trace — iter {iterations} — {ts_raw}\n\n"
                    f"## How the model read/analysed this iteration\n\n"
                    f"{thinking_text}\n\n"
                    f"## Final Tcl emitted (for cross-reference)\n\n"
                    f"```tcl\n{filtered}\n```\n",
                    encoding="utf-8",
                )
                print(f"  [LLM] Thinking trace saved to {thinking_path}", flush=True)

            # One-line JSONL token log for quick grepping / plotting.
            token_log = log_dir / "token_log.jsonl"
            with token_log.open("a", encoding="utf-8") as tfp:
                tfp.write(json.dumps({
                    "iteration": iterations,
                    "timestamp": ts_raw,
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                    "num_turns": claude_meta.get("num_turns") if isinstance(claude_meta, dict) else None,
                    "duration_ms": claude_meta.get("duration_ms") if isinstance(claude_meta, dict) else None,
                    "total_cost_usd": claude_meta.get("total_cost_usd") if isinstance(claude_meta, dict) else None,
                    "system_chars": len(system_prompt),
                    "user_chars": len(user_prompt),
                    "reply_raw_chars": len(reply_raw or ""),
                    "reply_filtered_chars": len(filtered),
                    "reply_path": str(reply_path),
                    "thinking_chars": (
                        (claude_meta.get("thinking_chars")
                         if isinstance(claude_meta, dict) else None)
                        or ((claude_meta.get("retry") or {}).get("thinking_chars")
                            if isinstance(claude_meta, dict) else None)
                        or ((claude_meta.get("first") or {}).get("thinking_chars")
                            if isinstance(claude_meta, dict) else None)
                    ),
                    "thinking_blocks": (
                        claude_meta.get("thinking_blocks")
                        if isinstance(claude_meta, dict) else None
                    ),
                }) + "\n")

            # Raw claude envelope (JSON) for deep analysis — contains any
            # per-turn breakdown and modelUsage keys.
            if envelope:
                (log_dir / f"envelope.iter{iterations}.{ts_raw}.json").write_text(
                    json.dumps(envelope, indent=2), encoding="utf-8",
                )
            print(f"  [LLM] Log saved to {reply_path}", flush=True)
        except Exception as e:
            print(f"  [WARN] Failed to save claude log: {e}", flush=True)

        print(f"  [DONE] ECO written to {ECO_OUT}", flush=True)


if __name__ == "__main__":
    main()
