"""
Context builder v5 — single-top-path TOON emitter.

Consumes:
  - A timing report (raw `report_checks -fields {net cap slew input fanout}` dump,
    or the JSON-format checks file supported by TimingReportParser).
  - node_details.csv (x_um, y_um per instance)
  - net_details.csv (per-net driver + sinks + fanout)
  - sibling slacks file produced per-iteration by `eco_top_paths_through`
    (format documented at _load_sibling_slacks below). Optional.
  - eco / run history JSON for recent_eco_actions. Optional.

Produces:
  TOON (Token-Oriented Object Notation) text matching the input schema in
  timing_eco_system_prompt.md §11. Written to the path given by --out.

CLI is kept backwards-compatible with the v4 arguments used by
auto_runme_loop_v5.py (same flag names); new v5-only flags are added.
"""

import argparse
import csv
import re
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from parsers.library_extractor import LibraryExtractor
from parsers.timing_rpt_parser import TimingReportParser
import os


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output-pin names across ASAP7 + common Liberty conventions.
OUTPUT_PINS = frozenset({
    "Y", "Q", "QN", "S", "SO", "CO", "SN", "Z", "ZN",
    "CON", "COUT", "S0", "S1", "SUM",
})

# A "register" for is_dff flagging. Stages whose cell starts with one of
# these are don't-touch per system prompt §6.
REGISTER_PREFIXES = ("DFF", "SDFF", "LATCH", "ICG", "SNL", "SRA")

# VT speed ordering (higher = faster per system prompt §1).
VT_SPEED = {"SL": 2, "L": 1, "R": 0}

# Parser for "<drive_strength>" like x1, x4, xp33, x1p5, x12f.
# Returns a sortable (int_mul_100, suffix_len) tuple.
_DRIVE_RE = re.compile(r"^x(?:p(\d+)|(\d+)(?:p(\d+))?([fslhntr]*)?)$")


def _parse_drive_rank(drive: str) -> Tuple[int, int]:
    """Sort key for drive strengths. Higher rank = stronger drive."""
    if not drive:
        return (0, 0)
    m = _DRIVE_RE.match(drive.lower())
    if not m:
        return (0, 0)
    pfrac, whole, frac, suf = m.groups()
    if pfrac is not None:                     # "xp33" → 0.33
        val = int(pfrac) / (10 ** len(pfrac))
    else:
        w = int(whole)
        f = int(frac) / (10 ** len(frac)) if frac else 0.0
        val = w + f
    return (int(val * 100), len(suf or ""))


# ---------------------------------------------------------------------------
# Timing report parsing — extract the top violating path with full columns
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """One path stage (one cell) with raw report fields + enrichment."""
    stage: int
    driver_pin: str                                # "inst/Y"
    cell: str                                      # cell_type (master)
    inst: str                                      # instance name
    fanout: Optional[int] = None
    load_cap_ff: Optional[float] = None
    in_slew_ps: Optional[float] = None
    out_slew_ps: Optional[float] = None
    cell_delay_ps: Optional[float] = None
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    neighbors_5um: int = -1
    is_dff: bool = False
    dont_touch: bool = False
    on_other_violating_paths: int = 0
    upsize: List[str] = field(default_factory=list)   # ["master@cap_ff", ...]
    vt_swap: Optional[str] = None                    # "master@cap_ff" or None
    input_cap_ff: Optional[float] = None             # current cell's input pin cap (fF)
    # Wire-related enrichments (filled in build() / extract_top_path_stages).
    wire_delay_ps: Optional[float] = None    # wire delay from THIS stage's
                                             #   driver output to the NEXT
                                             #   stage's input (ps). For the
                                             #   final stage, wire delay into
                                             #   the endpoint DFF/D pin.
    wire_cap_ff: Optional[float] = None      # wire component of load_cap_ff:
                                             #   load_cap_ff − Σ(sink pin caps).
    wire_length_um: Optional[float] = None   # routed/estimated length of the
                                             #   driver's output net (um).


def _pin_is_output(pin_short: str) -> bool:
    return pin_short in OUTPUT_PINS


def _is_register_cell(cell_type: str) -> bool:
    ct = (cell_type or "").upper()
    return any(ct.startswith(p) for p in REGISTER_PREFIXES)


def extract_top_path_stages(report_text: str) -> Tuple[List[Stage], Dict[str, Any]]:
    """Parse the raw report_checks text and return (stages, path_meta).

    path_meta keys: startpoint, endpoint, clock_period_ps, slack_ps,
                    tns_ps (None here; filled by caller), path_type.
    """
    # If it's a short path that exists on disk, load it. Anything too long
    # to be a filename is treated as raw report text (avoids OSError 36).
    if len(report_text) < 260:
        try:
            p = Path(report_text)
            if p.exists() and p.is_file():
                report_text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass

    # Find the first "Startpoint:" block (worst-first is the caller's job if
    # there are multiple; the existing parser already sorts by slack).
    sections = re.split(r"^Startpoint:", report_text, flags=re.MULTILINE)
    if len(sections) < 2:
        return [], {}
    body = "Startpoint:" + sections[1]
    lines = body.splitlines()

    def _grab(field_name: str) -> str:
        for ln in lines:
            if ln.strip().startswith(field_name):
                return ln.split(":", 1)[1].strip()
        return ""

    startpoint = _grab("Startpoint").split("(", 1)[0].strip()
    endpoint = _grab("Endpoint").split("(", 1)[0].strip()
    path_type = _grab("Path Type") or "max"

    # Clock period: pick the maximum "clock clk (rise edge)" delay across
    # all matches — the capture-edge rise-edge line carries the full period.
    # (The launch edge at time 0.00 would otherwise win if we took the first.)
    clock_period_ps = None
    for ln in lines:
        m = re.search(r"^\s+(\d+\.\d+)\s+(\d+\.\d+)\s+clock\s+\S+\s+\(rise edge\)", ln)
        if m:
            v = float(m.group(1))
            if clock_period_ps is None or v > clock_period_ps:
                clock_period_ps = v

    # Slack line: "<num>   slack (VIOLATED|MET)"
    slack_ps = None
    for ln in reversed(lines):
        m = re.search(r"(-?\d+\.\d+)\s+slack\s+\(", ln)
        if m:
            slack_ps = float(m.group(1))
            break

    # Parse transition rows (driver Y/Q rows and input A/B rows).
    # A timing row has one of ^ or v followed by "inst/pin (cell_type)".
    row_re = re.compile(
        r"^(.*?)\s+([\^v])\s+(\S+)/(\w+)\s+\((\S+)\)\s*$"
    )
    num_re = re.compile(r"-?\d+\.\d+")

    raw_rows: List[Dict[str, Any]] = []
    for ln in lines:
        # Stop before the data-required block so we don't pick up the
        # capture clock's /CLK row as the endpoint.
        if "data arrival time" in ln.lower():
            break
        m = row_re.match(ln)
        if not m:
            continue
        before, trans, inst, pin_short, cell_type = m.groups()
        nums = [float(x) for x in num_re.findall(before)]
        # 4 nums → fanout_cap_slew_delay + time, or cap_slew_delay_time;
        # 3 nums → slew_delay_time; 2 nums → delay_time.
        # In "report_checks -fields {net cap slew input fanout}" output,
        # driver-pin rows have integer fanout BEFORE the decimals — the
        # integer is NOT picked up by `\d+\.\d+`, so we parse it separately.
        fanout = None
        m_fo = re.match(r"\s*(\d+)\s+\d+\.\d+", before)
        if m_fo:
            fanout = int(m_fo.group(1))

        raw_rows.append({
            "inst": inst,
            "pin": pin_short,
            "cell_type": cell_type,
            "transition": trans,
            "nums": nums,
            "fanout": fanout,
            "is_out": _pin_is_output(pin_short),
        })

    # Walk rows and build stages. Each driver-row → one stage. in_slew is
    # taken from the PRECEDING input-row on the same inst (if any).
    # We also capture the wire (net) delay between consecutive stages:
    # input rows in report_checks carry [slew, delay, time] where `delay`
    # is the wire delay from the previous stage's driver output to this
    # input pin. We attribute that delay to the PREVIOUS stage (= the
    # driver whose output net we just traversed) at the moment the next
    # output row arrives. The final stage's wire_delay_ps (to the
    # endpoint DFF/D pin) is flushed after the loop.
    stages: List[Stage] = []
    last_input_row: Optional[Dict[str, Any]] = None
    prev_net_delay_ps: Optional[float] = None

    for r in raw_rows:
        if r["is_out"]:
            # Attribute the pending wire delay (from the most-recent input
            # row) to the previous stage's wire_delay_ps, then reset.
            if stages and prev_net_delay_ps is not None:
                stages[-1].wire_delay_ps = round(prev_net_delay_ps, 3)
            prev_net_delay_ps = None
            nums = r["nums"]
            # 5 decimals = [cap, slew, delay, time]  (fanout is integer,
            # already extracted). 4 decimals = [cap, slew, delay, time]
            # if no fanout col; 3 = [slew, delay, time].
            load_cap = out_slew = cell_delay = None
            if len(nums) >= 4:
                load_cap, out_slew, cell_delay = nums[0], nums[1], nums[2]
            elif len(nums) == 3:
                out_slew, cell_delay = nums[0], nums[1]
            elif len(nums) == 2:
                cell_delay = nums[0]

            in_slew = None
            if last_input_row and last_input_row["inst"] == r["inst"]:
                li = last_input_row["nums"]
                # input row: [slew, delay, time] (3) or [cap?, slew, delay, time]
                if len(li) >= 3:
                    in_slew = li[-3]
                elif len(li) == 2:
                    in_slew = li[0]

            stages.append(Stage(
                stage=len(stages) + 1,
                driver_pin=f"{r['inst']}/{r['pin']}",
                cell=r["cell_type"],
                inst=r["inst"],
                fanout=r["fanout"],
                load_cap_ff=load_cap,
                in_slew_ps=in_slew,
                out_slew_ps=out_slew,
                cell_delay_ps=cell_delay,
                is_dff=_is_register_cell(r["cell_type"]),
                dont_touch=_is_register_cell(r["cell_type"]),
            ))
        else:
            last_input_row = r
            li_nums = r["nums"]
            # Input row layout: [slew, delay, time] → delay is wire delay.
            # 4-num variant (rare; with explicit cap col): [cap, slew, delay, time].
            if len(li_nums) >= 3:
                prev_net_delay_ps = li_nums[-2]

    # Endpoint flush: the DFF/D input row's delay is the wire delay from
    # the last stage's driver output to the endpoint pin. No further output
    # row arrives, so we attribute it here.
    if stages and prev_net_delay_ps is not None:
        stages[-1].wire_delay_ps = round(prev_net_delay_ps, 3)

    # Extend endpoint with its actual input pin (e.g. "/D"), which is the
    # last input-side row matching the endpoint inst name.
    endpoint_with_pin = endpoint
    for r in reversed(raw_rows):
        if not r["is_out"] and r["inst"] == endpoint:
            endpoint_with_pin = f"{endpoint}/{r['pin']}"
            break

    meta = {
        "startpoint": startpoint,
        "endpoint": endpoint_with_pin,
        "path_type": path_type,
        "clock_period_ps": clock_period_ps,
        "slack_ps": slack_ps,
    }
    return stages, meta


def extract_top_n_paths(report_text_or_path: str, n: int = 3) -> List[Tuple[List[Stage], Dict[str, Any]]]:
    """Return up to N DISTINCT paths from a multi-path report, deduped by
    (startpoint, endpoint). Each entry is the (stages, meta) tuple produced by
    extract_top_path_stages on that block."""
    text = report_text_or_path
    if len(text) < 260:
        try:
            p = Path(text)
            if p.exists() and p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    sections = re.split(r"^Startpoint:", text, flags=re.MULTILINE)
    if len(sections) < 2:
        return []
    out: List[Tuple[List[Stage], Dict[str, Any]]] = []
    seen = set()
    for body in sections[1:]:
        block = "Startpoint:" + body
        stages, meta = extract_top_path_stages(block)
        if not stages:
            continue
        key = (meta.get("startpoint", ""), meta.get("endpoint", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append((stages, meta))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilderV5:
    def __init__(
        self,
        library: Optional[LibraryExtractor],
        node_file: Optional[Path] = None,
        net_file: Optional[Path] = None,
        sibling_slacks_file: Optional[Path] = None,
        fanout_rank_file: Optional[Path] = None,
        eco_history_file: Optional[Path] = None,
    ):
        self.library = library
        self.node_pos, self.node_meta = self._load_node_file(node_file)
        (
            self.net_fanout,
            self.inst_to_nets,
            self.net_length_um,
            self.net_sinks,
            self.inst_pin_to_net,
        ) = self._load_net_file(net_file)
        self.siblings_by_inst = self._load_sibling_slacks(sibling_slacks_file)
        self.fanout_rank_by_driver = self._load_fanout_rank(fanout_rank_file)
        self.recent_eco_actions = self._load_recent_eco_actions(eco_history_file)

    # ------------------------------- loaders -------------------------------

    @staticmethod
    def _load_node_file(path: Optional[Path]):
        pos: Dict[str, Tuple[float, float]] = {}
        meta: Dict[str, Dict[str, str]] = {}
        if not path or not path.exists():
            return pos, meta
        with path.open("r", encoding="utf-8", errors="ignore") as fp:
            for row in csv.DictReader(fp):
                name = (row.get("Name") or "").strip()
                if not name:
                    continue
                try:
                    x = float((row.get("llx") or "0").strip())
                    y = float((row.get("lly") or "0").strip())
                except ValueError:
                    continue
                pos[name] = (x, y)
                meta[name] = {
                    "master": (row.get("Master") or "").strip(),
                    "type": (row.get("Type") or "").strip(),
                }
        return pos, meta


    @staticmethod
    def _neighbor_density(node_pos, x: float, y: float, radius_um: float = 5.0) -> int:
        """Count cells whose (x,y) lies within `radius_um` of (x,y).
        Used to surface placement crowding to the LLM so it avoids dropping
        buffers / upsizing in already-dense regions."""
        r2 = radius_um * radius_um
        n = 0
        for px, py in node_pos.values():
            dx = px - x
            dy = py - y
            if dx * dx + dy * dy <= r2:
                n += 1
        return n - 1  # exclude self

    @staticmethod
    def _load_net_file(path: Optional[Path]):
        """Parse net_details.csv. Returns four dicts:
          net_fanout       : net  -> fanout (int)
          inst_to_nets     : inst -> [nets driven by this inst]
          net_length_um    : net  -> routed/estimated length (um), if present
          net_sinks        : net  -> ["inst/pin", ...] of all sinks
          inst_pin_to_net  : (inst, pin) -> net (uses driver's pin column)

        Expected CSV columns (header line, case-insensitive):
            Net, [LengthUm], [FanOut], Driver, Sink1, Sink2, ...
        Driver and Sinks are encoded as ``"inst pin"`` (whitespace-separated)
        per net_details.csv produced by the OpenROAD net dumper. We accept
        either form (``"inst pin"`` or ``"inst/pin"``).
        """
        net_fanout: Dict[str, int] = {}
        inst_to_nets: Dict[str, List[str]] = {}
        net_length_um: Dict[str, float] = {}
        net_sinks: Dict[str, List[str]] = {}
        inst_pin_to_net: Dict[Tuple[str, str], str] = {}
        if not path or not path.exists():
            return net_fanout, inst_to_nets, net_length_um, net_sinks, inst_pin_to_net

        def _split_inst_pin(tok: str) -> Tuple[str, str]:
            """Parse driver/sink token into (inst, pin). Accept both
            ``"inst pin"`` and ``"inst/pin"``; tolerate missing pin."""
            tok = tok.strip()
            if not tok:
                return "", ""
            # Prefer whitespace split (canonical net_details.csv form).
            ws = tok.split()
            if len(ws) >= 2:
                return ws[0], ws[1]
            # Fallback: slash-separated.
            if "/" in tok:
                inst, pin = tok.rsplit("/", 1)
                return inst, pin
            return tok, ""

        with path.open("r", encoding="utf-8", errors="ignore") as fp:
            first = fp.readline().strip()
            header = first.lower().startswith("net,")
            has_length = header and "length" in first.lower()
            has_fanout = header and ("fanout" in first.lower() or "fan" in first.lower())
            rest = fp.readlines()
            if not header:
                rest.insert(0, first + "\n")

            for ln in rest:
                parts = [p.strip() for p in ln.strip().split(",") if p]
                if len(parts) < 2:
                    continue
                net = parts[0]
                idx = 1
                length_val: Optional[float] = None
                if has_length:
                    try:
                        length_val = float(parts[idx])
                    except (ValueError, IndexError):
                        length_val = None
                    idx += 1
                fo = None
                if has_fanout:
                    try:
                        fo = int(parts[idx])
                    except (ValueError, IndexError):
                        fo = None
                    idx += 1
                driver_tok = parts[idx] if idx < len(parts) else ""
                sink_toks = parts[idx + 1:] if idx + 1 < len(parts) else []
                drv_inst, drv_pin = _split_inst_pin(driver_tok)
                if fo is None:
                    fo = max(len(sink_toks), 1)
                net_fanout[net] = fo
                if length_val is not None:
                    net_length_um[net] = length_val
                # Canonical "inst/pin" sink encoding for downstream lookups.
                canon_sinks: List[str] = []
                for st in sink_toks:
                    si, sp = _split_inst_pin(st)
                    if si:
                        canon_sinks.append(f"{si}/{sp}" if sp else si)
                if canon_sinks:
                    net_sinks[net] = canon_sinks
                if drv_inst:
                    inst_to_nets.setdefault(drv_inst, []).append(net)
                    if drv_pin:
                        inst_pin_to_net[(drv_inst, drv_pin)] = net
        return net_fanout, inst_to_nets, net_length_um, net_sinks, inst_pin_to_net

    @staticmethod
    def _load_sibling_slacks(path: Optional[Path]) -> Dict[str, List[float]]:
        """Load per-instance top-N slacks. Expected format (plain text, one
        block per inst):

            === <inst_name> ===
            <endpoint>   <slack_ps>
            <endpoint>   <slack_ps>
            ...

        Values may be in ns or ps — if |max|<10 we assume ns and convert.
        This file is what `eco_top_paths_through inst <x> 5` + a wrapper
        should dump each iteration for every inst on the worst path.
        """
        out: Dict[str, List[float]] = {}
        if not path or not path.exists():
            return out
        cur = None
        buf: List[float] = []
        hdr = re.compile(r"^===\s*(\S+)\s*===")
        row = re.compile(r"^\S.*?\s+(-?\d+\.\d+)\s*$")
        def _flush():
            if cur and buf:
                out[cur] = buf[:]
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = hdr.match(ln)
            if m:
                _flush()
                cur = m.group(1)
                buf = []
                continue
            m = row.match(ln)
            if m and cur:
                buf.append(float(m.group(1)))
        _flush()
        # ns → ps if needed
        for inst, slks in out.items():
            if slks and max(abs(s) for s in slks) < 10.0:
                out[inst] = [round(s * 1000.0, 2) for s in slks]
        return out

    @staticmethod
    def _load_fanout_rank(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        """Load per-driver slack-ranked fanout sinks. Expected format:

            === <driver_inst> FO=<n> ===
            <sink_iterm>   <slack_ps_or_ns>
            <sink_iterm>   <slack_ps_or_ns>
            ...

        Written by Tcl proc `eco_dump_fanout_ranks` (batched for all
        high-FO drivers on the worst path). First row is worst slack.
        Values auto-converted ns->ps if |max|<10.

        Returns {driver_inst: {"fanout": N, "sinks": [(sink, slack_ps)...]}}.
        """
        out: Dict[str, Dict[str, Any]] = {}
        if not path or not path.exists():
            return out
        hdr = re.compile(r"^===\s*(\S+)(?:\s+FO=(\d+))?\s*===")
        row = re.compile(r"^(\S+)\s+(-?\d+\.\d+)\s*$")
        cur: Optional[str] = None
        cur_fo: Optional[int] = None
        buf: List[Tuple[str, float]] = []

        def _flush():
            if cur and buf:
                out[cur] = {"fanout": cur_fo if cur_fo is not None else len(buf),
                            "sinks": buf[:]}

        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = hdr.match(ln)
            if m:
                _flush()
                cur = m.group(1)
                cur_fo = int(m.group(2)) if m.group(2) else None
                buf = []
                continue
            m = row.match(ln)
            if m and cur:
                buf.append((m.group(1), float(m.group(2))))
        _flush()
        # ns -> ps if needed (per-driver scale)
        for drv, data in out.items():
            sinks = data["sinks"]
            if sinks and max(abs(s) for _, s in sinks) < 10.0:
                data["sinks"] = [(k, round(v * 1000.0, 2)) for k, v in sinks]
        return out

    @staticmethod
    def _load_recent_eco_actions(path: Optional[Path], last_n: int = 6) -> List[Dict[str, Any]]:
        """Distill recent ECO actions from run_history.json / eco_history.json.
        Each record may contain {"iteration": N, "commands": ["replace_cell X Y", ...]}.
        We lift resize/replace commands into the documented per-action schema.
        """
        if not path or not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        actions: List[Dict[str, Any]] = []
        for rec in data[-last_n * 3:]:           # scan a bit more, we filter
            it = rec.get("iteration")
            for cmd in rec.get("commands", []) or []:
                toks = cmd.split()
                if not toks:
                    continue
                op = toks[0]
                if op in ("replace_cell", "eco_resize_gate") and len(toks) >= 3:
                    actions.append({"iter": it, "action": "resize", "inst": toks[1], "old": "", "new": toks[2]})
                elif op == "eco_insert_buffer" and len(toks) >= 5:
                    actions.append({"iter": it, "action": "insert_buffer", "inst": toks[4], "old": "", "new": toks[3]})
                elif op == "eco_remove_buffer" and len(toks) >= 2:
                    actions.append({"iter": it, "action": "remove_buffer", "inst": toks[1], "old": "", "new": ""})
                elif op == "eco_clone_gate" and len(toks) >= 3:
                    actions.append({"iter": it, "action": "clone", "inst": toks[1], "old": "", "new": toks[2]})
                elif op == "eco_clone_gate_worst_half" and len(toks) >= 3:
                    actions.append({"iter": it, "action": "clone_worst_half", "inst": toks[1], "old": "", "new": toks[2]})
        return actions[-last_n:]

    # ------------------------------ enrichment -----------------------------

    def _library_options(self, cur_cell_type: str):
        """Return (upsize_list, vt_swap_suggestion, cur_input_cap_ff).

        `upsize_list` and `vt_swap_suggestion` are strings of the form
        ``"<master>@<cap_ff>"`` so the prompt conveys the input-pin-cap
        cost of each swap candidate. `cur_input_cap_ff` is the current
        cell's own input-pin cap so the model can compute Δcap = new − cur.
        """
        empty = ([], None, None)
        if not self.library:
            return empty
        cur = self.library.cells.get(cur_cell_type)
        if cur is None:
            return empty
        cur_cap = round(cur.max_input_cap_ff, 3) if cur.max_input_cap_ff else None
        cur_rank = _parse_drive_rank(cur.drive_strength)
        cur_vt = VT_SPEED.get(cur.vt_flavor, -1)

        try:
            alts = self.library.get_equivalent_cells(cur_cell_type, exclude_same=True, cross_vt=True)
        except Exception:
            return empty

        upsize: List[Tuple[Tuple[int, int], str]] = []
        vt_candidates: List[Tuple[int, str]] = []
        for a in alts:
            a_rank = _parse_drive_rank(a.drive_strength)
            a_vt = VT_SPEED.get(a.vt_flavor, -1)
            a_cap = round(a.max_input_cap_ff, 3) if a.max_input_cap_ff else 0.0
            label = f"{a.name}@{a_cap}"
            if a.vt_flavor == cur.vt_flavor and a_rank > cur_rank:
                upsize.append((a_rank, label))
            if a.drive_strength == cur.drive_strength and a_vt > cur_vt:
                vt_candidates.append((a_vt, label))

        upsize.sort()
        up_labels = [lbl for _, lbl in upsize[:5]]
        vt_candidates.sort(reverse=True)            # fastest VT first
        vt_swap = vt_candidates[0][1] if vt_candidates else None
        return up_labels, vt_swap, cur_cap

    def _wire_decomp(self, stage: "Stage") -> Tuple[Optional[float], Optional[float]]:
        """Return (wire_cap_ff, wire_length_um) for `stage`'s output net.

        Method:
          1. Resolve the output net of `stage.inst` on `stage.driver_pin`
             using inst_pin_to_net (exact pin match) with a fallback to
             inst_to_nets[stage.inst][0] when pin info is missing.
          2. Look up the net's routed length (wire_length_um) from
             net_length_um.
          3. Compute wire_cap_ff = stage.load_cap_ff − Σ(sink pin caps).
             Sink pin caps come from library.cells[sink_master].input_caps_ff
             keyed by sink pin name. Sinks whose master is unknown (e.g.
             cells added by previous ECOs not yet re-dumped in
             node_details.csv) are skipped — the resulting wire_cap_ff is
             a lower bound but never negative.

        Returns (None, None) when load_cap_ff or the net mapping is
        unavailable. None values render as empty cells in the TOON.
        """
        if stage.load_cap_ff is None:
            return None, None
        # Extract pin name from driver_pin "inst/PIN".
        drv_pin_name = ""
        if "/" in (stage.driver_pin or ""):
            drv_pin_name = stage.driver_pin.rsplit("/", 1)[1]
        net = None
        if drv_pin_name:
            net = self.inst_pin_to_net.get((stage.inst, drv_pin_name))
        if net is None:
            nets = self.inst_to_nets.get(stage.inst, [])
            net = nets[0] if nets else None
        if net is None:
            return None, None

        length = self.net_length_um.get(net)
        # Sum sink pin caps.
        sink_pin_cap_sum = 0.0
        if self.library is not None:
            for sink in self.net_sinks.get(net, []):
                if "/" not in sink:
                    continue
                s_inst, s_pin = sink.rsplit("/", 1)
                s_master = self.node_meta.get(s_inst, {}).get("master")
                if not s_master:
                    continue
                cell_info = self.library.cells.get(s_master)
                if not cell_info:
                    continue
                cap = cell_info.input_caps_ff.get(s_pin)
                if cap is not None:
                    sink_pin_cap_sum += cap
        wire_cap = stage.load_cap_ff - sink_pin_cap_sum
        if wire_cap < 0:
            wire_cap = 0.0
        return round(wire_cap, 3), (round(length, 3) if length is not None else None)

    def _count_on_other_paths(self, inst: str, all_violated_paths: List[Any], top_idx: int) -> int:
        """How many OTHER violating paths contain this inst."""
        n = 0
        for i, p in enumerate(all_violated_paths):
            if i == top_idx:
                continue
            if any(pt.cell == inst for pt in p.points):
                n += 1
        return n

    # ------------------------------- builder -------------------------------

    def build(
        self,
        report_text_or_path: str,
        iteration: int,
        top_n_paths: int = 5,
    ) -> Dict[str, Any]:
        """Return a dict containing the top-N violating paths with full
        per-stage detail plus shared siblings / fanout_rank / recent ECO
        actions. The caller serializes via to_toon()."""

        # Use the existing parser for TNS + violation count across all paths.
        parser = TimingReportParser()
        all_paths = parser.parse_report(report_text_or_path)
        tns_ps = round(sum(p.slack for p in all_paths), 2) if all_paths else None
        n_viol = len(all_paths)

        top_paths = extract_top_n_paths(report_text_or_path, n=top_n_paths)
        paths_out: List[Dict[str, Any]] = []
        all_path_insts: set = set()

        for idx, (stages, meta) in enumerate(top_paths):
            for s in stages:
                if s.inst in self.node_pos:
                    x, y = self.node_pos[s.inst]
                    s.neighbors_5um = self._neighbor_density(self.node_pos, x, y, 5.0)
                    s.x_um = round(x, 3)
                    s.y_um = round(y, 3)
                if s.fanout is None:
                    nets = self.inst_to_nets.get(s.inst, [])
                    if nets:
                        s.fanout = max(self.net_fanout.get(nn, 1) for nn in nets)
                if not s.is_dff:
                    s.upsize, s.vt_swap, s.input_cap_ff = self._library_options(s.cell)
                # Wire-cap / wire-length decomposition (independent of cell type;
                # DFFs may still have meaningful wire info on Q/QN output nets).
                wc, wl = self._wire_decomp(s)
                s.wire_cap_ff = wc
                s.wire_length_um = wl
                s.on_other_violating_paths = self._count_on_other_paths(s.inst, all_paths, idx)
            path_insts = {s.inst for s in stages}
            all_path_insts |= path_insts
            paths_out.append({
                "path_index": idx + 1,
                "startpoint": meta.get("startpoint", ""),
                "endpoint": meta.get("endpoint", ""),
                "slack_ps": meta.get("slack_ps"),
                "num_stages": len(stages),
                "path": [asdict(s) for s in stages],
            })

        # ---- Nearby endpoints (ranks 6..N) + cell→nearby-paths map ----
        # The path[] block above is limited to TOP_N detailed paths so the
        # model can reason stage-by-stage. But paths just outside the top-N
        # (slack within a few ps of WNS) can tip into worst-violator status
        # when an upstream cell on the top-N paths is touched. We surface
        # those nearby paths in two blocks:
        #
        #   <nearby_endpoints[]>      — ranks 6..N: rank, sp, ep, slack,
        #                               and how many cells are shared with top-N
        #   <shared_cells_to_nearby[]> — per top-N instance that ALSO appears
        #                               on at least one nearby path: list of
        #                               nearby-path ranks + worst nearby slack.
        # The second is the "cascade graph" — model checks "if I touch X,
        # what other paths get perturbed and what is their current headroom".
        NEARBY_LIMIT = 50
        nearby_endpoints_out: List[Dict[str, Any]] = []
        shared_map: Dict[str, List[Tuple[int, float]]] = {}
        if all_paths:
            paths_sorted = sorted(all_paths, key=lambda p: p.slack)
            # Cells on the top-N paths (the ones that have detailed path[] entries).
            top_n_cell_set: set = set()
            for tp in paths_sorted[:top_n_paths]:
                for pt in tp.points:
                    if pt.cell:
                        top_n_cell_set.add(pt.cell)
            seen_key = set()
            for ridx, p in enumerate(paths_sorted):
                rank = ridx + 1
                if rank <= top_n_paths:
                    continue
                if rank > NEARBY_LIMIT:
                    break
                key = (p.startpoint, p.endpoint)
                if key in seen_key:
                    continue
                seen_key.add(key)
                this_cells = {pt.cell for pt in p.points if pt.cell}
                shared = this_cells & top_n_cell_set
                nearby_endpoints_out.append({
                    "rank": rank,
                    "startpoint": p.startpoint,
                    "endpoint": p.endpoint,
                    "slack_ps": round(p.slack, 2),
                    "n_shared_with_top5": len(shared),
                })
                slack_round = round(p.slack, 2)
                for inst in shared:
                    shared_map.setdefault(inst, []).append((rank, slack_round))

        # Sort the cell→nearby map: each cell\'s nearby-refs sorted by slack
        # (worst path first); the outer list sorted by worst-slack ascending
        # so the model sees the highest-risk cells first.
        shared_cells_out: List[Dict[str, Any]] = []
        for inst, refs in shared_map.items():
            refs_sorted = sorted(refs, key=lambda x: x[1])
            ranks = [r for r, _ in refs_sorted]
            worst_slack = refs_sorted[0][1]
            shared_cells_out.append({
                "inst": inst,
                "n_nearby_paths": len(refs_sorted),
                "nearby_path_ranks": ranks,
                "worst_nearby_slack_ps": worst_slack,
            })
        shared_cells_out.sort(key=lambda r: r["worst_nearby_slack_ps"])

        # Siblings: keyed by inst, limited to stages that appear on ANY of
        # the top-N paths. Lets the LLM check safety margins before picking
        # a fix that could push a near-miss path below the current WNS.
        siblings: Dict[str, List[float]] = {}
        for inst in all_path_insts:
            if inst in self.siblings_by_inst:
                siblings[inst] = self.siblings_by_inst[inst][:5]

        # Fanout-rank: ranked sink lists for high-FO drivers found on any
        # top-N path. Includes worst-half slice for clone_gate_worst_half.
        fanout_rank: Dict[str, Dict[str, Any]] = {}
        for inst in all_path_insts:
            rec = self.fanout_rank_by_driver.get(inst)
            if not rec:
                continue
            sinks = rec.get("sinks", []) or []
            fo = rec.get("fanout") or len(sinks)
            half = max(1, (len(sinks) + 1) // 2)
            fanout_rank[inst] = {
                "fanout": fo,
                "worst_half_count": half,
                "worst_half_sinks": [k for k, _ in sinks[:half]],
                "sinks": sinks,
            }

        top_meta = top_paths[0][1] if top_paths else {}
        return {
            "iteration": iteration,
            "wns_endpoint": top_meta.get("endpoint", ""),
            "wns_slack_ps": top_meta.get("slack_ps"),
            "clock_period_ps": top_meta.get("clock_period_ps"),
            "tns_ps": tns_ps,
            "num_violating_endpoints": n_viol,
            "paths": paths_out,
            # Back-compat for codex_exec allow-inst extraction: flattened top path.
            "path": paths_out[0]["path"] if paths_out else [],
            "nearby_endpoints": nearby_endpoints_out,
            "shared_cells_to_nearby": shared_cells_out,
            "siblings": siblings,
            "fanout_rank": fanout_rank,
            "recent_eco_actions": self.recent_eco_actions,
        }


# ---------------------------------------------------------------------------
# TOON serializer
# ---------------------------------------------------------------------------

def _toon_scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # Drop trailing zeros but keep precision.
        return f"{v:g}"
    s = str(v)
    # Quote if contains separators that could confuse CSV rows.
    if "," in s or "\n" in s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _toon_emit_table(key: str, rows: List[Dict[str, Any]], cols: List[str]) -> str:
    out = [f"{key}[{len(rows)}]{{{','.join(cols)}}}:"]
    for r in rows:
        vals: List[str] = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, list):
                # Lists inside a row → bracket-with-semicolon form.
                v = "[" + ";".join(_toon_scalar(x) for x in v) + "]"
            vals.append(_toon_scalar(v))
        out.append(",".join(vals))
    return "\n".join(out)


def to_toon(d: Dict[str, Any]) -> str:
    """Emit a TOON document for the dict shape produced by ContextBuilderV5.build().
    Emits up to 3 violating paths in a single `path[]` table (stage uses the
    "p{idx}.{stage}" format so paths are distinguishable while still flat)."""
    lines: List[str] = []

    # Scalars first (stable order).
    scalar_keys = ["iteration", "wns_endpoint", "wns_slack_ps", "clock_period_ps",
                   "tns_ps", "num_violating_endpoints"]
    for k in scalar_keys:
        lines.append(f"{k}: {_toon_scalar(d.get(k))}")
    lines.append("")

    # Per-path header lines listing slack/endpoint for each of the top-N paths.
    paths = d.get("paths") or []
    if paths:
        for p in paths:
            lines.append(
                f"# path{p.get('path_index')}: slack={_toon_scalar(p.get('slack_ps'))}ps "
                f"stages={p.get('num_stages')} "
                f"startpoint={p.get('startpoint')} endpoint={p.get('endpoint')}"
            )
        lines.append("")

    # path[] — tabular, across all 3 paths. "stage" encodes the path index:
    # e.g. "1.5" = stage 5 on path #1. dont_touch, is_dff, upsize, vt_swap,
    # on_other_violating_paths are per-stage identical across paths when the
    # same inst recurs.
    path_cols = [
        "stage", "driver_pin", "cell", "fanout",
        "load_cap_ff", "wire_cap_ff",
        "in_slew_ps", "out_slew_ps", "slew_jump_from_prev_ps",
        "cell_delay_ps", "wire_delay_ps", "wire_length_um",
        "x_um", "y_um", "neighbors_5um", "is_dff", "dont_touch",
        "on_other_violating_paths", "input_cap_ff", "upsize", "vt_swap",
    ]
    def _slew_jump(prev_row, curr_row):
        # slew_jump_from_prev_ps = curr.in_slew - prev.out_slew. Captures
        # WIRE-induced slew degradation between adjacent stages: when the
        # driver puts out a clean edge but the receiver sees a degraded one,
        # the wire is the culprit (long net + capacitive load) and a mid-net
        # buffer insertion will recover the edge.
        try:
            ci = curr_row.get("in_slew_ps")
            po = prev_row.get("out_slew_ps")
            if ci is None or po is None:
                return None
            return round(float(ci) - float(po), 2)
        except (TypeError, ValueError):
            return None

    merged_rows: List[Dict[str, Any]] = []
    if paths:
        for p in paths:
            pidx = p.get("path_index", 1)
            prev_row = None
            for row in p.get("path", []):
                r = dict(row)
                r["stage"] = f"{pidx}.{row.get('stage')}"
                r["slew_jump_from_prev_ps"] = _slew_jump(prev_row, row) if prev_row else None
                merged_rows.append(r)
                prev_row = row
    else:
        merged_rows = []
        prev_row = None
        for row in d.get("path", []):
            r = dict(row)
            r["slew_jump_from_prev_ps"] = _slew_jump(prev_row, row) if prev_row else None
            merged_rows.append(r)
            prev_row = row
    lines.append(_toon_emit_table("path", merged_rows, path_cols))
    lines.append("")

    # nearby_endpoints[]  — paths just outside top-N. Lets the model see
    # the sibling-tip risk before committing a move that perturbs upstream
    # loads. n_shared_with_top5 is the count of cells this path shares with
    # the detailed path[] block above.
    ne = d.get("nearby_endpoints", []) or []
    if ne:
        ne_cols = ["rank", "startpoint", "endpoint", "slack_ps",
                   "n_shared_with_top5"]
        lines.append(_toon_emit_table("nearby_endpoints", ne, ne_cols))
        lines.append("")

    # shared_cells_to_nearby[]  — per top-N instance, list of nearby-path
    # ranks that share it + worst slack among those. This IS the cascade
    # graph: touching `inst` perturbs the listed nearby paths.
    sc = d.get("shared_cells_to_nearby", []) or []
    if sc:
        sc_cols = ["inst", "n_nearby_paths", "nearby_path_ranks",
                   "worst_nearby_slack_ps"]
        lines.append(_toon_emit_table("shared_cells_to_nearby", sc, sc_cols))
        lines.append("")

    # siblings{inst: [s1..s5]}  — emit as a table with inst + slack columns
    sib = d.get("siblings", {}) or {}
    if sib:
        # Max columns we'll flatten to:
        maxn = max(len(v) for v in sib.values())
        sib_cols = ["inst"] + [f"s{i+1}_ps" for i in range(maxn)]
        rows = []
        for inst, slks in sib.items():
            row = {"inst": inst}
            for i, sv in enumerate(slks):
                row[f"s{i+1}_ps"] = sv
            rows.append(row)
        lines.append(_toon_emit_table("siblings", rows, sib_cols))
        lines.append("")

    # fanout_rank{inst: {...}} — per-driver clone guidance
    fr = d.get("fanout_rank", {}) or {}
    if fr:
        fr_cols = ["driver", "fanout", "worst_half_count", "worst_half_sinks"]
        fr_rows = []
        for drv, data in fr.items():
            fr_rows.append({
                "driver": drv,
                "fanout": data.get("fanout"),
                "worst_half_count": data.get("worst_half_count"),
                "worst_half_sinks": data.get("worst_half_sinks", []),
            })
        lines.append(_toon_emit_table("fanout_rank", fr_rows, fr_cols))
        lines.append("")

    # recent_eco_actions[]
    rea = d.get("recent_eco_actions", []) or []
    if rea:
        rea_cols = ["iter", "action", "inst", "old", "new"]
        lines.append(_toon_emit_table("recent_eco_actions", rea, rea_cols))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# __main__  (CLI compatible with auto_runme_loop_v5.py call site)
# ---------------------------------------------------------------------------

# Repo root resolved from this file's location: src/src/context_builder.py
_DEFAULT_WORK = Path(__file__).resolve().parents[1]

def main() -> None:
    ap = argparse.ArgumentParser()
    # Original v4 args (preserved for drop-in compatibility)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--max-paths", type=int, default=1,
                    help="Unused in v5 (always top-1); kept for CLI compat.")
    ap.add_argument("--node-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/node_details.csv"))
    ap.add_argument("--net-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/net_details.csv"))
    ap.add_argument("--neighbor-hops", type=int, default=1, help="Unused in v5 (CLI compat).")
    ap.add_argument("--bbox-margin-um", type=float, default=3.0, help="Unused in v5 (CLI compat).")
    ap.add_argument("--qor-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/qor_history.json"))
    ap.add_argument("--displacement-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/displacement.json"),
                    help="Unused in v5 (CLI compat).")
    ap.add_argument("--net-reports-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/net_reports.txt"),
                    help="Unused in v5 (CLI compat).")
    ap.add_argument("--out", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/context_v5.toon"))

    # v5-specific
    ap.add_argument("--timing-rpt", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/dynamic_timing_rpt.txt"))
    ap.add_argument("--sibling-slacks-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/sibling_slacks.txt"))
    ap.add_argument("--fanout-rank-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/dynamic/fanout_rank.txt"))
    ap.add_argument("--eco-history-file", type=str,
                    default=str(_DEFAULT_WORK / "prompts/run_history.json"))
    ap.add_argument("--lib-dir", type=str,
                    default=os.environ.get("ASAP7_PDK_LIB",
                        str(_DEFAULT_WORK / "asap7/lib/NLDM/")))
    ap.add_argument("--print", action="store_true",
                    help="Also print the TOON to stdout for inspection.")
    args = ap.parse_args()

    lib = None
    lib_path = Path(args.lib_dir)
    if lib_path.exists():
        try:
            lib = LibraryExtractor(lib_path)
        except Exception as e:
            print(f"# warning: LibraryExtractor failed: {e}")

    builder = ContextBuilderV5(
        library=lib,
        node_file=Path(args.node_file),
        net_file=Path(args.net_file),
        sibling_slacks_file=Path(args.sibling_slacks_file),
        fanout_rank_file=Path(args.fanout_rank_file),
        eco_history_file=Path(args.eco_history_file),
    )

    ctx = builder.build(args.timing_rpt, iteration=args.iteration)
    toon = to_toon(ctx)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Structured JSON for programmatic consumers (codex_exec_v5 loads this).
    out.write_text(json.dumps(ctx, indent=2), encoding="utf-8")
    # TOON rendering for LLM prompt embedding (token-compact view).
    toon_out = out.with_suffix(".toon")
    toon_out.write_text(toon, encoding="utf-8")

    if args.print or not lib:
        print(toon)


if __name__ == "__main__":
    main()