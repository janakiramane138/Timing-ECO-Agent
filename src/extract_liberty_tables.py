"""Extract a compact cell-delay reference from ASAP7 NLDM Liberty libraries.

For each (family, drive, VT) cell that the LLM might reasonably propose
during ECO, we record:
  - input_cap_ff (worst across input pins)
  - cell_delay_ps + out_slew_ps at a 4 x 3 grid (load_ff x in_slew_ps)
    averaged across rise/fall arcs and across input pins (worst-case)

The output is written to prompts/static/cell_delay_reference.toon and
injected into the LLM's system prompt as a cached reference block. This
lets the model do Liberty-accurate first-principles ΔWNS predictions
instead of guessing scaling ratios from training-data intuition.

Run once per technology — re-run only if Liberty files change. The
output file is treated as data, not generated artifact.
"""
from __future__ import annotations
import gzip
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration — repo-relative paths (no hardcoded absolutes).
#
# This file lives at <repo>/src/extract_liberty_tables.py; the repo root is
# one level up. LIB_DIR points at the bundled ASAP7 NLDM Liberty directory;
# override it with the ASAP7_PDK environment variable to target a different
# PDK. Re-run this script once per technology (see the repo README).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PDK_ROOT = Path(os.environ.get("ASAP7_PDK", _REPO_ROOT / "asap7"))

LIB_DIR = _PDK_ROOT / "lib" / "NLDM"
OUTPUT  = _REPO_ROOT / "prompts" / "static" / "cell_delay_reference.toon"

# Only TT corner — matches what OpenROAD loads for STA in this flow.
LIB_PATTERN = "*{group}_{vt}_{corner}_nldm_*.lib*"

# Combinational families we dump. Curated for setup-WNS ECO relevance.
# Sequential (DFF/LATCH) and clock cells are excluded — model cannot touch
# them. SRAM is excluded — also untouchable.
FAMILIES = [
    "BUF", "INV",                               # buffers and inverters (most-used)
    "NAND2", "NAND3", "NAND4",                  # basic NAND
    "NOR2", "NOR3", "NOR4",                     # basic NOR
    "AND2", "AND3", "AND4", "AND5",             # basic AND
    "OR2", "OR3", "OR4", "OR5",                 # basic OR
    "XOR2", "XNOR2",                            # XOR/XNOR
    "AO21", "AO22", "AO221", "AO222", "AO32",   # AND-OR
    "OA21", "OA22", "OA211", "OA221", "OA222",  # OR-AND
    "AOI21", "AOI22", "AOI221", "AOI222",       # AND-OR-INVERT
    "OAI21", "OAI22", "OAI221", "OAI222",       # OR-AND-INVERT
    "MAJI3",                                    # majority
    "FA", "HA",                                 # adders (hardmacros, included for VT-swap consideration)
]

# Grid we present (chosen to match path[] typical ranges in the design).
TARGET_LOADS_FF = [1.5, 5.0, 15.0, 40.0]
TARGET_SLEWS_PS = [10.0, 30.0, 80.0]

# Library group (.lib filename prefix) per family.
FAMILY_GROUP = {
    "BUF": "INVBUF", "INV": "INVBUF",
    "NAND2": "SIMPLE", "NAND3": "SIMPLE", "NAND4": "SIMPLE",
    "NOR2": "SIMPLE", "NOR3": "SIMPLE", "NOR4": "SIMPLE",
    "AND2": "SIMPLE", "AND3": "SIMPLE", "AND4": "SIMPLE", "AND5": "SIMPLE",
    "OR2": "SIMPLE", "OR3": "SIMPLE", "OR4": "SIMPLE", "OR5": "SIMPLE",
    "XOR2": "SIMPLE", "XNOR2": "SIMPLE",
    "MAJI3": "SIMPLE",
    "FA": "SIMPLE", "HA": "SIMPLE",
    "AO21": "AO", "AO22": "AO", "AO221": "AO", "AO222": "AO", "AO32": "AO",
    "AOI21": "AO", "AOI22": "AO", "AOI221": "AO", "AOI222": "AO",
    "OA21": "OA", "OA22": "OA", "OA211": "OA", "OA221": "OA", "OA222": "OA",
    "OAI21": "OA", "OAI22": "OA", "OAI221": "OA", "OAI222": "OA",
}

VTS = [("L", "LVT"), ("R", "RVT"), ("SL", "SLVT")]   # suffix → lib substring


# ---------------------------------------------------------------------------
# Liberty parsing
# ---------------------------------------------------------------------------

def _read_lib(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_cell_blocks(text: str):
    """Yield (cell_name, body_text) for every cell. Uses brace counting since
    cell blocks have arbitrarily-nested braces."""
    i = 0
    pat_start = re.compile(r"  cell \(([^)]+)\) \{")
    while True:
        m = pat_start.search(text, i)
        if not m:
            return
        name = m.group(1)
        start = m.end()
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{": depth += 1
            elif c == "}": depth -= 1
            j += 1
        # j is one past the matching closing brace
        body = text[start:j - 1]
        yield name, body
        i = j


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_table(body: str, table_kind: str) -> Optional[Tuple[List[float], List[float], List[List[float]]]]:
    """Locate the first `<table_kind> (...) { index_1 ... index_2 ... values ... }`
    inside body and return (slew_axis, load_axis, values_2d).
    Returns None if not found / unparseable.
    """
    m = re.search(rf"{table_kind}\s*\(\s*\S+\s*\)\s*\{{(.*?)\}}", body, re.DOTALL)
    if not m:
        return None
    blk = m.group(1)
    m1 = re.search(r'index_1\s*\(\s*"([^"]+)"', blk)
    m2 = re.search(r'index_2\s*\(\s*"([^"]+)"', blk)
    mv = re.search(r"values\s*\((.*?)\);", blk, re.DOTALL)
    if not (m1 and m2 and mv):
        return None
    slews = [float(x) for x in _NUM.findall(m1.group(1))]
    loads = [float(x) for x in _NUM.findall(m2.group(1))]
    rows: List[List[float]] = []
    # values() body is rows of `"a, b, c", \n "d, e, f", ...`
    for line in mv.group(1).split('"'):
        nums = [float(x) for x in _NUM.findall(line)]
        if len(nums) == len(loads):
            rows.append(nums)
    if len(rows) != len(slews):
        return None
    return slews, loads, rows


def _iter_braced_blocks(text: str, opener_pattern: str):
    """Yield (header_match, body_text) for every brace-block whose opening line
    matches opener_pattern (which must include the trailing '{'). Uses brace
    counting so nested braces inside the block are handled correctly.
    """
    pat = re.compile(opener_pattern)
    i = 0
    while True:
        m = pat.search(text, i)
        if not m:
            return
        start = m.end()
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{": depth += 1
            elif c == "}": depth -= 1
            j += 1
        yield m, text[start:j - 1]
        i = j


def _parse_cell(name: str, body: str) -> Optional[Dict]:
    """Extract: input_cap_ff (max over input pins) + per-arc delay/transition tables.
    Returns the cell's reduced fingerprint or None if the cell has no usable timing arcs.
    """
    # Input pin capacitances — pick max input pin cap (most pessimistic upstream load).
    # Use brace-counting because pin blocks contain nested timing/table blocks.
    input_caps: List[float] = []
    output_bodies: List[str] = []
    for hdr, pin_body in _iter_braced_blocks(body, r"pin \(([^)]+)\)\s*\{"):
        if "direction : input" in pin_body:
            cap_m = re.search(r"\bcapacitance\s*:\s*([-+\d.eE]+)\s*;", pin_body)
            if cap_m:
                try:
                    input_caps.append(float(cap_m.group(1)))
                except ValueError:
                    pass
        elif "direction : output" in pin_body:
            output_bodies.append(pin_body)
    if not input_caps:
        return None
    input_cap_ff = max(input_caps)

    # Output pin timing tables. For multi-input gates we keep the WORST-case
    # arc (highest cell_delay at our reference grid point of (slew=20, load=23)).
    best_arc = None
    best_score = -1.0
    for out_body in output_bodies:
      for _hdr, arc_body in _iter_braced_blocks(out_body, r"timing\s*\(\)\s*\{"):
        # parse rise + fall
        cr = _parse_table(arc_body, "cell_rise")
        cf = _parse_table(arc_body, "cell_fall")
        rt = _parse_table(arc_body, "rise_transition")
        ft = _parse_table(arc_body, "fall_transition")
        if not (cr and cf and rt and ft):
          continue
        # Score by avg delay near (slew=20, load=23) — middle of the lib grid
        slews, loads, rvals = cr
        try:
          sidx = min(range(len(slews)), key=lambda i: abs(slews[i] - 20.0))
          lidx = min(range(len(loads)), key=lambda i: abs(loads[i] - 23.0))
          score = rvals[sidx][lidx]
        except (IndexError, ValueError):
          score = 0
        if score > best_score:
          best_score = score
          best_arc = (cr, cf, rt, ft)
    if best_arc is None:
        return None
    cr, cf, rt, ft = best_arc
    return dict(
        input_cap_ff=input_cap_ff,
        cell_rise=cr, cell_fall=cf,
        rise_trans=rt, fall_trans=ft,
    )


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------

def _interp1(x: float, xs: List[float], ys: List[float]) -> float:
    """Linear interpolation/extrapolation."""
    if x <= xs[0]:
        # extrapolate at low end using first two points
        if len(xs) >= 2 and xs[1] != xs[0]:
            t = (x - xs[0]) / (xs[1] - xs[0])
            return ys[0] + t * (ys[1] - ys[0])
        return ys[0]
    if x >= xs[-1]:
        if len(xs) >= 2 and xs[-1] != xs[-2]:
            t = (x - xs[-2]) / (xs[-1] - xs[-2])
            return ys[-2] + t * (ys[-1] - ys[-2])
        return ys[-1]
    # locate bracket
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _bilinear(slew_q: float, load_q: float,
              slew_ax: List[float], load_ax: List[float],
              vals: List[List[float]]) -> float:
    """Look up vals[slew_idx][load_idx] at (slew_q, load_q) with linear interp."""
    # Interp along load for each slew row, then interp those results along slew
    per_slew = [_interp1(load_q, load_ax, row) for row in vals]
    return _interp1(slew_q, slew_ax, per_slew)


def _reduce_cell(parsed: Dict, target_slews: List[float], target_loads: List[float]) -> Dict:
    """Average rise+fall, interpolate to target grid. Returns:
       { 'input_cap_ff':..., 'delay_grid':[[d_ps,...] for each slew],
         'slew_grid':[[s_ps,...] for each slew] }
    """
    cr_s, cr_l, cr_v = parsed["cell_rise"]
    cf_s, cf_l, cf_v = parsed["cell_fall"]
    rt_s, rt_l, rt_v = parsed["rise_trans"]
    ft_s, ft_l, ft_v = parsed["fall_trans"]

    delay_grid: List[List[float]] = []
    slew_grid: List[List[float]] = []
    for s in target_slews:
        d_row: List[float] = []
        sl_row: List[float] = []
        for L in target_loads:
            d_rise = _bilinear(s, L, cr_s, cr_l, cr_v)
            d_fall = _bilinear(s, L, cf_s, cf_l, cf_v)
            s_rise = _bilinear(s, L, rt_s, rt_l, rt_v)
            s_fall = _bilinear(s, L, ft_s, ft_l, ft_v)
            d_row.append(0.5 * (d_rise + d_fall))
            sl_row.append(0.5 * (s_rise + s_fall))
        delay_grid.append(d_row)
        slew_grid.append(sl_row)
    return dict(
        input_cap_ff=parsed["input_cap_ff"],
        delay_grid=delay_grid,
        slew_grid=slew_grid,
    )


# ---------------------------------------------------------------------------
# Drive-strength ordering (for the "upsize ladder")
# ---------------------------------------------------------------------------

def _drive_strength(cell_name: str) -> float:
    """Parse cell name into numeric drive strength. e.g. BUFx6f → 6, HAxp5 → 0.5."""
    s = re.sub(r"_ASAP7_75t_\w+$", "", cell_name)
    m = re.search(r"x(p?)(\d+)(f?)$", s)
    if not m:
        return 0.0
    p_prefix = m.group(1)
    digits = float(m.group(2))
    if p_prefix == "p":
        return digits / 10.0  # xp5 = 0.5
    return digits


def _family_of(cell_name: str) -> str:
    s = re.sub(r"_ASAP7_75t_\w+$", "", cell_name)
    m = re.match(r"^(.+?)x[a-z]?\d+\w*$", s)
    if m:
        return m.group(1)
    return s


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_all() -> Dict[str, Dict[str, Dict]]:
    """Return { family: { full_cell_name: reduced_dict } } across all VTs."""
    results: Dict[str, Dict[str, Dict]] = {}
    groups_needed = sorted({FAMILY_GROUP[f] for f in FAMILIES if f in FAMILY_GROUP})
    for grp in groups_needed:
        for vt_suffix, vt_substring in VTS:
            matches = list(LIB_DIR.glob(f"*{grp}_{vt_substring}_TT*.lib*"))
            if not matches:
                continue
            lib_path = matches[0]
            try:
                text = _read_lib(lib_path)
            except Exception as e:
                print(f"  [WARN] could not read {lib_path}: {e}", file=sys.stderr)
                continue
            for name, body in _iter_cell_blocks(text):
                family = _family_of(name)
                if family not in FAMILIES:
                    continue
                parsed = _parse_cell(name, body)
                if not parsed:
                    continue
                reduced = _reduce_cell(parsed, TARGET_SLEWS_PS, TARGET_LOADS_FF)
                results.setdefault(family, {})[name] = reduced
    return results


# ---------------------------------------------------------------------------
# TOON-style output formatter
# ---------------------------------------------------------------------------

def format_reference(data: Dict[str, Dict[str, Dict]]) -> str:
    out: List[str] = []
    out.append("# CELL DELAY REFERENCE (ASAP7, TT corner — matches what OpenROAD loads for STA)")
    out.append("#")
    out.append("# For each cell: input_cap_ff and 'cell_delay_ps / out_slew_ps' at a 4x3 grid.")
    out.append("# Grid columns (output_load_ff): " +
               ", ".join(f"{L}" for L in TARGET_LOADS_FF))
    out.append("# Grid rows (input_slew_ps):    " +
               ", ".join(f"{s}" for s in TARGET_SLEWS_PS))
    out.append("#")
    out.append("# Reading example:")
    out.append("#   `BUFx6f_SL  ic=1.297  (1.5,10)=8.1/9 (5,10)=11.0/15 ...`")
    out.append("#   means: BUFx6f at the SL corner, input_cap_ff = 1.297. At output load")
    out.append("#   = 1.5 ff and input_slew = 10 ps, cell_delay = 8.1 ps and out_slew = 9 ps.")
    out.append("#")
    out.append("# Use these tables to predict ΔWNS first-principles, then apply the")
    out.append("# move-type calibration ratio from <prediction_calibration>.")
    out.append("#")
    out.append("# VT swap: same family+drive across L/R/SL preserves drive strength;")
    out.append("# typical input_cap delta is < 5%. R is slowest, SL is fastest.")
    out.append("#")

    for family in FAMILIES:
        cells = data.get(family) or {}
        if not cells:
            continue
        # Sort cells by drive strength then VT (SL → L → R for human readability)
        vt_order = {"_SL": 0, "_L": 1, "_R": 2}
        def _sortkey(name):
            ds = _drive_strength(name)
            vt = ""
            for k in vt_order:
                if name.endswith(k):
                    vt = k; break
            return (ds, vt_order.get(vt, 9), name)
        sorted_names = sorted(cells.keys(), key=_sortkey)

        out.append("")
        out.append(f"family {family} ({len(cells)} cells)")
        # Compact upsize ladder for each VT (highlights what drives exist)
        for vt_suffix in ("SL", "L", "R"):
            ladder = [n for n in sorted_names if n.endswith(f"_{vt_suffix}")]
            if not ladder:
                continue
            short = []
            for n in ladder:
                ic = cells[n]["input_cap_ff"]
                # short label = drive part only
                drv = re.search(r"(x[a-z]?\d+f?)_ASAP7", n)
                lbl = drv.group(1) if drv else n
                short.append(f"{lbl}(ic={ic:.3f})")
            out.append(f"  {vt_suffix} drive ladder: " + " → ".join(short))

        # Per-cell delay/slew table — compact form:
        #   short_name (e.g. "BUFx2_SL" not "BUFx2_ASAP7_75t_SL") since
        #   ic already shown in the ladder above. Per row we only show
        #   in_slew + 4 columns of "delay/slew".
        L_axis = TARGET_LOADS_FF
        S_axis = TARGET_SLEWS_PS
        out.append(f"  load_ff cols: " + ", ".join(f"{L:g}" for L in L_axis))
        for name in sorted_names:
            cd = cells[name]
            # Strip "_ASAP7_75t" infix from name — keep only drive_VT
            short = name.replace("_ASAP7_75t", "")
            for si, sval in enumerate(S_axis):
                cells_str = []
                for li, _L in enumerate(L_axis):
                    d = cd["delay_grid"][si][li]
                    sl = cd["slew_grid"][si][li]
                    cells_str.append(f"{d:.1f}/{sl:.0f}")
                out.append(f"  {short} is={sval:g}: " + " ".join(cells_str))
    out.append("")
    return "\n".join(out)


def main():
    print(f"[liberty] extracting cells from {LIB_DIR}", file=sys.stderr)
    data = extract_all()
    n_cells = sum(len(v) for v in data.values())
    n_families = sum(1 for v in data.values() if v)
    print(f"[liberty] extracted {n_cells} cells across {n_families} families",
          file=sys.stderr)
    out = format_reference(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(out, encoding="utf-8")
    print(f"[liberty] wrote {OUTPUT} ({len(out):,} chars, ~{len(out)//4:,} tokens)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
