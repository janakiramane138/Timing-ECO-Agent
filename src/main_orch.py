"""Main ECO loop: OpenROAD ↔ LLM iterative timing closure.

Flow per iteration:
  1. Save pre-ECO cell positions
  2. Run LLM (LLM_call.py) → produces llm_eco.tcl
  3. Source ECO in OpenROAD with error capture
  4. Run incremental placement + global route + parasitic estimation
  5. Generate timing report + metrics
  6. Compute cell displacement
  7. Score QoR change, track best-ever, revert on sustained regression
  8. Rebuild context for next iteration
"""
import json
import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime
from time import sleep
from typing import Optional, List, Dict, Any, Tuple
import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths — repo-root self-locating (no hardcoded absolute paths)
#
# This file lives at <repo>/src/main_orch.py. We resolve the repo root
# from __file__ so the loop works wherever the user has cloned this repo.
# The ASAP7 PDK is expected at <repo>/asap7 by default; override with
# the ASAP7_PDK environment variable if it lives elsewhere on your system.
# ---------------------------------------------------------------------------
import os as _os
_REPO_ROOT = Path(__file__).resolve().parents[1]
workdir = str(_REPO_ROOT) + "/"
ASAP7_PDK = _os.environ.get("ASAP7_PDK", str(_REPO_ROOT / "asap7"))

# OpenROAD binary — resolved from the environment so no source edit is needed.
# Set OPENROAD_BIN to an absolute path to a local build, or leave it unset to
# use whatever `openroad` is on PATH (e.g. an OpenROAD-flow-scripts install).
OpenROAD_bin = "/data/jethiraj/OpenROAD/build/bin/openroad"
#OpenROAD_bin = _os.environ.get("OPENROAD_BIN", "openroad")
OpenROAD_design_tcl = f"{workdir}/OpenROAD_utils/OpenROAD_load_design.tcl"
OpenROAD_design_tcl_best = f"{workdir}/OpenROAD_utils/OpenROAD_load_design_best.tcl"

ECO_out = f"{workdir}/prompts/dynamic/llm_eco.tcl"
ANALYSIS_out = f"{workdir}/prompts/dynamic/llm_analysis.md"
PREV_TARGET_JSON = f"{workdir}/prompts/dynamic/prev_target_path_status.json"
PREV_TARGET_REPORT_TXT = f"{workdir}/prompts/dynamic/prev_target_path_report.txt"
RPT_out = f"{workdir}/prompts/dynamic/dynamic_timing_rpt.txt"
verilog_out = f"{workdir}/prompts/dynamic/eco_applied.v"

NODE_FILE = f"{workdir}/prompts/dynamic/node_details.csv"
NODE_FILE_PRE = f"{workdir}/prompts/dynamic/node_details_pre.csv"
NET_FILE = f"{workdir}/prompts/dynamic/net_details.csv"
NET_REPORTS_FILE = f"{workdir}/prompts/dynamic/net_reports.txt"
DISPLACEMENT_FILE = f"{workdir}/prompts/dynamic/displacement.json"
SIBLING_SLACKS_FILE = f"{workdir}/prompts/dynamic/sibling_slacks.txt"
FANOUT_RANK_FILE = f"{workdir}/prompts/dynamic/fanout_rank.txt"
# Per-iter capture of probe-command outputs (eco_rank_fanout_by_slack,
# eco_top_paths_through, eco_net_sink_report). Cleared at iter start;
# read by LLM_call.py into the <probe_responses> block next iter.
PROBE_RESPONSES_FILE = f"{workdir}/prompts/dynamic/probe_responses.txt"

LLM_CALL = f"{workdir}/src/LLM_call.py"
CONTEXT_BUILD = f"{workdir}/src/context_builder.py"
DYNAMIC_CONTEXT_JSON = f"{workdir}/prompts/dynamic/context.json"

slack_track = Path(workdir) / "prompts/wns_llmcalls.txt"
slack_history_png = Path(workdir) / "prompts/slack_history.png"
METRICS_out = f"{workdir}/prompts/dynamic/metrics.txt"

# Per-iteration QoR dump (one file per iter) + aggregated CSV.
ITER_QOR_DIR = Path(workdir) / "prompts/dynamic/iter_qor"
ITER_METRICS_CSV = Path(workdir) / "prompts/dynamic/iter_metrics.csv"
TOKEN_LOG_JSONL = Path(workdir) / "prompts/dynamic/claude_logs/token_log.jsonl"


# Merged lean history used by context_builder (replaces eco+qor for LLM consumption)
ECO_HISTORY_JSON = Path(workdir) / "prompts/eco_history.json"
RUN_HISTORY_JSON = Path(workdir) / "prompts/run_history.json"


# Final output files (post-loop)
FINAL_VERILOG = f"{workdir}/outputs/final_eco.v"
FINAL_SPEF = f"{workdir}/outputs/final_eco.spef"
FINAL_DEF = f"{workdir}/outputs/final_eco.def"
FINAL_ROUTE_RPT = f"{workdir}/outputs/final_route_report.txt"
FINAL_TIMING_RPT = f"{workdir}/outputs/final_timing_report.txt"
FINAL_DRC_RPT = f"{workdir}/outputs/final_drc_report.txt"
FINAL_SUMMARY = f"{workdir}/outputs/final_summary.txt"

# ---------------------------------------------------------------------------
# Design selection — fail-fast validation BEFORE OpenROAD is spawned.
#
# DESIGN_DIR points to the OpenROAD-flow-scripts result dir for the design we
# want to ECO. It lives outside this repo because flow-scripts produces it.
# TOP is the checkpoint stem (e.g. "5_1_grt" → reads ${DESIGN_DIR}/5_1_grt.odb).
#
# We also pass these (plus ASAP7_PDK) into the OpenROAD subprocess via the
# environment so OpenROAD_load_design.tcl can pick them up via $::env(...).
# ---------------------------------------------------------------------------
#DESIGN_DIR = _os.environ.get("DESIGN_DIR")
DESIGN_DIR = Path(workdir) / _os.environ.get("DESIGN_DIR")
DESIGN_TOP = _os.environ.get("TOP", "6_final")

def _validate_design_dir() -> None:
    """Hard-error if DESIGN_DIR / TOP is missing or its .odb is unreadable.
    Called from run_loop() before spawning OpenROAD so a misconfigured run
    fails immediately instead of looking like a dead iteration."""
    if not DESIGN_DIR:
        raise SystemExit(
            "[FATAL] DESIGN_DIR environment variable is not set.\n"
            "        Set it to the OpenROAD-flow-scripts result dir of the\n"
            "        design you want to ECO, for example:\n"
            "          export DESIGN_DIR=/path/to/OpenROAD-flow-scripts/flow/\n"
            "                            results/asap7/aes/base\n"
            "        Optional: export TOP=6_final   (checkpoint stem; default shown)"
        )
    dd = Path(DESIGN_DIR)
    if not dd.is_dir():
        raise SystemExit(
            f"[FATAL] DESIGN_DIR='{DESIGN_DIR}' is not a directory.\n"
            f"        Check the path and re-export DESIGN_DIR before retrying."
        )
    odb = dd / f"{DESIGN_TOP}.odb"
    sdc = dd / f"{DESIGN_TOP}.sdc"
    missing = [str(f) for f in (odb, sdc) if not f.is_file()]
    if missing:
        raise SystemExit(
            f"[FATAL] Required design files missing under DESIGN_DIR:\n"
            + "".join(f"          - {f}\n" for f in missing)
            + f"        TOP='{DESIGN_TOP}'. Either re-export TOP to a stem that\n"
              f"        has a corresponding .odb + .sdc in {DESIGN_DIR}, or run\n"
              f"        the OpenROAD-flow-scripts upstream stage to produce them."
        )
    print(f"[DESIGN] DESIGN_DIR = {DESIGN_DIR}", flush=True)
    print(f"[DESIGN] TOP        = {DESIGN_TOP}", flush=True)
    print(f"[DESIGN]   .odb     = {odb}", flush=True)
    print(f"[DESIGN]   .sdc     = {sdc}", flush=True)


# Sentinel strings for synchronizing with OpenROAD stdout
# Per-iteration .odb snapshots — only the current best-WNS one is kept.
SNAPSHOT_DIR = Path(workdir) / "prompts/dynamic/snapshots"
# best_grt.odb: best GRT iter WITHIN the current DRT cycle (resets each
#   cycle); this state is what the cycle's DRT routes.
# best_drt.odb: best DRT odb across all cycles — the global best, shipped as
#   finals and used for crash/backtrace recovery. BEST_ODB aliases it so all
#   existing best-recovery code paths resolve to the best-DRT snapshot.
BEST_GRT_ODB = SNAPSHOT_DIR / "best_grt.odb"
BEST_DRT_ODB = SNAPSHOT_DIR / "best_drt.odb"
BEST_ODB = BEST_DRT_ODB
BACKTRACE_NOTICE = Path(workdir) / "prompts/dynamic/backtrace_notice.json"
OPENROAD_SESSION_LOG = Path(workdir) / "prompts/dynamic/openroad_session.log"
LLM_SESSION_LOG = Path(workdir) / "prompts/dynamic/llm_session.log"

MARK = "___ECO_DONE___"
PARA = "__PARA_DONE__"

TOP_PATHS = 5            # how many paths get FULL stage detail in path[]
NEARBY_PATHS = 50        # how many paths the report dumps; ranks 6..N
                         # populate <nearby_endpoints> and 
                         # <shared_cells_to_nearby> blocks. Visibility
                         # into 'paths just outside top-5' lets the model
                         # see sibling-tip risk before committing a move.

# QoR scoring weights
W_WNS = 0.7
W_TNS = 0.5
W_NEIGHBOR = 0.5
W_NEW_VIOL = 0.8

# If no new best-ever WNS is set for this many iterations, kill the
# OpenROAD subprocess and restart it from the best-ever .odb snapshot.
# Backtrace fires AT MOST ONCE per run; if stagnation hits the threshold
# a second time after the backtrace, the loop exits and finals are
# written from the restored best snapshot.
BACKTRACE_THRESHOLD = 4

# Rip-up + detailed-route cycle: every DRT_CYCLE ECO iterations, rip all
# signal-net wires, run detailed_route -droute_end_iter 1, extract
# parasitics, and re-read the SPEF so the LLM can correlate actual
# post-route delays with GRT estimates over the next cycle.
DRT_CYCLE = 8

# Control: when DRT_EVERY_ITER is enabled (env DRT_EVERY_ITER=1/true/yes/on),
# the loop drops the intermediate GRT-estimate iterations entirely. EVERY ECO
# iteration routes the FULL detailed_route + OpenRCX SPEF inline (post-route
# accurate report/scoring), crowns best inline, and the separate periodic-DRT
# block is skipped. Default OFF -> normal DRT_CYCLE behavior, byte-identical.
DRT_EVERY_ITER = _os.environ.get("DRT_EVERY_ITER", "0").strip().lower() in (
    "1", "true", "yes", "on")
SPEF_CYCLE_DIR = Path(workdir) / "prompts/dynamic/spef_cycles"
# Written at the start of every ECO iteration; consumed by LLM_call.py
# to inject <parasitic_context> into the prompt.
DRT_STATE_FILE = Path(workdir) / "prompts/dynamic/drt_state.json"

# Knob: when enabled, append per-DRT-cycle GRT<->DRT correlation rows to
# GRT_DRT_CORR_FILE — capturing BOTH the GRT->DRT shift at each DRT boundary
# (last GRT estimate vs the routed SPEF result) AND the DRT->GRT shift at the
# next cycle's first ECO iter (the SPEF checkpoint vs its first GRT re-estimate).
# Disable with env SAVE_GRT_DRT_CORR=0.
SAVE_GRT_DRT_CORR = _os.environ.get("SAVE_GRT_DRT_CORR", "1") not in ("0", "false", "False", "")
GRT_DRT_CORR_FILE = Path(workdir) / "prompts/dynamic/grt_drt_correlation.csv"

# Dedicated WNS/TNS capture file (parsed for perfect per-iteration tracking).
WNS_TNS_FILE = f"{workdir}/prompts/dynamic/wns_tns.txt"

# ---------------------------------------------------------------------------
# Tcl block: define procs for dumping node/net CSVs inside OpenROAD
# ---------------------------------------------------------------------------
NODE_NET_TCL_BLOCK = r'''
proc write_net_details { file_name } {
  set fp [open $file_name w]
  set block [odb::get_block]
  puts $fp "Net,LengthUm,FanOut,Driver,Sinks"

  foreach net [$block getNets] {
    set sig_type [$net getSigType]
    if {$sig_type == "POWER" || $sig_type == "GROUND"} { continue }

    set net_name [$net getName]
    set net_name [string map {\\ "" [ "" ] "" $ ""} $net_name]

    set driver ""
    set sinks {}
    set xs {}
    set ys {}

    foreach iterm [$net getITerms] {
      set inst [$iterm getInst]
      set inst_name [string map {\\ "" [ "" ] "" $ ""} [$inst getName]]
      set mterm [$iterm getMTerm]
      set pin_name [$mterm getName]
      set io_type [$mterm getIoType]

      set ll [$inst getLocation]
      lappend xs [lindex $ll 0]
      lappend ys [lindex $ll 1]

      if {$io_type == "OUTPUT" || $io_type == "INOUT"} {
        if {$driver == ""} {
          set driver "$inst_name $pin_name"
        }
      } else {
        lappend sinks "$inst_name $pin_name"
      }
    }

    foreach bterm [$net getBTerms] {
      set io_name [$bterm getName]
      set io_type [$bterm getIoType]

      set bpins [$bterm getBPins]
      if {[llength $bpins] > 0} {
        set bpin [lindex $bpins 0]
        set bbox [$bpin getBBox]
        lappend xs [$bbox xMin]
        lappend ys [$bbox yMin]
      }

      if {$io_type == "INPUT"} {
        if {$driver == ""} { set driver "$io_name _IO_" }
      } else {
        lappend sinks "$io_name _IO_"
      }
    }

    set length_um 0.0
    if {[llength $xs] >= 2} {
      set sorted_xs [lsort -integer $xs]
      set sorted_ys [lsort -integer $ys]
      set hpwl_dbu [expr {[lindex $sorted_xs end] - [lindex $sorted_xs 0] \
                        + [lindex $sorted_ys end] - [lindex $sorted_ys 0]}]
      set length_um [format "%.3f" [$block dbuToMicrons $hpwl_dbu]]
    }

    catch {
      set wire [$net getWire]
      if {$wire ne "NULL" && $wire ne ""} {
        set wlen [$wire getLength]
        if {[string is integer -strict $wlen] && $wlen > 0} {
          set length_um [format "%.3f" [$block dbuToMicrons $wlen]]
        }
      }
    }

    set fanout [llength $sinks]
    if {$driver != "" || $fanout > 0} {
      set line "$net_name,$length_um,$fanout,$driver"
      foreach sink $sinks { append line ",$sink" }
      puts $fp $line
    }
  }
  close $fp
}

proc dump_net_reports { reports_file {min_fanout 3} {min_wirelength_um 10.0} } {
  # Emit `report_net` output for every signal net meeting the gates. Targeted at
  # nets the ECO decision matrix actually cares about — keeps the dump bounded
  # (a few hundred nets for a 16k-net design) while still capturing everything
  # with any chance of being picked for buffer insertion.
  set block [odb::get_block]
  set fp [open $reports_file w]

  set emitted 0
  foreach net [$block getNets] {
    set sig_type [$net getSigType]
    if {$sig_type == "POWER" || $sig_type == "GROUND"} { continue }

    # Fanout gate (sinks = loads + outgoing bterms).
    set fo 0
    foreach iterm [$net getITerms] {
      set mt [$iterm getMTerm]
      set io [$mt getIoType]
      if {$io == "INPUT"} { incr fo }
    }
    foreach bterm [$net getBTerms] {
      if {[$bterm getIoType] == "OUTPUT"} { incr fo }
    }
    if {$fo < $min_fanout} { continue }

    # Wirelength gate (HPWL from instance locations — matches write_net_details).
    set xs {}
    set ys {}
    foreach iterm [$net getITerms] {
      set ll [[$iterm getInst] getLocation]
      lappend xs [lindex $ll 0]
      lappend ys [lindex $ll 1]
    }
    if {[llength $xs] < 2} { continue }
    set sorted_xs [lsort -integer $xs]
    set sorted_ys [lsort -integer $ys]
    set hpwl_dbu [expr {[lindex $sorted_xs end] - [lindex $sorted_xs 0]                       + [lindex $sorted_ys end] - [lindex $sorted_ys 0]}]
    set wl_um [$block dbuToMicrons $hpwl_dbu]
    if {$wl_um < $min_wirelength_um} { continue }

    set net_name [string map {\\ "" [ \\[ ] \\]} [$net getName]]
    sta::redirect_string_begin
    if {[catch {report_net $net_name}]} {
      # Skip nets with names containing chars that confuse OpenROAD internals
      sta::redirect_string_end
      continue
    }
    set out [sta::redirect_string_end]
    puts $fp "=== $net_name ==="
    puts $fp $out
    puts $fp ""
    incr emitted
  }
  close $fp
  puts "# dump_net_reports: $emitted nets (fanout>=$min_fanout, wl>=${min_wirelength_um}um) -> $reports_file"
}

proc write_node_and_net_files { node_file_name net_file_name } {
  set block [odb::get_block]
  set fp_node [open $node_file_name w]
  puts $fp_node "Name,Master,Type,llx,lly"

  foreach inst [$block getInsts] {
    set name [string map {\\ ""} [$inst getName]]
    set master [$inst getMaster]
    set master_name [$master getName]
    set ll [$inst getLocation]
    set llx [$block dbuToMicrons [lindex $ll 0]]
    set lly [$block dbuToMicrons [lindex $ll 1]]
    set type [expr {[$master isBlock] ? "Macro" : "Inst"}]
    puts $fp_node "$name,$master_name,$type,$llx,$lly"
  }

  foreach bterm [$block getBTerms] {
    set name [$bterm getName]
    set bpins [$bterm getBPins]
    if {[llength $bpins] > 0} {
      set bpin [lindex $bpins 0]
      set bbox [$bpin getBBox]
      set x [$block dbuToMicrons [$bbox xMin]]
      set y [$block dbuToMicrons [$bbox yMin]]
      puts $fp_node "$name,NA,IO,$x,$y"
    }
  }

  close $fp_node
  write_net_details $net_file_name
}

proc write_node_net_and_reports { node_file_name net_file_name reports_file_name } {
  if {[catch {write_node_and_net_files $node_file_name $net_file_name} _wnf_err]} {
    puts "# write_node_and_net_files failed: $_wnf_err"
  }
  if {[catch {dump_net_reports $reports_file_name} _dnr_err]} {
    puts "# dump_net_reports failed: $_dnr_err"
  }
}

proc dump_iter_qor { iter_file } {
  # Per-iter QoR dump. Each report goes into the file via a single-command
  # `tee -file -append`, which:
  #   * avoids OpenROAD's brace-block tee bug (Utl.tcl line ~126 uses
  #     {*}[lindex $args 0], flattening multi-command bodies into one argv)
  #   * synchronously redirects the utl::Logger sink (teeFileBegin →
  #     setRedirectSink), so report_power / report_design_area output is
  #     captured immediately — no spdlog async-flush race.
  set fp [open $iter_file w]
  puts $fp "# iter_qor dump at [clock format [clock seconds]]"
  close $fp

  set fp [open $iter_file a]; puts $fp "=== WNS ===";   close $fp
  catch {tee -file $iter_file -append "report_wns -digits 3"}

  set fp [open $iter_file a]; puts $fp "=== TNS ===";   close $fp
  catch {tee -file $iter_file -append "report_tns -digits 3"}

  set fp [open $iter_file a]; puts $fp "=== POWER ==="; close $fp
  catch {tee -file $iter_file -append "report_power"}

  set fp [open $iter_file a]; puts $fp "=== AREA ===";  close $fp
  catch {tee -file $iter_file -append "report_design_area"}
}
'''

RIP_ALL_WIRES_PROC = r'''
# Rip all signal-net routing wires + guides (leaves power/ground intact).
# Called before detailed_route to clear ECO-dirty wires for re-routing.
proc rip_all_signal_wires {} {
    set block [odb::get_block]
    set count 0
    foreach net [$block getNets] {
        set sig_type [$net getSigType]
        if {$sig_type eq {POWER} || $sig_type eq {GROUND}} { continue }
        set wire [$net getWire]
        if {$wire ne {NULL}} {
            odb::dbWire_destroy $wire
            incr count
        }
        $net clearGuides
    }
    puts "\[RIP_ALL] Destroyed wires on $count signal nets"
}
'''

# ---------------------------------------------------------------------------
# Node/net CSV helpers
# ---------------------------------------------------------------------------

def load_node_positions(node_csv: Path) -> Dict[str, tuple]:
    """Read node CSV → {inst_name: (x_um, y_um)}."""
    pos = {}
    if not node_csv.exists():
        return pos
    with node_csv.open("r", encoding="utf-8", errors="ignore") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            try:
                x = float((row.get("llx") or "0").strip())
                y = float((row.get("lly") or "0").strip())
            except ValueError:
                continue
            pos[name] = (x, y)
    return pos


# ---------------------------------------------------------------------------
# ECO command parsing (for displacement tracking)
# ---------------------------------------------------------------------------

def parse_eco_targets(eco_commands: List[str]) -> Dict[str, str]:
    """Extract instance names touched by ECO commands and classify the action.

    Returns {inst_name: action} where action is one of:
      'replaced', 'removed', 'inserted', 'created', 'connected', 'disconnected'
    """
    targets = {}
    for cmd in eco_commands:
        parts = cmd.split()
        if not parts:
            continue
        op = parts[0]

        if op == "replace_cell" and len(parts) >= 3:
            targets[parts[1]] = "replaced"

        elif op == "remove_buffers" and len(parts) >= 2:
            for inst in parts[1:]:
                targets[inst] = "removed"

        elif op == "insert_buffer":
            # insert_buffer -load_pins inst/pin -buffer_cell ... -buffer_name eco_buf_1 ...
            # insert_buffer -net net_name -buffer_cell ... -buffer_name eco_buf_1 ...
            # Extract -buffer_name value as the created instance
            for i, tok in enumerate(parts):
                if tok == "-buffer_name" and i + 1 < len(parts):
                    targets[parts[i + 1]] = "inserted"
                # Extract load pin instance for tracking
                if tok == "-load_pins" and i + 1 < len(parts):
                    pin_tok = parts[i + 1]
                    if "/" in pin_tok:
                        inst = pin_tok.split("/")[0]
                        targets.setdefault(inst, "buffered_load")

    return targets


def _tokenize_tcl(line: str) -> List[str]:
    """Split a Tcl command line into tokens, keeping {braced} groups intact.
    Handles nested braces like `{driver/pin}`. Does NOT evaluate $vars or []."""
    toks: List[str] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c == "{":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if line[j] == "{":
                    depth += 1
                elif line[j] == "}":
                    depth -= 1
                j += 1
            toks.append(line[i:j])
            i = j
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            toks.append(line[i:j])
            i = j
    return toks


# Position (0-indexed, including the proc name) of the `buf_name` arg
# for each eco_* insertion proc. See AGENTS.md arg-count cheat-sheet.
_BUF_NAME_POS: Dict[str, int] = {
    "eco_insert_buffer": 3,                # <net> {sink} BUFNAME <cell>
    "eco_insert_buffer_midpoint": 4,       # <net> {sink} {driver} BUFNAME <cell>
    "eco_insert_buffer_optimal_alpha": 4,
    "eco_buffer_driver_fanout": 4,         # <net> {driver} <cell> BUFNAME
    "eco_buffer_sink_cluster": 4,          # <net> {driver} <cell> BUFNAME {sinks}
}


def extract_inserted_buffer_names(eco_cmds: List[str]) -> List[str]:
    """For each buffer-insert command in the batch, return the `buf_name` arg.
    These names are the argument passed to `eco_remove_buffer` on revert.
    Order preserved; removal should iterate in REVERSE order so the latest
    insertions are undone first (safer when buffers chain, though chaining
    on _eco_* nets is already forbidden)."""
    inserted: List[str] = []
    for raw in eco_cmds or []:
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        toks = _tokenize_tcl(line)
        if not toks:
            continue
        proc_name = toks[0]
        idx = _BUF_NAME_POS.get(proc_name)
        if idx is None or idx >= len(toks):
            continue
        name = toks[idx].strip("{}").strip()
        if name:
            inserted.append(name)
    return inserted


# ---------------------------------------------------------------------------
# Displacement computation
# ---------------------------------------------------------------------------

def compute_displacement(
    pre_pos: Dict[str, tuple],
    post_pos: Dict[str, tuple],
    eco_commands: List[str],
    path_instances: set | None = None,
) -> Dict[str, Any]:
    """Compute cell displacement between pre-ECO and post-ECO positions.

    Categorizes displaced cells into:
      1. ECO target cells (directly modified)
      2. Cells on worst timing paths (collateral from legalization)
      3. Other collateral
    """
    eco_targets = parse_eco_targets(eco_commands)

    eco_displaced = []
    path_displaced = []
    collateral_displaced = []

    for inst, (px, py) in post_pos.items():
        if inst not in pre_pos:
            continue
        ox, oy = pre_pos[inst]
        dx = round(px - ox, 3)
        dy = round(py - oy, 3)
        manhattan = round(abs(dx) + abs(dy), 3)
        if manhattan < 0.001:
            continue

        entry = {
            "inst": inst,
            "dx_um": dx,
            "dy_um": dy,
            "manhattan_um": manhattan,
        }

        if inst in eco_targets:
            entry["eco_action"] = eco_targets[inst]
            eco_displaced.append(entry)
        elif path_instances and inst in path_instances:
            entry["on_worst_path"] = True
            path_displaced.append(entry)
        else:
            collateral_displaced.append(entry)

    # Sort each category by displacement (largest first)
    eco_displaced.sort(key=lambda x: -x["manhattan_um"])
    path_displaced.sort(key=lambda x: -x["manhattan_um"])
    collateral_displaced.sort(key=lambda x: -x["manhattan_um"])

    # New / removed instances
    new_insts = sorted(set(post_pos.keys()) - set(pre_pos.keys()))
    removed_insts = sorted(set(pre_pos.keys()) - set(post_pos.keys()))
    new_eco = [i for i in new_insts if i in eco_targets]
    removed_eco = [i for i in removed_insts if i in eco_targets]

    # Summary stats
    all_displaced = eco_displaced + path_displaced + collateral_displaced
    all_um = [d["manhattan_um"] for d in all_displaced]
    path_um = [d["manhattan_um"] for d in path_displaced]

    summary = {
        "eco_targets_displaced": len(eco_displaced),
        "path_cells_displaced": len(path_displaced),
        "collateral_displaced": len(collateral_displaced),
        "total_displaced": len(all_displaced),
        "max_displacement_um": round(max(all_um), 3) if all_um else 0.0,
        "avg_displacement_um": round(sum(all_um) / len(all_um), 3) if all_um else 0.0,
        "path_max_displacement_um": round(max(path_um), 3) if path_um else 0.0,
        "path_avg_displacement_um": round(sum(path_um) / len(path_um), 3) if path_um else 0.0,
        "new_instances": len(new_insts),
        "removed_instances": len(removed_insts),
    }

    return {
        "summary": summary,
        "eco_targets": eco_displaced[:10],
        "path_cells_moved": path_displaced[:20],
        "collateral_moved": collateral_displaced[:10],
        "new_instances": new_eco[:10],
        "removed_instances": removed_eco[:10],
    }


# ---------------------------------------------------------------------------
# OpenROAD process helpers
# ---------------------------------------------------------------------------

def write_best_odb_snapshot(proc):
    """Persist current OpenROAD state as the best-ever .odb. Called only when
    a new best WNS is set. Overwrites the previous best.odb."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    send(proc, f'write_db "{BEST_ODB}"\nputs {PARA}\n')



def write_drt_state(
    cycle_num: int,
    iter_in_cycle: int,
    current_iter: int,
    wns_ps: Optional[float] = None,
    tns_ps: Optional[float] = None,
    last_drt: Optional[dict] = None,
) -> None:
    """Write DRT cycle state to DRT_STATE_FILE for LLM_call.py to include in prompt.

    Fields:
      cycle_num         — how many DRT cycles have completed (0=none yet)
      iter_in_cycle     — 0-based index within current DRT_CYCLE (0=just after DRT or
                          just after baseline SPEF load, i.e. LLM sees SPEF timing)
      current_iter      — global ECO iteration number (1-based)
      parasitic_source  — 'SPEF_extracted' when iter_in_cycle==0 (LLM sees post-DRT
                          or baseline SPEF report_checks), else 'GRT_estimate'
      iters_until_drt   — ECO iters until the next DRT refresh
      last_drt          — None if no DRT cycle done; else dict with calibration data
    """
    source = "SPEF_extracted" if iter_in_cycle == 0 else "GRT_estimate"
    iters_until_drt = DRT_CYCLE - iter_in_cycle - 1  # 0 means DRT fires THIS iter end
    state = {
        "cycle_num": cycle_num,
        "iter_in_cycle": iter_in_cycle,
        "iters_until_drt": iters_until_drt,
        "current_iter": current_iter,
        "parasitic_source": source,
        "last_drt": last_drt,
    }
    try:
        DRT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DRT_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"  [WARN] write_drt_state failed: {e}", flush=True)


_CORR_COLS = ["cycle", "at_iter", "transition", "wns_from", "wns_to",
              "dwns_ps", "tns_from", "tns_to", "dtns_ps"]


def append_corr_row(record: Dict[str, Any]) -> None:
    """Append one GRT<->DRT correlation row to GRT_DRT_CORR_FILE (CSV).

    transition is 'GRT->DRT' (logged at a DRT boundary: how much WNS/TNS
    changed when the cycle's last GRT estimate was replaced by the routed
    SPEF result) or 'DRT->GRT' (logged at the next cycle's first ECO iter:
    how much the SPEF/DRT checkpoint shifts under the first GRT re-estimate).
    dwns/dtns are (to - from): positive = improved (less negative)."""
    if not SAVE_GRT_DRT_CORR:
        return
    try:
        GRT_DRT_CORR_FILE.parent.mkdir(parents=True, exist_ok=True)
        new = not GRT_DRT_CORR_FILE.exists()
        with GRT_DRT_CORR_FILE.open("a", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=_CORR_COLS)
            if new:
                w.writeheader()
            w.writerow({k: record.get(k) for k in _CORR_COLS})
    except OSError as e:
        print(f"  [WARN] append_corr_row failed: {e}", flush=True)


def generate_load_tcl_for(odb_path, dst_path):
    """Write a copy of the design-load tcl with every ACTIVE `read_db` line
    commented out and an explicit `read_db <odb_path>` inserted in its place,
    then return the dst Path. The base design-load tcl hardcodes
    `read_db ${design_dir}/${top}.odb` and does NOT honor any env override, so
    a plain `source` of it always reloads the baseline. Sourcing the generated
    tcl instead is the reliable way to restart OpenROAD from a snapshot
    (best_grt.odb mid-cycle rollback, or best_drt.odb crash recovery).
    """
    src = Path(OpenROAD_design_tcl)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"design load tcl not found: {src}")
    lines_ = src.read_text(encoding="utf-8").splitlines()
    out_lines = []
    inserted = False
    for ln in lines_:
        stripped = ln.lstrip()
        if stripped.startswith("read_db ") and not stripped.startswith("#"):
            out_lines.append("# [LOAD-PATCH] commented by generate_load_tcl_for(): " + ln)
            if not inserted:
                out_lines.append(f"read_db {odb_path}    ;# [LOAD-PATCH] inserted snapshot")
                inserted = True
            continue
        # [LOAD-PATCH] Drop the baseline read_spef too: the snapshot is
        # re-routed and re-extracted (extract_parasitics + read_spef <cycle>)
        # by the DRT block before any report, so a load-time SPEF is stale
        # (mismatched to the ECO-modified netlist -> STA-1650/1651 warnings)
        # and could leave wrong caps on ECO nets that report_power would pick
        # up. read_sdc is KEPT (timing constraints are still needed).
        if stripped.startswith("read_spef ") and not stripped.startswith("#"):
            out_lines.append("# [LOAD-PATCH] dropped baseline read_spef (re-extracted post-load): " + ln)
            continue
        out_lines.append(ln)
    if not inserted:
        last_rdb = -1
        for i, ln in enumerate(out_lines):
            if ln.lstrip().startswith("#read_db"):
                last_rdb = i
        insert_at = (last_rdb + 1) if last_rdb >= 0 else 0
        out_lines.insert(insert_at,
                         f"read_db {odb_path}    ;# [LOAD-PATCH] inserted snapshot")
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"  [LOAD] generated {dst.name} with read_db -> {odb_path}", flush=True)
    return dst


def generate_best_load_tcl():
    """Generate a load tcl whose read_db points at the global best (best-DRT)
    snapshot. Used by crash/backtrace recovery."""
    return generate_load_tcl_for(BEST_ODB, OpenROAD_design_tcl_best)


def shutdown_openroad(proc):
    """Cleanly close stdin and wait for OpenROAD to exit."""
    try:
        proc.stdin.write("exit\n")
        proc.stdin.flush()
        proc.stdin.close()
    except (BrokenPipeError, ValueError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def restart_openroad_from_odb(odb_path):
    """Spawn a fresh OpenROAD with BACKTRACE_ODB set so OpenROAD_load_design.tcl
    reads from `odb_path` instead of ${DESIGN_DIR}/${TOP}.odb."""
    env = _os.environ.copy()
    if DESIGN_DIR:
        env["DESIGN_DIR"] = str(DESIGN_DIR)
    env["TOP"] = DESIGN_TOP
    env["ASAP7_PDK"] = ASAP7_PDK
    env["BACKTRACE_ODB"] = str(odb_path)
    return subprocess.Popen(
        [OpenROAD_bin, "-no_init"],
        cwd=workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )


def restart_openroad_from_best():
    """Restart OpenROAD reading from the global best (best-DRT) snapshot."""
    return restart_openroad_from_odb(BEST_ODB)


def write_backtrace_notice(best_iter, best_wns, last_n_records):
    """Drop a sidecar JSON the next prompt-builder reads to inject a
    <backtrace_notice> block in the user prompt. Consumed once, then deleted."""
    BACKTRACE_NOTICE.write_text(json.dumps({
        "best_iter": best_iter,
        "best_wns": best_wns,
        "last_iterations": last_n_records,
    }, indent=2))


def start_openroad():
    """Spawn OpenROAD with DESIGN_DIR / TOP / ASAP7_PDK pushed into its env so
    OpenROAD_load_design.tcl resolves them via $::env(...)."""
    env = _os.environ.copy()
    if DESIGN_DIR:
        env["DESIGN_DIR"] = DESIGN_DIR
    env["TOP"] = DESIGN_TOP
    env["ASAP7_PDK"] = ASAP7_PDK
    return subprocess.Popen(
        [OpenROAD_bin, "-no_init"],
        cwd=workdir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )


def _ts():
    return datetime.now().isoformat(timespec="seconds")


def log_event(msg: str, also_llm: bool = True):
    """Write a banner to OPENROAD_SESSION_LOG (and LLM_SESSION_LOG if also_llm)
    so backtrace/restart/exit events are visible in both timelines."""
    banner = f"\n{'='*72}\n=== {_ts()} | {msg}\n{'='*72}\n"
    try:
        OPENROAD_SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with OPENROAD_SESSION_LOG.open("a", encoding="utf-8") as fp:
            fp.write(banner); fp.flush()
        if also_llm:
            with LLM_SESSION_LOG.open("a", encoding="utf-8") as fp:
                fp.write(banner); fp.flush()
    except OSError:
        pass


def send(proc, tcl: str):
    """Send Tcl to OpenROAD and tee the sent text into OPENROAD_SESSION_LOG."""
    try:
        with OPENROAD_SESSION_LOG.open("a", encoding="utf-8") as fp:
            fp.write(f"\n>>> SEND [{_ts()}]\n{tcl}")
            if not tcl.endswith("\n"):
                fp.write("\n")
            fp.flush()
    except OSError:
        pass
    proc.stdin.write(tcl + "\n")
    proc.stdin.flush()


class OpenROADCrash(RuntimeError):
    """Raised when OpenROAD process dies before emitting the expected sentinel."""


def wait_for_sentinel(proc, sentinel: str = PARA, tag: str = ""):
    """Read proc.stdout line-by-line, teeing into OPENROAD_SESSION_LOG, until
    `sentinel` is seen. Raises OpenROADCrash if the process dies first.

    When OpenROAD crashes (e.g. SIGSEGV from FlexPA::updateDirtyInsts),
    stdout reaches EOF without the sentinel. Previously the loop exited
    silently, leaving the orchestrator to continue with stale data. Now
    it detects EOF-without-sentinel and raises so the caller can restart.
    """
    try:
        fp = OPENROAD_SESSION_LOG.open("a", encoding="utf-8")
        if tag:
            fp.write(f"\n--- [WAIT {sentinel} | {tag}] ---\n")
            fp.flush()
    except OSError:
        fp = None
    sentinel_found = False
    try:
        for line in proc.stdout:
            if fp is not None:
                fp.write(line); fp.flush()
            if sentinel in line:
                sentinel_found = True
                break
    finally:
        if fp is not None:
            fp.close()
    if not sentinel_found:
        rc = proc.poll()
        rc_str = f"rc={rc}" if rc is not None else "process still alive (hung?)"
        raise OpenROADCrash(
            f"OpenROAD stdout closed before sentinel '{sentinel}' "
            f"({rc_str}) — process likely crashed (check openroad_session.log)"
        )


def run_llm(max_iter=1, timeout=60, hard_timeout_s: float = 600.0):
    """Run the LLM subprocess with a hard wall-clock timeout.
    Returns a tuple (ok: bool, info: str) — caller checks ok and writes a
    sentinel ECO file on failure so wait_for_file_update can proceed.
    """
    try:
        cp = subprocess.run(
            ["python3", LLM_CALL, "--max-iter", str(max_iter), "--timeout", str(timeout)],
            check=False,
            timeout=hard_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return (False, f"timeout after {hard_timeout_s:.0f}s")
    if cp.returncode != 0:
        return (False, f"non-zero exit {cp.returncode}")
    return (True, "ok")


def _write_eco_sentinel(eco_path: Path, reason: str) -> None:
    """Write a placeholder ECO file so wait_for_file_update can move on
    when the LLM crashed/timed out. Also wipe llm_analysis.md and
    prev_target_path_status.json so the next iter's prompt does not include
    stale content from the previous successful iter (which would otherwise
    appear in <recent_analysis> / <previous_target_path>).
    """
    ts = datetime.now().isoformat()
    msg = f"# CLAUDE_FAILED: {reason} at {ts}\n"
    try:
        eco_path.write_text(msg, encoding="utf-8")
        print(f"  [LLM-RECOVER] wrote sentinel ECO ({reason}); iter will be a no-op", flush=True)
    except Exception as e:
        print(f"  [LLM-RECOVER] could not write sentinel: {e}", flush=True)
    # Wipe analysis sidecar so main_orch does not persist stale prior analysis
    try:
        ap = Path(ANALYSIS_out)
        if ap.exists():
            ap.write_text(f"# CLAUDE_FAILED: {reason} at {ts}\n", encoding="utf-8")
    except Exception:
        pass
    # Wipe prev_target_path so next iter does not show last-iter's path diff
    try:
        pp = Path(PREV_TARGET_JSON)
        if pp.exists():
            pp.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Report / metrics parsing
# ---------------------------------------------------------------------------

def make_report_cmd(report_file: str, endpoint_count: int = TOP_PATHS) -> str:
    # -group_count gives us N distinct worst endpoints; -endpoint_path_count 1
    # returns one path per endpoint. That avoids the common degenerate case
    # where the top-N are all the same endpoint.
    return (
        "report_checks "
        "-slack_max 0 -fields {slew input_pins cap fanout net} "
        "-path_group reg2reg "
        f"-group_path_count {endpoint_count} -endpoint_path_count 1 "
        f"> {report_file}"
    )


def extract_worst_slack(rpt_path: Path) -> Optional[float]:
    worst = None
    for line in rpt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "slack" not in line.lower():
            continue
        m = re.search(r"(-?\d+(?:\.\d+)?)\s+slack", line)
        if not m:
            continue
        val = float(m.group(1))
        if worst is None or val < worst:
            worst = val
    return worst


def parse_report_path_slacks(rpt_path: Path) -> List[float]:
    text = rpt_path.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"Startpoint:", text)[1:]
    slacks = []
    for sec in sections:
        lines = sec.splitlines()
        candidates = [ln for ln in lines if "slack" in ln.lower()]
        if not candidates:
            continue
        m = re.search(r"(-?\d+(?:\.\d+)?)", candidates[-1])
        if m:
            slacks.append(float(m.group(1)))
    return slacks


def parse_tns_from_metrics(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    tns = None
    for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"tns\s+max\s+(-?\d+(?:\.\d+)?)", line)
        if m:
            tns = float(m.group(1))
    return tns


def read_eco_commands(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def parse_predicted_delta_ps(analysis_text: str) -> Optional[float]:
    """Pull the first Move plan row's predicted Δslack (in ps) from the
    model's analysis. Handles formats like '+3.5 ps', '~7ps', '~10-11ps',
    '5-7 ps', '-2.0 ps'. Returns None if nothing parseable in the table."""
    if not analysis_text:
        return None
    in_move_plan = False
    for raw in analysis_text.splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith("### move plan") or low.startswith("### ranked"):
            in_move_plan = True
            continue
        if line.startswith("### ") and in_move_plan:
            # End of move-plan section
            break
        if not in_move_plan:
            continue
        if not line.startswith("|"):
            continue
        # Skip the header and separator rows
        if "---" in line or line.lower().count("rationale") or "δslack" in low or "delta" in low:
            # Header row containing column names like "Δslack est."
            if "ps" not in low:
                continue
        # Look for cell with a numeric ps value
        # Patterns: ~10-11ps, +3.5ps, -2.0ps, 5-7 ps, ~7ps
        # Range pattern first: ~?<num>[-–]<num>\s*ps
        m = re.search(r"[~+\-]?(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*ps", line)
        if m:
            try:
                return (float(m.group(1)) + float(m.group(2))) / 2.0
            except ValueError:
                pass
        # Single value with optional sign/tilde
        m = re.search(r"([~+\-]?)(\d+(?:\.\d+)?)\s*ps", line)
        if m:
            sign = -1.0 if m.group(1) == "-" else 1.0
            try:
                v = sign * float(m.group(2))
                if -100 < v < 100:
                    return v
            except ValueError:
                pass
    return None


def extract_path1_endpoints(rpt_path: Path) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Pull (startpoint, endpoint, slack_ps) for the worst path (path 1) from
    a report_checks dump. Returns (None, None, None) if not parseable."""
    if not rpt_path.exists():
        return None, None, None
    text = rpt_path.read_text(encoding="utf-8", errors="ignore")
    sections = re.split(r"Startpoint:", text)[1:]
    if not sections:
        return None, None, None
    sec = sections[0]  # path 1 only
    # First line of section is the startpoint
    lines = sec.splitlines()
    if not lines:
        return None, None, None
    sp_line = lines[0].strip()
    sp = sp_line.split("(")[0].strip()
    ep = None
    for ln in lines:
        if "Endpoint:" in ln:
            ep = ln.split("Endpoint:", 1)[1].split("(")[0].strip()
            break
    slack = None
    for ln in lines[::-1]:
        if "slack" in ln.lower():
            m = re.search(r"(-?\d+(?:\.\d+)?)", ln)
            if m:
                try:
                    slack = float(m.group(1))
                    break
                except ValueError:
                    pass
    return sp, ep, slack


def _path_to_toon(rpt_text: str, label_prefix: str = "") -> str:
    """Convert a report_checks dump containing ONE path into a compact
    TOON row format matching the schema the model already reads in
    <iteration_context>. Returns '' if the report is unparseable.
    Schema: stage,pin,cell,cell_delay_ps,cumulative_ps,transition
    """
    if not rpt_text:
        return ""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from parsers.timing_rpt_parser import TimingReportParser
    except Exception as e:
        return f"# parser unavailable: {e}"
    try:
        parser = TimingReportParser()
        paths = parser.parse_report(rpt_text)
    except Exception as e:
        return f"# parse error: {e}"
    if not paths:
        return "# no path parsed"
    pth = paths[0]  # the targeted path (only one in the re-report)
    rows = []
    for i, pt in enumerate(pth.points, start=1):
        # delay_ps and cumulative_ps come in as the report's units (ps).
        rows.append(
            f"{i},{pt.pin},{pt.cell_type},"
            f"{pt.delay:.3f},{pt.cumulative_time:.3f},{pt.transition}"
        )
    header = (
        f"{label_prefix}path[{len(rows)}]"
        "{stage,pin,cell,cell_delay_ps,cumulative_ps,transition}:\n"
    )
    return header + "\n".join(rows)


# ---------------------------------------------------------------------------
# JSON history helpers
# ---------------------------------------------------------------------------

def append_json_record(path: Path, record: Dict[str, Any], max_len: int = 300):
    data = []
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    data.append(record)
    data = data[-max_len:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# QoR scoring
# ---------------------------------------------------------------------------

def _trim_history_after(path: Path, keep_through_iter: int) -> None:
    """Drop records whose integer `iteration` exceeds keep_through_iter.

    Used on a mid-cycle rollback: when the loop reverts to the cycle's
    best-GRT iter and discards the trailing iters, the on-disk run/eco
    histories must not keep advertising the reverted moves (buffers/cells
    that no longer exist) to the next cycle's LLM. Records with a non-int
    iteration (markers) are kept."""
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(data, list):
        return
    kept = []
    for rec in data:
        it = rec.get("iteration") if isinstance(rec, dict) else None
        if isinstance(it, int) and it > keep_through_iter:
            continue
        kept.append(rec)
    path.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def compute_qor_score(prev: Optional[Dict[str, Any]], curr: Dict[str, Any]) -> Optional[float]:
    if prev is None:
        return None
    dwns = 0.0 if prev.get("wns") is None or curr.get("wns") is None else curr["wns"] - prev["wns"]
    dtns = 0.0 if prev.get("tns") is None or curr.get("tns") is None else curr["tns"] - prev["tns"]
    dnei = curr.get("neighbor_delta", 0.0)
    newv = float(curr.get("new_violations", 0))
    return W_WNS * dwns + W_TNS * dtns + W_NEIGHBOR * dnei - W_NEW_VIOL * newv


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_slack_plot(slack_history: List[float], out_path: Path):
    if not slack_history:
        return
    its = np.arange(1, len(slack_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(its, slack_history, marker="o", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Worst Slack (ps)")
    plt.title("WNS vs LLM calls")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# File update polling
# ---------------------------------------------------------------------------

def wait_for_file_update(path: Path, last_mtime: float) -> float:
    print("inside wait_for_file_update", flush=True)
    print(f"  waiting for update to {path} (last mtime {last_mtime})", flush=True)
    print(f"  exists: {path.exists()}, current mtime: {path.stat().st_mtime if path.exists() else 'N/A'}", flush=True)
    while (not path.exists()) or (path.stat().st_mtime <= last_mtime):
        print(f"  still waiting... exists: {path.exists()}, mtime: {path.stat().st_mtime if path.exists() else 'N/A'}", flush=True)
        sleep(0.2)
        print(f"  checking again... exists: {path.exists()}, mtime: {path.stat().st_mtime if path.exists() else 'N/A'}", flush=True)
    return path.stat().st_mtime


# ---------------------------------------------------------------------------
# Precise WNS/TNS parsing — from a dedicated `report_wns`/`report_tns` dump.
# The big timing report file is parsed by regex over many "slack" lines which
# yields values that sometimes disagree with OpenROAD's own report_wns output.
# This helper reads the dedicated two-line dump instead.
# ---------------------------------------------------------------------------

def parse_wns_tns_file(path: Path) -> Dict[str, Optional[float]]:
    """Parse `wns_tns.txt` containing the output of:
           report_wns -digits 3
           report_tns -digits 3
       Each `report_wns` / `report_tns` prints a single number.
    """
    out: Dict[str, Optional[float]] = {"wns": None, "tns": None}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    # report_wns/report_tns each emit a single-line floating-point number,
    # sometimes preceded by a short label. Grab all numbers and assign the
    # first to wns, second to tns. OpenROAD prints wns first.
    nums = re.findall(r"-?\d+\.\d+", text)
    if len(nums) >= 1:
        out["wns"] = float(nums[0])
    if len(nums) >= 2:
        out["tns"] = float(nums[1])
    return out


# ---------------------------------------------------------------------------
# Run-state cleanup — wipes all histories/artifacts before a fresh run so
# iteration 0 doesn't inherit stale qor/run/eco history from a prior session.
# ---------------------------------------------------------------------------

def clear_run_state():
    """Remove every artifact that carries over between sessions. Anything
    auto-generated under prompts/dynamic/ or history JSON files listed here."""
    to_remove = [
        ECO_HISTORY_JSON,
        RUN_HISTORY_JSON,
        Path(workdir) / "prompts" / "history.json",
        slack_track,
        slack_history_png,
        Path(METRICS_out),
        ITER_METRICS_CSV,
        Path(WNS_TNS_FILE),
        Path(DISPLACEMENT_FILE),
        Path(f"{workdir}/prompts/dynamic/eco_errors.log"),
        Path(f"{workdir}/prompts/dynamic/last_prompt.txt"),
        Path(DYNAMIC_CONTEXT_JSON),
        Path(ECO_out),
        Path(RPT_out),
        Path(verilog_out),
        Path(NODE_FILE),
        Path(NODE_FILE_PRE),
        Path(NET_FILE),
        Path(f"{workdir}/prompts/dynamic/best_eco.v"),
        # Sidecars added in later patches — must also be wiped or the next
        # session's iter 1 reads stale content from the previous design
        # (this is the root cause of the JPEG-latest iter-1 contamination).
        Path(ANALYSIS_out),
        Path(PREV_TARGET_JSON),
        Path(PREV_TARGET_REPORT_TXT),
        Path(PROBE_RESPONSES_FILE),
        # Per-DRT-cycle GRT<->DRT correlation log — wipe so a new run does not
        # append onto the previous run's rows (they share the same file).
        GRT_DRT_CORR_FILE,
    ]
    # Wipe per-run .odb snapshots, iter QoR dumps, and stale backtrace notice.
    if SNAPSHOT_DIR.exists():
        import shutil as _sh
        _sh.rmtree(SNAPSHOT_DIR, ignore_errors=True)
    if ITER_QOR_DIR.exists():
        import shutil as _sh2
        _sh2.rmtree(ITER_QOR_DIR, ignore_errors=True)
    if BACKTRACE_NOTICE.exists():
        try:
            BACKTRACE_NOTICE.unlink()
        except OSError:
            pass
    for _log in (OPENROAD_SESSION_LOG, LLM_SESSION_LOG):
        try:
            if _log.exists():
                _log.unlink()
        except OSError:
            pass

    for p in to_remove:
        try:
            if p.exists():
                p.unlink()
        except OSError as e:
            print(f"  [CLEAN] could not remove {p}: {e}")

    # Clear claude_logs subdir contents (keep the directory)
    claude_logs_dir = Path(f"{workdir}/prompts/dynamic/claude_logs")
    if claude_logs_dir.exists():
        for child in claude_logs_dir.iterdir():
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    import shutil
                    shutil.rmtree(child)
            except OSError:
                pass

    print("  [CLEAN] run state cleared — fresh session")


# ---------------------------------------------------------------------------
# Per-iteration QoR (WNS/TNS/Power/Area) parsing
# ---------------------------------------------------------------------------

import time as _time
import re as _re

_ANSI_RE = _re.compile(r"\x1b\[[0-9;]*[mGKHF]")

def parse_iter_qor(path: Path) -> Dict[str, Optional[float]]:
    """Parse the per-iteration QoR dump produced by `dump_iter_qor`.

    Layout (sections delimited by `=== HEADER ===`):
      === WNS ===   → single float (worst slack)
      === TNS ===   → single float (total negative slack)
      === POWER === → report_power table; we grab the `Total` row
                       (Internal, Switching, Leakage, Total — Watts)
      === AREA ===  → `Design area <X> u^2 <Y>% utilization.`
    Missing sections return None for their fields (never crashes).
    """
    out: Dict[str, Optional[float]] = {
        "wns_ps": None, "tns_ps": None,
        "internal_pw_w": None, "switching_pw_w": None,
        "leakage_pw_w": None, "total_pw_w": None,
        "design_area_um2": None, "util_pct": None,
    }
    if not path.exists():
        return out
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _ANSI_RE.sub("", raw)
    # Split into sections
    sections: Dict[str, List[str]] = {}
    cur = None
    for line in raw.splitlines():
        m = _re.match(r"\s*===\s*(\w+)\s*===\s*$", line)
        if m:
            cur = m.group(1).upper()
            sections[cur] = []
            continue
        if cur is not None:
            sections[cur].append(line)
    # WNS / TNS: first signed-float we see in the section
    for key, target in (("WNS", "wns_ps"), ("TNS", "tns_ps")):
        for line in sections.get(key, []):
            m = _re.search(r"-?\d+\.\d+", line)
            if m:
                out[target] = float(m.group(0))
                break
    # POWER: find the Total row and pull 4 floats (internal, switching, leakage, total)
    for line in sections.get("POWER", []):
        if line.strip().lower().startswith("total"):
            nums = _re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", line)
            # Strip a leading percentage if it sneaks into the first slot.
            floats = []
            for n in nums:
                try:
                    floats.append(float(n))
                except ValueError:
                    pass
            if len(floats) >= 4:
                out["internal_pw_w"] = floats[0]
                out["switching_pw_w"] = floats[1]
                out["leakage_pw_w"] = floats[2]
                out["total_pw_w"] = floats[3]
            break
    # AREA: `Design area <X> u^2 <Y>% utilization.`
    for line in sections.get("AREA", []):
        m = _re.search(r"[Dd]esign\s+area\s+(-?\d+\.?\d*)\s*u", line)
        if m:
            out["design_area_um2"] = float(m.group(1))
        m2 = _re.search(r"(-?\d+\.?\d*)\s*%\s*utilization", line)
        if m2:
            out["util_pct"] = float(m2.group(1))
        if out["design_area_um2"] is not None:
            break
    return out


# ---------------------------------------------------------------------------
# Token log delta parsing — `LLM_call.py` writes its own iteration count to
# token_log.jsonl (always 1 because it\'s a fresh subprocess each invocation).
# Rather than modify LLM_call.py, we snapshot the file line-count before each
# run_llm() and treat anything appended after as belonging to this iteration.
# ---------------------------------------------------------------------------

def token_log_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        return sum(1 for _ in fp)


def read_new_token_log_rows(path: Path, from_line: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for idx, line in enumerate(fp):
            if idx < from_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def aggregate_token_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "duration_ms": 0,
        "total_cost_usd": 0.0,
        "num_calls": 0,
    }
    for r in rows:
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens", "duration_ms"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                agg[k] += int(v)
        c = r.get("total_cost_usd")
        if isinstance(c, (int, float)):
            agg["total_cost_usd"] += float(c)
        agg["num_calls"] += 1
    return agg


# ---------------------------------------------------------------------------
# iter_metrics.csv — single source of truth for the end-of-run summary table
# ---------------------------------------------------------------------------

ITER_METRICS_HEADER = [
    "iteration", "phase",
    "wns_ps", "tns_ps",
    "total_pw_w", "internal_pw_w", "switching_pw_w", "leakage_pw_w",
    "design_area_um2", "util_pct",
    "iter_runtime_s", "llm_runtime_s",
    "llm_calls", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_creation_tokens",
    "llm_cost_usd",
]


def ensure_iter_metrics_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerow(ITER_METRICS_HEADER)


def append_iter_metrics_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_iter_metrics_header(path)
    with path.open("a", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([row.get(k, "") for k in ITER_METRICS_HEADER])


def _fmt(v, fmt="{:.3f}"):
    if v is None or v == "":
        return "-"
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def render_summary_table(rows: List[Dict[str, Any]]) -> str:
    """Render the headline summary as a compact ASCII table.

    `rows` should already be filtered to the rows we want to show
    (typically: baseline, best_iter, last_iter, post_route).
    """
    if not rows:
        return "(no metrics rows)\n"
    cols = [
        ("Phase",          lambda r: str(r.get("phase") or r.get("iteration") or "-")),
        ("Iter",           lambda r: str(r.get("iteration") if r.get("iteration") not in (None, "") else "-")),
        ("WNS_ps",         lambda r: _fmt(r.get("wns_ps"))),
        ("TNS_ps",         lambda r: _fmt(r.get("tns_ps"))),
        ("TotalPwr_mW",    lambda r: _fmt(float(r["total_pw_w"]) * 1000.0 if r.get("total_pw_w") not in (None, "") else None)),
        ("Area_um2",       lambda r: _fmt(r.get("design_area_um2"))),
        ("Util_%",         lambda r: _fmt(r.get("util_pct"), "{:.1f}")),
        ("IterRT_s",       lambda r: _fmt(r.get("iter_runtime_s"), "{:.1f}")),
        ("LLM_RT_s",       lambda r: _fmt(r.get("llm_runtime_s"), "{:.1f}")),
        ("InTok",          lambda r: _fmt(r.get("input_tokens"), "{:.0f}")),
        ("OutTok",         lambda r: _fmt(r.get("output_tokens"), "{:.0f}")),
        ("Cost_USD",       lambda r: _fmt(r.get("llm_cost_usd"), "{:.4f}")),
    ]
    rendered: List[List[str]] = [[h for h, _ in cols]]
    for r in rows:
        rendered.append([fn(r) for _, fn in cols])
    widths = [max(len(row[c]) for row in rendered) for c in range(len(cols))]
    def fmt_row(row):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
    sep = "  ".join("-" * widths[i] for i in range(len(cols)))
    lines = [fmt_row(rendered[0]), sep]
    for r in rendered[1:]:
        lines.append(fmt_row(r))
    return "\n".join(lines) + "\n"


def load_iter_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            out.append(dict(row))
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(loop_max=48):
    log_event(f"RUN_LOOP START — DESIGN_DIR={DESIGN_DIR} TOP={DESIGN_TOP} loop_max={loop_max}")
    run_loop_t0 = _time.time()  # total wall-clock for the run
    # Fail fast if the design isn't set up correctly. This runs BEFORE we
    # touch any state so a bad invocation can't poison run_history.json.
    _validate_design_dir()
    # Fresh run — discard prior session artifacts so histories don't pollute
    # iteration 0 analysis.
    clear_run_state()

    proc = start_openroad()

    # Load design
    send(proc, f"source {OpenROAD_design_tcl}\nputs {PARA}\n")
    wait_for_sentinel(proc)

    # Define CSV dump procs and write initial node/net files + timing report
    send(proc, NODE_NET_TCL_BLOCK)
    send(proc, f'write_node_net_and_reports "{NODE_FILE}" "{NET_FILE}" "{NET_REPORTS_FILE}"')
    send(proc, make_report_cmd(RPT_out, NEARBY_PATHS))
    # Dump sibling-slack safety margins + high-fanout slack-ranked sinks
    send(
        proc,
        f'eco_dump_path_siblings "{SIBLING_SLACKS_FILE}" 5\n'
        f'eco_dump_fanout_ranks "{FANOUT_RANK_FILE}" 20\n'
        f'puts {PARA}\n',
    )
    wait_for_sentinel(proc)
    send(
        proc,
        f"""
    set fp [open "{METRICS_out}" a]
    puts $fp "=== Iteration 0 ==="
    close $fp
    tee -file "{METRICS_out}" -append {{ report_tns -digits 3 }}
    # Dedicated WNS/TNS dump — single source of truth for slack tracking.
    set fp_wt [open "{WNS_TNS_FILE}" w]
    close $fp_wt
    tee -file "{WNS_TNS_FILE}" {{ report_wns -digits 3 }}
    tee -file "{WNS_TNS_FILE}" -append {{ report_tns -digits 3 }}
    # Per-iter QoR dump (written directly via per-command tee).
    dump_iter_qor "{ITER_QOR_DIR}/iter_0.txt"
    puts {PARA}
    """,
    )
    ITER_QOR_DIR.mkdir(parents=True, exist_ok=True)
    SPEF_CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    send(proc, RIP_ALL_WIRES_PROC)  # define rip_all_signal_wires proc
    wait_for_sentinel(proc)

    eco_path = Path(ECO_out)
    rpt_path = Path(RPT_out)

    last_eco_mtime = eco_path.stat().st_mtime if eco_path.exists() else 0.0
    last_rpt_mtime = rpt_path.stat().st_mtime if rpt_path.exists() else 0.0
    #last_rpt_mtime = wait_for_file_update(rpt_path, last_rpt_mtime)

    # Build initial context (iteration 0)
    subprocess.run(
        [
            "python3", CONTEXT_BUILD,
            "--iteration", "0",
            "--max-paths", "20",
            "--node-file", NODE_FILE,
            "--net-file", NET_FILE,
            "--net-reports-file", NET_REPORTS_FILE,
            "--displacement-file", DISPLACEMENT_FILE,
            "--sibling-slacks-file", SIBLING_SLACKS_FILE,
            "--fanout-rank-file", FANOUT_RANK_FILE,
            "--out", DYNAMIC_CONTEXT_JSON,
        ],
        check=True,
    )

    slacks = []
    prev_qor = None
    prev_path_slacks = parse_report_path_slacks(rpt_path)

    # Prefer the dedicated WNS/TNS file; fall back to the report-scrape only
    # if OpenROAD didn't produce it (older runs).
    wt0 = parse_wns_tns_file(Path(WNS_TNS_FILE))
    baseline_wns = wt0.get("wns") if wt0.get("wns") is not None else extract_worst_slack(rpt_path)
    baseline_tns = wt0.get("tns") if wt0.get("tns") is not None else parse_tns_from_metrics(Path(METRICS_out))
    baseline_qor = {"iteration": 0, "wns": baseline_wns, "tns": baseline_tns}

    best_qor = dict(baseline_qor)
    stagnation_count = 0  # iterations without a new best WNS — drives backtrace
    backtrace_used = False  # backtrace is one-shot per run
    consecutive_degradations = 0  # logged-only for diagnostics; no longer drives control flow
    cycle_num = 0  # counts completed DRT refresh cycles
    last_drt_info: Optional[dict] = None  # calibration data from last DRT
    pre_drt_wns: Optional[float] = None   # GRT WNS right before DRT fires
    pre_drt_tns: Optional[float] = None   # GRT TNS right before DRT fires
    # best_qor is the GLOBAL best-DRT record ({cycle, iteration, wns, tns});
    # it is crowned ONLY at DRT boundaries from SPEF-accurate values.
    # cycle_best_grt_* track the best GRT iter WITHIN the current cycle — that
    # state (best_grt.odb) is what the cycle's DRT routes.
    cycle_best_grt_wns: Optional[float] = None
    cycle_best_grt_iter: Optional[int] = None
    iters_since_drt = 0
    best_drt_spef: Optional[Path] = None  # SPEF paired with the best-DRT odb

    # Baseline iter_metrics row (iteration 0). Token cost is 0 — no LLM call.
    baseline_qor_dump = parse_iter_qor(ITER_QOR_DIR / "iter_0.txt")
    baseline_row = {
        "iteration": 0,
        "phase": "baseline",
        "wns_ps": baseline_qor_dump.get("wns_ps") if baseline_qor_dump.get("wns_ps") is not None else baseline_wns,
        "tns_ps": baseline_qor_dump.get("tns_ps") if baseline_qor_dump.get("tns_ps") is not None else baseline_tns,
        "total_pw_w": baseline_qor_dump.get("total_pw_w"),
        "internal_pw_w": baseline_qor_dump.get("internal_pw_w"),
        "switching_pw_w": baseline_qor_dump.get("switching_pw_w"),
        "leakage_pw_w": baseline_qor_dump.get("leakage_pw_w"),
        "design_area_um2": baseline_qor_dump.get("design_area_um2"),
        "util_pct": baseline_qor_dump.get("util_pct"),
        "iter_runtime_s": round(_time.time() - run_loop_t0, 3),
        "llm_runtime_s": 0.0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "llm_cost_usd": 0.0,
    }
    append_iter_metrics_row(ITER_METRICS_CSV, baseline_row)
    print(f"  [METRICS] baseline iter_0 → wns={baseline_row['wns_ps']} "
          f"tns={baseline_row['tns_ps']} "
          f"total_pw_mW={(baseline_row['total_pw_w'] or 0)*1000:.3f} "
          f"area_um2={baseline_row['design_area_um2']}", flush=True)

    # Token-log line offset at the start of the iteration loop. Every
    # call to run_llm() appends rows; we diff line counts to attribute
    # rows to the right orchestrator iteration.
    token_log_offset = token_log_line_count(TOKEN_LOG_JSONL)

    # Write initial DRT state (SPEF loaded from baseline 6_final.odb).
    write_drt_state(cycle_num=0, iter_in_cycle=0, current_iter=0,
                    last_drt=None)

    # Write baseline best.odb BEFORE the loop starts. This guarantees
    # crash recovery always has a restore point even if WNS never
    # improves in the first DRT_CYCLE ECO iterations.
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    write_best_odb_snapshot(proc)
    print(f"  [INIT] Baseline best.odb saved to {BEST_ODB}", flush=True)

    # Seed best_grt.odb from the baseline too. The DRT reload guard keys off
    # BEST_GRT_ODB.exists(); without this seed an early-DRT that fires on the
    # very FIRST iteration (GRT WNS>=0 at iter 1) finds no snapshot yet, the
    # guard is False, and the DRT silently runs IN-PLACE -- leaking the GRT
    # estimate_parasitics state into report_power (the carry-forward bug the
    # reload exists to prevent). Seeding here guarantees the guard is true
    # from iter 1 onward; the per-iter GRT-best write overwrites it as soon
    # as a real best appears.
    send(proc, f'write_db "{BEST_GRT_ODB}"\nputs {PARA}\n')
    wait_for_sentinel(proc)
    print(f"  [INIT] Baseline best_grt.odb seeded to {BEST_GRT_ODB}", flush=True)

    for i in range(loop_max):
        print(f"=== Iteration {i+1} ===")
        iter_t0 = _time.time()

        # DRT-cycle tracking — iter_in_cycle drives best_qor / stagnation logic
        # and parasitic_context (SPEF vs GRT). Routing is always full global_route:
        # global_route -end_incremental calls FlexPA::updateDirtyInsts() which
        # segfaults (Signal 11) when eco_resize_gate changes a cell master.
        # TCL catch cannot intercept SIGSEGV — only avoiding incremental mode prevents it.
        # DRT_EVERY_ITER: every iter is a fresh SPEF-routed checkpoint, so pin
        # iter_in_cycle to 0 (parasitic_source=SPEF_extracted in drt_state).
        iter_in_cycle = 0 if DRT_EVERY_ITER else (iters_since_drt % DRT_CYCLE)
        is_first_in_cycle = (iter_in_cycle == 0)

        # Write DRT state so LLM_call.py can inject <parasitic_context> this iter.
        write_drt_state(
            cycle_num=cycle_num,
            iter_in_cycle=iter_in_cycle,
            current_iter=i + 1,
            last_drt=last_drt_info,
        )

        # ---- Save pre-ECO positions ----
        send(proc, f'write_node_net_and_reports "{NODE_FILE_PRE}" "{NET_FILE}" "{NET_REPORTS_FILE}"\nputs {PARA}\n')
        wait_for_sentinel(proc)
        sleep(0.1)

        # ---- Snapshot path 1 (the worst path the LLM will target) BEFORE ----
        path1_sp_pre, path1_ep_pre, path1_slack_pre = extract_path1_endpoints(Path(RPT_out))
        path1_before_text = ""
        try:
            full_rpt = Path(RPT_out).read_text(encoding="utf-8", errors="ignore")
            # Cut just the first Startpoint:... section
            chunks = re.split(r"(?=Startpoint:)", full_rpt)
            if len(chunks) >= 2:
                path1_before_text = chunks[1]
        except Exception:
            pass
        print(f"  [TARGET] path1: sp={path1_sp_pre} ep={path1_ep_pre} slack={path1_slack_pre}")

        # ---- Call LLM to produce ECO commands ----
        llm_t0 = _time.time()
        pre_token_lines = token_log_line_count(TOKEN_LOG_JSONL)
        llm_ok, llm_info = run_llm(max_iter=1, timeout=60, hard_timeout_s=600.0)
        llm_runtime_s = round(_time.time() - llm_t0, 3)
        if not llm_ok:
            print(f"  [LLM] FAILED: {llm_info} (after {llm_runtime_s}s)", flush=True)
            _write_eco_sentinel(eco_path, llm_info)
            llm_consecutive_failures = locals().get('llm_consecutive_failures', 0) + 1
            if llm_consecutive_failures >= 3:
                print(f"  [LLM] 3 consecutive failures — terminating loop", flush=True)
                break
        else:
            llm_consecutive_failures = 0
        last_eco_mtime = wait_for_file_update(eco_path, last_eco_mtime)

        # ---- Source ECO in OpenROAD with error capture ----
        eco_error_log = Path(f"{workdir}/prompts/dynamic/eco_errors.log")
        # ---- Post-ECO routing sequence ----
        # DEFAULT: fast incremental GRT estimate (optimistic; the periodic
        # DRT-cycle block re-routes every DRT_CYCLE iters). DRT_EVERY_ITER ON:
        # every iter routes the FULL detailed_route + OpenRCX SPEF so this
        # iteration's report/scoring are post-route accurate, and the separate
        # periodic-DRT block is skipped (gated by not DRT_EVERY_ITER above).
        _report_cmd = make_report_cmd(RPT_out, NEARBY_PATHS)
        if DRT_EVERY_ITER:
            spef_iter_path = SPEF_CYCLE_DIR / f"spef_iter_{i+1}.spef"
            _route_seq = f"""remove_fillers
            detailed_placement -incremental
            rip_all_signal_wires
            global_route
            detailed_route -droute_end_iter 1
            define_process_corner -ext_model_index 0 X
            extract_parasitics -ext_model_file {ASAP7_PDK}/rcx_patterns.rules
            if {{[catch {{write_spef {spef_iter_path}}} _ws_err]}} {{
                puts "\[DRT-ITER\] write_spef failed ($_ws_err) - in-memory RC"
            }} else {{
                read_spef {spef_iter_path}
            }}
            {_report_cmd}"""
        else:
            spef_iter_path = None
            _route_seq = f"""remove_fillers
            detailed_placement -incremental
            global_route
            estimate_parasitics -global_routing
            {_report_cmd}"""
        send(
            proc,
            f"""
            # Per-command catch: read ECO file line-by-line and execute
            # each non-comment line in its own [catch]. One command\'s
            # failure (e.g. bad hierarchical name) no longer aborts the
            # remaining commands in the batch.
            # Probe commands (read-only queries the LLM emits to ask OpenROAD
            # for fanout, sink, or shared-path data) are detected by name and
            # their stdout is tee\'d into PROBE_RESPONSES_FILE so next iter
            # can show the LLM what its probe returned.
            set _eco_errors [list]
            set _eco_ran 0
            set _probe_cmds [list "eco_rank_fanout_by_slack" "eco_top_paths_through" "eco_net_sink_report"]
            set _probe_file "{PROBE_RESPONSES_FILE}"
            set _pfh_init [open $_probe_file w]
            puts $_pfh_init "# probe_responses for iter {i+1}"
            close $_pfh_init
            set _eco_fp [open {ECO_out} r]
            while {{[gets $_eco_fp _eco_line] >= 0}} {{
                set _eco_line [string trim $_eco_line]
                if {{$_eco_line eq "" || [string index $_eco_line 0] eq "#"}} {{
                    continue
                }}
                set _eco_head [lindex [split $_eco_line] 0]
                set _is_probe [expr {{[lsearch -exact $_probe_cmds $_eco_head] >= 0}}]
                if {{$_is_probe}} {{
                    # Probe procs (eco_rank_fanout_by_slack, eco_top_paths_through)
                    # RETURN a Tcl list of pairs (sink_or_endpoint, slack) rather
                    # than printing to stdout, so we capture the return value
                    # and write it directly. Errors are written too — even an
                    # error message is useful feedback ("you asked for a sink
                    # report on a net that doesn\'t exist").
                    set _pfh [open $_probe_file a]
                    puts $_pfh ""
                    puts $_pfh "=== probe: $_eco_line ==="
                    if {{[catch {{uplevel #0 $_eco_line}} _eco_result]}} {{
                        puts $_pfh "ERROR: $_eco_result"
                        lappend _eco_errors "FAILED: $_eco_line  ERROR: $_eco_result"
                    }} else {{
                        # Pretty-print the return value. If it\'s a list of
                        # pairs, one per line ("sink  slack_ps"). Otherwise
                        # dump as-is. Skip the proc\'s ZERO-length case (some
                        # probes only side-effect — write a 0-row marker so the
                        # LLM sees the probe ran but produced nothing).
                        if {{[catch {{
                            set _row_count 0
                            foreach _r $_eco_result {{
                                if {{[llength $_r] >= 2}} {{
                                    puts $_pfh "[lindex $_r 0]  [lindex $_r 1]"
                                }} else {{
                                    puts $_pfh $_r
                                }}
                                incr _row_count
                            }}
                            if {{$_row_count == 0}} {{ puts $_pfh "(no rows returned)" }}
                        }}]}} {{
                            puts $_pfh $_eco_result
                        }}
                        incr _eco_ran
                    }}
                    close $_pfh
                }} else {{
                    if {{[catch {{uplevel #0 $_eco_line}} _eco_cerr]}} {{
                        lappend _eco_errors "FAILED: $_eco_line  ERROR: $_eco_cerr"
                    }} else {{
                        incr _eco_ran
                    }}
                }}
            }}
            close $_eco_fp
            set fp_err [open "{eco_error_log}" w]
            if {{[llength $_eco_errors] == 0}} {{
                puts $fp_err "OK"
            }} else {{
                puts $fp_err "PARTIAL: $_eco_ran command(s) succeeded, [llength $_eco_errors] failed"
                foreach _eco_e $_eco_errors {{ puts $fp_err $_eco_e }}
            }}
            close $fp_err
            {_route_seq}
            write_verilog {verilog_out}
            puts {MARK}
            """,
        )
        send(
            proc,
            f"""
            set fp [open "{METRICS_out}" a]
            puts $fp "=== Iteration {i+1} ==="
            close $fp
            tee -file "{METRICS_out}" -append {{
            report_tns -digits 3
            }}
            # Fresh WNS/TNS dump — overwritten every iteration so the parser
            # only ever sees the current values.
            set fp_wt [open "{WNS_TNS_FILE}" w]
            close $fp_wt
            tee -file "{WNS_TNS_FILE}" {{ report_wns -digits 3 }}
            tee -file "{WNS_TNS_FILE}" -append {{ report_tns -digits 3 }}
            # Per-iter QoR dump (written directly via per-command tee).
            dump_iter_qor "{ITER_QOR_DIR}/iter_{i+1}.txt"
            """,
        )

        try:
            wait_for_sentinel(proc, MARK)
        except OpenROADCrash as _crash:
            print(f"  [CRASH] OpenROAD died at iter {i+1}: {_crash}", flush=True)
            log_event(f"OPENROAD CRASH iter={i+1}: {_crash}")
            # Restart from the best snapshot we have. If no best.odb yet,
            # we cannot safely continue — exit the loop.
            if not BEST_ODB.exists():
                print("  [CRASH] No best.odb to restart from — ending loop.", flush=True)
                break
            print(
                f"  [CRASH] Restarting from best.odb "
                f"(iter {best_qor.get('iteration')}, WNS={best_qor.get('wns')})",
                flush=True)
            shutdown_openroad(proc)
            proc = restart_openroad_from_best()
            send(proc, f"source {OpenROAD_design_tcl}\nputs {PARA}\n")
            wait_for_sentinel(proc)
            send(proc, NODE_NET_TCL_BLOCK)
            send(proc, RIP_ALL_WIRES_PROC)
            send(proc, make_report_cmd(RPT_out, NEARBY_PATHS))
            send(proc,
                f'eco_dump_path_siblings "{SIBLING_SLACKS_FILE}" 5\n'
                f'eco_dump_fanout_ranks "{FANOUT_RANK_FILE}" 20\n'
                f'puts {PARA}\n')
            wait_for_sentinel(proc)
            wt_crash = parse_wns_tns_file(Path(WNS_TNS_FILE))
            wns = wt_crash.get("wns") or best_qor.get("wns")
            tns = wt_crash.get("tns") or best_qor.get("tns")
            curr_qor = {"iteration": i + 1, "wns": wns, "tns": tns,
                        "score": 0, "score_vs_baseline": 0, "score_vs_best": 0,
                        "commands": [], "neighbor_delta": 0.0, "new_violations": 0}
            outcome = "crash_restart"
            stagnation_count = 0  # fresh start after crash
            print(f"  [CRASH] Restarted. WNS={wns} TNS={tns}", flush=True)
            # Skip the rest of this iteration's processing; context will
            # rebuild at end of loop for next iter.
            prev_qor = curr_qor
            prev_path_slacks = parse_report_path_slacks(rpt_path)
            subprocess.run(
                ["python3", CONTEXT_BUILD,
                 "--iteration", str(i + 1),
                 "--max-paths", "20",
                 "--node-file", NODE_FILE,
                 "--net-file", NET_FILE,
                 "--net-reports-file", NET_REPORTS_FILE,
                 "--displacement-file", DISPLACEMENT_FILE,
                 "--sibling-slacks-file", SIBLING_SLACKS_FILE,
                 "--fanout-rank-file", FANOUT_RANK_FILE,
                 "--out", DYNAMIC_CONTEXT_JSON],
                check=False)
            continue  # skip to next iteration

        last_rpt_mtime = wait_for_file_update(rpt_path, last_rpt_mtime)

        # ---- Read ECO error log ----
        eco_error_msg = None
        if eco_error_log.exists():
            err_text = eco_error_log.read_text(encoding="utf-8", errors="ignore").strip()
            if err_text and err_text != "OK":
                eco_error_msg = err_text[:300]
                print(f"  [ECO ERROR] {eco_error_msg[:100]}")

        # ---- Refresh post-ECO node/net files ----
        send(proc, f'write_node_net_and_reports "{NODE_FILE}" "{NET_FILE}" "{NET_REPORTS_FILE}"\nputs {PARA}\n')
        wait_for_sentinel(proc)

        # Refresh sibling-slack + fanout-rank dumps for next-iter context
        send(
            proc,
            f'eco_dump_path_siblings "{SIBLING_SLACKS_FILE}" 5\n'
            f'eco_dump_fanout_ranks "{FANOUT_RANK_FILE}" 20\n'
            f'puts {PARA}\n',
        )
        wait_for_sentinel(proc)

        # ---- Compute displacement ----
        eco_cmds = read_eco_commands(eco_path)

        # ---- Re-report the targeted worst path so the LLM sees how its
        #      move actually moved the SAME path (including upstream
        #      cascade if any). Writes PREV_TARGET_JSON consumed by
        #      LLM_call.py's <previous_target_path> block.
        try:
            if path1_sp_pre and path1_ep_pre:
                # Escape any TCL special chars in identifiers (backslashes
                # in flattened hierarchical names need to be passed cleanly).
                # Tcl vars in {} so [], $, \ in hierarchical names are literal.
                # Truncate the file FIRST so a silent failure can never leave
                # stale content; append "# REPORT_END" after report_checks via
                # a separate open/puts/close which synchronously flushes — the
                # orchestrator reads only when the marker is present.
                send(
                    proc,
                    f'set _ptp_sp {{{path1_sp_pre}}}\n'
                    f'set _ptp_ep {{{path1_ep_pre}}}\n'
                    f'set _ptp_file "{PREV_TARGET_REPORT_TXT}"\n'
                    # Truncate up-front so stale content from prior iter cannot leak through
                    f'set _ptp_fh [open $_ptp_file w]; close $_ptp_fh\n'
                    f'if {{[catch {{report_checks -from $_ptp_sp -to $_ptp_ep '
                    f'-fields {{slew input_pins cap fanout net}} > $_ptp_file}} _ptp_err]}} {{'
                    f'  set _ptp_efp [open $_ptp_file w]; '
                    f'  puts $_ptp_efp "# report_checks failed: $_ptp_err"; '
                    f'  close $_ptp_efp'
                    f'}}\n'
                    # Append completion marker via a separate atomic open/puts/close.
                    # close() does fsync on the file handle, so by the time PARA fires,
                    # the marker (and the report_checks content above it) is on disk.
                    f'set _ptp_efp [open $_ptp_file a]; puts $_ptp_efp "# REPORT_END"; close $_ptp_efp\n'
                    f'puts {PARA}\n'
                )
                wait_for_sentinel(proc)
                # Poll until the marker is on disk (up to 5 s) before reading.
                after_text = ""
                import time as _t
                _deadline = _t.time() + 5.0
                while _t.time() < _deadline:
                    try:
                        after_text = Path(PREV_TARGET_REPORT_TXT).read_text(
                            encoding='utf-8', errors='ignore'
                        )
                        if "# REPORT_END" in after_text:
                            break
                    except Exception:
                        pass
                    _t.sleep(0.1)
                if "# REPORT_END" not in after_text:
                    print(f"  [WARN] prev_target_path report missing REPORT_END marker; using whatever we have", flush=True)
                # Parse the post-ECO slack of the same path
                after_slack = None
                for ln in reversed(after_text.splitlines()):
                    if 'slack' in ln.lower():
                        m = re.search(r'(-?\d+(?:\.\d+)?)', ln)
                        if m:
                            try:
                                after_slack = float(m.group(1))
                                break
                            except ValueError:
                                pass
                before_toon = _path_to_toon(path1_before_text, label_prefix="before_")
                after_toon  = _path_to_toon(after_text,         label_prefix="after_")
                ptp = {
                    'targeted_iteration': i + 1,
                    'startpoint': path1_sp_pre,
                    'endpoint': path1_ep_pre,
                    'slack_before_ps': path1_slack_pre,
                    'slack_after_ps': after_slack,
                    'delta_ps': round((after_slack - path1_slack_pre), 3)
                                if (after_slack is not None and path1_slack_pre is not None)
                                else None,
                    'commands_applied': eco_cmds,
                    'before_path_toon': before_toon,
                    'after_path_toon':  after_toon,
                }
                Path(PREV_TARGET_JSON).write_text(json.dumps(ptp, indent=2), encoding='utf-8')
                print(f"  [PREV_PATH] slack_before={path1_slack_pre} slack_after={after_slack} delta={ptp['delta_ps']}")
        except Exception as e:
            print(f"  [WARN] prev_target_path snapshot failed: {e}", flush=True)

        analysis_path = Path(ANALYSIS_out)
        analysis_text = ""
        if analysis_path.exists():
            try:
                analysis_text = analysis_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"  [WARN] failed to read analysis sidecar: {e}", flush=True)
        pre_pos = load_node_positions(Path(NODE_FILE_PRE))
        post_pos = load_node_positions(Path(NODE_FILE))

        # Extract instances on worst paths from the most recent context
        path_insts = set()
        try:
            ctx = json.loads(Path(DYNAMIC_CONTEXT_JSON).read_text(encoding="utf-8"))
            for viol in ctx.get("dynamic", ctx).get("worst_violations", []):
                for pt in viol.get("points", []):
                    cell = pt.get("cell", "")
                    if cell:
                        path_insts.add(cell)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

        disp_data = compute_displacement(pre_pos, post_pos, eco_cmds, path_instances=path_insts)
        disp_data["iteration"] = i + 1

        disp_path = Path(DISPLACEMENT_FILE)
        disp_path.parent.mkdir(parents=True, exist_ok=True)
        disp_path.write_text(json.dumps(disp_data, indent=2), encoding="utf-8")

        s = disp_data["summary"]
        if s["total_displaced"] > 0:
            print(f"  [DISPLACEMENT] eco_targets={s['eco_targets_displaced']}, "
                  f"path_cells={s['path_cells_displaced']}, "
                  f"collateral={s['collateral_displaced']}, "
                  f"path_max={s['path_max_displacement_um']:.1f}um")

        # ---- Parse timing results ----
        # Prefer the dedicated report_wns / report_tns dump for per-iteration
        # tracking (avoids misparses from the big multi-path report file).
        wt = parse_wns_tns_file(Path(WNS_TNS_FILE))
        wns = wt.get("wns") if wt.get("wns") is not None else extract_worst_slack(rpt_path)
        tns = wt.get("tns") if wt.get("tns") is not None else parse_tns_from_metrics(Path(METRICS_out))

        # DRT->GRT parasitic-model shift: at the first ECO iter of a new cycle,
        # compare the previous cycle's routed SPEF/DRT WNS against this iter's
        # first GRT re-estimate (how much the model jump moves the numbers).
        if (SAVE_GRT_DRT_CORR and iter_in_cycle == 0 and cycle_num >= 1
                and last_drt_info):
            _pdw = last_drt_info.get("post_drt_wns_ps")
            _pdt = last_drt_info.get("post_drt_tns_ps")
            append_corr_row({
                "cycle": cycle_num, "at_iter": i + 1, "transition": "DRT->GRT",
                "wns_from": _pdw, "wns_to": wns,
                "dwns_ps": (round(wns - _pdw, 3) if wns is not None and _pdw is not None else None),
                "tns_from": _pdt, "tns_to": tns,
                "dtns_ps": (round(tns - _pdt, 3) if tns is not None and _pdt is not None else None),
            })
            print(f"  [CORR] DRT->GRT cycle {cycle_num} start (iter {i+1}): "
                  f"WNS {_pdw} -> {wns}", flush=True)
        curr_path_slacks = parse_report_path_slacks(rpt_path)

        # Neighbor delta proxy (average slack change of paths 2-6)
        neighbor_delta = 0.0
        if prev_path_slacks and curr_path_slacks:
            prev_sorted = sorted(prev_path_slacks)[:6]
            curr_sorted = sorted(curr_path_slacks)[:6]
            if len(prev_sorted) > 1 and len(curr_sorted) > 1:
                prev_nei = prev_sorted[1:min(6, len(prev_sorted))]
                curr_nei = curr_sorted[1:min(6, len(curr_sorted))]
                k = min(len(prev_nei), len(curr_nei))
                if k > 0:
                    neighbor_delta = sum(curr_nei[:k]) / k - sum(prev_nei[:k]) / k

        new_viol = max(0, len(curr_path_slacks) - len(prev_path_slacks)) if prev_path_slacks else 0

        # ---- Score this iteration ----
        curr_qor = {
            "iteration": i + 1,
            "wns": wns,
            "tns": tns,
            "neighbor_delta": neighbor_delta,
            "new_violations": new_viol,
            "commands": eco_cmds,
        }
        curr_qor["score"] = compute_qor_score(prev_qor, curr_qor)
        curr_qor["score_vs_baseline"] = compute_qor_score(baseline_qor, curr_qor)
        curr_qor["score_vs_best"] = compute_qor_score(best_qor, curr_qor)

    # Classify outcome
        outcome = "unknown"
        if prev_qor is not None and isinstance(curr_qor["score"], (int, float)):
            if curr_qor["score"] > 0:
                outcome = "improved"
                consecutive_degradations = 0
            elif curr_qor["score"] < 0:
                outcome = "degraded"
                consecutive_degradations += 1
            else:
                outcome = "unchanged"
                consecutive_degradations = 0

        # ---- Track best GRT iter WITHIN this cycle ----
        # GRT estimates are optimistic vs SPEF/DRT, so they NEVER crown the
        # global best (that happens only at DRT boundaries). Instead we
        # record the best GRT iter of the current cycle and snapshot it to
        # best_grt.odb; at the DRT boundary the loop rolls back to that
        # state and routes IT (not the regressed iter-8 cumulative state).
        if iter_in_cycle == 0:
            # New cycle — reset the cycle-local GRT-best tracker.
            cycle_best_grt_wns = None
            cycle_best_grt_iter = None
        if wns is not None and (cycle_best_grt_wns is None or wns > cycle_best_grt_wns):
            cycle_best_grt_wns = wns
            cycle_best_grt_iter = i + 1
            send(proc, f'write_db "{BEST_GRT_ODB}"\nputs {PARA}\n')
            wait_for_sentinel(proc)
            print(f"  [GRT-BEST] cycle GRT best WNS={wns} at iter {i+1} "
                  f"-> {BEST_GRT_ODB}", flush=True)

        # ---- Backtrace on stagnation (no new best for N iters) ----
        # If the loop has gone N iterations without setting a new best WNS,
        # we are stuck oscillating below the best ever found. Restart from
        # the best.odb snapshot and tell the LLM what the last 10 moves
        # looked like so it can pick a different strategy. Backtrace fires
        # at most once per run; the second stagnation event ends the loop
        # and finals are written from the restored best.odb.
        if stagnation_count >= BACKTRACE_THRESHOLD:
            if backtrace_used:
                print(
                    f"  [STAGNATION] {stagnation_count} iters since last new "
                    f"best WNS, and backtrace already used. Ending loop; "
                    f"finals will be written from best.odb (iter "
                    f"{best_qor.get('iteration')}, WNS={best_qor.get('wns')})."
                )
                break
            if not BEST_ODB.exists():
                print(
                    f"  [BACKTRACE] {stagnation_count} iters without new best, "
                    f"but no best.odb snapshot exists — aborting."
                )
                break
            print(
                f"  [BACKTRACE] {stagnation_count} iters since last new best. "
                f"Restarting OpenROAD from best.odb "
                f"(iter {best_qor.get('iteration')}, WNS={best_qor.get('wns')})."
            )
            log_event(
                f"BACKTRACE — restarting from best.odb "
                f"iter={best_qor.get('iteration')} WNS={best_qor.get('wns')} "
                f"stagnation_count={stagnation_count}"
            )
            try:
                hist = json.loads(RUN_HISTORY_JSON.read_text())
                last10 = hist[-10:] if isinstance(hist, list) else []
            except Exception:
                last10 = []
            write_backtrace_notice(best_qor.get("iteration"),
                                   best_qor.get("wns"), last10)

            shutdown_openroad(proc)
            proc = restart_openroad_from_best()

            # Re-load design (TCL reads BACKTRACE_ODB env var via $::env).
            send(proc, f"source {OpenROAD_design_tcl}\nputs {PARA}\n")
            wait_for_sentinel(proc)
            send(proc, NODE_NET_TCL_BLOCK)
            send(proc, f'write_node_net_and_reports "{NODE_FILE}" "{NET_FILE}" "{NET_REPORTS_FILE}"')
            send(proc, make_report_cmd(RPT_out, NEARBY_PATHS))
            send(proc,
                f'eco_dump_path_siblings "{SIBLING_SLACKS_FILE}" 5\n'
                f'eco_dump_fanout_ranks "{FANOUT_RANK_FILE}" 20\n'
                f'puts {PARA}\n')
            wait_for_sentinel(proc)
            last_rpt_mtime = wait_for_file_update(rpt_path, last_rpt_mtime)
            wns = extract_worst_slack(rpt_path)
            tns = parse_tns_from_metrics(Path(METRICS_out))
            curr_path_slacks = parse_report_path_slacks(rpt_path)
            prev_path_slacks = curr_path_slacks
            prev_qor = {"iteration": i + 1, "wns": wns, "tns": tns}
            stagnation_count = 0
            consecutive_degradations = 0
            backtrace_used = True

        # ---- Parse per-iter QoR dump + token cost, append iter_metrics row ----
        iter_qor_path = ITER_QOR_DIR / f"iter_{i+1}.txt"
        iqd = parse_iter_qor(iter_qor_path)
        new_tok_rows = read_new_token_log_rows(TOKEN_LOG_JSONL, pre_token_lines)
        tok_agg = aggregate_token_rows(new_tok_rows)
        iter_runtime_s = round(_time.time() - iter_t0, 3)
        iter_row = {
            "iteration": i + 1,
            "phase": "eco",
            # Prefer the iter_qor parse for wns/tns (it was dumped at the same time as power/area).
            "wns_ps": iqd.get("wns_ps") if iqd.get("wns_ps") is not None else wns,
            "tns_ps": iqd.get("tns_ps") if iqd.get("tns_ps") is not None else tns,
            "total_pw_w": iqd.get("total_pw_w"),
            "internal_pw_w": iqd.get("internal_pw_w"),
            "switching_pw_w": iqd.get("switching_pw_w"),
            "leakage_pw_w": iqd.get("leakage_pw_w"),
            "design_area_um2": iqd.get("design_area_um2"),
            "util_pct": iqd.get("util_pct"),
            "iter_runtime_s": iter_runtime_s,
            "llm_runtime_s": llm_runtime_s,
            "llm_calls": tok_agg.get("num_calls", 0),
            "input_tokens": tok_agg.get("input_tokens", 0),
            "output_tokens": tok_agg.get("output_tokens", 0),
            "cache_read_tokens": tok_agg.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": tok_agg.get("cache_creation_input_tokens", 0),
            "llm_cost_usd": round(tok_agg.get("total_cost_usd", 0.0), 6),
        }
        append_iter_metrics_row(ITER_METRICS_CSV, iter_row)
        print(f"  [METRICS] iter {i+1} → wns={iter_row['wns_ps']} "
              f"tns={iter_row['tns_ps']} "
              f"total_pw_mW={(iter_row['total_pw_w'] or 0)*1000:.3f} "
              f"area_um2={iter_row['design_area_um2']} "
              f"iter_rt={iter_runtime_s}s llm_rt={llm_runtime_s}s "
              f"cost=${iter_row['llm_cost_usd']:.4f} "
              f"tok={iter_row['input_tokens']}/{iter_row['output_tokens']}", flush=True)

        # ---- DRT_EVERY_ITER: crown the GLOBAL best inline ----
        # In every-iter mode the per-iter route IS a full DRT + SPEF, so wns/tns
        # here are post-route accurate and the periodic DRT block (which normally
        # crowns best_qor + writes finals) is skipped. Crown here instead: on a
        # new best, write best_drt.odb + shipped deliverables in place; otherwise
        # advance stagnation. Mirrors the DRT-block crowning logic.
        if DRT_EVERY_ITER:
            if wns is not None and (
                best_qor.get("wns") is None or wns > best_qor["wns"]
            ):
                best_qor = {"cycle": None, "iteration": i + 1,
                            "wns": wns, "tns": tns}
                best_drt_spef = spef_iter_path
                stagnation_count = 0
                try:
                    shutil.copy(RPT_out, FINAL_TIMING_RPT)
                except OSError as _e:
                    print(f"  [DRT-ITER][WARN] copy FINAL_TIMING_RPT: {_e}", flush=True)
                send(
                    proc,
                    f'write_db "{BEST_ODB}"\n'
                    f'write_verilog {FINAL_VERILOG}\n'
                    f'write_def {FINAL_DEF}\n'
                    f'set _drc_fp [open "{FINAL_DRC_RPT}" w]; close $_drc_fp\n'
                    f'tee -file "{FINAL_DRC_RPT}" {{ check_drc }}\n'
                    f'puts {PARA}\n'
                )
                wait_for_sentinel(proc)
                print(f"  [DRT-ITER] best improved -> WNS={wns} at iter {i+1}; "
                      f"best_drt.odb + finals written.", flush=True)
            else:
                stagnation_count += 1
                print(f"  [DRT-ITER] WNS={wns} did not improve best "
                      f"({best_qor.get('wns')}); stagnation_count="
                      f"{stagnation_count}.", flush=True)

        # ---- Record ECO outcome ----

        # Persist only the per-iteration record actually consumed by the LLM.
        predicted_delta_ps = parse_predicted_delta_ps(analysis_text)
        run_record = {
            "iteration": i + 1,
            "wns": wns,
            "tns": tns,
            "score": curr_qor["score"],
            "outcome": outcome,
            "commands": eco_cmds,
            "analysis": analysis_text,
            "predicted_delta_ps": predicted_delta_ps,
            "eco_error": eco_error_msg,   # None if clean
        }
        append_json_record(RUN_HISTORY_JSON, run_record, max_len=100)

        # Detailed ECO record for offline QoR tracing. Not sent to the LLM.
        eco_record = {
            "iteration": i + 1,
            "commands": curr_qor["commands"],
            "slack_before": None if prev_qor is None else prev_qor.get("wns"),
            "slack_after": wns,
            "tns_before": None if prev_qor is None else prev_qor.get("tns"),
            "tns_after": tns,
            "score": curr_qor["score"],
            "score_vs_baseline": curr_qor["score_vs_baseline"],
            "score_vs_best": curr_qor["score_vs_best"],
            "outcome": outcome,
            "best_wns_ever": best_qor.get("wns"),
            "best_wns_iteration": best_qor.get("iteration"),
            "consecutive_degradations": consecutive_degradations,
            "displacement_summary": disp_data["summary"],
            "command_types_used": list(set(
                cmd.split()[0] for cmd in curr_qor["commands"] if cmd.split()
            )),
        }
        if eco_error_msg:
            eco_record["eco_error"] = eco_error_msg
        append_json_record(ECO_HISTORY_JSON, eco_record, max_len=400)

        with open(slack_track, "a", encoding="utf-8") as f:
            f.write(f"{i+1}\t{wns}\n")

        if wns is not None:
            slacks.append(wns)

        prev_qor = curr_qor
        prev_path_slacks = curr_path_slacks

        # ---- DRT refresh ----
        # Fires when EITHER (a) DRT_CYCLE ECO iters have run since the last
        # DRT, (b) GRT timing just closed (WNS >= 0) — route NOW instead of
        # wasting the remaining GRT iters of the cycle, or (c) this is the
        # final loop iteration (so the run always ends DRT-confirmed).
        iters_since_drt += 1
        grt_timing_closed = (wns is not None and wns >= 0)
        # DRT_EVERY_ITER: the per-iter route is already a full DRT+SPEF, so the
        # periodic-DRT block below is disabled (gated by not DRT_EVERY_ITER).
        run_drt_now = (iters_since_drt >= DRT_CYCLE) or grt_timing_closed or ((i + 1) == loop_max)
        if run_drt_now and grt_timing_closed and iters_since_drt < DRT_CYCLE:
            print(f"  [EARLY-DRT] GRT WNS={wns} >= 0 at iter {i+1} "
                  f"({iters_since_drt}/{DRT_CYCLE} iters into the cycle) — "
                  f"skipping remaining GRT iters, routing immediately.",
                  flush=True)
            log_event(f"EARLY DRT at iter {i+1}: GRT WNS={wns} >= 0")
        if run_drt_now and not DRT_EVERY_ITER:
            cycle_num += 1
            spef_cycle_path = SPEF_CYCLE_DIR / f"spef_cycle_{cycle_num}.spef"
            drt_cycle_odb = SPEF_CYCLE_DIR / f"drt_cycle_{cycle_num}.odb"
            # The DRT routes the cycle's BEST GRT state, so the GRT estimate
            # it is calibrated against is that best-GRT WNS (not iter-8 wns).
            pre_drt_wns = cycle_best_grt_wns if cycle_best_grt_wns is not None else wns
            pre_drt_tns = tns

            # ---- ALWAYS reload OpenROAD from the cycle's best-GRT snapshot ----
            # before DRT, for TWO reasons:
            #   (1) regression: if the best GRT iter is NOT the last iter, the
            #       in-memory design has drifted past the best — reloading routes
            #       the best state (and we trim the discarded iters' history).
            #   (2) power correctness: an IN-PLACE DRT leaves the GRT
            #       estimate_parasitics state in memory, which report_power keeps
            #       using even AFTER read_spef — so the captured DRT power equals
            #       the GRT estimate (verified carry-forward leak). A FRESH
            #       process has no GRT ghost and reports true SPEF power. Every
            #       restart-cycle reported correct SPEF power; every in-place
            #       cycle leaked GRT. So reload EVERY cycle, even when best-GRT
            #       IS the last iter (best_grt.odb == current state, just loaded
            #       into a clean process).
            _regressed = (cycle_best_grt_iter is not None
                          and cycle_best_grt_iter != (i + 1))
            # Reload is keyed ONLY on the snapshot existing -- NOT on
            # cycle_best_grt_iter being set. An early-DRT can fire before a
            # cycle-best is recorded (e.g. GRT WNS>=0 at iter 1); best_grt.odb
            # is seeded at baseline so it always exists. If it is somehow
            # missing we fall back to an IN-PLACE DRT but say so LOUDLY rather
            # than silently leaking GRT-parasitic power (see the [DRT-WARN]
            # else branch below).
            if BEST_GRT_ODB.exists():
                if _regressed:
                    print(f"  [ROLLBACK] cycle {cycle_num}: reverting to best-GRT "
                          f"iter {cycle_best_grt_iter} (WNS={cycle_best_grt_wns}) "
                          f"before DRT; discarding iters "
                          f"{cycle_best_grt_iter + 1}..{i + 1}.", flush=True)
                    log_event(f"CYCLE {cycle_num} ROLLBACK to best-GRT "
                              f"iter={cycle_best_grt_iter} WNS={cycle_best_grt_wns}")
                else:
                    print(f"  [DRT-RELOAD] cycle {cycle_num}: best-GRT is the last "
                          f"iter — reloading fresh from best_grt.odb anyway so "
                          f"report_power uses SPEF (no in-place GRT-parasitic carry).",
                          flush=True)
                    log_event(f"CYCLE {cycle_num} DRT-RELOAD (clean SPEF power) "
                              f"from best_grt.odb iter={cycle_best_grt_iter}")
                shutdown_openroad(proc)
                proc = restart_openroad_from_odb(BEST_GRT_ODB)
                _grt_load_tcl = generate_load_tcl_for(
                    BEST_GRT_ODB,
                    f"{workdir}/OpenROAD_utils/OpenROAD_load_design_best.tcl")
                send(proc, f"source {_grt_load_tcl}\nputs {PARA}\n")
                wait_for_sentinel(proc)
                send(proc, NODE_NET_TCL_BLOCK)
                send(proc, RIP_ALL_WIRES_PROC)
                send(proc, f"puts {PARA}\n")
                wait_for_sentinel(proc)
                if _regressed:
                    _trim_history_after(RUN_HISTORY_JSON, cycle_best_grt_iter)
                    _trim_history_after(ECO_HISTORY_JSON, cycle_best_grt_iter)
            else:
                # best_grt.odb missing despite the baseline seed -- DRT must
                # run in-place on the current process. report_power may carry
                # the GRT estimate_parasitics ghost (power will read as the GRT
                # estimate, not true SPEF). Warn loudly so this is never silent.
                print(f"  [DRT-WARN] cycle {cycle_num}: no best_grt.odb snapshot "
                      f"-- running DRT IN-PLACE; report_power may leak GRT "
                      f"parasitics (SPEF power suspect this cycle).", flush=True)
                log_event(f"CYCLE {cycle_num} IN-PLACE DRT (no best_grt.odb) "
                          f"-- power may be GRT-leaked")
            drt_mark = f"__DRT_CYCLE_{cycle_num}_DONE__"
            print(
                f"\n  [DRT-CYCLE {cycle_num}] Full rip-and-reroute DRT "
                f"after iter {i+1}...", flush=True)
            log_event(
                f"DRT CYCLE {cycle_num} START — after iter {i+1}, "
                f"spef={spef_cycle_path}")

            send(
                proc,
                f"""
                # === DRT Cycle {cycle_num}: Full rip-and-reroute sequence ===
                # Step 1: Re-legalize placement after 8 ECO iterations of
                # cumulative changes (clones, inserts, resizes).
                remove_fillers
                puts "\[DRT\] Step 1: detailed_placement -incremental..."
                detailed_placement -incremental

                # Step 2: Rip all signal net wires + clear routing guides.
                # This gives detailed_route a clean slate with no stale wires.
                puts "\[DRT\] Step 2: rip_all_signal_wires..."
                rip_all_signal_wires

                # Step 3: Global route — fresh GRT guides for ALL nets,
                # including new ECO nets from clones and buffer inserts.
                puts "\[DRT\] Step 3: global_route..."
                global_route

                # Step 4: Detailed route — routes all nets from GRT guides.
                # After full rip-up, all nets need routing (1 opt iteration).
                puts "\[DRT\] Step 4: detailed_route -droute_end_iter 1..."
                detailed_route -droute_end_iter 1

                # Step 5: OpenRCX extraction — accurate RC from properly
                # routed wires. All new ECO nets now have dbWire objects,
                # so the iterm index is complete and write_spef will not crash.
                puts "\[DRT\] Step 5: extract_parasitics (OpenRCX)..."
                define_process_corner -ext_model_index 0 X
                extract_parasitics -ext_model_file {ASAP7_PDK}/rcx_patterns.rules

                # Step 6: Write SPEF + read back. Catch as safety net;
                # after full DRT all iterms are indexed so this rarely fails.
                if {{[catch {{write_spef {spef_cycle_path}}} _ws_err]}} {{
                    puts "\[DRT\] write_spef failed ($_ws_err) — using in-memory RC"
                }} else {{
                    puts "\[DRT\] Step 6: read_spef {spef_cycle_path}"
                    read_spef {spef_cycle_path}
                }}

                # === Post-DRT Timing Report ===
                puts "\[DRT\] Generating post-DRT timing report..."
                {make_report_cmd(RPT_out, NEARBY_PATHS)}

                # === Post-DRT WNS/TNS ===
                set fp_wt [open "{WNS_TNS_FILE}" w]
                close $fp_wt
                tee -file "{WNS_TNS_FILE}" {{ report_wns -digits 3 }}
                tee -file "{WNS_TNS_FILE}" -append {{ report_tns -digits 3 }}

                # === Per-iter QoR dump ===
                dump_iter_qor "{ITER_QOR_DIR}/drt_cycle_{cycle_num}.txt"

                puts {drt_mark}
                """,
            )
            try:
                wait_for_sentinel(proc, drt_mark)
            except OpenROADCrash as _drt_crash:
                print(
                    f"  [DRT-CRASH] OpenROAD died during DRT cycle {cycle_num}: "
                    f"{_drt_crash}",
                    flush=True)
                log_event(f"DRT CRASH cycle={cycle_num} iter={i+1}: {_drt_crash}")
                cycle_num -= 1  # DRT did not complete; roll back cycle counter
                if BEST_ODB.exists():
                    print(
                        f"  [DRT-CRASH] Restarting from best.odb "
                        f"(iter {best_qor.get('iteration')}, "
                        f"WNS={best_qor.get('wns')})",
                        flush=True)
                    shutdown_openroad(proc)
                    proc = restart_openroad_from_best()
                    # Use generate_best_load_tcl() so the restarted process
                    # loads from best.odb, not the hardcoded baseline odb.
                    _drt_load_tcl = generate_best_load_tcl()
                    send(proc, f"source {_drt_load_tcl}\nputs {PARA}\n")
                    wait_for_sentinel(proc)
                    send(proc, NODE_NET_TCL_BLOCK)
                    send(proc, RIP_ALL_WIRES_PROC)
                else:
                    print("  [DRT-CRASH] No best.odb — ending loop.", flush=True)
                    break
                stagnation_count = 0
                prev_qor = {"iteration": i + 1, "wns": best_qor.get("wns"),
                            "tns": best_qor.get("tns")}
                # Skip DRT post-processing; continue to context rebuild.
                # ---- Rebuild context for next iteration ----
                subprocess.run(
                    ["python3", CONTEXT_BUILD,
                     "--iteration", str(i + 1),
                     "--max-paths", "20",
                     "--node-file", NODE_FILE,
                     "--net-file", NET_FILE,
                     "--net-reports-file", NET_REPORTS_FILE,
                     "--displacement-file", DISPLACEMENT_FILE,
                     "--sibling-slacks-file", SIBLING_SLACKS_FILE,
                     "--fanout-rank-file", FANOUT_RANK_FILE,
                     "--out", DYNAMIC_CONTEXT_JSON],
                    check=False)
                continue

            # TCL completed: global_route → estimate_parasitics -global_routing
            # → extract_parasitics (OpenRCX) → write_spef → read_spef
            # → report_checks → report_wns/tns → dump_iter_qor. SPEF loaded.
            print(
                f"  [DRT-CYCLE {cycle_num}] OpenRCX SPEF extracted and loaded: "
                f"{spef_cycle_path}",
                flush=True)

            # Refresh node/net files with post-DRT data
            send(
                proc,
                f'write_node_net_and_reports \
                    "{NODE_FILE}" "{NET_FILE}" "{NET_REPORTS_FILE}"\n'
                f'puts {PARA}\n'
            )
            wait_for_sentinel(proc)

            # Parse post-DRT WNS/TNS.  report_wns/tns ran AFTER read_spef
            # inside the TCL block, so these values are SPEF-extracted.
            drt_wt = parse_wns_tns_file(Path(WNS_TNS_FILE))
            drt_wns = drt_wt.get("wns")
            drt_tns = drt_wt.get("tns")
            print(
                f"  [DRT-CYCLE {cycle_num}] Post-DRT WNS (SPEF-extracted)="
                f"{drt_wns} TNS={drt_tns}",
                flush=True)

            # Save this cycle's routed DRT odb (pairs with spef_cycle_N.spef).
            send(proc, f'write_db "{drt_cycle_odb}"\nputs {PARA}\n')
            wait_for_sentinel(proc)
            print(f"  [DRT-CYCLE {cycle_num}] Saved routed odb -> "
                  f"{drt_cycle_odb}", flush=True)

            # Append DRT cycle metrics row to iter_metrics.csv
            drt_iqd = parse_iter_qor(ITER_QOR_DIR / f"drt_cycle_{cycle_num}.txt")
            drt_row = {
                "iteration": f"drt_{cycle_num}",
                "phase": f"drt_cycle_{cycle_num}",
                "wns_ps": (
                    drt_iqd.get("wns_ps")
                    if drt_iqd.get("wns_ps") is not None else drt_wns
                ),
                "tns_ps": (
                    drt_iqd.get("tns_ps")
                    if drt_iqd.get("tns_ps") is not None else drt_tns
                ),
                "total_pw_w": drt_iqd.get("total_pw_w"),
                "internal_pw_w": drt_iqd.get("internal_pw_w"),
                "switching_pw_w": drt_iqd.get("switching_pw_w"),
                "leakage_pw_w": drt_iqd.get("leakage_pw_w"),
                "design_area_um2": drt_iqd.get("design_area_um2"),
                "util_pct": drt_iqd.get("util_pct"),
                "iter_runtime_s": round(_time.time() - iter_t0, 3),
                "llm_runtime_s": 0.0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "llm_cost_usd": 0.0,
            }
            append_iter_metrics_row(ITER_METRICS_CSV, drt_row)

            # Update prev_qor to DRT-corrected timing so next iteration
            # scores against actual post-route parasitics, not GRT estimates.
            prev_qor = {
                "iteration": f"drt_{cycle_num}",
                "wns": drt_wns,
                "tns": drt_tns,
            }
            prev_path_slacks = parse_report_path_slacks(rpt_path)
            # Also update wns for the end-of-loop timing-met check.
            wns = drt_wns

            # Crown the GLOBAL best-DRT from SPEF-accurate ground truth. This
            # is the ONLY place best_qor is updated. When it improves we write
            # best_drt.odb AND the shipped deliverables in place (no post-loop
            # route): verilog/def/drc here + FINAL_SPEF copied from this
            # cycle's SPEF in the post-loop. Stagnation advances per
            # non-improving DRT cycle.
            if drt_wns is not None and (
                best_qor.get("cycle") is None
                or best_qor.get("wns") is None
                or drt_wns > best_qor["wns"]
            ):
                best_qor = {"cycle": cycle_num,
                            "iteration": cycle_best_grt_iter or (i + 1),
                            "wns": drt_wns, "tns": drt_tns}
                best_drt_spef = spef_cycle_path
                stagnation_count = 0
                try:
                    shutil.copy(RPT_out, FINAL_TIMING_RPT)
                except OSError as _e:
                    print(f"  [DRT-BEST][WARN] copy FINAL_TIMING_RPT: {_e}", flush=True)
                send(
                    proc,
                    f'write_db "{BEST_ODB}"\n'
                    f'write_verilog {FINAL_VERILOG}\n'
                    f'write_def {FINAL_DEF}\n'
                    f'set _drc_fp [open "{FINAL_DRC_RPT}" w]; close $_drc_fp\n'
                    f'tee -file "{FINAL_DRC_RPT}" {{ check_drc }}\n'
                    f'puts {PARA}\n'
                )
                wait_for_sentinel(proc)
                print(
                    f"  [DRT-BEST] best-DRT improved -> WNS={drt_wns} at cycle "
                    f"{cycle_num}. best_drt.odb + finals written.",
                    flush=True)
            else:
                stagnation_count += 1
                print(
                    f"  [DRT-CYCLE {cycle_num}] SPEF WNS={drt_wns} did not "
                    f"improve best-DRT ({best_qor.get('wns')}); "
                    f"stagnation_count={stagnation_count}.",
                    flush=True)

            log_event(
                f"DRT CYCLE {cycle_num} DONE — "
                f"post-DRT WNS={drt_wns} TNS={drt_tns} "
                f"spef={spef_cycle_path}"
            )

            # Update DRT calibration record for LLM prompt.
            # 0.0 = unmeasured (no real report value / stale GRT estimate),
            # NOT a measurement — never feed 0.0 into the grt_error diff or it
            # corrupts every downstream wire-dominated derating.
            grt_err_wns = (
                round(pre_drt_wns - drt_wns, 3)
                if pre_drt_wns not in (None, 0, 0.0) and drt_wns not in (None, 0, 0.0) else None
            )
            grt_err_tns = (
                round(pre_drt_tns - drt_tns, 3)
                if pre_drt_tns not in (None, 0, 0.0) and drt_tns not in (None, 0, 0.0) else None
            )
            last_drt_info = {
                "cycle": cycle_num,
                "at_iter": i + 1,
                "post_drt_wns_ps": drt_wns,
                "post_drt_tns_ps": drt_tns,
                "pre_drt_grt_wns_ps": pre_drt_wns,
                "pre_drt_grt_tns_ps": pre_drt_tns,
                "grt_error_wns_ps": grt_err_wns,
                "grt_error_tns_ps": grt_err_tns,
            }
            if SAVE_GRT_DRT_CORR:
                _dw = (round(drt_wns - pre_drt_wns, 3)
                       if drt_wns is not None and pre_drt_wns is not None else None)
                _dt = (round(drt_tns - pre_drt_tns, 3)
                       if drt_tns is not None and pre_drt_tns is not None else None)
                append_corr_row({
                    "cycle": cycle_num, "at_iter": i + 1, "transition": "GRT->DRT",
                    "wns_from": pre_drt_wns, "wns_to": drt_wns, "dwns_ps": _dw,
                    "tns_from": pre_drt_tns, "tns_to": drt_tns, "dtns_ps": _dt,
                })
                print(f"  [CORR] GRT->DRT cycle {cycle_num}: WNS {pre_drt_wns} -> "
                      f"{drt_wns} (d={_dw})  TNS d={_dt}", flush=True)
            # Re-write state: next cycle starts at iter_in_cycle=0 (SPEF source).
            write_drt_state(
                cycle_num=cycle_num,
                iter_in_cycle=0,
                current_iter=i + 1,
                last_drt=last_drt_info,
            )
            iters_since_drt = 0

        # ---- Rebuild context for next iteration ----
        subprocess.run(
            [
                "python3", CONTEXT_BUILD,
                "--iteration", str(i + 1),
                "--max-paths", "20",
                "--node-file", NODE_FILE,
                "--net-file", NET_FILE,
                "--net-reports-file", NET_REPORTS_FILE,
                "--displacement-file", DISPLACEMENT_FILE,
                "--sibling-slacks-file", SIBLING_SLACKS_FILE,
                "--fanout-rank-file", FANOUT_RANK_FILE,
                "--out", DYNAMIC_CONTEXT_JSON,
            ],
            check=True,
        )

        # Only exit on a DRT-CONFIRMED non-negative WNS. GRT estimates are
        # optimistic, so a positive GRT slack mid-cycle is not trustworthy;
        # wait for the cycle's DRT (wns is set to drt_wns at the boundary).
        if (i + 1) % DRT_CYCLE == 0 and wns is not None and wns >= 0:
            print("Timing met (DRT-confirmed)! Ending loop.")
            break

    # ------------------------------------------------------------------
    # Post-loop finals — NO routing. The loop ends on a DRT boundary
    # (loop_max is a multiple of DRT_CYCLE), and the best-DRT odb plus its
    # deliverables (verilog/def/drc/timing) were already written in place at
    # crown time inside the DRT block. Here we only assemble FINAL_SPEF and
    # the summary from artifacts already on disk. No OpenROAD work.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== POST-LOOP: Final Output (no re-route) ===")
    print("=" * 60)

    out_dir = Path(f"{workdir}/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Close OpenROAD — no further routing/extraction needed.
    shutdown_openroad(proc)

    # FINAL_SPEF = the best-DRT cycle's SPEF (paired with best_drt.odb).
    if best_drt_spef is not None and Path(best_drt_spef).exists():
        try:
            shutil.copy(best_drt_spef, FINAL_SPEF)
            print(f"  [POST] FINAL_SPEF <- {best_drt_spef}")
        except OSError as _e:
            print(f"  [POST][WARN] could not copy FINAL_SPEF: {_e}")

    total_runtime_s = round(_time.time() - run_loop_t0, 3)
    all_new_tok_rows = read_new_token_log_rows(TOKEN_LOG_JSONL, token_log_offset)
    total_tok_agg = aggregate_token_rows(all_new_tok_rows)

    # The 'final' QoR row is the best-DRT cycle's metrics row, already dumped
    # per-cycle (no re-dump). Fall back to best_qor if the CSV row is missing.
    rows = load_iter_metrics_csv(ITER_METRICS_CSV)
    by_iter: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_iter[str(r.get("iteration"))] = r
    best_cycle = best_qor.get("cycle")
    if DRT_EVERY_ITER:
        best_cycle_key = str(best_qor.get("iteration"))
        _best_phase = f"best(iter {best_qor.get('iteration')})"
    else:
        best_cycle_key = f"drt_{best_cycle}" if best_cycle else None
        _best_phase = f"best_drt(cycle {best_cycle})"

    summary_rows: List[Dict[str, Any]] = []
    if "0" in by_iter:
        summary_rows.append({**by_iter["0"], "phase": "baseline"})
    if best_cycle_key and best_cycle_key in by_iter:
        summary_rows.append({**by_iter[best_cycle_key], "phase": _best_phase})
    final_row = summary_rows[-1] if summary_rows else {}
    table = render_summary_table(summary_rows)

    llm_cost = round(total_tok_agg.get("total_cost_usd", 0.0), 6)
    print("\n" + "=" * 60, flush=True)
    print("=== Final QoR Summary (best-DRT, no post-loop route) ===", flush=True)
    print("=" * 60, flush=True)
    print(table, flush=True)
    print(f"Best-DRT cycle {best_cycle} (GRT-best iter {best_qor.get('iteration')}) WNS={best_qor.get('wns')} TNS={best_qor.get('tns')}", flush=True)
    print(f"Total run wall-clock: {total_runtime_s} s", flush=True)
    print(f"Total LLM cost: ${llm_cost:.4f} over {total_tok_agg.get('num_calls', 0)} call(s)", flush=True)

    # Write final summary file.
    summary_path = Path(FINAL_SUMMARY)
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write("=== Final ECO Summary (best-DRT, no post-loop route) ===\n")
        fp.write(f"Best-DRT cycle: {best_cycle} (GRT-best iter {best_qor.get('iteration')})\n")
        fp.write(f"Total iterations: {len(slacks)}\n")
        fp.write(f"Total wall-clock: {total_runtime_s} s\n")
        fp.write(f"Total LLM cost (USD): {llm_cost:.4f}\n")
        fp.write(f"Baseline WNS: {baseline_wns}  TNS: {baseline_tns}\n")
        fp.write(f"Best WNS (SPEF/DRT): {best_qor.get('wns')}  TNS: {best_qor.get('tns')}\n")
        fp.write("\n=== QoR Summary Table ===\n")
        fp.write(table)
        if slacks:
            fp.write("\nWNS per iteration:\n")
            for idx, s in enumerate(slacks, 1):
                fp.write(f"  iter {idx}: {s}\n")

    print(f"\n  [OUTPUT] Final verilog:      {FINAL_VERILOG}")
    print(f"  [OUTPUT] Final DEF:          {FINAL_DEF}")
    print(f"  [OUTPUT] Final SPEF:         {FINAL_SPEF}")
    print(f"  [OUTPUT] Post-route timing:  {FINAL_TIMING_RPT}")
    print(f"  [OUTPUT] DRC report:         {FINAL_DRC_RPT}")
    print(f"  [OUTPUT] Summary:            {FINAL_SUMMARY}")

    save_slack_plot(slacks, slack_history_png)
    print(f"  [OUTPUT] Slack plot:         {slack_history_png}")

    print("\n[DONE] ECO loop complete (best-DRT shipped; no post-loop route).")

if __name__ == "__main__":
    # loop_max MUST be a multiple of DRT_CYCLE so the final iteration is a
    # DRT boundary — finals are shipped from the best-DRT odb, never an
    # un-routed GRT state.
    run_loop(loop_max=48)
