# Timing-ECO-Agent

An LLM-driven **post-place-and-route timing ECO** (Engineering Change Order)
agent. It closes setup-timing violations on a routed design by iterating a
closed loop between [EDA tool](https://github.com/The-OpenROAD-Project/OpenROAD)
and a large language model (Anthropic Claude, via the `claude` CLI):

```
          ┌──────────────────────────────────────────────────────────┐
          │                      per iteration                        │
          │                                                           │
  EDA tool ──► timing report + node/net/parasitic context ──► LLM     │
    ▲                                                          │      │
    │                                                          ▼      │
    └────────── incremental place + route + STA ◄── ECO Tcl (resize / │
               (score QoR, keep best, revert on regress)  buffer /    │
                                                           clone)      │
          └──────────────────────────────────────────────────────────┘
```

Each iteration the orchestrator builds a compact, token-efficient context
(worst timing paths, sibling slacks, fanout ranks, per-stage cell-vs-wire
delay breakdown, and a Liberty cell-delay reference), asks the model for a
small batch of ECO moves, applies them in OpenROAD, re-times, scores the
change, and keeps the best-ever database. It periodically runs full detailed
routing + parasitic extraction (SPEF) so the model's predictions stay
calibrated against real post-route RC.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/main_orch.py` | Main orchestrator — the EDA tool ↔ LLM ECO loop, scoring, revert/backtrace, finals. |
| `src/LLM_call.py` | LLM driver — builds the prompt, calls the `claude` CLI, sanitizes the returned Tcl. |
| `src/context_builder.py` | Emits the per-iteration design state context from timing/node/net files. |
| `src/extract_liberty_tables.py` | Builds the Liberty cell-delay reference (see [Onboarding a new PDK](#onboarding-a-new-pdk)). |
| `src/parsers/` | Timing-report and Liberty parsing helpers. |
| `OpenROAD_utils/OpenROAD_load_design.tcl` | Loads the design checkpoint + PDK into OpenROAD (see [Onboarding a new EDA tool](#onboarding-a-new-eda-tool)). |
| `OpenROAD_utils/eco_procs.tcl` | gate level ECO commands procs (`eco_resize_gate`, `eco_insert_buffer`, `eco_clone_gate`, …). |
| `prompts/AGENTS.md` | Strategy guide (loaded into the system prompt). |
| `prompts/static/cell_delay_reference.toon` | Liberty cell-delay reference for the active PDK. |
| `prompts/static/few_shot/few_shot.toon` | Worked ECO example injected into the system prompt. |
| `asap7/` | Bundled ASAP7 PDK subset (LEF, NLDM Liberty, `setRC.tcl`, RCX rules). |
| `benchmark/` | Detailed routed design checkpoints (`6_final.odb/.sdc/.spef`, …). |

---

## Prerequisites

1. **OpenROAD** — a local build or an
   [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
   install providing the `openroad` binary.
2. **Python ≥ 3.9** with the packages in [`requirements.txt`](requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```
3. **Claude CLI** — the agent shells out to `claude -p`. Install the native
   build:
   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   # add the install dir to your PATH if the installer prompts you to:
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
   claude --version          # verify
   ```
   Authenticate once interactively (`claude`) before running the loop.

---

## Configuration (environment variables)

The code contains **no hardcoded absolute paths** — everything resolves
relative to the repo root, and machine-specific locations come from the
environment:

| Variable | Required | Default | Meaning |
|----------|----------|---------|---------|
| `DESIGN_DIR` | **yes** | — | Design checkpoint directory, **relative to the repo root**. For the bundled design: `benchmark/JPEG_RDF`. |
| `TOP` | no | `6_final` | Checkpoint stem — reads `<DESIGN_DIR>/<TOP>.odb/.sdc/.spef`. |
| `OPENROAD_BIN` | no | `openroad` (on `PATH`) | Absolute path to the OpenROAD binary. |
| `ASAP7_PDK` | no | `<repo>/asap7` | PDK root (holds `lef/`, `lib/NLDM/`, `setRC.tcl`). |
| `DRT_EVERY_ITER` | no | `0` | `1` routes full detailed-route + SPEF every iteration (accurate but slow). |

---

## Running

```bash
export OPENROAD_BIN=/path/to/OpenROAD/build/bin/openroad   # or leave unset to use PATH
export DESIGN_DIR=benchmark/JPEG_RDF                        # relative to repo root

python3 src/main_orch.py
```

The orchestrator fails fast if `DESIGN_DIR`/`TOP` are missing or their `.odb`/
`.sdc` are unreadable. Per-iteration artifacts (prompts, timing reports, token
logs, QoR) are written under `prompts/dynamic/`; finals land in `outputs/`.

---

## Onboarding a new PDK

The bundled Liberty cell-delay reference
(`prompts/static/cell_delay_reference.toon`) is technology-specific — it is the
NLDM lookup the model uses to make first-principles delay predictions. **When
you run against a different PDK you must regenerate it** for that PDK's Liberty
libraries:

```bash
# Point ASAP7_PDK (or edit the variable) at the new PDK's NLDM lib dir, then:
ASAP7_PDK=/path/to/new_pdk python3 src/extract_liberty_tables.py
```

This rewrites `prompts/static/cell_delay_reference.toon` from the Liberty files
under `<ASAP7_PDK>/lib/NLDM/`. Re-run it **once per technology** (or whenever the
Liberty files change) — the output is treated as data, not a build artifact.
You will also want to review the cell families / drive-strength ladders in
`src/extract_liberty_tables.py` (`FAMILIES`, `FAMILY_GROUP`, `VTS`) if the new
PDK uses different naming conventions.

---

## Onboarding a new EDA tool

The loop currently drives OpenROAD. To target a different place-and-route /
STA engine, provide an equivalent **load-design script** modeled on
[`OpenROAD_utils/OpenROAD_load_design.tcl`](OpenROAD_utils/OpenROAD_load_design.tcl).
That script (adapted from the OpenROAD project's design-load flow) is the single
integration point: it reads the design checkpoint plus the PDK LEF/Liberty,
restores parasitics (SPEF) and constraints (SDC), and sources the ECO helper
procs. A new tool's equivalent must:

1. Resolve all paths from the environment (`ASAP7_PDK`, `DESIGN_DIR`, `TOP`,
   and `BACKTRACE_ODB` for snapshot recovery) — **no hardcoded absolute paths**.
2. Load the design database, tech + standard-cell libraries, timing
   constraints, and parasitics.
3. Expose the same ECO primitive command surface the model emits
   (resize / insert-buffer / remove-buffer / clone — see
   `OpenROAD_utils/eco_procs.tcl` and `prompts/AGENTS.md`).
4. Support reloading from a saved snapshot when `BACKTRACE_ODB` is set, so the
   orchestrator's crash / stagnation recovery works.

---

## How it works (loop internals)

- **Context** — each iteration dumps node positions, net topology, and a
  timing report; `context_builder.py` distills them into a compact TOON block
  (top worst paths with per-stage cell/wire delay, sibling slacks, fanout ranks).
- **Prompt** — `LLM_call.py` assembles a cached system prompt (`prompts/AGENTS.md`
  strategy + `cell_delay_reference.toon` + few-shot) plus the per-iteration
  context, calls `claude`, and extracts only the fenced ```tcl``` block.
- **Apply + re-time** — the orchestrator sources the ECO Tcl in OpenROAD, runs
  incremental placement + global route + parasitic estimation, and re-reports.
- **Score + keep best** — a QoR score (WNS/TNS/neighbor/new-violation weighted)
  decides whether to keep the move; the best-ever `.odb` is snapshotted and used
  for revert / backtrace on sustained regression.
- **DRT calibration** — every `DRT_CYCLE` iterations (or every iteration under
  `DRT_EVERY_ITER=1`) it runs full detailed routing + OpenRCX SPEF so the model
  sees actual post-route RC and can derate its global-route estimates.

---

## Bundled PDK attribution

The `asap7/` directory contains a subset of the **ASAP7** predictive process
design kit, © 2020 Lawrence T. Clark, Vinay Vashishtha, and Arizona State
University, distributed under the **BSD 3-Clause License** (see the header of
each file under `asap7/`). Those files retain their original license and
copyright.
