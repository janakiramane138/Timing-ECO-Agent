########################################################################
# OpenROAD load-design script  (Timing-ECO-Agent)
#
# Adapted from the OpenROAD project's design-load flow
# (https://github.com/The-OpenROAD-Project/OpenROAD). It reads a
# post-place-and-route checkpoint (.odb) together with the ASAP7 PDK
# LEF/Liberty, restores parasitics (SPEF) and timing constraints (SDC),
# and sources the ECO helper procs so the orchestrator can drive
# incremental timing repair.
#
# ----------------------------------------------------------------------
# Path resolution — NO hardcoded absolute paths.
#
# All paths resolve relative to this script's location so the repo runs
# wherever it is cloned. The orchestrator (src/main_orch.py) also pushes
# the following environment variables into the OpenROAD subprocess; when
# set, they override the in-repo defaults:
#
#   ASAP7_PDK     PDK root that holds lef/, lib/NLDM/, setRC.tcl
#                 (default: <repo>/asap7)
#   DESIGN_DIR    directory holding <TOP>.odb / <TOP>.sdc / <TOP>.spef
#                 (default: <repo>/benchmark/JPEG_RDF)
#   TOP           checkpoint stem, e.g. 6_final  (default: 6_final)
#   BACKTRACE_ODB when set, read THIS .odb instead of <DESIGN_DIR>/<TOP>.odb
#                 (used by crash / stagnation recovery to restart from the
#                  best-ever snapshot)
########################################################################

# <repo>/OpenROAD_utils/OpenROAD_load_design.tcl -> repo root is two levels up
set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file dirname $script_dir]

# ---- Resolve PDK / design roots (env override -> in-repo default) ----
if {[info exists ::env(ASAP7_PDK)] && $::env(ASAP7_PDK) ne ""} {
  set pdk_root $::env(ASAP7_PDK)
} else {
  set pdk_root [file join $repo_root asap7]
}

if {[info exists ::env(DESIGN_DIR)] && $::env(DESIGN_DIR) ne ""} {
  set design_dir $::env(DESIGN_DIR)
} else {
  set design_dir [file join $repo_root benchmark JPEG_RDF]
}

if {[info exists ::env(TOP)] && $::env(TOP) ne ""} {
  set top $::env(TOP)
} else {
  set top "6_final"
}

puts "\[LOAD] repo_root  = $repo_root"
puts "\[LOAD] pdk_root   = $pdk_root"
puts "\[LOAD] design_dir = $design_dir"
puts "\[LOAD] top        = $top"

# ---- LEF: technology + standard-cell macros ----
set tech_lef [list \
  [file join $pdk_root lef asap7_tech_1x_201209.lef]]

set std_lef [list \
  [file join $pdk_root lef asap7sc7p5t_28_R_1x_220121a.lef] \
  [file join $pdk_root lef asap7sc7p5t_28_L_1x_220121a.lef] \
  [file join $pdk_root lef asap7sc7p5t_28_SL_1x_220121a.lef] \
  [file join $pdk_root lef asap7sc7p5t_28_SRAM_1x_220121a.lef]]

# ---- Liberty: NLDM, FF corner (matches the checkpoint's STA corner) ----
set lib_files [list \
  [file join $pdk_root lib NLDM asap7sc7p5t_AO_LVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_AO_SLVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_INVBUF_LVT_FF_nldm_220122.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_INVBUF_SLVT_FF_nldm_220122.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_OA_LVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_OA_SLVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_SEQ_LVT_FF_nldm_220123.lib] \
  [file join $pdk_root lib NLDM asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib] \
  [file join $pdk_root lib NLDM asap7sc7p5t_SEQ_SLVT_FF_nldm_220123.lib] \
  [file join $pdk_root lib NLDM asap7sc7p5t_SIMPLE_LVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz] \
  [file join $pdk_root lib NLDM asap7sc7p5t_SIMPLE_SLVT_FF_nldm_211120.lib.gz]]

# ---- Read the design checkpoint ----
# BACKTRACE_ODB (set by crash/stagnation recovery) wins over the default
# <design_dir>/<top>.odb so a restart resumes from the best-ever snapshot.
if {[info exists ::env(BACKTRACE_ODB)] && $::env(BACKTRACE_ODB) ne ""} {
  set odb_file $::env(BACKTRACE_ODB)
  puts "\[LOAD] BACKTRACE_ODB set -> reading snapshot: $odb_file"
} else {
  set odb_file [file join $design_dir ${top}.odb]
}
read_db $odb_file

# ---- Load tech + standard-cell LEFs ----
read_lef $tech_lef

foreach lef $std_lef {
  puts "READ_LEF: $lef"
  read_lef $lef
}

# ---- Load Liberty libraries ----
foreach lib $lib_files {
  puts "READ_LIBERTY: $lib"
  read_liberty $lib
}

# ---- Parasitic RC model + timing setup ----
source [file join $pdk_root setRC.tcl]

define_process_corner -ext_model_index 0 X

read_sdc  [file join $design_dir ${top}.sdc]
read_spef [file join $design_dir ${top}.spef]

# ---- ECO helper procs (mechanical mutators used by the orchestrator) ----
source [file join $script_dir eco_procs.tcl]
