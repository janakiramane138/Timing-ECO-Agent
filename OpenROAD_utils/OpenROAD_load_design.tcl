########################################################################
# OpenROAD load design script
########################################################################
set design "JPEG_RDF_70_400"
set design_dir "/data/jethiraj/OpenROAD-flow-scripts/flow/results/asap7/${design}/base"
set top "6_final"



set tech_lef [list \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lef/asap7_tech_1x_201209.lef]

set std_lef [list \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lef/asap7sc7p5t_28_L_1x_220121a.lef \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lef/asap7sc7p5t_28_SL_1x_220121a.lef \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lef/asap7sc7p5t_28_SRAM_1x_220121a.lef]

set lib_files [list \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_AO_LVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_AO_RVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_AO_SLVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_INVBUF_LVT_FF_nldm_220122.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_INVBUF_RVT_FF_nldm_220122.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_INVBUF_SLVT_FF_nldm_220122.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_OA_LVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_OA_RVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_OA_SLVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SEQ_LVT_FF_nldm_220123.lib \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SEQ_RVT_FF_nldm_220123.lib \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SEQ_SLVT_FF_nldm_220123.lib \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SIMPLE_LVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SIMPLE_RVT_FF_nldm_211120.lib.gz \
  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SIMPLE_SLVT_FF_nldm_211120.lib.gz]


#read_db ${design_dir}/${top}.odb
read_db /data/jethiraj/OpenROAD-flow-scripts/flow/rdf_tuned/output/results/asap7/jpeg/DESIGN_jpeg__CLK_400__UTIL_70__AR_1__TECH_ASAP7__LB_ADDON_0.2__TIMING_EFFORT_71__POWER_EFFORT_0__HIER_SYNTH_0__GP_PAD_0__DP_PAD_0__RD_1__TD_1__DPO_1__CTS_CSIZE_20__CTS_CDIA_111__PIN_ADJ_0.281__UP_ADJ_0.195/6_final.odb
puts "read_db ${design_dir}/${top}.odb"


#Load tech + stdcell LEFs
read_lef $tech_lef

foreach lef $std_lef {
  puts "READ_LEF: $lef"
  read_lef $lef
  }

# Read all libs
foreach lib $lib_files {
  puts "READ_LIBERTY: $lib"
  read_liberty $lib
  }

#read_verilog ${design_dir}/${top}.v
#read_def ${design_dir}/${top}.def

#source /data/jethiraj/agentic_AI/asap7_27/setRC.tcl
source /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/setRC.tcl

define_process_corner -ext_model_index 0 X
#extract_parasitics -ext_model_file  /data/jethiraj/agentic_AI/asap7_27/rcx_patterns.rules
#extract_parasitics -ext_model_file  /data/jethiraj/OpenROAD-flow-scripts/flow/platforms/asap7/rcx_patterns.rules
#extract_parasitics -lef_rc

#read_sdc ${design_dir}/${top}.sdc
#read_sdc /data/jethiraj/OpenROAD-flow-scripts/flow/results/asap7/${design}/base/6_1_fill.sdc
read_sdc /data/jethiraj/OpenROAD-flow-scripts/flow/rdf_tuned/output/results/asap7/jpeg/DESIGN_jpeg__CLK_400__UTIL_70__AR_1__TECH_ASAP7__LB_ADDON_0.2__TIMING_EFFORT_71__POWER_EFFORT_0__HIER_SYNTH_0__GP_PAD_0__DP_PAD_0__RD_1__TD_1__DPO_1__CTS_CSIZE_20__CTS_CDIA_111__PIN_ADJ_0.281__UP_ADJ_0.195/6_1_fill.sdc
#read_spef ${design_dir}/${top}.spef
read_spef /data/jethiraj/OpenROAD-flow-scripts/flow/rdf_tuned/output/results/asap7/jpeg/DESIGN_jpeg__CLK_400__UTIL_70__AR_1__TECH_ASAP7__LB_ADDON_0.2__TIMING_EFFORT_71__POWER_EFFORT_0__HIER_SYNTH_0__GP_PAD_0__DP_PAD_0__RD_1__TD_1__DPO_1__CTS_CSIZE_20__CTS_CDIA_111__PIN_ADJ_0.281__UP_ADJ_0.195/6_final.spef
#estimate_parasitics -global_routing


source /data/jethiraj/jpeg_exp/repo/OpenROAD_utils/eco_procs.tcl