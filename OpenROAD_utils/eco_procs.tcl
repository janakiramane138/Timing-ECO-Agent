########################################################################
# ECO Procs for LLM-driven Timing Repair  (v7 — simplified)
#
# Source once into OpenROAD after the design is loaded.
#
# Four primitives that mirror the repair_timing move set:
#   eco_insert_buffer   <net> <driver_pin> <buf_cell> <buf_name> ?-sinks list? ?-at loc?
#   eco_remove_buffer   <buf_inst>
#   eco_resize_gate     <inst> <new_cell>
#   eco_clone_gate      <orig_inst> <clone_name> <sink_pins>
#
# Tcl quoting: names with $, [, ] MUST be braced at the call site, e.g.
#   eco_insert_buffer _00623_ {sa12_sr[0]$_DFF_P_/QN} BUFx4_ASAP7_75t_L b1
#
# OpenROAD object model used here (mini-glossary):
#   dbInst     — a placed gate (has master, location in DBU, ITerms)
#   dbMaster   — a library cell (e.g., BUFx4_ASAP7_75t_SL)
#   dbMTerm    — a pin DEFINITION on a master (name + IoType)
#   dbITerm    — an Instance Terminal: pin on a specific dbInst
#   dbBTerm    — a Block Terminal: a top-level design port
#   dbNet      — a net; reachable ITerms via getITerms, ports via getBTerms
#
# The STA pin graph (get_pins) can go stale after an ECO insert/delete.
# ODB is always the source of truth — these procs walk ODB for sink
# enumeration and bridge via sta::sta_to_db_inst / sta::sta_to_db_net.
#
# Design principle: these procs are *mechanical mutators*. All timing
# judgement (which gate to resize, which sinks to clone off, whether a
# buffer is safe to remove) is the caller's responsibility — the LLM
# does that from the timing report. Matches how repair_timing separates
# its doMove() routines from the repair loop that decides when to call.
########################################################################


# ----------------------------------------------------------------------
# Helpers — tiny, no wire-geometry parsing (legalizer handles precision)
# ----------------------------------------------------------------------

# STA pin → ODB dbInst. Errors if the STA→ODB bridge returns NULL.
proc _eco_pin_to_dbinst { sta_pin } {
    set sta_inst [[lindex $sta_pin 0] instance]
    set dbinst [sta::sta_to_db_inst $sta_inst]
    if {$dbinst eq "NULL" || $dbinst eq ""} {
        error "_eco_pin_to_dbinst: sta_to_db_inst returned NULL"
    }
    return $dbinst
}

# STA pin → instance location in microns (for placement).
proc _eco_pin_to_location_um { sta_pin } {
    set dbinst [_eco_pin_to_dbinst $sta_pin]
    set loc [$dbinst getLocation]
    set x_um [ord::dbu_to_microns [lindex $loc 0]]
    set y_um [ord::dbu_to_microns [lindex $loc 1]]
    return [list $x_um $y_um]
}

# Split "inst/pin" string without relying on get_pins (safe post-ECO).
proc _eco_split_sink_key { sp } {
    set t [string trim $sp "{} "]
    set slash [string last "/" $t]
    if {$slash < 0} { return {} }
    set iname [string range $t 0 [expr {$slash - 1}]]
    set pname [string range $t [expr {$slash + 1}] end]
    return [list $iname $pname]
}


# ======================================================================
# eco_insert_buffer
# ----------------------------------------------------------------------
# Insert a buffer on <net>. Moves either ALL sinks or a specified subset
# behind the new buffer. Placement location is a caller-driven choice:
#
#   -at driver           → 0.2 µm right of driver (default; shields fanout)
#   -at centroid         → geometric mean of moved sink (x,y) — mirrors
#                          repair_timing's clone/split-load centroid logic
#   -at {x_um y_um}      → explicit coordinates
#
# Args:
#   net_name      the net the driver currently drives
#   driver_pin    "inst/pin" of the driver
#   buf_cell      library cell (e.g., BUFx4_ASAP7_75t_SL)
#   buf_name      desired name of the new buffer instance
#   args          optional: -sinks <list>  -at <driver|centroid|{x y}>
#
# Returns the new buffer instance name.
# ======================================================================
proc eco_insert_buffer { net_name driver_pin buf_cell buf_name args } {

    # ---- parse optional flags -----------------------------------------
    set sink_pins "all"
    set at_mode   "driver"
    set at_xy     ""
    for {set i 0} {$i < [llength $args]} {incr i} {
        set k [lindex $args $i]
        set v [lindex $args [expr {$i + 1}]]
        switch -- $k {
            "-sinks" { set sink_pins $v; incr i }
            "-at"    {
                if {$v eq "driver" || $v eq "centroid"} {
                    set at_mode $v
                } else {
                    set at_mode "explicit"
                    set at_xy $v
                }
                incr i
            }
            default { error "eco_insert_buffer: unknown option '$k'" }
        }
    }

    set block [ord::get_db_block]

    # ---- resolve driver ----------------------------------------------
    set drv_pin [get_pins $driver_pin]
    if {$drv_pin eq "" || $drv_pin eq "NULL"} {
        error "eco_insert_buffer: driver pin '$driver_pin' not found"
    }
    set drv_dbinst [_eco_pin_to_dbinst $drv_pin]
    set drv_odb_name [$drv_dbinst getName]
    lassign [_eco_pin_to_location_um $drv_pin] drv_x drv_y

    set drv_net_sta [[lindex $drv_pin 0] net]
    set db_net [sta::sta_to_db_net $drv_net_sta]
    if {$db_net eq "NULL" || $db_net eq ""} {
        error "eco_insert_buffer: cannot get ODB net from driver"
    }

    # ---- decide which sinks to move ----------------------------------
    # ODB walk is ground truth. For a sink subset, we parse "inst/pin"
    # and match against ITerms without going through get_pins.
    set move_all [expr {$sink_pins eq "all"}]
    set requested {}
    if {!$move_all} {
        foreach sp $sink_pins {
            set kp [_eco_split_sink_key $sp]
            if {$kp eq ""} {
                puts "# eco_insert_buffer: WARNING bad sink '$sp', skipping"
                continue
            }
            dict set requested "[lindex $kp 0]/[lindex $kp 1]" 1
        }
    }

    set sink_iterms {}
    foreach iterm [$db_net getITerms] {
        set inst_name [[$iterm getInst] getName]
        set pin_name  [[$iterm getMTerm] getName]
        set io        [[$iterm getMTerm] getIoType]
        # skip the driver's own output terminal
        if {$inst_name eq $drv_odb_name && ($io eq "OUTPUT" || $io eq "INOUT")} { continue }
        if {$move_all} {
            lappend sink_iterms $iterm
        } elseif {[dict exists $requested "${inst_name}/${pin_name}"]} {
            lappend sink_iterms $iterm
        }
    }

    set sink_bterms {}
    if {$move_all} {
        foreach bt [$db_net getBTerms] { lappend sink_bterms $bt }
    }

    set total [expr {[llength $sink_iterms] + [llength $sink_bterms]}]
    if {$total == 0} {
        error "eco_insert_buffer: no sinks to move on '$net_name'"
    }

    # ---- compute placement location BEFORE rewire --------------------
    set loc_x $drv_x
    set loc_y $drv_y
    switch -- $at_mode {
        "driver"   { set loc_x [expr {$drv_x + 0.2}] }
        "explicit" { lassign $at_xy loc_x loc_y }
        "centroid" {
            set sx 0.0; set sy 0.0; set n 0
            foreach it $sink_iterms {
                set L [[$it getInst] getLocation]
                set sx [expr {$sx + [ord::dbu_to_microns [lindex $L 0]]}]
                set sy [expr {$sy + [ord::dbu_to_microns [lindex $L 1]]}]
                incr n
            }
            if {$n > 0} { set loc_x [expr {$sx/$n}]; set loc_y [expr {$sy/$n}] }
        }
    }

    # ---- create buffer + new net, rewire -----------------------------
    set new_net_name "${buf_name}_net"
    set new_net [make_net $new_net_name]
    make_instance $buf_name $buf_cell

    set old_net [get_nets [get_name $drv_net_sta]]
    connect_pin $old_net [get_pins "${buf_name}/A"]
    connect_pin $new_net [get_pins "${buf_name}/Y"]

    set moved 0
    foreach it $sink_iterms {
        set full "[[$it getInst] getName]/[[$it getMTerm] getName]"
        if {[catch {
            set p [get_pins $full]
            disconnect_pin $old_net $p
            connect_pin    $new_net $p
            incr moved
        } err]} { puts "# eco_insert_buffer: WARN rewire $full: $err" }
    }
    foreach bt $sink_bterms {
        set name [$bt getName]
        if {[catch {
            set p [get_ports $name]
            disconnect_pin $old_net $p
            connect_pin    $new_net $p
            incr moved
        } err]} { puts "# eco_insert_buffer: WARN rewire port $name: $err" }
    }

    place_inst -name $buf_name -cell $buf_cell \
        -location [list $loc_x $loc_y] -status PLACED

    puts "# eco_insert_buffer: $buf_name ($buf_cell) at ($loc_x,$loc_y) moved $moved/$total sinks (-at $at_mode)"
    return $buf_name
}


# ======================================================================
# eco_remove_buffer
# ----------------------------------------------------------------------
# Safely delete a buffer and reconnect all its output-net loads back to
# the input net. Caller must have checked that this is safe (fanout/cap
# on the upstream driver, no slack regression) — repair_timing's gates
# are its job, not ours.
#
# Returns the surviving (input) net name.
# ======================================================================
proc eco_remove_buffer { buf_inst_name } {

    set buf_a [get_pins "${buf_inst_name}/A"]
    set buf_y [get_pins "${buf_inst_name}/Y"]
    if {$buf_a eq "" || $buf_y eq ""} {
        error "eco_remove_buffer: pins for '$buf_inst_name' not found"
    }

    set in_net_sta  [[lindex $buf_a 0] net]
    set out_net_sta [[lindex $buf_y 0] net]
    set in_name  [get_name $in_net_sta]
    set out_name [get_name $out_net_sta]

    set buf_odb_name [[_eco_pin_to_dbinst $buf_a] getName]
    set out_db_net [sta::sta_to_db_net $out_net_sta]

    # collect all sinks on the output net EXCEPT the buffer's own Y
    set iterms {}
    foreach it [$out_db_net getITerms] {
        if {[[$it getInst] getName] eq $buf_odb_name} { continue }
        lappend iterms $it
    }
    set bterms [$out_db_net getBTerms]

    set in_obj  [get_nets $in_name]
    set out_obj [get_nets $out_name]

    disconnect_pin $in_obj  $buf_a
    disconnect_pin $out_obj $buf_y

    set moved 0
    foreach it $iterms {
        set full "[[$it getInst] getName]/[[$it getMTerm] getName]"
        if {[catch {
            set p [get_pins $full]
            disconnect_pin $out_obj $p
            connect_pin    $in_obj  $p
            incr moved
        } err]} { puts "# eco_remove_buffer: WARN $full: $err" }
    }
    foreach bt $bterms {
        set nm [$bt getName]
        if {[catch {
            set p [get_ports $nm]
            disconnect_pin $out_obj $p
            connect_pin    $in_obj  $p
            incr moved
        } err]} { puts "# eco_remove_buffer: WARN port $nm: $err" }
    }

    delete_instance $buf_odb_name
    delete_net $out_name

    puts "# eco_remove_buffer: removed $buf_inst_name, reconnected $moved sink(s) to $in_name"
    return $in_name
}


# ======================================================================
# eco_resize_gate
# ----------------------------------------------------------------------
# Swap a gate's master cell. Use for size-up (INVx4 → INVx8), size-down,
# or VT swap (e.g., *_SL → *_R). The new master must have pin-compatible
# names, which is the case for ASAP7 drive-strength variants.
#
# No rewiring needed — replace_cell preserves all net connections.
# ======================================================================
proc eco_resize_gate { inst_name new_cell } {
    set inst [get_cells $inst_name]
    if {$inst eq "" || $inst eq "NULL"} {
        error "eco_resize_gate: instance '$inst_name' not found"
    }
    replace_cell $inst $new_cell
    puts "# eco_resize_gate: $inst_name -> $new_cell"
    return $inst_name
}


# ======================================================================
# eco_clone_gate
# ----------------------------------------------------------------------
# Duplicate a gate and move a specified sink subset to the clone's
# output. Mirrors repair_timing's CloneMove: original keeps the critical
# (slack-poor) loads; caller passes the non-critical subset as <sink_pins>.
#
# Steps:
#   1. Create a new instance <clone_name> with the same master as <orig>.
#   2. Connect every INPUT pin of the clone to the same nets as <orig>.
#   3. Create a new output net; connect clone's OUTPUT to it.
#   4. Move each pin in <sink_pins> from orig's output net to the new net.
#   5. Place clone at the centroid of its new loads (matches repair_timing).
#
# Args:
#   orig_inst_name   name of the gate to clone
#   clone_name       desired name for the new clone instance
#   sink_pins        list of "inst/pin" strings to move to the clone
#
# Returns the clone instance name.
# ======================================================================
proc eco_clone_gate { orig_inst_name clone_name sink_pins } {

    if {[llength $sink_pins] == 0} {
        error "eco_clone_gate: sink_pins must be non-empty"
    }

    set block [ord::get_db_block]
    set orig [$block findInst $orig_inst_name]
    if {$orig eq "NULL" || $orig eq ""} {
        error "eco_clone_gate: instance '$orig_inst_name' not found"
    }
    set master_name [[$orig getMaster] getName]

    # -- identify the original's OUTPUT pin and its net ----------------
    # NOTE: some masters expose more than one OUTPUT/INOUT mterm and only
    # the connected one drives a real net. We must `continue` past empty
    # ones rather than break on the first OUTPUT-direction iterm we see.
    set out_pin_name ""
    set orig_out_net_name ""
    set onet ""
    set _dbg_iterms {}
    foreach it [$orig getITerms] {
        set mt [$it getMTerm]
        set sig [$mt getSigType]
        set io [$mt getIoType]
        set pname [$mt getName]
        set n [$it getNet]
        set nname ""
        if {$n ne "NULL" && $n ne ""} { catch {set nname [$n getName]} }
        lappend _dbg_iterms "pin=$pname io=$io sig=$sig net=$nname"
        # Skip power/ground pins — they are often INOUT-typed and would
        # otherwise hijack the "output" slot and hand us the VDD/VSS net.
        if {$sig eq "POWER" || $sig eq "GROUND"} { continue }
        if {$io ne "OUTPUT" && $io ne "INOUT"} { continue }
        if {$nname eq ""} { continue }
        set out_pin_name $pname
        set onet $n
        set orig_out_net_name $nname
        break
    }
    if {$out_pin_name eq "" || $orig_out_net_name eq ""} {
        puts "# eco_clone_gate: iterm dump for '$orig_inst_name':"
        foreach r $_dbg_iterms { puts "#    $r" }
        error "eco_clone_gate: cannot locate connected output pin/net on '$orig_inst_name'"
    }

    # -- create the clone ---------------------------------------------
    make_instance $clone_name $master_name

    # -- wire every INPUT of the clone to the same net as orig --------
    foreach it [$orig getITerms] {
        set mt [$it getMTerm]
        set sig [$mt getSigType]
        if {$sig eq "POWER" || $sig eq "GROUND"} { continue }
        if {[$mt getIoType] ne "INPUT"} { continue }
        set inet [$it getNet]
        if {$inet eq "NULL" || $inet eq ""} { continue }
        set pname [$mt getName]
        connect_pin [get_nets [$inet getName]] [get_pins "${clone_name}/${pname}"]
    }

    # -- create clone's output net and connect its output pin ---------
    set clone_net_name "${clone_name}_net"
    set clone_net [make_net $clone_net_name]
    connect_pin $clone_net [get_pins "${clone_name}/${out_pin_name}"]

    # -- move requested sinks from orig's output net to clone's net ---
    set orig_out_obj [get_nets $orig_out_net_name]
    set moved 0
    set sum_x 0.0; set sum_y 0.0; set n 0
    foreach sp $sink_pins {
        set kp [_eco_split_sink_key $sp]
        if {$kp eq ""} {
            puts "# eco_clone_gate: WARN bad sink '$sp', skipping"
            continue
        }
        set iname [lindex $kp 0]
        set pname [lindex $kp 1]
        set full "${iname}/${pname}"

        # centroid accumulation (uses ODB directly, not STA)
        set sinst [$block findInst $iname]
        if {$sinst ne "NULL" && $sinst ne ""} {
            set L [$sinst getLocation]
            set sum_x [expr {$sum_x + [ord::dbu_to_microns [lindex $L 0]]}]
            set sum_y [expr {$sum_y + [ord::dbu_to_microns [lindex $L 1]]}]
            incr n
        }

        if {[catch {
            set p [get_pins $full]
            disconnect_pin $orig_out_obj $p
            connect_pin    $clone_net   $p
            incr moved
        } err]} { puts "# eco_clone_gate: WARN move $full: $err" }
    }

    if {$moved == 0} {
        error "eco_clone_gate: no sinks were actually moved to clone"
    }

    # -- place clone at centroid of its new loads ---------------------
    # Fall back to the original's location if we couldn't read any sink coords.
    if {$n > 0} {
        set cx [expr {$sum_x / $n}]
        set cy [expr {$sum_y / $n}]
    } else {
        set L [$orig getLocation]
        set cx [ord::dbu_to_microns [lindex $L 0]]
        set cy [ord::dbu_to_microns [lindex $L 1]]
    }
    place_inst -name $clone_name -cell $master_name \
        -location [list $cx $cy] -status PLACED

    puts "# eco_clone_gate: $clone_name ($master_name) clone of $orig_inst_name, moved $moved sinks, at ($cx,$cy)"
    return $clone_name
}

# ======================================================================
# eco_top_paths_through
# ----------------------------------------------------------------------
# Top-N worst-slack timing paths through an instance, pin, or net.
# Use before an ECO move to check sibling slacks so the LLM can size
# its delay delta against the 2nd-worst path sharing the node.
#
# Wraps `report_checks -through ... -format end`, which prints a table:
#     sa13_sr[1]/D (DFFHQNx1_...)  483.83  609.60  -125.77 (VIOLATED)
# Endpoint labels can contain $, [, ], /, escaped chars, and the cell
# suffix may be absent (e.g., BTerm endpoints). We anchor on the fixed
# trailing shape: <num> <num> <num> (STATUS).
#
# Returns: list of {endpoint slack_ns} sorted worst-first.
# ======================================================================
proc eco_top_paths_through { obj_type obj_name {n 5} {mode "max"} } {

    switch -- $obj_type {
        "inst" { set thru [get_cells $obj_name] }
        "pin"  { set thru [get_pins  $obj_name] }
        "net"  { set thru [get_nets  $obj_name] }
        default { error "eco_top_paths_through: obj_type must be inst|pin|net" }
    }
    if {$thru eq "" || $thru eq "NULL"} {
        error "eco_top_paths_through: '$obj_name' not found"
    }

    set tmp "/tmp/eco_paths_[pid]_[clock microseconds].rpt"

    if {[catch {
        report_checks -through $thru \
                      -group_path_count $n \
                      -path_delay $mode \
                      -format end > $tmp
    } err]} {
        catch {file delete $tmp}
        error "eco_top_paths_through: report_checks failed: $err"
    }
    if {![file exists $tmp]} {
        error "eco_top_paths_through: no output file produced ($tmp)"
    }

    set fh [open $tmp r]
    set raw [read $fh]
    close $fh
    file delete $tmp

    # Anchor on trailing "<num> <num> <num> (STATUS)". Everything before
    # the three numbers is the endpoint label (may include a "(cell)"
    # suffix, which we strip after the match).
    set num {[-+]?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?}
    set pat "^(.+?)\\s+${num}\\s+${num}\\s+(${num})\\s*\\(\[^)\]+\\)\\s*\$"

    set rows {}
    foreach line [split $raw "\n"] {
        if {[regexp $pat $line -> ep_full slk]} {
            # Strip a trailing "(CELL_NAME)" from the endpoint if present.
            regsub {\s*\([^)]+\)\s*$} $ep_full "" ep
            lappend rows [list [string trim $ep] $slk]
        }
    }
    return $rows
}


# ======================================================================
# eco_dump_path_siblings
# ----------------------------------------------------------------------
# Dump per-instance top-N sibling slacks for every non-register instance
# on the worst violating setup path whose output-net fanout is > 1.
#
# Writes a plain-text file consumed by context_builder_v5.py's
# sibling_slacks loader. Format (one block per inst):
#
#     === <inst_name> ===
#     <endpoint>   <slack_ns>
#     <endpoint>   <slack_ns>
#     ...
#
# Register (DFF/LATCH/ICG/SDFF) instances are skipped — they're
# don't-touch per system prompt §6, so sibling data doesn't influence
# any legal move. Fanout=1 instances are skipped because insertion
# and clone moves (the only decisions that consume sibling slacks)
# don't apply to single-sink drivers.
#
# Args:
#   out_file   absolute path to write
#   n          top-N slacks to emit per inst (default 5)
# ======================================================================

proc eco_dump_path_siblings { out_file {n 5} } {

    # Get the worst setup path's instance list via report_checks.
    set tmp "/tmp/eco_wpath_[pid]_[clock microseconds].rpt"
    if {[catch {
        report_checks -path_delay max -group_path_count 1 \
                      -fields {net cap slew input fanout} \
                      -format full > $tmp
    } err]} {
        catch {file delete $tmp}
        error "eco_dump_path_siblings: report_checks failed: $err"
    }

    set fh [open $tmp r]; set raw [read $fh]; close $fh; file delete $tmp

    # Walk lines; match output-pin rows matching  "<num> ... ^|v <inst>/<pin> (<cell>)".
    # Register cells are skipped (DFF*/SDFF*/LATCH*/ICG*).
    set seen [dict create]
    set ordered {}
    foreach line [split $raw "\n"] {
        # Only consider transition rows (have ^ or v then "inst/pin (cell)")
        if {![regexp {[\^v]\s+(\S+)/(\w+)\s+\((\S+)\)} $line -> inst pin cell]} {
            continue
        }
        # Skip register cells.
        if {[regexp -nocase {^(DFF|SDFF|LATCH|ICG|SNL|SRA)} $cell]} { continue }
        # Only output pins (the driver rows with fanout/cap/slew columns).
        if {![regexp {^(Y|Q|QN|S|SO|CO|SN|Z|ZN|CON|COUT|S0|S1|SUM)$} $pin]} { continue }
        if {[dict exists $seen $inst]} { continue }
        dict set seen $inst 1
        lappend ordered $inst
    }

    # Dump per-inst top-N (only for fanout > 1).
    set of [open $out_file w]
    puts $of "# Generated by eco_dump_path_siblings — [clock format [clock seconds]]"

    set block [ord::get_db_block]
    foreach inst $ordered {
        # Resolve inst → output-net fanout via ODB.
        set dbi [$block findInst $inst]
        if {$dbi eq "NULL" || $dbi eq ""} { continue }
        set fo 0
        foreach it [$dbi getITerms] {
            set mt [$it getMTerm]
            set sig [$mt getSigType]
            if {$sig eq "POWER" || $sig eq "GROUND"} { continue }
            set io [$mt getIoType]
            if {$io ne "OUTPUT" && $io ne "INOUT"} { continue }
            set net [$it getNet]
            if {$net eq "NULL" || $net eq ""} { continue }
            # Fanout = number of sink ITerms (all non-this-inst connections).
            set fo [llength [$net getITerms]]
            incr fo -1   ;# subtract the driver itself
            break
        }
        if {$fo <= 1} { continue }

        # Get top-N siblings through this inst.
        if {[catch {set rows [eco_top_paths_through inst $inst $n max]} err]} {
            puts "# eco_dump_path_siblings: WARN $inst: $err"
            continue
        }
        puts $of "=== $inst ==="
        foreach row $rows {
            set ep [lindex $row 0]
            set sl [lindex $row 1]
            puts $of "$ep  $sl"
        }
        puts $of ""
    }
    close $of
    puts "# eco_dump_path_siblings: wrote [llength $ordered] inst blocks to $out_file"
    return $out_file
}


# ======================================================================
# eco_rank_fanout_by_slack
# ----------------------------------------------------------------------
# Rank a driver's fanout sinks by worst-path slack through that sink.
# The worst-slack sinks are the ones that need to be *moved* to a clone
# (the critical cluster); slack-rich sinks stay on the original driver.
#
# For each sink ITerm on the driver's output net, calls
#   report_checks -through <sink_pin> -group_path_count 1 -path_delay max
# and extracts the single worst slack value.
#
# Args:
#   driver_inst   instance name whose output-net fanout we rank
#   out_file      optional; if non-empty, write a dumpfile in the same
#                 format as eco_dump_path_siblings (so context_builder
#                 can pick it up). Default "" → no file written.
#   mode          "max" (setup, default) or "min" (hold)
#
# Returns:
#   Tcl list of {sink_iterm_name slack_ns}, sorted worst-slack first.
# ======================================================================
proc eco_rank_fanout_by_slack { driver_inst {out_file ""} {mode "max"} } {
    set block [ord::get_db_block]
    set dbi [$block findInst $driver_inst]
    if {$dbi eq "NULL" || $dbi eq ""} {
        error "eco_rank_fanout_by_slack: instance '$driver_inst' not found"
    }

    # -- find the driver's output net (skip VDD/VSS power iterms) -----
    set onet ""
    foreach it [$dbi getITerms] {
        set mt [$it getMTerm]
        set sig [$mt getSigType]
        if {$sig eq "POWER" || $sig eq "GROUND"} { continue }
        set io [$mt getIoType]
        if {$io eq "OUTPUT" || $io eq "INOUT"} {
            set n [$it getNet]
            if {$n ne "NULL" && $n ne ""} { set onet $n; break }
        }
    }
    if {$onet eq ""} {
        error "eco_rank_fanout_by_slack: no output net on '$driver_inst'"
    }
    set onet_name [$onet getName]

    # -- collect every sink iterm (inst/pin) on the net ---------------
    set sinks {}
    foreach it [$onet getITerms] {
        set mt [$it getMTerm]
        set sig [$mt getSigType]
        if {$sig eq "POWER" || $sig eq "GROUND"} { continue }
        set io [$mt getIoType]
        if {$io ne "INPUT"} { continue }
        set sinst [$it getInst]
        if {$sinst eq "NULL" || $sinst eq ""} { continue }
        if {[$sinst getName] eq $driver_inst} { continue }
        set full "[$sinst getName]/[[$it getMTerm] getName]"
        lappend sinks $full
    }
    if {[llength $sinks] == 0} {
        error "eco_rank_fanout_by_slack: net '$onet_name' has no sinks"
    }

    # -- per-sink worst-slack via report_checks -through --------------
    set num {[-+]?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?}
    set pat "^(.+?)\\s+${num}\\s+${num}\\s+(${num})\\s*\\(\[^)\]+\\)\\s*\$"
    set tmp "/tmp/eco_fanout_[pid]_[clock microseconds].rpt"

    set rows {}
    foreach sk $sinks {
        set pobj ""
        if {[catch {set pobj [get_pins $sk]}]} { set pobj "" }
        if {$pobj eq "" || $pobj eq "NULL"} {
            # Pin not in STA graph (rare); skip gracefully.
            lappend rows [list $sk 999.0]
            continue
        }
        if {[catch {
            report_checks -through $pobj \
                          -group_path_count 1 \
                          -path_delay $mode \
                          -format end > $tmp
        } err]} {
            lappend rows [list $sk 999.0]
            continue
        }
        set fh [open $tmp r]; set raw [read $fh]; close $fh
        set slk 999.0
        foreach line [split $raw "\n"] {
            if {[regexp $pat $line -> _ s]} { set slk $s; break }
        }
        lappend rows [list $sk $slk]
    }
    catch {file delete $tmp}

    # -- sort ascending (worst slack first) ---------------------------
    set rows [lsort -real -index 1 $rows]

    # -- optional dumpfile in siblings format -------------------------
    if {$out_file ne ""} {
        set of [open $out_file w]
        puts $of "# eco_rank_fanout_by_slack driver=$driver_inst fanout=[llength $rows] [clock format [clock seconds]]"
        puts $of "=== $driver_inst ==="
        foreach r $rows {
            puts $of "[lindex $r 0]  [lindex $r 1]"
        }
        close $of
        puts "# eco_rank_fanout_by_slack: $driver_inst fanout=[llength $rows] -> $out_file"
    }
    return $rows
}


# ======================================================================
# eco_clone_gate_worst_half
# ----------------------------------------------------------------------
# Convenience wrapper: rank fanouts by slack (worst-first), take the
# worst `split` fraction, hand that sink list to eco_clone_gate. The
# clone then drives the critical cluster; the original driver keeps
# the slack-rich sinks. Placement falls out of eco_clone_gate
# (centroid of moved sinks).
#
# Args:
#   driver_inst   instance to clone
#   clone_name    name for the new clone instance
#   split         fraction (0<split<1) of worst-slack sinks to move to
#                 the clone. Default 0.5 (half).
#   mode          "max" (setup) or "min" (hold). Default max.
#
# Returns: list of sink pins moved to clone.
# ======================================================================
proc eco_clone_gate_worst_half { driver_inst clone_name {split 0.5} {mode "max"} } {
    if {$split <= 0.0 || $split >= 1.0} {
        error "eco_clone_gate_worst_half: split must be strictly between 0 and 1"
    }
    set ranked [eco_rank_fanout_by_slack $driver_inst "" $mode]
    set n [llength $ranked]
    if {$n < 2} {
        error "eco_clone_gate_worst_half: $driver_inst fanout=$n — nothing to split"
    }
    set k [expr {int(ceil($n * $split))}]
    if {$k < 1} { set k 1 }
    if {$k >= $n} { set k [expr {$n - 1}] }

    set move {}
    for {set i 0} {$i < $k} {incr i} {
        lappend move [lindex [lindex $ranked $i] 0]
    }
    puts "# eco_clone_gate_worst_half: $driver_inst fanout=$n, moving $k worst-slack sinks to $clone_name"
    eco_clone_gate $driver_inst $clone_name $move
    return $move
}


# ======================================================================
# eco_dump_fanout_ranks
# ----------------------------------------------------------------------
# Batch wrapper: walk the worst violating setup path and, for every
# non-register inst whose output-net fanout >= min_fanout, dump a
# slack-ranked sink list. Writes a multi-block file consumed by
# context_builder_v5.py's fanout_rank loader.
#
# Format (one block per high-FO driver):
#
#     === <driver_inst> FO=<n> ===
#     <sink_iterm>   <slack_ns>
#     <sink_iterm>   <slack_ns>
#     ...
#
# The sink listed first is the MOST critical (worst slack). A clone
# move should move the top ceil(FO/2) rows to the clone via
# eco_clone_gate.
#
# Args:
#   out_file     absolute path to write
#   min_fanout   only dump drivers with FO >= this (default 20, matches §5 P5)
# ======================================================================
proc eco_dump_fanout_ranks { out_file {min_fanout 20} } {

    set tmp "/tmp/eco_frpath_[pid]_[clock microseconds].rpt"
    if {[catch {
        report_checks -path_delay max -group_path_count 1 \
                      -fields {net cap slew input fanout} \
                      -format full > $tmp
    } err]} {
        catch {file delete $tmp}
        error "eco_dump_fanout_ranks: report_checks failed: $err"
    }

    set fh [open $tmp r]; set raw [read $fh]; close $fh; file delete $tmp

    # Walk transition rows (same logic as eco_dump_path_siblings) to
    # get the ordered list of path instances.
    set seen [dict create]
    set ordered {}
    foreach line [split $raw "\n"] {
        if {![regexp {[\^v]\s+(\S+)/(\w+)\s+\((\S+)\)} $line -> inst pin cell]} {
            continue
        }
        if {[regexp -nocase {^(DFF|SDFF|LATCH|ICG|SNL|SRA)} $cell]} { continue }
        if {![regexp {^(Y|Q|QN|S|SO|CO|SN|Z|ZN|CON|COUT|S0|S1|SUM)$} $pin]} { continue }
        if {[dict exists $seen $inst]} { continue }
        dict set seen $inst 1
        lappend ordered $inst
    }

    set of [open $out_file w]
    puts $of "# Generated by eco_dump_fanout_ranks min_fanout=$min_fanout [clock format [clock seconds]]"

    set block [ord::get_db_block]
    set dumped 0
    foreach inst $ordered {
        set dbi [$block findInst $inst]
        if {$dbi eq "NULL" || $dbi eq ""} { continue }
        set fo 0
        foreach it [$dbi getITerms] {
            set mt [$it getMTerm]
            set sig [$mt getSigType]
            if {$sig eq "POWER" || $sig eq "GROUND"} { continue }
            set io [$mt getIoType]
            if {$io ne "OUTPUT" && $io ne "INOUT"} { continue }
            set net [$it getNet]
            if {$net eq "NULL" || $net eq ""} { continue }
            set fo [expr {[llength [$net getITerms]] - 1}]
            break
        }
        if {$fo < $min_fanout} { continue }

        if {[catch {set rows [eco_rank_fanout_by_slack $inst "" max]} err]} {
            puts "# eco_dump_fanout_ranks: WARN $inst: $err"
            continue
        }
        puts $of "=== $inst FO=$fo ==="
        foreach row $rows {
            puts $of "[lindex $row 0]  [lindex $row 1]"
        }
        puts $of ""
        incr dumped
    }
    close $of
    puts "# eco_dump_fanout_ranks: $dumped high-FO drivers (>=$min_fanout) -> $out_file"
    return $out_file
}
