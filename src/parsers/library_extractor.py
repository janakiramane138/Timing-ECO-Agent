# library_extractor.py
from pathlib import Path
import gzip
import re
from typing import Dict, List
from dataclasses import dataclass, field
import os


@dataclass
class CellInfo:
    """Information about a standard cell"""
    name: str
    type: str           # base function, e.g., "XOR2", "AOI21", "BUF"
    drive_strength: str  # e.g., "x1", "x2", "xp5"
    vt_flavor: str       # "R", "L", or "SL"
    tech_base: str       # "ASAP7_75t" — shared across VTs
    area: float
    pins: Dict[str, str] = field(default_factory=dict)       # pin_name -> direction
    input_caps_ff: Dict[str, float] = field(default_factory=dict)  # pin_name -> cap (fF)

    @property
    def max_input_cap_ff(self) -> float:
        return max(self.input_caps_ff.values(), default=0.0)


class LibraryExtractor:
    """Extract cell information from .lib and .lib.gz files"""

    def __init__(self, lib_dir: Path):
        self.lib_dir = lib_dir
        self.cells: Dict[str, CellInfo] = {}
        self._parse_libraries()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_libraries(self):
        """Parse all .lib and .lib.gz files in lib_dir."""
        for lib_file in sorted(self.lib_dir.iterdir()):
            if lib_file.name.endswith(".lib"):
                self._parse_lib_file(lib_file, compressed=False)
            elif lib_file.name.endswith(".lib.gz"):
                self._parse_lib_file(lib_file, compressed=True)

    @staticmethod
    def _find_balanced_end(text: str, open_idx: int) -> int:
        """Return the index of the matching '}' for the '{' at `open_idx`,
        or -1 if unbalanced."""
        depth = 0
        n = len(text)
        for j in range(open_idx, n):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return j
        return -1

    def _parse_lib_file(self, lib_file: Path, compressed: bool = False):
        """Parse a single .lib or .lib.gz file."""
        if compressed:
            with gzip.open(lib_file, "rt", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            with open(lib_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        # Find every `cell (NAME) {` header, then use manual brace balancing
        # to capture the full (nested) cell body — the original one-level regex
        # truncated at the first timing-arc closing brace, so pins were lost.
        header_re = re.compile(r"cell\s*\(\s*([^\s)]+)\s*\)\s*\{")
        for m in header_re.finditer(content):
            cell_name = m.group(1)
            open_idx = m.end() - 1  # position of the '{'
            close_idx = self._find_balanced_end(content, open_idx)
            if close_idx < 0:
                continue
            cell_content = content[open_idx + 1 : close_idx]
            cell_info = self._parse_cell(cell_name, cell_content)
            if cell_info:
                self.cells[cell_name] = cell_info

    def _parse_cell(self, name: str, content: str) -> CellInfo:
        """Parse individual cell information."""
        # Extract area
        area_match = re.search(r'area\s*:\s*([\d.]+)', content)
        area = float(area_match.group(1)) if area_match else 0.0

        # ---- Decompose ASAP7 cell name ----
        # E.g. "XOR2x2_ASAP7_75t_R"  → type="XOR2", drive="x2", vt="R", tech="ASAP7_75t"
        # E.g. "AOI21xp5_ASAP7_75t_L" → type="AOI21", drive="xp5", vt="L", tech="ASAP7_75t"
        # E.g. "DFFHQNx2_ASAP7_75t_SL" → type="DFFHQN", drive="x2", vt="SL"
        type_match = re.match(
            r'^(.+?)(x(?:\d+p\d+|\d+[a-zA-Z]*|p\d+))_(.+)_(R|L|SL)$', name
        )
        if type_match:
            cell_type = type_match.group(1)
            drive_strength = type_match.group(2)
            tech_base = type_match.group(3)
            vt_flavor = type_match.group(4)
        else:
            cell_type = name
            drive_strength = ""
            tech_base = ""
            vt_flavor = ""

        # Extract pins (direction + input capacitance) with brace balancing
        # so nested timing arcs inside a pin do not truncate the pin body.
        pins: Dict[str, str] = {}
        input_caps: Dict[str, float] = {}
        pin_header_re = re.compile(r"pin\s*\(\s*([^\s)]+)\s*\)\s*\{")
        for pm in pin_header_re.finditer(content):
            pin_name = pm.group(1)
            open_idx = pm.end() - 1
            close_idx = LibraryExtractor._find_balanced_end(content, open_idx)
            if close_idx < 0:
                continue
            pin_body = content[open_idx + 1 : close_idx]
            # Only consider the direct pin body, not nested sub-blocks.
            # The `capacitance` and `direction` we want are at the pin's top
            # level, which happens to be the first occurrence.
            dir_m = re.search(r"direction\s*:\s*(\w+)", pin_body)
            if not dir_m:
                continue
            pins[pin_name] = dir_m.group(1)
            if dir_m.group(1) == "input":
                cap_m = re.search(r"capacitance\s*:\s*([\d.]+)", pin_body)
                if cap_m:
                    input_caps[pin_name] = float(cap_m.group(1))

        return CellInfo(
            name=name,
            type=cell_type,
            drive_strength=drive_strength,
            vt_flavor=vt_flavor,
            tech_base=tech_base,
            area=area,
            pins=pins,
            input_caps_ff=input_caps,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_equivalent_cells(
        self, cell_name: str, exclude_same: bool = True, cross_vt: bool = True
    ) -> List[CellInfo]:
        """Get functionally equivalent cells (same base type, different drive/VT).

        Parameters
        ----------
        cell_name : str
            Reference cell, e.g. "XOR2x2_ASAP7_75t_R"
        exclude_same : bool
            If True, exclude the cell itself from the result.
        cross_vt : bool
            If True (default), include _L and _SL variants.
            If False, only return cells with same VT flavor.
        """
        if cell_name not in self.cells:
            return []

        base = self.cells[cell_name]
        equivalents = []

        for name, info in self.cells.items():
            if exclude_same and name == cell_name:
                continue
            # Must be same base function type (e.g., "XOR2" == "XOR2")
            if info.type != base.type:
                continue
            # Must share the same tech node (e.g., "ASAP7_75t")
            if info.tech_base != base.tech_base:
                continue
            # If cross_vt is False, restrict to same VT flavor
            if not cross_vt and info.vt_flavor != base.vt_flavor:
                continue
            equivalents.append(info)

        # Sort: LVT first (speed priority), then by area within each VT
        vt_order = {"L": 0, "R": 1, "SL": 2}
        return sorted(
            equivalents,
            key=lambda c: (vt_order.get(c.vt_flavor, 9), c.area),
        )

    def get_buffer_cells(self) -> List[CellInfo]:
        """Get all buffer cells sorted by VT (L first) then area."""
        buffers = [c for c in self.cells.values() if c.type == "BUF"]
        vt_order = {"L": 0, "R": 1, "SL": 2}
        return sorted(
            buffers,
            key=lambda c: (vt_order.get(c.vt_flavor, 9), c.area),
        )

    def get_inverter_cells(self) -> List[CellInfo]:
        """Get all inverter cells sorted by VT then area."""
        invs = [c for c in self.cells.values() if c.type == "INV"]
        vt_order = {"L": 0, "R": 1, "SL": 2}
        return sorted(invs, key=lambda c: (vt_order.get(c.vt_flavor, 9), c.area))


if __name__ == "__main__":
    import sys

    lib_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        os.environ.get("ASAP7_PDK_LIB",
            str(Path(__file__).resolve().parents[2] / "asap7/lib/NLDM/"))
    )
    lib = LibraryExtractor(lib_dir)

    print(f"Loaded {len(lib.cells)} cells from {lib_dir}")

    # Show VT distribution
    from collections import Counter
    vt_counts = Counter(c.vt_flavor for c in lib.cells.values())
    print(f"VT distribution: {dict(vt_counts)}")

    # Test cross-VT equivalents
    test_cell = "XOR2x2_ASAP7_75t_R"
    if test_cell in lib.cells:
        eqs = lib.get_equivalent_cells(test_cell)
        print(f"\nEquivalents for {test_cell}:")
        for c in eqs:
            print(f"  {c.name}  (VT={c.vt_flavor}, drive={c.drive_strength}, area={c.area})")
    else:
        print(f"\n{test_cell} not found. Sample cells:")
        for name in list(lib.cells.keys())[:20]:
            c = lib.cells[name]
            print(f"  {name}  type={c.type} vt={c.vt_flavor} drive={c.drive_strength}")