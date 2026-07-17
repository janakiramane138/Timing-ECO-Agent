# timing_parser.py #OpenRoad timing report parser
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class PathPoint:
    """Represents a point in a timing path"""
    delay: float
    cumulative_time: float
    pin: str
    cell: str
    cell_type: str
    transition: str  # ^ or v
    x_um: Optional[float] = None
    y_um: Optional[float] = None

    
@dataclass
class TimingPath:
    """Complete timing path information"""
    startpoint: str
    endpoint: str
    path_group: str
    path_type: str  # max (setup) or min (hold)
    data_arrival: float
    data_required: float
    slack: float
    points: List[PathPoint]
    clock_period: float
    
    @property
    def is_violated(self) -> bool:
        return self.slack < 0
    
    @property
    def violation_type(self) -> str:
        return "setup" if self.path_type == "max" else "hold"

class TimingReportParser:
    """Parse OpenRoad timing reports"""
    
    def parse_report(self, report_text: str) -> List[TimingPath]:
        """Parse timing report text (or a file path) and extract all violated paths"""
        paths = []

        # Allow passing a file path for convenience
        if os.path.exists(report_text):
            with open(report_text, "r", encoding="utf-8") as f:
                report_text = f.read()

        stripped = report_text.lstrip()
        if stripped.startswith("{") and "\"checks\"" in stripped:
            return self._parse_json(report_text)
        
        # Split by "Startpoint:" to get individual paths
        path_sections = re.split(r'Startpoint:', report_text)[1:]
        
        for section in path_sections:
            path = self._parse_single_path(section)
            if path and path.is_violated:
                paths.append(path)
        
        return sorted(paths, key=lambda p: p.slack)  # Worst first
    
    def _parse_single_path(self, section: str) -> Optional[TimingPath]:
        """Parse a single path section"""
        lines = section.strip().split('\n')
        
        # Extract startpoint
        startpoint = lines[0].split('(')[0].strip()
        
        # Extract endpoint
        endpoint_line = [l for l in lines if 'Endpoint:' in l][0]
        endpoint = endpoint_line.split(':')[1].split('(')[0].strip()
        
        # Extract path group and type
        path_group = self._extract_field(lines, 'Path Group:')
        path_type = self._extract_field(lines, 'Path Type:')
        
        # Extract slack
        slack_line = [l for l in lines if 'slack' in l.lower()][-1]
        slack = float(re.search(r'(-?\d+\.?\d*)', slack_line).group(1))
        
        # Extract clock period
        clock_line = [l for l in lines if 'clock' in l and 'rise edge' in l][0]
        clock_period = float(re.search(r'(\d+\.?\d+)', clock_line).group(1))
        
        # Extract arrival and required times
        arrival_line = [l for l in lines if 'data arrival time' in l][0]
        required_line = [l for l in lines if 'data required time' in l][0]
        
        data_arrival = float(re.search(r'(-?\d+\.?\d+)', arrival_line).group(1))
        data_required = float(re.search(r'(-?\d+\.?\d+)', required_line).group(1))
        
        # Parse path points
        points = self._parse_path_points(lines)
        
        return TimingPath(
            startpoint=startpoint,
            endpoint=endpoint,
            path_group=path_group,
            path_type=path_type,
            data_arrival=data_arrival,
            data_required=data_required,
            slack=slack,
            points=points,
            clock_period=clock_period
        )
    
    def _parse_path_points(self, lines: List[str]) -> List[PathPoint]:
        """Parse individual points in the timing path"""
        points : List[PathPoint] = []
        
        start_idx = None
        for i, line in enumerate(lines):
            l = line.lower()
            if "description" in l and "time" in l and "delay" in l:
                start_idx = i + 1
                break
        if start_idx is None:
            return points

        while start_idx < len(lines) and set(lines[start_idx].strip()) <= {"-"}:
            start_idx += 1

        # 2) Parse timing lines until we hit end-of-path summary
        for line in lines[start_idx:]:
            low = line.lower()
            if not line.strip():
                continue

            # Stop when we enter the summary / footer area
            if ("data arrival time" in low or
                "data required time" in low or
                "slack" in low):
                break
            if set(line.strip()) <= {"-"}:
                # another separator usually means we're done with the main point list
                continue

            # 3) We only care about path point lines that contain a transition marker ^ or v
            m_tr = re.search(r"[\^v]", line)
            if not m_tr:
                continue

            transition = line[m_tr.start()]             # '^' or 'v'
            before = line[:m_tr.start()]                # columns before ^/v
            rest = line[m_tr.start() + 1:].strip()      # text after ^/v

            # 4) Extract decimal numbers BEFORE the transition marker.
            #    This avoids getting confused by cell names like INVx1 (not decimals).
            nums = re.findall(r"-?\d+\.\d+", before)

            # Map based on count (supports both report styles + blank cap cases)
            delay = None
            time_val = None
            if len(nums) == 4:
                # Cap Slew Delay Time
                delay = float(nums[2])
                time_val = float(nums[3])
            elif len(nums) == 3:
                # Slew Delay Time (Cap blank)
                delay = float(nums[1])
                time_val = float(nums[2])
            elif len(nums) == 2:
                # Delay Time
                delay = float(nums[0])
                time_val = float(nums[1])
            else:
                continue

            # 5) Extract pin and cell info from the rest:  instance/pin (cell_type)
            pin_match = re.search(r"(\S+)/(\w+)\s+\((\S+)\)", rest)
            if not pin_match:
                continue

            cell = pin_match.group(1)
            pin = pin_match.group(2)
            cell_type = pin_match.group(3)

            points.append(PathPoint(
                delay=delay,
                cumulative_time=time_val,
                pin=f"{cell}/{pin}",
                cell=cell,
                cell_type=cell_type,
                transition=transition
            ))

        return points

    def _parse_json(self, report_text: str) -> List[TimingPath]:
        data = json.loads(report_text)
        paths = []
        for chk in data.get("checks", []):
            path_points = chk.get("source_path") or chk.get("path") or []
            points = []
            prev_arrival = None

            for p in path_points:
                arrival = float(p.get("arrival", 0.0))
                delay = 0.0 if prev_arrival is None else (arrival - prev_arrival)
                prev_arrival = arrival

                x = p.get("x")
                y = p.get("y")
                x_um = x * 1e6 if x is not None else None
                y_um = y * 1e6 if y is not None else None

                points.append(PathPoint(
                    delay=delay,
                    cumulative_time=arrival,
                    pin=p.get("pin", ""),
                    cell=p.get("instance", "") or "",
                    cell_type=p.get("cell", "") or "",
                    transition="",
                    x_um=x_um,
                    y_um=y_um,
                ))

            slack = chk.get("slack", 0.0)
            startpoint = chk.get("startpoint", "")
            endpoint = chk.get("endpoint", "")
            path_group = chk.get("path_group", "")
            path_type = chk.get("path_type", "max")

            data_arrival = chk.get("data_arrival_time", points[-1].cumulative_time if points else 0.0)
            data_required = chk.get("required_time", 0.0)

            paths.append(TimingPath(
                startpoint=startpoint,
                endpoint=endpoint,
                path_group=path_group,
                path_type=path_type,
                data_arrival=float(data_arrival),
                data_required=float(data_required),
                slack=float(slack),
                points=points,
                clock_period=0.0
            ))

        return sorted(paths, key=lambda p: p.slack)

    
    def _extract_field(self, lines: List[str], field: str) -> str:
        """Extract a field value from lines"""
        line = [l for l in lines if field in l][0]
        return line.split(':')[1].strip()


if __name__ == "__main__":
    # Example usage
    parser = TimingReportParser()
    violated_paths = parser.parse_report(
        str(Path(__file__).resolve().parents[2] /
            "prompts" / "dynamic" / "dynamic_timing_rpt.txt"))
    for path in violated_paths:
        print(f"Startpoint : {path.startpoint}\nEndpoint : {path.endpoint}\nSlack : {path.slack}")