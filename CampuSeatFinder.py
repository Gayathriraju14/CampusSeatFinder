#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CampusSeatFinder — Smart study-space locator (no reservations).

Problem it solves:
- At uni, most casual study areas are not bookable. Students walk around wasting
  time trying to find a free seat (desk, PC, monitor desk with power).
- It can also help safety because people shouldn't leave laptops unattended:
  they "claim" a seat when they're physically there, and "vacate" it when they leave.

What this program does:
1. Live-style occupancy dashboard for the whole campus
   - Shows how many seats are Vacant vs Occupied in each building
   - Can filter by seat type:
        desk_only      -> plain desk/table
        unikey_pc      -> PC that requires UniKey login
        monitor_usb    -> desk with monitor/power/USB
   - Includes a simple bar visual and colours (green/red)

2. Manual seat claiming / releasing (simulation of QR + UniKey login)
   - Menu option 3 lets you toggle any seat as "occupied" or "vacant"
   - This acts like: student scans QR code at the seat and presses "I'm here"
     or "I'm leaving"
   - After toggling, the new state is saved permanently to a JSON file, so it
     persists next time you run the app

3. Auto-refresh display mode
   - Menu option 4 runs a background thread that keeps redrawing the dashboard
     every few seconds like a live status screen
   - This mode also continuously logs snapshots into a CSV file

4. Analytics / trends
   - Menu option 5 reads the CSV log and shows
        • average vacancy per building
        • which building tends to be busiest
        • which one tends to stay quiet
   - This is useful for facilities planning (e.g. add more powered desks here)

5. Recommender
   - Menu option 6 suggests the "best" building right now for a given seat type
   - "Best" = highest vacancy ratio (more chance you can sit immediately)

6. Navigation (graph shortest path)
   - Menu option 8 uses BFS on a campus graph to suggest a path from one building
     to another (e.g. "J12 -> A02")
   - This supports the idea that the system can guide you to the best available area

Tech highlights:
- Pure Python standard library (no external installs required)
- Object-oriented design with @dataclass
- JSON persistence of state (current occupancy)
- CSV logging and simple analytics
- BFS shortest path in a campus graph
- Threading for auto-refresh dashboard

Author: <Your Name> (COMP9001 Final Project, 2025)
"""

from __future__ import annotations
import os, sys, json, csv, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
from collections import deque
from datetime import datetime

# Filenames for persistence
DATA_FILE = "campus_data.json"
LOG_FILE  = "occupancy_log.csv"

# ---------------------------- Domain Model ----------------------------

SPACE_TYPES = {"desk_only", "unikey_pc", "monitor_usb"}

@dataclass
class Space:
    """
    Represents a single physical study position / seat.

    Attributes:
        id        : seat ID, e.g. "J12-201-03"
        type      : "desk_only", "unikey_pc", or "monitor_usb"
        occupied  : True if currently taken by a student, False if free

    In a real deployment:
    - occupied = True would happen when a student scans the QR code at that seat
      and authenticates with UniKey ("Claim seat").
    - occupied = False when they press "Vacate" or after timeout.
    """
    id: str
    type: str
    occupied: bool = False

    def __post_init__(self):
        if self.type not in SPACE_TYPES:
            raise ValueError(
                f"Invalid space type '{self.type}'. Allowed: {SPACE_TYPES}"
            )


@dataclass
class Room:
    """
    Represents a room or zone inside a building that contains multiple seats.

    Attributes:
        id        : room ID, e.g. "J12-201"
        type_mix  : list of seat types offered in this room
        spaces    : list[Space] describing each seat in the room

    Methods:
        counts(desired_type):
            Returns (vacant_count, occupied_count), optionally filtered by seat type.
    """
    id: str
    type_mix: List[str]
    spaces: List[Space] = field(default_factory=list)

    def counts(self, desired_type: Optional[str] = None) -> Tuple[int, int]:
        """Count how many seats are vacant vs occupied, optionally filtering by seat type."""
        vacant = 0
        occupied = 0
        for s in self.spaces:
            if desired_type and s.type != desired_type:
                continue
            if s.occupied:
                occupied += 1
            else:
                vacant  += 1
        return vacant, occupied


@dataclass
class Building:
    """
    Represents a single campus building.

    Attributes:
        id     : e.g. "J12"
        name   : friendly name, e.g. "J12 — Engineering"
        rooms  : list[Room] inside this building

    Methods:
        aggregate_counts(desired_type):
            Sum vacant/occupied across every room in this building.
    """
    id: str
    name: str
    rooms: List[Room] = field(default_factory=list)

    def aggregate_counts(self, desired_type: Optional[str] = None) -> Tuple[int, int]:
        """Aggregate (vacant, occupied) across all rooms, optionally filtering by seat type."""
        v_total = 0
        o_total = 0
        for r in self.rooms:
            rv, ro = r.counts(desired_type)
            v_total += rv
            o_total += ro
        return v_total, o_total


class CampusGraph:
    """
    A simple undirected graph of buildings, used for navigation.

    Example:
        graph = {
            "J12": ["J03", "A02"],
            "J03": ["J12"],
            "A02": ["J12"]
        }

    shortest_path("J03", "A02") -> ["J03", "J12", "A02"]

    This supports the "navigation" part of the idea: guiding a student to a target
    building that still has free capacity.
    """
    def __init__(self, adj: Dict[str, List[str]]):
        self.adj = {k: list(v) for k, v in adj.items()}

    def neighbors(self, node: str) -> Iterable[str]:
        return self.adj.get(node, [])

    def shortest_path(self, start: str, goal: str) -> Optional[List[str]]:
        """
        Breadth-First Search (BFS) for shortest path in an unweighted graph.

        Returns:
            list of nodes from start -> goal,
            or None if no path exists.
        """
        if start == goal:
            return [start]

        q = deque([start])
        prev: Dict[str, Optional[str]] = {start: None}

        while q:
            cur = q.popleft()
            for nb in self.neighbors(cur):
                if nb not in prev:
                    prev[nb] = cur
                    if nb == goal:
                        # reconstruct path
                        path = [goal]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        path.reverse()
                        return path
                    q.append(nb)

        return None


class SeatFinder:
    """
    High-level controller for the system.

    Responsibilities:
    - Store Building objects
    - Let us query which rooms/buildings have the most vacancies
    - Toggle seat occupancy (simulating claim/vacate in real life)
    - Export/import state for persistence
    - Provide building IDs for iteration
    """
    def __init__(self, buildings: List[Building], graph: CampusGraph):
        self.buildings = {b.id: b for b in buildings}
        self.graph = graph

    # ---- Query helpers ----

    def building_ids(self) -> List[str]:
        """Return a sorted list of building IDs."""
        return sorted(self.buildings.keys())

    def find_matching_rooms(self, desired_type: Optional[str]) -> List[Tuple[str, str, int, int]]:
        """
        For each room in each building:
            Count how many of the requested seat type are vacant/occupied.
        Return a list of tuples:
            (building_id, room_id, vacant, occupied)
        Sorted primarily by highest vacancy.
        """
        rows = []
        for b in self.buildings.values():
            for r in b.rooms:
                v, o = r.counts(desired_type)
                if v + o > 0:
                    rows.append((b.id, r.id, v, o))

        # Sort by:
        # - vacancies descending
        # - occupied ascending
        # - building, room name as tie-breakers
        rows.sort(key=lambda x: (-x[2], x[3], x[0], x[1]))
        return rows

    # ---- Mutations ----

    def toggle_occupancy(self, room_id: str, space_id: str) -> bool:
        """
        Flip a single seat from occupied -> vacant or vacant -> occupied.

        Returns True if it found that seat and toggled it.
        Returns False if not found.

        Real-world meaning:
        - When a student sits at a seat and scans the QR code, the system
          would mark that seat occupied under their UniKey.
        - When they leave (or press "Vacate"), it flips back to vacant.
        For this assignment, we simulate that with manual toggling.
        """
        for b in self.buildings.values():
            for r in b.rooms:
                if r.id == room_id:
                    for s in r.spaces:
                        if s.id == space_id:
                            s.occupied = not s.occupied
                            return True
        return False

    # ---- Persistence ----

    def to_dict(self) -> Dict:
        """
        Serialize current state so we can write it to a JSON file.
        This includes buildings, rooms, spaces, and the campus graph.
        """
        return {
            "buildings": [
                {
                    "id": b.id,
                    "name": b.name,
                    "rooms": [
                        {
                            "id": r.id,
                            "type_mix": r.type_mix,
                            "spaces": [
                                {"id": s.id, "type": s.type, "occupied": s.occupied}
                                for s in r.spaces
                            ],
                        }
                        for r in b.rooms
                    ],
                }
                for b in self.buildings.values()
            ],
            "graph": self.graph.adj,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SeatFinder":
        """
        Inverse of to_dict():
        Rebuild all Building/Room/Space objects from a plain dictionary
        (e.g. from campus_data.json).
        """
        buildings: List[Building] = []

        for b in data.get("buildings", []):
            rooms: List[Room] = []
            for r in b.get("rooms", []):
                spaces = [Space(**s) for s in r.get("spaces", [])]
                rooms.append(Room(
                    id=r["id"],
                    type_mix=r.get("type_mix", []),
                    spaces=spaces
                ))
            buildings.append(Building(
                id=b["id"],
                name=b.get("name", b["id"]),
                rooms=rooms
            ))

        graph = CampusGraph(data.get("graph", {}))
        return SeatFinder(buildings, graph)


# ---------------------------- Sample Data ----------------------------

# This is the "default campus" if no saved data file exists yet.
SAMPLE = {
    "buildings": [
        {
            "id": "J12", "name": "J12 — Engineering",
            "rooms": [
                {
                    "id": "J12-201",
                    "type_mix": ["desk_only", "monitor_usb"],
                    "spaces": [
                        {"id": "J12-201-01", "type": "desk_only",    "occupied": False},
                        {"id": "J12-201-02", "type": "desk_only",    "occupied": True},
                        {"id": "J12-201-03", "type": "monitor_usb",  "occupied": False}
                    ]
                },
                {
                    "id": "J12-202",
                    "type_mix": ["unikey_pc"],
                    "spaces": [
                        {"id": "J12-202-01", "type": "unikey_pc",    "occupied": False},
                        {"id": "J12-202-02", "type": "unikey_pc",    "occupied": True}
                    ]
                }
            ]
        },
        {
            "id": "J03", "name": "J03 — Learning Hub",
            "rooms": [
                {
                    "id": "J03-101",
                    "type_mix": ["desk_only"],
                    "spaces": [
                        {"id": "J03-101-01", "type": "desk_only",    "occupied": True},
                        {"id": "J03-101-02", "type": "desk_only",    "occupied": False},
                        {"id": "J03-101-03", "type": "desk_only",    "occupied": False}
                    ]
                }
            ]
        },
        {
            "id": "A02", "name": "A02 — Library",
            "rooms": [
                {
                    "id": "A02-3F",
                    "type_mix": ["monitor_usb", "unikey_pc"],
                    "spaces": [
                        {"id": "A02-3F-01", "type": "monitor_usb",   "occupied": True},
                        {"id": "A02-3F-02", "type": "unikey_pc",     "occupied": False}
                    ]
                }
            ]
        }
    ],
    "graph": {
        "J12": ["J03", "A02"],
        "J03": ["J12"],
        "A02": ["J12"]
    }
}

# ---------------------------- Persistence Helpers ----------------------------

def ensure_data_file(path: str = DATA_FILE) -> None:
    """
    If we don't have a campus_data.json yet, create one using SAMPLE.
    This gives us a starting campus layout.
    """
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE, f, indent=2)

def load_seat_finder(path: str = DATA_FILE) -> SeatFinder:
    """
    Load campus layout + current occupancy from disk and return SeatFinder.
    """
    ensure_data_file(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SeatFinder.from_dict(data)

def save_seat_finder(sf: SeatFinder, path: str = DATA_FILE) -> None:
    """
    Save the campus layout + current occupancy to disk.
    This lets us persist updates between runs.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sf.to_dict(), f, indent=2)

# ---------------------------- UI Helpers ----------------------------

def print_header(title: str) -> None:
    """Pretty console header divider."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def choose_type() -> Optional[str]:
    """
    Ask the user which seat type they care about.
    Returns:
        "desk_only", "unikey_pc", "monitor_usb", or None for ANY.
    """
    print("Filter by type:")
    print(" [1] desk_only")
    print(" [2] unikey_pc")
    print(" [3] monitor_usb")
    print(" [Enter] ANY")
    c = input("Choice: ").strip()
    return {
        "1": "desk_only",
        "2": "unikey_pc",
        "3": "monitor_usb"
    }.get(c)

def bar(pct: float, width: int = 20) -> str:
    """
    Make a small ASCII bar for fullness (how crowded a building is).
    pct is clamped to [0,1].
    """
    clamped = max(0.0, min(1.0, pct))
    filled = int(clamped * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def pr_live_dashboard(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Print a single snapshot of building-level availability.

    - Vacant vs Occupied per building
    - Colored bar (green if more free than taken, red otherwise)
    - Overall total vacancy/occupancy at the bottom
    """
    print_header(f"Campus Occupancy Dashboard (type={seat_type or 'ANY'})")
    total_v = 0
    total_o = 0

    for bid in sf.building_ids():
        b = sf.buildings[bid]
        v, o = b.aggregate_counts(seat_type)
        total_v += v
        total_o += o

        total = v + o
        fullness = (o / total) if total else 0.0  # 1.0 means totally full
        color = "\033[92m" if v > o else "\033[91m"  # green if more free, red if more taken
        reset = "\033[0m"

        print(
            f"{b.name:24} [{bid}]  "
            f"Vacant:{v:2d}  Occ:{o:2d}  "
            f"{color}{bar(1 - fullness)}{reset}"
        )

    tot = total_v + total_o
    pct_full = f"{(total_o / tot * 100):.0f}%" if tot else "—"
    print("-" * 60)
    print(f"Total: Vacant={total_v}  Occupied={total_o}  Fullness={pct_full}")

def pr_room_list(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Show each room, sorted by vacancy, so a user can decide where to walk.
    """
    print_header(f"Rooms (type={seat_type or 'ANY'}) — sorted by vacancies")
    rows = sf.find_matching_rooms(seat_type)
    if not rows:
        print("No rooms with seats found.")
        return

    for bid, rid, v, o in rows:
        total = v + o
        fullness_pct = f"{(o / total * 100):.0f}%" if total else "—"
        print(
            f"[{bid}] {rid:<12} "
            f"Vacant:{v:2d}  Occ:{o:2d}  Fullness:{fullness_pct}"
        )

# ---------------------------- Logging & Analytics ----------------------------

def append_log_snapshot(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Log a snapshot of each building's (vacant, occupied) to a CSV file.
    Columns:
        timestamp_iso, seat_type_filter, building_id, vacant, occupied

    This historical data feeds into our usage trends summary.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    for bid in sf.building_ids():
        b = sf.buildings[bid]
        v, o = b.aggregate_counts(seat_type)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, seat_type or "ANY", bid, v, o])

def pr_trends_summary(seat_type: Optional[str] = None) -> None:
    """
    Read the CSV log and summarise average vacancy and occupancy per building.
    Shows:
    - avg vacant seats
    - avg occupied seats
    - avg fullness %
    - which building is busiest vs quietest
    """
    print_header(f"Usage Trends (type={seat_type or 'ANY'})")

    by_building: Dict[str, Dict[str, int]] = {}
    try:
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                # expect: ts, filter_type, building_id, vacant, occupied
                if len(row) != 5:
                    continue
                _, logged_type, bid, v_raw, o_raw = row

                # filter if user asked for a specific seat type
                if seat_type and logged_type != seat_type:
                    continue

                v = int(v_raw)
                o = int(o_raw)

                agg = by_building.setdefault(bid, {"v": 0, "o": 0, "n": 0})
                agg["v"] += v
                agg["o"] += o
                agg["n"] += 1
    except FileNotFoundError:
        print("No log yet. View the dashboard first to create log data.")
        return

    if not by_building:
        print("No data recorded for this filter yet.")
        return

    # Build (bid, avg_vacant, avg_occ, avg_fullness_ratio)
    rows = []
    for bid, a in by_building.items():
        n = max(a["n"], 1)
        avg_v = a["v"] // n
        avg_o = a["o"] // n
        denom = (a["v"] + a["o"])
        fullness_ratio = (a["o"] / denom) if denom > 0 else 0.0
        rows.append((bid, avg_v, avg_o, fullness_ratio))

    # Sort by fullness ratio descending (busiest first)
    rows.sort(key=lambda x: x[3], reverse=True)

    for bid, avg_v, avg_o, occ_ratio in rows:
        print(
            f"[{bid}] avg vacant={avg_v:2d}  "
            f"avg occupied={avg_o:2d}  "
            f"avg fullness={occ_ratio * 100:3.0f}%"
        )

    print("\nBusiest (highest fullness):", rows[0][0])
    print("Quietest (lowest fullness):", rows[-1][0])

def recommend_building(sf: SeatFinder, seat_type: Optional[str] = None) -> Optional[Tuple[float, str]]:
    """
    Recommend the best building to go to right now.

    We compute a 'vacancy ratio' for each building:
        ratio = (vacant seats) / (total seats)
    The building with the highest ratio is suggested first.

    Returns:
        (score, building_id) or None if nothing suitable.
    """
    best = None  # (score, bid)
    for bid in sf.building_ids():
        v, o = sf.buildings[bid].aggregate_counts(seat_type)
        total = v + o
        if total == 0 or v == 0:
            # Either building has no seats of that type
            # or it's completely full (v == 0).
            continue
        ratio = v / total
        if (best is None) or (ratio > best[0]):
            best = (ratio, bid)
    return best

# ---------------------------- Auto Refresher ----------------------------

class AutoRefresher:
    """
    Auto-refresh console dashboard in a background thread every N seconds.

    This simulates a live "wall display":
    - Continuously shows current availability
    - Continuously appends snapshots to CSV for analytics

    You stop it by pressing ENTER in the menu after it starts.
    """
    def __init__(self, sf: SeatFinder, interval_sec: int = 5, seat_type: Optional[str] = None):
        self.sf = sf
        self.interval = max(1, int(interval_sec))
        self.seat_type = seat_type
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self) -> None:
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            try:
                self._t.join(timeout=1.0)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            # Clear terminal screen and redraw at (0,0)
            sys.stdout.write("\033[2J\033[H")
            pr_live_dashboard(self.sf, self.seat_type)
            append_log_snapshot(self.sf, self.seat_type)
            print("(Auto-refresh ON — press ENTER in menu to stop)")
            time.sleep(self.interval)

# ---------------------------- CLI / Main Loop ----------------------------

def main() -> None:
    """
    Entry point for the CampusSeatFinder interactive console app.

    Menu options:
    1. Live occupancy dashboard (refresh once)
       -> Shows per-building Vacant / Occupied counts, plus a bar

    2. List rooms (filterable, sorted by most vacancies first)
       -> Helps a student choose which *room* to walk to

    3. Toggle a seat (simulate 'Claim' / 'Vacate')
       -> This is the "QR scan / UniKey login" simulation step:
          in reality you would scan a QR code at the seat,
          log in, and mark yourself as present
       -> We flip occupied <-> free for a specific space ID and save

    4. Auto-refreshing dashboard
       -> Continuously redraws the dashboard every few seconds
       -> Also logs history to CSV in the background

    5. Trends summary (busiest / quietest)
       -> Reads the CSV log and summarises average building occupancy

    6. Recommend best building right now
       -> Suggests where you have the best chance to immediately sit

    7. Save & Exit
       -> Saves current state (JSON) and quits

    8. Navigation: shortest path between buildings
       -> Uses BFS on the campus graph to suggest how to get from one
          building to another (e.g. J03 -> A02)
    """
    sf = load_seat_finder()

    while True:
        print_header("CampusSeatFinder")
        print("1) Live occupancy dashboard (refresh once)")
        print("2) List rooms (filterable, sorted by vacancies)")
        print("3) Toggle a seat (simulate claim/vacate)")
        print("4) Auto-refreshing dashboard (advanced)")
        print("5) Trends summary (busiest / quietest)")
        print("6) Recommend best building (highest vacancy ratio)")
        print("7) Save & Exit")
        print("8) Navigation: shortest path between buildings")
        cmd = input("Select: ").strip()

        if cmd == "1":
            # Show dashboard snapshot
            t = choose_type()
            pr_live_dashboard(sf, t)
            append_log_snapshot(sf, t)

        elif cmd == "2":
            # Show per-room availability sorted
            t = choose_type()
            pr_room_list(sf, t)

        elif cmd == "3":
            # Simulate QR claim / release
            rid = input("Room ID (e.g., J12-201): ").strip()
            sid = input("Space ID (e.g., J12-201-01): ").strip()
            if sf.toggle_occupancy(rid, sid):
                save_seat_finder(sf)
                print("Updated. Data saved.")
                pr_live_dashboard(sf, None)
                append_log_snapshot(sf, None)
            else:
                print("Room/Space not found. No changes made.")

        elif cmd == "4":
            # Continuous dashboard in background thread
            t = choose_type()
            try:
                interval = input("Refresh interval seconds [Enter=5]: ").strip()
                interval = int(interval) if interval else 5
            except ValueError:
                interval = 5
            ar = AutoRefresher(sf, interval_sec=interval, seat_type=t)
            try:
                ar.start()
                input()  # wait until user presses Enter
            finally:
                ar.stop()

        elif cmd == "5":
            # Analytics/trends
            t = choose_type()
            pr_trends_summary(t)

        elif cmd == "6":
            # Recommend building
            t = choose_type()
            rec = recommend_building(sf, t)
            if rec:
                score, bid = rec
                bname = sf.buildings[bid].name
                print(f"Best pick: {bname} [{bid}]  "
                      f"(vacancy ratio {score:.0%} free right now)")
            else:
                print("No suitable building found right now for that filter.")

        elif cmd == "7":
            # Persist and exit
            save_seat_finder(sf)
            print("Saved. Bye!")
            return

        elif cmd == "8":
            # Navigation / path suggestion between buildings
            start = input("From building ID (e.g., J03): ").strip()
            goal  = input("To building ID (e.g., A02): ").strip()
            path = sf.graph.shortest_path(start, goal)
            if path:
                print("Suggested path across campus: " + " -> ".join(path))
            else:
                print("No path found between those buildings.")

        else:
            print("Invalid option. Please try again.")

# ---------------------------- Script Entry ----------------------------
if __name__ == "__main__":
    main()
