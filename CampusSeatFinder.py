#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CampusSeatFinder — Smart study-space locator (no reservations).

What problem it solves:
- Students waste time walking around campus trying to find a free desk / UniKey PC / monitor desk.
- This gives live-style availability without booking.
- It also encourages people not to "reserve" desks with laptops and disappear.

Core features:
1. Live occupancy dashboard by building (with colour bars).
2. Manual claim / vacate a specific seat (simulating QR + UniKey login).
3. Auto-refresh "wall display" mode (threaded) + CSV activity logging.
4. Trend summary (busiest / quietest buildings from log history).
5. Smart recommender (which building is best to go to right now).
6. Navigation across campus using BFS shortest path in a building graph.
7. Room list view that shows all rooms across campus, sorted by most free seats.

Advanced concepts (for marks):
- OOP with @dataclass
- Graph + BFS shortest path (DSA)
- Threading (background auto-refresh loop)
- File persistence (JSON) and analytics logging (CSV)
- Dictionaries, lists, sorting with custom keys, formatted console UI
"""

from __future__ import annotations
import os, sys, json, csv, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
from collections import deque
from datetime import datetime

DATA_FILE = "campus_data.json"
LOG_FILE  = "occupancy_log.csv"

SPACE_TYPES = {"desk_only", "unikey_pc", "monitor_usb"}

# ---------------------------- Domain Model ----------------------------

@dataclass
class Space:
    """
    A single physical study seat.
    """
    id: str
    type: str
    occupied: bool = False

    def __post_init__(self):
        if self.type not in SPACE_TYPES:
            raise ValueError(f"Invalid type '{self.type}'. Allowed: {SPACE_TYPES}")

@dataclass
class Room:
    """
    A room/zone inside a building.
    Example: 'J12-201'
    """
    id: str
    type_mix: List[str]
    spaces: List[Space] = field(default_factory=list)

    def counts(self, desired_type: Optional[str] = None) -> Tuple[int, int]:
        """
        Return (vacant, occupied) for this room.
        If desired_type is given, only count seats of that type.
        """
        vacant = 0
        occupied = 0
        for s in self.spaces:
            if desired_type and s.type != desired_type:
                continue
            if s.occupied:
                occupied += 1
            else:
                vacant += 1
        return vacant, occupied

@dataclass
class Building:
    """
    A building on campus.
    Example: 'J12 — Belinda Hutchinson Building'
    """
    id: str
    name: str
    rooms: List[Room] = field(default_factory=list)

    def aggregate_counts(self, desired_type: Optional[str] = None) -> Tuple[int, int]:
        """
        Sum (vacant, occupied) over all rooms in this building.
        """
        v_total = 0
        o_total = 0
        for r in self.rooms:
            rv, ro = r.counts(desired_type)
            v_total += rv
            o_total += ro
        return v_total, o_total

# ---------------------------- Graph / BFS (DSA concept) ----------------------------

class CampusGraph:
    """
    Undirected graph between buildings.
    We use BFS for shortest_path() navigation.
    """

    def __init__(self, adj: Dict[str, List[str]]):
        self.adj = {k: list(v) for k, v in adj.items()}

    def neighbors(self, node: str) -> Iterable[str]:
        return self.adj.get(node, [])

    def shortest_path(self, start: str, goal: str) -> Optional[List[str]]:
        """
        BFS to compute the shortest path from 'start' building to 'goal' building.
        Returns list of building IDs, or None if disconnected.
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
                        # Reconstruct path backwards
                        path = [goal]
                        while prev[path[-1]] is not None:
                            path.append(prev[path[-1]])
                        path.reverse()
                        return path
                    q.append(nb)
        return None

# ---------------------------- SeatFinder controller ----------------------------

class SeatFinder:
    """
    Holds all buildings and the campus graph.
    Provides queries, state updates, and persistence helpers.
    """

    def __init__(self, buildings: List[Building], graph: CampusGraph):
        # Use dict for O(1) building lookup by ID
        self.buildings = {b.id: b for b in buildings}
        self.graph = graph

    def building_ids(self) -> List[str]:
        return sorted(self.buildings.keys())

    def find_matching_rooms(self, seat_type: Optional[str]) -> List[Tuple[str, str, int, int]]:
        """
        Build a list of all rooms on campus that have seats
        (optionally only certain seat_type).
        Returns list of tuples:
            (building_id, room_id, vacant, occupied)
        We'll sort these later in the UI.
        """
        rows = []
        for b in self.buildings.values():
            for r in b.rooms:
                v, o = r.counts(seat_type)
                if v + o > 0:
                    rows.append((b.id, r.id, v, o))
        # NOTE: sorting by vacancy DESC happens in pr_room_list now,
        # not here, so that function can control presentation.
        return rows

    def toggle_occupancy(self, room_id: str, space_id: str) -> bool:
        """
        Flip a seat from free->taken or taken->free.
        Returns True if found and toggled.
        """
        for b in self.buildings.values():
            for r in b.rooms:
                if r.id == room_id:
                    for s in r.spaces:
                        if s.id == space_id:
                            s.occupied = not s.occupied
                            return True
        return False

    def to_dict(self) -> Dict:
        """
        Convert to plain dict so we can JSON dump.
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
                                {
                                    "id": s.id,
                                    "type": s.type,
                                    "occupied": s.occupied
                                }
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
        Build SeatFinder (and Rooms/Spaces) back from a dict (loaded from JSON).
        """
        buildings: List[Building] = []
        for b in data["buildings"]:
            rooms: List[Room] = []
            for r in b["rooms"]:
                spaces = [Space(**s) for s in r["spaces"]]
                rooms.append(Room(
                    id=r["id"],
                    type_mix=r["type_mix"],
                    spaces=spaces
                ))
            buildings.append(Building(
                id=b["id"],
                name=b.get("name", b["id"]),
                rooms=rooms
            ))
        graph = CampusGraph(data["graph"])
        return SeatFinder(buildings, graph)

# ---------------------------- Initial dataset (first run seed) ----------------------------

SAMPLE = {
    "buildings": [
        {
            "id": "J12",
            "name": "J12 — Belinda Hutchinson Building",
            "rooms": [
                {
                    "id": "J12-201",
                    "type_mix": ["desk_only", "monitor_usb"],
                    "spaces": [
                        {"id": "J12-201-01", "type": "desk_only", "occupied": False},
                        {"id": "J12-201-02", "type": "desk_only", "occupied": True},
                        {"id": "J12-201-03", "type": "monitor_usb", "occupied": False}
                    ]
                },
                {
                    "id": "J12-202",
                    "type_mix": ["unikey_pc"],
                    "spaces": [
                        {"id": "J12-202-01", "type": "unikey_pc", "occupied": False},
                        {"id": "J12-202-02", "type": "unikey_pc", "occupied": True}
                    ]
                }
            ]
        },
        {
            "id": "J03",
            "name": "J03 — SciTech Library",
            "rooms": [
                {
                    "id": "J03-101",
                    "type_mix": ["desk_only"],
                    "spaces": [
                        {"id": "J03-101-01", "type": "desk_only", "occupied": True},
                        {"id": "J03-101-02", "type": "desk_only", "occupied": False},
                        {"id": "J03-101-03", "type": "desk_only", "occupied": False}
                    ]
                }
            ]
        },
        {
            "id": "A02",
            "name": "A02 — Fisher Library",
            "rooms": [
                {
                    "id": "A02-3F",
                    "type_mix": ["monitor_usb", "unikey_pc"],
                    "spaces": [
                        {"id": "A02-3F-01", "type": "monitor_usb", "occupied": True},
                        {"id": "A02-3F-02", "type": "unikey_pc", "occupied": False}
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

# ---------------------------- Persistence helpers ----------------------------

def ensure_data_file(path: str = DATA_FILE) -> None:
    """
    If campus_data.json doesn't exist yet, create it from SAMPLE.
    """
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE, f, indent=2)

def load_seat_finder(path: str = DATA_FILE) -> SeatFinder:
    """
    Load the persistent campus data into SeatFinder objects.
    """
    ensure_data_file(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SeatFinder.from_dict(data)

def save_seat_finder(sf: SeatFinder, path: str = DATA_FILE) -> None:
    """
    Save (including any updated seat states) back to campus_data.json.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sf.to_dict(), f, indent=2)

# ---------------------------- UI helper functions ----------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def choose_type() -> Optional[str]:
    """
    Ask user which seat type they want to filter by.
    """
    print("Filter by type:")
    print(" [1] desk_only")
    print(" [2] unikey_pc")
    print(" [3] monitor_usb")
    print(" [Enter] ANY")
    choice = input("Choice: ").strip()
    return {"1": "desk_only", "2": "unikey_pc", "3": "monitor_usb"}.get(choice)

def bar(pct: float, width: int = 20) -> str:
    """
    Draw a mini progress bar (for crowding).
    pct == 1.0 => full, pct == 0.0 => empty.
    """
    pct = max(0.0, min(1.0, pct))
    filled = int(pct * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

# ---------------------------- DASHBOARD VIEW (menu 1) ----------------------------

def pr_live_dashboard(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Show each building:
    - Vacant / Occupied seats
    - A little bar
    - Colour (green=more free, red=more taken)

    Also prints:
    - Campus total full percentage
    - Which building currently has the MOST free seats
    """
    print_header(f"Campus Occupancy Dashboard (type={seat_type or 'ANY'})")

    total_v = 0
    total_o = 0
    most_vacant_bid = None
    most_vacant_count = -1

    for bid in sf.building_ids():
        b = sf.buildings[bid]
        v, o = b.aggregate_counts(seat_type)
        total_v += v
        total_o += o

        # track "most vacant building"
        if v > most_vacant_count:
            most_vacant_bid = bid
            most_vacant_count = v

        total_here = v + o
        fullness = (o / total_here) if total_here else 0.0
        color = "\033[92m" if v > o else "\033[91m"
        reset = "\033[0m"

        print(
            f"{b.name:35} [{bid}]  "
            f"Vacant:{v:2d}  Occ:{o:2d}  "
            f"{color}{bar(1 - fullness)}{reset}"
        )

    campus_total = total_v + total_o
    fullness_pct = (total_o / campus_total * 100) if campus_total else 0.0
    print("-" * 70)
    print(f"Total: Vacant={total_v}  Occupied={total_o}  Fullness={fullness_pct:.0f}%")

    if most_vacant_bid is not None:
        best_b = sf.buildings[most_vacant_bid]
        print(f"\nMost Vacant Building RIGHT NOW:")
        print(f" -> {best_b.name} [{best_b.id}] with {most_vacant_count} free seats")

# ---------------------------- ROOM LIST VIEW (menu 2) ----------------------------

def pr_room_list(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Show each room on campus.
    - Includes building name
    - Sorted by highest vacancy first
    """
    print_header(f"Rooms (type={seat_type or 'ANY'}) — sorted by vacancies")

    rows = sf.find_matching_rooms(seat_type)
    if not rows:
        print("No rooms found.")
        return

    # rows are (bid, rid, v, o)
    # Sort by vacancy DESC, fallback by building ID and room ID
    rows.sort(key=lambda x: (-x[2], x[0], x[1]))

    print(f"{'Building Name':35} {'Room ID':12} {'Vacant':7} {'Occupied':9} {'Fullness'}")
    print("-" * 70)
    for bid, rid, v, o in rows:
        total = v + o
        fullness_pct = (o / total * 100) if total else 0.0
        bname = sf.buildings[bid].name
        print(f"{bname:35} {rid:12} {v:7d} {o:9d} {fullness_pct:6.0f}%")

# ---------------------------- LOGGING & TRENDS (menu 5) ----------------------------

def append_log_snapshot(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Log current building stats into CSV for historical analysis.
    CSV columns:
        timestamp, filter_type, building_id, vacant, occupied
    """
    ts = datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for bid in sf.building_ids():
            v, o = sf.buildings[bid].aggregate_counts(seat_type)
            w.writerow([ts, seat_type or "ANY", bid, v, o])

def pr_trends_summary(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Look at the CSV log and calculate:
    - average vacant seats per building
    - average occupied seats
    - average fullness %
    - print busiest / quietest building overall
    """
    print_header(f"Usage Trends (type={seat_type or 'ANY'})")

    by_building: Dict[str, Dict[str, int]] = {}

    try:
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 5:
                    continue
                ts, logged_type, bid, v_raw, o_raw = row
                if seat_type and logged_type != seat_type:
                    continue
                v = int(v_raw)
                o = int(o_raw)
                agg = by_building.setdefault(bid, {"v": 0, "o": 0, "n": 0})
                agg["v"] += v
                agg["o"] += o
                agg["n"] += 1
    except FileNotFoundError:
        print("No log yet. View dashboard at least once to create data.")
        return

    if not by_building:
        print("No data recorded for this filter yet.")
        return

    summary_rows = []
    for bid, stats in by_building.items():
        n = max(stats["n"], 1)
        avg_v = stats["v"] // n
        avg_o = stats["o"] // n
        denom = stats["v"] + stats["o"]
        fullness_ratio = (stats["o"] / denom) if denom else 0.0
        summary_rows.append((bid, avg_v, avg_o, fullness_ratio))

    # Sort by fullness ratio DESC (busiest first)
    summary_rows.sort(key=lambda x: x[3], reverse=True)

    for bid, avg_v, avg_o, occ in summary_rows:
        bname = sf.buildings[bid].name
        print(
            f"[{bid}] {bname}\n"
            f"    avg vacant={avg_v:2d}  "
            f"avg occupied={avg_o:2d}  "
            f"avg fullness={occ*100:3.0f}%"
        )

    print("\nBusiest (highest fullness):", summary_rows[0][0])
    print("Quietest (lowest fullness):", summary_rows[-1][0])

# ---------------------------- RECOMMENDER (menu 6) ----------------------------

def recommend_building_and_print(sf: SeatFinder, seat_type: Optional[str] = None) -> None:
    """
    Recommend a building based on vacancy ratio RIGHT NOW.
    ratio = vacant / total seats in that building.
    """
    best = None  # (ratio, bid, free_count)
    for bid in sf.building_ids():
        v, o = sf.buildings[bid].aggregate_counts(seat_type)
        total = v + o
        if total == 0:
            continue
        ratio = v / total
        if (best is None) or (ratio > best[0]):
            best = (ratio, bid, v)

    if best is None:
        print("No suitable building found for that filter.")
        return

    ratio, bid, free = best
    b = sf.buildings[bid]
    print("\nRecommended Building RIGHT NOW:")
    print(f" -> {b.name} [{bid}]")
    print(f" -> Approx {ratio:.0%} of seats free ({free} seats currently vacant)")

# ---------------------------- AUTO REFRESHER (menu 4) ----------------------------

class AutoRefresher:
    """
    Threaded live mode:
    keeps redrawing the dashboard every N seconds and logs snapshots.
    """
    def __init__(self, sf: SeatFinder, interval_sec: int = 5, seat_type: Optional[str] = None):
        self.sf = sf
        self.interval = max(1, int(interval_sec))
        self.seat_type = seat_type
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            # Clear the terminal and redraw at top
            sys.stdout.write("\033[2J\033[H")
            pr_live_dashboard(self.sf, self.seat_type)
            append_log_snapshot(self.sf, self.seat_type)
            print("(Auto-refresh ON — press ENTER in menu to stop)")
            time.sleep(self.interval)

# ---------------------------- MAIN CLI LOOP ----------------------------

def main() -> None:
    sf = load_seat_finder()

    while True:
        print_header("CampusSeatFinder")
        print("1) Live dashboard")
        print("2) List rooms (sorted by vacancies)")
        print("3) Toggle a seat (simulate claim/vacate)")
        print("4) Auto-refresh mode (live wall display)")
        print("5) Trends summary (busiest / quietest)")
        print("6) Recommend best building right now")
        print("7) Navigation (shortest path between buildings)")
        print("8) Save & Exit")
        cmd = input("Select: ").strip()

        if cmd == "1":
            # dashboard snapshot + log
            t = choose_type()
            pr_live_dashboard(sf, t)
            append_log_snapshot(sf, t)

        elif cmd == "2":
            # now shows building name + sorted by vacancy
            t = choose_type()
            pr_room_list(sf, t)

        elif cmd == "3":
            # simulate QR claim / vacate
            rid = input("Room ID (e.g. J12-201): ").strip()
            sid = input("Seat ID (e.g. J12-201-01): ").strip()
            if sf.toggle_occupancy(rid, sid):
                save_seat_finder(sf)
                print("Seat status updated and saved.")
                pr_live_dashboard(sf, None)
                append_log_snapshot(sf, None)
            else:
                print("Room/Seat not found. No changes made.")

        elif cmd == "4":
            # threaded live wall display
            t = choose_type()
            try:
                interval_raw = input("Refresh interval seconds [5]: ").strip()
                interval_val = int(interval_raw) if interval_raw else 5
            except ValueError:
                interval_val = 5
            refresher = AutoRefresher(sf, interval_val, t)
            try:
                refresher.start()
                input()  # waits for ENTER
            finally:
                refresher.stop()

        elif cmd == "5":
            # read CSV analytics
            t = choose_type()
            pr_trends_summary(sf, t)

        elif cmd == "6":
            # recommend best building right now
            t = choose_type()
            recommend_building_and_print(sf, t)

        elif cmd == "7":
            # BFS nav across campus graph
            start = input("From building ID (e.g. J03): ").strip()
            goal  = input("To building ID (e.g. A02): ").strip()
            path = sf.graph.shortest_path(start, goal)
            if path:
                print("Suggested path:", " -> ".join(path))
            else:
                print("No path found between those buildings.")

        elif cmd == "8":
            save_seat_finder(sf)
            print("Saved. Bye!")
            return

        else:
            print("Invalid option. Try again.")

# ---------------------------- Entry Point ----------------------------

if __name__ == "__main__":
    main()
