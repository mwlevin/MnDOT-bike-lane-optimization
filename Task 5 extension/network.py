"""TNTP input, physical construction units, and exact detour preprocessing.

Only the Python standard library is used here. No equity inputs or old model
modules are imported. TNTP link length is the default construction cost, as in
the Task 4 code; opposite directions share one physical construction decision.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Arc:
    u: int
    v: int
    length: float
    unit: int


@dataclass(frozen=True)
class Trip:
    origin: int
    destination: int
    demand: float
    shortest: float
    arcs: tuple[int, ...]


@dataclass
class Problem:
    directory: Path
    nodes: tuple[int, ...]
    arcs: tuple[Arc, ...]
    units: tuple[tuple[int, int], ...]
    costs: tuple[float, ...]
    originals: dict[int, str]
    trips: tuple[Trip, ...]
    coverage_share: float
    epsilon: float
    uncovered_share: float
    network_fingerprint: str
    problem_fingerprint: str

    @property
    def total_demand(self) -> float:
        return sum(w.demand for w in self.trips)


def distances(start: int, adjacency: dict[int, list[tuple[int, float]]]) -> dict[int, float]:
    result = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != result[node]:
            continue
        for neighbor, length in adjacency.get(node, []):
            candidate = distance + length
            if candidate < result.get(neighbor, math.inf):
                result[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return result


def _find(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.is_file():
        return direct
    matches = sorted(directory.glob("*_" + name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot uniquely find {name} in {directory}")


def load_problem(directory: Path, *, coverage_share: float = 0.8,
                 epsilon: float = 0.2, uncovered_share: float = 0.2,
                 od_limit: int = 0, seed: int = 42,
                 cost_file: Path | None = None) -> Problem:
    if not math.isfinite(coverage_share) or not 0 < coverage_share <= 1:
        raise ValueError("coverage-share must be in (0, 1]")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not math.isfinite(uncovered_share) or not 0 <= uncovered_share < 1:
        raise ValueError("t must be in [0, 1)")
    if od_limit < 0:
        raise ValueError("od-limit must be nonnegative")
    directory = directory.expanduser().resolve()
    net_path, trips_path = _find(directory, "net.tntp"), _find(directory, "trips.tntp")
    raw_arcs = []
    started = False
    seen = set()
    for line in net_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.upper().startswith("<END OF METADATA>"):
            started = True
            continue
        if not started or not line or line.startswith(("~", "<")):
            continue
        parts = line.replace(";", " ").split()
        if len(parts) < 4:
            raise ValueError(f"Malformed network row: {line}")
        u, v, length = int(parts[0]), int(parts[1]), float(parts[3])
        if u == v or not math.isfinite(length) or length <= 0:
            raise ValueError(f"Arc {u}->{v} must have positive finite length and distinct endpoints")
        if (u, v) in seen:
            raise ValueError(f"Parallel arc {u}->{v}: this input requires unique directed node pairs")
        seen.add((u, v))
        raw_arcs.append((u, v, length))
    if not raw_arcs:
        raise ValueError("The network has no arcs")

    units = tuple(sorted({tuple(sorted((u, v))) for u, v, _ in raw_arcs}))
    unit_index = {pair: j for j, pair in enumerate(units)}
    costs = [0.0] * len(units)
    arcs = []
    forward: dict[int, list[tuple[int, float]]] = {}
    reverse: dict[int, list[tuple[int, float]]] = {}
    for u, v, length in raw_arcs:
        j = unit_index[tuple(sorted((u, v)))]
        costs[j] = max(costs[j], length)
        arcs.append(Arc(u, v, length, j))
        forward.setdefault(u, []).append((v, length))
        reverse.setdefault(v, []).append((u, length))
    nodes = tuple(sorted({i for u, v, _ in raw_arcs for i in (u, v)}))
    originals = {i: str(i) for i in nodes}
    mapping = directory / "n2024_polygon_id_map.csv"
    if mapping.exists():
        with mapping.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                originals[int(row["numeric_id"])] = row["original_id"]

    if cost_file:
        replacements = {}
        with cost_file.expanduser().open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                pair = tuple(sorted((int(row["numeric_u"]), int(row["numeric_v"]))))
                value = float(row["cost"])
                if pair not in unit_index or pair in replacements or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid or duplicate construction cost row: {row}")
                replacements[pair] = value
        if set(replacements) != set(units):
            raise ValueError("cost-file must give exactly one cost for EVERY physical construction unit")
        costs = [replacements[pair] for pair in units]

    demand: dict[tuple[int, int], float] = {}
    origin = None
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pair_pattern = re.compile(r"(\d+)\s*:\s*(" + number + r")")
    for line in trips_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith(("<", "~")):
            continue
        match = re.match(r"Origin\s+(\d+)", line, flags=re.IGNORECASE)
        if match:
            origin = int(match.group(1))
            line = line[match.end():]
        pairs = pair_pattern.findall(line)
        if pairs and origin is None:
            raise ValueError("Trip demand occurs before an Origin row")
        for destination, quantity in pairs:
            destination, quantity = int(destination), float(quantity)
            if not math.isfinite(quantity) or quantity < 0:
                raise ValueError("Trip demand must be finite and nonnegative")
            if quantity > 0 and origin != destination:
                key = (origin, destination)
                demand[key] = demand.get(key, 0.0) + quantity
    if not demand:
        raise ValueError("No positive non-diagonal OD demand")
    keys = sorted(demand)
    if od_limit and od_limit < len(keys):
        keys = sorted(random.Random(seed).sample(keys, od_limit))
    from_origin = {o: distances(o, forward) for o in {od[0] for od in keys}}
    to_destination = {d: distances(d, reverse) for d in {od[1] for od in keys}}
    trips = []
    for o, d in keys:
        shortest = from_origin[o].get(d, math.inf)
        if not math.isfinite(shortest) or shortest <= 0:
            raise ValueError(f"OD {o}->{d} is unreachable or has zero shortest length")
        cap = (1 + epsilon) * shortest
        feasible = tuple(a for a, arc in enumerate(arcs)
                         if arc.v != o and arc.u != d
                         and from_origin[o].get(arc.u, math.inf) + arc.length
                         + to_destination[d].get(arc.v, math.inf) <= cap + 1e-8)
        trips.append(Trip(o, d, demand[(o, d)], shortest, feasible))

    network_text = json.dumps([raw_arcs, units, costs, sorted(originals.items())], sort_keys=True)
    network_hash = hashlib.sha256(network_text.encode()).hexdigest()
    problem_text = json.dumps([network_hash, [(o, d, demand[(o, d)]) for o, d in keys],
                               coverage_share, epsilon, uncovered_share])
    return Problem(directory, nodes, tuple(arcs), units, tuple(costs), originals,
                   tuple(trips), coverage_share, epsilon, uncovered_share,
                   network_hash, hashlib.sha256(problem_text.encode()).hexdigest())


def load_design(path: Path, problem: Problem) -> set[int]:
    """Accept Task 4/Task 5 solution JSON or a construction-unit CSV.

    Original node IDs take precedence, allowing safe remapping when numeric
    node IDs changed. Unknown roads fail loudly; none are silently discarded.
    """
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict):
            if "chosen_design_units" in document:
                rows = document["chosen_design_units"]
            elif "chosen_links" in document:
                rows = document["chosen_links"]
            else:
                raise ValueError("Design JSON needs chosen_design_units or chosen_links")
        else:
            rows = document
    if not isinstance(rows, list):
        raise ValueError("Construction units must be a list")
    lookup = {pair: j for j, pair in enumerate(problem.units)}
    reverse_ids: dict[str, list[int]] = {}
    for node in problem.nodes:
        original = problem.originals[node]
        # A zone node and a road node can legitimately share an original ID.
        reverse_ids.setdefault(original, []).append(node)
    chosen = set()
    for row in rows:
        if isinstance(row, dict):
            if row.get("original_u") not in (None, "") and row.get("original_v") not in (None, ""):
                original_u, original_v = str(row["original_u"]), str(row["original_v"])
                numeric_u, numeric_v = row.get("numeric_u"), row.get("numeric_v")
                numeric_pair = None
                if numeric_u is not None and numeric_v is not None:
                    u0, v0 = int(numeric_u), int(numeric_v)
                    if (u0 in problem.nodes and v0 in problem.nodes
                            and problem.originals[u0] == original_u and problem.originals[v0] == original_v
                            and tuple(sorted((u0, v0))) in lookup):
                        numeric_pair = (u0, v0)
                if numeric_pair is not None:
                    u, v = numeric_pair
                else:
                    candidates = {tuple(sorted((u0, v0)))
                                  for u0 in reverse_ids.get(original_u, [])
                                  for v0 in reverse_ids.get(original_v, [])
                                  if tuple(sorted((u0, v0))) in lookup}
                    if len(candidates) != 1:
                        raise ValueError(f"Existing design has unknown or ambiguous original road IDs: {row}")
                    u, v = next(iter(candidates))
            else:
                u, v = int(row["numeric_u"]), int(row["numeric_v"])
        else:
            u, v = map(int, row)
        pair = tuple(sorted((u, v)))
        if pair not in lookup:
            raise ValueError(f"Existing design road {pair} is absent from this network")
        chosen.add(lookup[pair])
    return chosen


def load_certificate(path: Path, problem: Problem, chosen: set[int]) -> dict | None:
    """Read a feasible routing hint from old/new JSON, checking it on current data.

    This is only a MIP start. Unverified or renumbered paths are omitted; they
    never change the mathematical model or the independent final validation.
    """
    if path.suffix.lower() != ".json":
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return None
    source_stats, source_paths = document.get("od_stats", {}), document.get("od_paths", {})
    arc_lookup = {(arc.u, arc.v): arc for arc in problem.arcs}
    stats, paths = {}, {}
    for trip in problem.trips:
        key = f"{trip.origin}->{trip.destination}"
        source = source_stats.get(key)
        if not isinstance(source, dict):
            continue
        covered = bool(source.get("covered", source.get("coverage_required", source.get("z_value", 0) > 0.5)))
        if not covered:
            stats[key], paths[key] = {"covered": False}, []
            continue
        route, node, visited = [], trip.origin, {trip.origin}
        valid = True
        for row in source_paths.get(key, []):
            if isinstance(row, dict):
                u, v = int(row["numeric_u"]), int(row["numeric_v"])
                if (row.get("original_u") is not None and problem.originals.get(u) != str(row["original_u"])) or (
                    row.get("original_v") is not None and problem.originals.get(v) != str(row["original_v"])):
                    valid = False
                    break
            else:
                u, v = map(int, row)
            if u != node or v in visited or (u, v) not in arc_lookup:
                valid = False
                break
            route.append([u, v])
            node = v
            visited.add(v)
        if not valid or node != trip.destination:
            continue
        length = sum(arc_lookup[tuple(pair)].length for pair in route)
        uncovered = sum(arc_lookup[tuple(pair)].length for pair in route
                        if arc_lookup[tuple(pair)].unit not in chosen)
        if length > (1 + problem.epsilon) * trip.shortest + 1e-5 or uncovered > problem.uncovered_share * trip.shortest + 1e-5:
            continue
        stats[key], paths[key] = {"covered": True}, route
    if not stats:
        return None
    return {"solution_available": True, "od_stats": stats, "od_paths": paths,
            "chosen_design_units": [{"numeric_u": problem.units[j][0], "numeric_v": problem.units[j][1]}
                                    for j in sorted(chosen)]}
