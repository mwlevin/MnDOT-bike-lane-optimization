from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    gp = None
    GRB = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

Arc = Tuple[int, int]
OD = Tuple[int, int]

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NETWORK_DIR = os.path.join("data", "cropped_subgraph")
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "gurobi_cropped_subgraph")
DEFAULT_ZONE_GEOMETRIES_PATH = os.path.join(DEFAULT_NETWORK_DIR, "OD_geometries.csv")


def resolve_project_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(PACKAGE_ROOT, expanded)


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


class TNTPNetworkData:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.net_path = self._resolve_file(["net.tntp"])
        self.trips_path = self._resolve_file(["trips.tntp"])
        self.node_path = self._resolve_file(["node.tntp"], required=False)

        self.nodes: List[int] = []
        self.arcs: List[Arc] = []
        self.arc_len: Dict[int, float] = {}
        self.arc_cost: Dict[int, float] = {}
        self.arc_type: Dict[int, int] = {}
        self.arc_to_idx: Dict[Arc, int] = {}
        self.idx_to_arc: Dict[int, Arc] = {}
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.od_demand: Dict[OD, float] = {}
        self.G = nx.DiGraph()
        self.out_arcs: Dict[int, List[int]] = {}
        self.in_arcs: Dict[int, List[int]] = {}

        self._parse_network()
        self._parse_trips()
        if self.node_path is not None and os.path.exists(self.node_path):
            self._parse_nodes()

    def _resolve_file(self, candidates: List[str], required: bool = True) -> Optional[str]:
        for name in candidates:
            path = os.path.join(self.input_dir, name)
            if os.path.exists(path):
                return path
        if required:
            raise FileNotFoundError(f"Cannot find any of {candidates} in {self.input_dir}")
        return None

    def _parse_network(self) -> None:
        lines = _read_lines(self.net_path)
        data_started = False
        arcs: List[Tuple[int, int, float, int]] = []
        nodes_set = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("<END OF METADATA>"):
                data_started = True
                continue
            if not data_started or line.startswith("~") or ";" not in line:
                continue
            line = line.replace(";", " ")
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 10:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                length = float(parts[3])
                link_type = int(float(parts[9]))
            except Exception:
                continue
            arcs.append((u, v, length, link_type))
            nodes_set.add(u)
            nodes_set.add(v)

        self.nodes = sorted(nodes_set)
        self.out_arcs = {i: [] for i in self.nodes}
        self.in_arcs = {i: [] for i in self.nodes}
        for idx, (u, v, length, link_type) in enumerate(arcs):
            arc = (u, v)
            self.arcs.append(arc)
            self.arc_to_idx[arc] = idx
            self.idx_to_arc[idx] = arc
            self.arc_len[idx] = length
            self.arc_cost[idx] = length
            self.arc_type[idx] = link_type
            self.G.add_edge(u, v, idx=idx, length=length)
            self.out_arcs[u].append(idx)
            self.in_arcs[v].append(idx)

    def _parse_nodes(self) -> None:
        lines = _read_lines(self.node_path)
        for line in lines:
            line = line.strip().replace(";", "")
            if not line or line.lower().startswith("node"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            try:
                n = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except Exception:
                continue
            self.pos[n] = (x, y)

    def _parse_trips(self) -> None:
        lines = _read_lines(self.trips_path)
        origin = None
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("<"):
                continue
            if raw.startswith("Origin"):
                matches = re.findall(r"Origin\s+(\d+)", raw)
                if matches:
                    origin = int(matches[0])
                continue
            if origin is None:
                continue
            for d_str, q_str in re.findall(r"(\d+)\s*:\s*([0-9.]+)", raw):
                dest = int(d_str)
                demand = float(q_str)
                if origin != dest and demand > 1e-6:
                    self.od_demand[(origin, dest)] = demand


def select_ods(all_od_demand: Dict[OD, float], od_limit: int, seed: int = 42) -> Dict[OD, float]:
    ods = list(all_od_demand.keys())
    if od_limit <= 0 or od_limit >= len(ods):
        return dict(all_od_demand)
    rng = random.Random(seed)
    chosen = rng.sample(ods, od_limit)
    return {od: all_od_demand[od] for od in chosen}


def shortest_path_length(data: TNTPNetworkData, od: OD) -> float:
    o, d = od
    return nx.shortest_path_length(data.G, o, d, weight="length")


def normalize_demand_share(value: float) -> float:
    share = float(value)
    if share > 1.0 + 1e-12:
        if share <= 100.0 + 1e-12:
            share /= 100.0
        else:
            raise ValueError(f"covered demand share must be in [0, 1] or [0, 100], got {value}.")
    if share < -1e-12 or share > 1.0 + 1e-12:
        raise ValueError(f"covered demand share must be in [0, 1], got {share}.")
    return min(max(share, 0.0), 1.0)




def compute_selected_od_io_stats(od_keys: List[OD]) -> Dict[str, object]:
    """Return origin/destination balance statistics for a selected OD set."""
    origins: Dict[int, int] = {}
    destinations: Dict[int, int] = {}
    for o, d in od_keys:
        origins[o] = origins.get(o, 0) + 1
        destinations[d] = destinations.get(d, 0) + 1

    selected_nodes = sorted(set(origins) | set(destinations))
    balanced_nodes = [node for node in selected_nodes if origins.get(node, 0) > 0 and destinations.get(node, 0) > 0]
    origin_only_nodes = [node for node in selected_nodes if origins.get(node, 0) > 0 and destinations.get(node, 0) == 0]
    destination_only_nodes = [node for node in selected_nodes if destinations.get(node, 0) > 0 and origins.get(node, 0) == 0]
    missing_origin_flow_nodes = [node for node in selected_nodes if origins.get(node, 0) == 0]
    missing_destination_flow_nodes = [node for node in selected_nodes if destinations.get(node, 0) == 0]

    return {
        "selected_od_nodes": selected_nodes,
        "balanced_od_nodes": balanced_nodes,
        "origin_only_nodes": origin_only_nodes,
        "destination_only_nodes": destination_only_nodes,
        "missing_origin_flow_nodes": missing_origin_flow_nodes,
        "missing_destination_flow_nodes": missing_destination_flow_nodes,
        "origin_counts": origins,
        "destination_counts": destinations,
        "all_selected_nodes_have_origin_and_destination_flow": (
            len(origin_only_nodes) == 0 and len(destination_only_nodes) == 0
        ),
    }


def select_ods_with_balanced_io(
    od_demand: Dict[OD, float],
    od_limit: int,
    *,
    seed: int = 42,
    require_balanced_io: bool = True,
) -> Tuple[Dict[OD, float], Dict[str, object]]:
    """Select OD pairs while trying to keep each selected OD node balanced.

    When require_balanced_io=True, the sampler first selects OD pairs so that
    each targeted OD node has at least one outgoing selected OD demand and one
    incoming selected OD demand. This prevents a random OD subset from leaving
    some OD nodes only as origins or only as destinations. The rule is exact for
    complete directed OD tables when od_limit is at least the number of OD nodes;
    otherwise the function returns the best feasible balanced subset it can build.
    """
    candidates = {
        od: float(demand)
        for od, demand in od_demand.items()
        if od[0] != od[1] and float(demand) > 1e-12
    }
    if not candidates:
        return {}, {
            "balanced_sampling_requested": require_balanced_io,
            "warning": "No positive non-self OD demand is available.",
        }

    if od_limit is None or od_limit <= 0 or od_limit >= len(candidates):
        selected = dict(sorted(candidates.items()))
        stats = compute_selected_od_io_stats(list(selected.keys()))
        stats.update({
            "balanced_sampling_requested": require_balanced_io,
            "target_od_limit": od_limit,
            "selected_od_count": len(selected),
            "candidate_od_count": len(candidates),
            "warning": "All available OD pairs were selected." if od_limit is None or od_limit <= 0 or od_limit >= len(candidates) else "",
        })
        return selected, stats

    if not require_balanced_io:
        selected = select_ods(candidates, od_limit, seed=seed)
        stats = compute_selected_od_io_stats(list(selected.keys()))
        stats.update({
            "balanced_sampling_requested": False,
            "target_od_limit": od_limit,
            "selected_od_count": len(selected),
            "candidate_od_count": len(candidates),
            "warning": "Balanced OD sampling was disabled.",
        })
        return selected, stats

    rng = random.Random(seed)
    candidate_nodes = sorted(set(o for o, _ in candidates) | set(d for _, d in candidates))
    nodes_with_in_and_out = [
        node
        for node in candidate_nodes
        if any(o == node for o, _ in candidates) and any(d == node for _, d in candidates)
    ]

    if len(nodes_with_in_and_out) < 2 or od_limit < 2:
        selected = select_ods(candidates, od_limit, seed=seed)
        stats = compute_selected_od_io_stats(list(selected.keys()))
        stats.update({
            "balanced_sampling_requested": True,
            "target_od_limit": od_limit,
            "selected_od_count": len(selected),
            "candidate_od_count": len(candidates),
            "warning": "The requested OD limit or available OD graph is too small to guarantee balanced OD-node flow.",
        })
        return selected, stats

    target_node_count = min(len(nodes_with_in_and_out), max(2, od_limit))
    shuffled_nodes = nodes_with_in_and_out[:]
    rng.shuffle(shuffled_nodes)
    target_nodes = sorted(shuffled_nodes[:target_node_count])
    target_node_set = set(target_nodes)
    internal_candidates = [
        od for od in candidates
        if od[0] in target_node_set and od[1] in target_node_set and od[0] != od[1]
    ]

    selected_set: Set[OD] = set()
    requirements = {("out", node) for node in target_nodes} | {("in", node) for node in target_nodes}

    # Prefer a random directed cycle through the target nodes. In complete OD
    # tables, this immediately gives every OD node one outgoing and one incoming
    # selected demand.
    cycle_nodes = target_nodes[:]
    rng.shuffle(cycle_nodes)
    for idx, u in enumerate(cycle_nodes):
        if len(selected_set) >= od_limit:
            break
        v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
        od = (u, v)
        if od in candidates:
            selected_set.add(od)
            requirements.discard(("out", u))
            requirements.discard(("in", v))

    # Greedy set-cover repair for sparse OD tables or missing cycle arcs.
    while requirements and len(selected_set) < od_limit:
        scored: List[Tuple[int, OD]] = []
        for od in internal_candidates:
            if od in selected_set:
                continue
            o, d = od
            gain = int(("out", o) in requirements) + int(("in", d) in requirements)
            if gain > 0:
                scored.append((gain, od))
        if not scored:
            break
        best_gain = max(gain for gain, _ in scored)
        best_ods = [od for gain, od in scored if gain == best_gain]
        od = rng.choice(best_ods)
        selected_set.add(od)
        requirements.discard(("out", od[0]))
        requirements.discard(("in", od[1]))

    # Fill the remaining budget using OD pairs internal to the balanced target
    # node set. This preserves the selected-node balance once all requirements
    # are satisfied.
    remaining_internal = [od for od in internal_candidates if od not in selected_set]
    rng.shuffle(remaining_internal)
    for od in remaining_internal:
        if len(selected_set) >= od_limit:
            break
        selected_set.add(od)

    selected = {od: candidates[od] for od in sorted(selected_set)}
    stats = compute_selected_od_io_stats(list(selected.keys()))
    warning = ""
    if len(selected) < od_limit:
        warning = (
            "Balanced OD sampling selected fewer OD pairs than requested because adding extra OD pairs "
            "would introduce OD nodes without both incoming and outgoing selected demand."
        )
    if not stats["all_selected_nodes_have_origin_and_destination_flow"]:
        warning = (
            "Balanced OD sampling could not fully guarantee incoming and outgoing selected demand for every selected OD node. "
            + warning
        ).strip()

    stats.update({
        "balanced_sampling_requested": True,
        "target_od_limit": od_limit,
        "selected_od_count": len(selected),
        "candidate_od_count": len(candidates),
        "target_balanced_node_count": len(target_nodes),
        "warning": warning,
    })
    return selected, stats

def read_zone_count(net_path: str) -> int:
    with open(net_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.match(r"\s*<NUMBER OF ZONES>\s+(\d+)", line)
            if match:
                return int(match.group(1))
    raise ValueError(f"Cannot read <NUMBER OF ZONES> from {net_path}")


def load_original_id_map(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["numeric_id"])] = row["original_id"]
    return mapping


def parse_wkt_polygon_points(wkt: str) -> List[Tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in re.findall(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", wkt)
    ]


def load_zone_geometries(path: str) -> Dict[str, Dict[str, object]]:
    if not path or not os.path.exists(path):
        return {}
    geometries: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points = parse_wkt_polygon_points(row.get("geometry", ""))
            if points:
                geometries[row["id"]] = {
                    "name": row.get("name", ""),
                    "polygon": points,
                }
    return geometries


def load_zone_centers_by_original_id(network_dir: str) -> Dict[str, Tuple[float, float]]:
    centers: Dict[str, Tuple[float, float]] = {}
    for filename in ("kept_od_zone_nearest_road_node.csv", "od_zone_to_nearest_road_node.csv"):
        path = os.path.join(network_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                original_id = row.get("old_zone_original_id", "")
                x = row.get("old_zone_x", "")
                y = row.get("old_zone_y", "")
                if original_id and x and y:
                    centers[original_id] = (float(x), float(y))
    return centers


def load_boundary_polygon(network_dir: str) -> List[Tuple[float, float]]:
    path = os.path.join(network_dir, "boundary_nodes.csv")
    if not os.path.exists(path):
        return []
    points: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("x") and row.get("y"):
                points.append((float(row["x"]), float(row["y"])))
    if len(points) < 3:
        return points
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    return sorted(points, key=lambda point: math.atan2(point[1] - cy, point[0] - cx))


def fallback_polygon_centroid(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    # This is only a display fallback. The explicit TAZ center file is used when available.
    usable = points[:-1] if len(points) > 1 and points[0] == points[-1] else points
    if not usable:
        usable = points
    return (
        sum(x for x, _ in usable) / len(usable),
        sum(y for _, y in usable) / len(usable),
    )


def shortest_path_arc_indices(data: TNTPNetworkData, od: OD) -> Tuple[List[int], List[int]]:
    node_path = nx.shortest_path(data.G, od[0], od[1], weight="length")
    arc_indices: List[int] = []
    for u, v in zip(node_path, node_path[1:]):
        arc_indices.append(data.arc_to_idx[(u, v)])
    return arc_indices, node_path

def build_undirected_design_units(
    data: TNTPNetworkData,
) -> Tuple[List[Arc], Dict[int, int], Dict[int, List[int]], Dict[int, float]]:
    """Build one construction variable for each opposite-direction physical link pair.

    The traffic/path variables remain directed arc variables. The design variable is
    indexed by the unordered node pair {u, v}; therefore arcs (u, v) and (v, u)
    share the same construction decision when both directions exist. If a one-way
    arc appears in the data, it is treated as a single-arc design unit.

    Returns
    -------
    design_units:
        List of canonical unordered node pairs.
    arc_to_design_unit:
        Map from directed arc index to its undirected design-unit index.
    design_unit_arcs:
        Map from design-unit index to all directed arc indices represented by it.
    design_unit_cost:
        One-time construction cost for the undirected design unit. When two
        directions have different arc costs, the larger value is used as a
        conservative physical-link construction cost.
    """
    unit_to_arcs: Dict[Arc, List[int]] = {}
    arc_to_key: Dict[int, Arc] = {}
    for a, (u, v) in enumerate(data.arcs):
        key = (u, v) if u <= v else (v, u)
        unit_to_arcs.setdefault(key, []).append(a)
        arc_to_key[a] = key

    design_units = sorted(unit_to_arcs.keys())
    key_to_unit = {key: q for q, key in enumerate(design_units)}
    arc_to_design_unit = {a: key_to_unit[key] for a, key in arc_to_key.items()}
    design_unit_arcs = {key_to_unit[key]: sorted(arcs) for key, arcs in unit_to_arcs.items()}
    design_unit_cost = {
        q: max(float(data.arc_cost[a]) for a in arcs)
        for q, arcs in design_unit_arcs.items()
    }
    return design_units, arc_to_design_unit, design_unit_arcs, design_unit_cost



def compute_od_feasible_arcs(
    data: TNTPNetworkData,
    ods: List[OD],
    B: Dict[OD, float],
    uncovered_cap_if_required: Dict[OD, float],
    *,
    use_detour_pruning: bool = True,
    use_node_reachability_pruning: bool = True,
    use_source_sink_variable_pruning: bool = True,
    use_directed_subgraph_trimming: bool = True,
    drop_long_uncovered_arcs: bool = False,
    tolerance: float = 1e-9,
) -> Tuple[Dict[OD, List[int]], Dict[OD, Dict[str, int]]]:
    """Compute OD-specific arc sets for sparse link-based path variables.

    This function performs preprocessing before Gurobi variables are created.
    Any OD-arc pair removed here has no x_{od,a} or c_{od,a} variable in the
    optimization model.

    Exact source/sink pruning removes, for OD (o,d), all arcs entering the origin
    and all arcs leaving the destination:
        a=(u,v), v=o  or  u=d.

    Exact node-based reachability pruning keeps only nodes that can lie on at
    least one path satisfying the OD detour bound:
        dist(o, i) + dist(i, d) <= B_od.

    Exact detour pruning then keeps only arcs that can appear on at least one path
    satisfying the OD detour bound:
        dist(o, u) + length(u, v) + dist(v, d) <= B_od.

    Exact directed-subgraph trimming is applied after the above filters. It keeps
    only arcs that remain on some directed o-to-d path inside the already-pruned
    OD-specific subgraph.

    The optional long-uncovered-arc filter removes, for each OD, every arc with
    length greater than C_od = t * K_od. This is an aggressive speed-oriented
    restriction. It is not an exact dominance rule because such an arc could still
    be feasible when its undirected design unit is built. It is disabled by default.
    """
    no_filtering = (
        not use_detour_pruning
        and not use_node_reachability_pruning
        and not use_source_sink_variable_pruning
        and not use_directed_subgraph_trimming
        and not drop_long_uncovered_arcs
    )
    if no_filtering:
        all_arcs = list(range(len(data.arcs)))
        return (
            {od: all_arcs[:] for od in ods},
            {
                od: {
                    "total_arcs": len(all_arcs),
                    "kept_arcs": len(all_arcs),
                    "removed_by_source_sink": 0,
                    "removed_by_node_reachability": 0,
                    "removed_by_detour": 0,
                    "removed_by_directed_subgraph_trimming": 0,
                    "removed_by_long_uncovered": 0,
                    "removed_unreachable": 0,
                }
                for od in ods
            },
        )

    reverse_G = data.G.reverse(copy=True)
    feasible_arcs: Dict[OD, List[int]] = {}
    pruning_stats: Dict[OD, Dict[str, int]] = {}

    for od in ods:
        o, d = od
        dist_from_o = nx.single_source_dijkstra_path_length(data.G, o, weight="length")
        dist_to_d = nx.single_source_dijkstra_path_length(reverse_G, d, weight="length")

        keep: List[int] = []
        removed_by_source_sink = 0
        removed_by_node_reachability = 0
        removed_by_detour = 0
        removed_by_directed_subgraph_trimming = 0
        removed_by_long_uncovered = 0
        removed_unreachable = 0
        C = uncovered_cap_if_required[od]

        if use_node_reachability_pruning:
            usable_nodes = {
                node
                for node in data.nodes
                if node in dist_from_o
                and node in dist_to_d
                and float(dist_from_o[node]) + float(dist_to_d[node]) <= B[od] + tolerance
            }
        else:
            usable_nodes = set(data.nodes)

        for a, (u, v) in enumerate(data.arcs):
            ell = float(data.arc_len[a])

            if u not in dist_from_o or v not in dist_to_d:
                removed_unreachable += 1
                continue

            # Exact source/sink variable pruning: arcs entering the origin and
            # arcs leaving the destination are forced to zero by the source/sink
            # constraints, so they are not defined for this OD.
            if use_source_sink_variable_pruning and (v == o or u == d):
                removed_by_source_sink += 1
                continue

            if use_node_reachability_pruning and (u not in usable_nodes or v not in usable_nodes):
                removed_by_node_reachability += 1
                continue

            if use_detour_pruning:
                min_path_len_using_arc = float(dist_from_o[u]) + ell + float(dist_to_d[v])
                if min_path_len_using_arc > B[od] + tolerance:
                    removed_by_detour += 1
                    continue

            if drop_long_uncovered_arcs and ell > C + tolerance:
                removed_by_long_uncovered += 1
                continue

            keep.append(a)

        if use_directed_subgraph_trimming and keep:
            keep_set = set(keep)

            # Nodes reachable from the origin within the current OD-specific
            # directed subgraph.
            reachable_from_origin: Set[int] = set()
            stack = [o]
            while stack:
                node = stack.pop()
                if node in reachable_from_origin:
                    continue
                reachable_from_origin.add(node)
                for a in data.out_arcs.get(node, []):
                    if a not in keep_set:
                        continue
                    _, nxt = data.idx_to_arc[a]
                    if nxt not in reachable_from_origin:
                        stack.append(nxt)

            # Nodes that can reach the destination within the current OD-specific
            # directed subgraph.
            can_reach_destination: Set[int] = set()
            stack = [d]
            while stack:
                node = stack.pop()
                if node in can_reach_destination:
                    continue
                can_reach_destination.add(node)
                for a in data.in_arcs.get(node, []):
                    if a not in keep_set:
                        continue
                    prev, _ = data.idx_to_arc[a]
                    if prev not in can_reach_destination:
                        stack.append(prev)

            trimmed_keep: List[int] = []
            for a in keep:
                u, v = data.idx_to_arc[a]
                if u in reachable_from_origin and v in can_reach_destination:
                    trimmed_keep.append(a)
                else:
                    removed_by_directed_subgraph_trimming += 1
            keep = trimmed_keep

        feasible_arcs[od] = keep
        pruning_stats[od] = {
            "total_arcs": len(data.arcs),
            "kept_arcs": len(keep),
            "removed_by_source_sink": removed_by_source_sink,
            "removed_by_node_reachability": removed_by_node_reachability,
            "removed_by_detour": removed_by_detour,
            "removed_by_directed_subgraph_trimming": removed_by_directed_subgraph_trimming,
            "removed_by_long_uncovered": removed_by_long_uncovered,
            "removed_unreachable": removed_unreachable,
        }

        if not keep:
            raise ValueError(
                f"OD {o}->{d} has no remaining arcs after OD-specific pruning. "
                "Try relaxing epsilon/t or disabling OD-specific pruning."
            )

    return feasible_arcs, pruning_stats

def extract_path_for_od_sparse(
    data: TNTPNetworkData,
    x: Dict[Tuple[OD, int], gp.Var],
    od: OD,
    feasible_arc_set: Set[int],
    *,
    tolerance: float = 0.5,
) -> List[int]:
    """Extract an OD path from sparse OD-arc variables.

    The imported extractor assumes that x[(od, a)] exists for every arc. After
    OD-specific pruning, x exists only for feasible_arcs[od], so this local
    extractor filters outgoing arcs accordingly.
    """
    origin, destination = od
    path: List[int] = []
    current = origin
    visited_nodes = {origin}
    used_arcs: Set[int] = set()

    while current != destination:
        candidates = []
        for a in data.out_arcs.get(current, []):
            if a not in feasible_arc_set or a in used_arcs:
                continue
            var = x.get((od, a))
            if var is not None and var.X > tolerance:
                candidates.append(a)

        if not candidates:
            break

        # In an integer solution the out-degree constraint should leave at most
        # one selected outgoing arc. If numerical noise leaves more than one,
        # follow the largest x value.
        a = max(candidates, key=lambda cur: x[(od, cur)].X)
        path.append(a)
        used_arcs.add(a)
        _, next_node = data.idx_to_arc[a]

        if next_node in visited_nodes and next_node != destination:
            break

        visited_nodes.add(next_node)
        current = next_node

    return path




def compute_path_design_metrics(
    data: TNTPNetworkData,
    path_arc_indices: List[int],
    chosen_design_unit_set: Set[int],
    arc_to_design_unit: Dict[int, int],
    shortest_length: float,
) -> Dict[str, float]:
    """Compute display metrics for a path under a selected bicycle-lane design."""
    path_length = float(sum(data.arc_len[a] for a in path_arc_indices))
    uncovered_length = float(
        sum(
            data.arc_len[a]
            for a in path_arc_indices
            if arc_to_design_unit[a] not in chosen_design_unit_set
        )
    )
    covered_length = max(0.0, path_length - uncovered_length)
    cover_ratio = covered_length / path_length if path_length > 1e-12 else 0.0
    length_to_shortest_ratio = path_length / shortest_length if shortest_length > 1e-12 else math.nan
    return {
        "path_length": path_length,
        "uncovered_length": uncovered_length,
        "covered_length": covered_length,
        "cover_ratio": cover_ratio,
        "length_to_shortest_ratio": length_to_shortest_ratio,
    }


def find_min_uncovered_reference_path(
    data: TNTPNetworkData,
    od: OD,
    chosen_design_unit_set: Set[int],
    arc_to_design_unit: Dict[int, int],
    max_path_length: float,
    *,
    tolerance: float = 1e-9,
    max_labels_per_node: int = 5000,
) -> List[int]:
    """Find a reference path minimizing non-bicycle-lane length subject to a length bound.

    This path is used only for post-solution visualization of ODs not selected by
    the optimization model. It minimizes uncovered length subject to total path
    length not exceeding B_od = (1 + epsilon) K_od. Ties are broken by shorter
    total length.
    """
    origin, destination = od
    counter = 0
    labels: Dict[int, Tuple[int, float, float, int | None, int | None]] = {}
    labels_by_node: Dict[int, List[Tuple[float, float, int]]] = {origin: [(0.0, 0.0, 0)]}
    labels[0] = (origin, 0.0, 0.0, None, None)
    heap: List[Tuple[float, float, int, int]] = [(0.0, 0.0, 0, origin)]

    def is_dominated(node: int, new_len: float, new_unc: float) -> bool:
        for old_len, old_unc, _ in labels_by_node.get(node, []):
            if old_len <= new_len + tolerance and old_unc <= new_unc + tolerance:
                return True
        return False

    def add_label(node: int, new_len: float, new_unc: float, pred_id: int, pred_arc: int) -> None:
        nonlocal counter
        if is_dominated(node, new_len, new_unc):
            return
        current = labels_by_node.setdefault(node, [])
        current[:] = [
            item for item in current
            if not (new_len <= item[0] + tolerance and new_unc <= item[1] + tolerance)
        ]
        counter += 1
        label_id = counter
        labels[label_id] = (node, new_len, new_unc, pred_id, pred_arc)
        current.append((new_len, new_unc, label_id))
        if len(current) > max_labels_per_node:
            current.sort(key=lambda item: (item[1], item[0]))
            del current[max_labels_per_node:]
        heapq.heappush(heap, (new_unc, new_len, label_id, node))

    best_label_id: int | None = None
    while heap:
        unc, length, label_id, node = heapq.heappop(heap)
        stored = labels.get(label_id)
        if stored is None:
            continue
        stored_node, stored_len, stored_unc, _, _ = stored
        if stored_node != node or abs(stored_len - length) > 1e-7 or abs(stored_unc - unc) > 1e-7:
            continue
        if node == destination:
            best_label_id = label_id
            break
        for a in data.out_arcs.get(node, []):
            _, next_node = data.idx_to_arc[a]
            ell = float(data.arc_len[a])
            new_len = length + ell
            if new_len > max_path_length + tolerance:
                continue
            new_unc = unc + (0.0 if arc_to_design_unit[a] in chosen_design_unit_set else ell)
            add_label(next_node, new_len, new_unc, label_id, a)

    if best_label_id is None:
        try:
            path_arcs, _ = shortest_path_arc_indices(data, od)
            return path_arcs
        except Exception:
            return []

    path: List[int] = []
    cur = best_label_id
    while cur is not None:
        _node, _length, _unc, pred, pred_arc = labels[cur]
        if pred_arc is not None:
            path.append(pred_arc)
        cur = pred
    path.reverse()
    return path


def compute_visual_positions(data: TNTPNetworkData, zone_count: int) -> Dict[int, Tuple[float, float]]:
    pos = dict(data.pos)
    through_nodes = {node for node in data.nodes if node > zone_count}

    for zone in range(1, zone_count + 1):
        neighbors: List[int] = []
        for arc_index in data.out_arcs.get(zone, []):
            if data.arc_type.get(arc_index, 1) == 1:
                continue
            _, v = data.idx_to_arc[arc_index]
            if v in through_nodes and v in pos:
                neighbors.append(v)
        for arc_index in data.in_arcs.get(zone, []):
            if data.arc_type.get(arc_index, 1) == 1:
                continue
            u, _ = data.idx_to_arc[arc_index]
            if u in through_nodes and u in pos:
                neighbors.append(u)
        unique_neighbors = sorted(set(neighbors))
        if unique_neighbors:
            xs = [pos[node][0] for node in unique_neighbors]
            ys = [pos[node][1] for node in unique_neighbors]
            pos[zone] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return pos


def sanitize_result(result: Dict[str, object], original_id_map: Dict[int, str]) -> Dict[str, object]:
    clean = {
        "status": result["status"],
        "status_name": result["status_name"],
        "runtime": result["runtime"],
        "objective": result["objective"],
        "best_bound": result["best_bound"],
        "gap": result["gap"],
        "num_ods": result["num_ods"],
        "num_arcs": result["num_arcs"],
        "num_nodes": result["num_nodes"],
        "chosen_links": [
            {
                "numeric_u": u,
                "numeric_v": v,
                "original_u": original_id_map.get(u, str(u)),
                "original_v": original_id_map.get(v, str(v)),
            }
            for u, v in result.get("chosen_links", [])
        ],
    }
    if "chosen_design_units" in result:
        clean["chosen_design_units"] = [
            {
                "numeric_u": u,
                "numeric_v": v,
                "original_u": original_id_map.get(u, str(u)),
                "original_v": original_id_map.get(v, str(v)),
            }
            for u, v in result.get("chosen_design_units", [])
        ]
    if "warm_start_design_units" in result:
        clean["warm_start_design_units"] = [
            {
                "numeric_u": u,
                "numeric_v": v,
                "original_u": original_id_map.get(u, str(u)),
                "original_v": original_id_map.get(v, str(v)),
            }
            for u, v in result.get("warm_start_design_units", [])
        ]

    for key in (
        "total_demand",
        "demand_target",
        "demand_target_share",
        "covered_demand",
        "covered_demand_share",
        "covered_od_count",
        "chosen_design_unit_count",
        "num_design_units",
        "num_od_arc_variables_full",
        "num_od_arc_variables_after_pruning",
        "od_arc_pruning_removed_by_source_sink",
        "use_source_sink_variable_pruning",
        "od_arc_pruning_removed_by_node_reachability",
        "use_node_reachability_pruning",
        "od_arc_pruning_removed_by_detour",
        "od_arc_pruning_removed_by_directed_subgraph_trimming",
        "use_directed_subgraph_trimming",
        "num_design_units_full",
        "num_design_units_after_pruning",
        "design_units_removed_unused",
        "use_unused_design_unit_pruning",
        "od_arc_pruning_removed_by_long_uncovered",
        "od_arc_pruning_removed_unreachable",
        "origin_destination_strengthening_count",
        "opposite_direction_x_cut_count",
        "single_arc_uncovered_implied_cut_count",
        "minimum_covered_length_cut_count",
        "single_arc_detour_implied_cut_count",
        "balanced_od_sampling",
        "uncovered_reference_path_count",
        "uncovered_reference_path_failure_count",
    ):
        if key in result:
            clean[key] = result[key]
    for key in ("od_nodes", "od_origin_nodes", "od_destination_nodes"):
        if key in result:
            clean[key] = result[key]
    if "od_sampling_stats" in result:
        clean["od_sampling_stats"] = result["od_sampling_stats"]
    for key in ("covered_ods", "relaxed_ods", "warm_start_covered_ods"):
        if key in result:
            clean[key] = [f"{o}->{d}" for (o, d) in result[key]]
    if "od_paths" in result:
        clean["od_paths"] = {
            f"{o}->{d}": [
                {
                    "numeric_u": u,
                    "numeric_v": v,
                    "original_u": original_id_map.get(u, str(u)),
                    "original_v": original_id_map.get(v, str(v)),
                }
                for u, v in arcs
            ]
            for (o, d), arcs in result["od_paths"].items()
        }
    if "od_stats" in result:
        clean["od_stats"] = {f"{o}->{d}": stats for (o, d), stats in result["od_stats"].items()}
    return clean


INTERACTIVE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Direct Bicycle-Lane Interactive Viewer</title>
  <style>
    :root {
      --bg: #f3f7fb;
      --panel: rgba(255, 255, 255, 0.92);
      --border: rgba(15, 23, 42, 0.12);
      --text: #0f172a;
      --muted: #475569;
      --road: #cbd5e1;
      --connector: #f59e0b;
      --bike: #16a34a;
      --through: #94a3b8;
      --od-fill: #f59e0b;
      --od-stroke: #7c2d12;
      --taz-fill: rgba(251, 191, 36, 0.12);
      --taz-stroke: rgba(146, 64, 14, 0.96);
      --taz-center: #ef4444;
      --boundary: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, #eaf1f8 0%, #f8fbfe 100%);
      color: var(--text);
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      padding: 18px 18px 14px;
      background: var(--panel);
      border-right: 1px solid var(--border);
      backdrop-filter: blur(8px);
      overflow-y: auto;
    }
    .main {
      position: relative;
      min-width: 0;
      min-height: 100vh;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 24px;
      line-height: 1.2;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 16px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat {
      padding: 12px 12px 10px;
      border-radius: 12px;
      background: #f8fbff;
      border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .stat .label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
      margin-bottom: 5px;
    }
    .stat .value {
      font-size: 20px;
      font-weight: 700;
    }
    .controls {
      margin: 14px 0 18px;
      padding: 14px;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
    }
    .controls h2, .legend h2, .details h2 {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #334155;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
      font-size: 14px;
      color: #1e293b;
    }
    .check:last-child { margin-bottom: 0; }
    .field {
      display: block;
      margin-top: 12px;
    }
    .field-label {
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      font-weight: 600;
      color: #334155;
    }
    select {
      width: 100%;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: #f8fbff;
      color: #0f172a;
      font-size: 13px;
    }
    .hint {
      margin-top: 8px;
      font-size: 12px;
      color: #64748b;
      line-height: 1.45;
    }
    button {
      width: 100%;
      margin-top: 12px;
      padding: 10px 12px;
      border: 0;
      border-radius: 12px;
      background: #0f172a;
      color: white;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #1e293b; }
    .legend, .details {
      margin-bottom: 16px;
      padding: 14px;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
      font-size: 14px;
    }
    .legend-row:last-child { margin-bottom: 0; }
    .swatch-line {
      width: 34px;
      border-top: 4px solid;
    }
    .swatch-dot {
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid;
    }
    .swatch-area {
      width: 34px;
      height: 16px;
      border: 2px solid var(--taz-stroke);
      background: var(--taz-fill);
      border-radius: 4px;
    }
    .swatch-boundary {
      width: 34px;
      height: 16px;
      border: 3px dashed var(--boundary);
      background: white;
      border-radius: 4px;
    }
    .details {
      font-size: 14px;
      color: #1e293b;
      line-height: 1.5;
      min-height: 160px;
    }
    .details .muted {
      color: #64748b;
    }
    .path-box {
      max-height: 220px;
      overflow-y: auto;
      padding: 10px 12px;
      border-radius: 10px;
      background: #f8fbff;
      border: 1px solid rgba(148, 163, 184, 0.18);
      white-space: normal;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .canvas-wrap {
      position: absolute;
      inset: 0;
      padding: 16px;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      border-radius: 18px;
      background: white;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    }
    .tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      max-width: 280px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.92);
      color: white;
      font-size: 13px;
      line-height: 1.45;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.28);
      z-index: 10;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .main { min-height: 70vh; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>Direct Bicycle-Lane Viewer</h1>
      <div class="subtitle">OD nodes can be displayed separately from ordinary road nodes.</div>
      <div class="stats" id="stats"></div>
      <div class="controls">
        <h2>Layers</h2>
        <label class="check"><input type="checkbox" id="showBoundary" checked> Crop boundary</label>
        <label class="check"><input type="checkbox" id="showTAZ" checked> TAZ areas</label>
        <label class="check"><input type="checkbox" id="showTAZCenters" checked> TAZ centers</label>
        <label class="check"><input type="checkbox" id="showSnapping" checked> TAZ-to-road OD links</label>
        <label class="check"><input type="checkbox" id="showRoad" checked> Road network</label>
        <label class="check"><input type="checkbox" id="showConnectors" checked> Connectors</label>
        <label class="check"><input type="checkbox" id="showBike" checked> Chosen bicycle lanes</label>
        <label class="check"><input type="checkbox" id="showCoveredPaths" checked> Covered OD paths</label>
        <label class="check"><input type="checkbox" id="showRelaxedPaths" checked> Uncovered OD reference paths</label>
        <label class="check"><input type="checkbox" id="showRoadNodes"> Road nodes</label>
        <label class="check"><input type="checkbox" id="showODNodes" checked> OD nodes</label>
        <label class="field">
          <span class="field-label">Focus OD</span>
          <select id="odSelect"></select>
        </label>
        <div class="hint" id="odSummary"></div>
        <button id="fitBtn" type="button">Fit To View</button>
      </div>
      <div class="legend">
        <h2>Legend</h2>
        <div class="legend-row"><span class="swatch-boundary"></span><span>Crop boundary</span></div>
        <div class="legend-row"><span class="swatch-area"></span><span>TAZ area</span></div>
        <div class="legend-row"><span class="swatch-dot" style="background: var(--taz-center); border-color:#7f1d1d"></span><span>TAZ center</span></div>
        <div class="legend-row"><span class="swatch-line" style="border-color: var(--road)"></span><span>Road edge</span></div>
        <div class="legend-row"><span class="swatch-line" style="border-color: var(--connector)"></span><span>Connector</span></div>
        <div class="legend-row"><span class="swatch-line" style="border-color: var(--bike)"></span><span>Chosen bicycle lane</span></div>
        <div class="legend-row"><span class="swatch-line" style="border-color: #2563eb"></span><span>Covered OD optimized path</span></div>
        <div class="legend-row"><span class="swatch-line" style="border-color: #dc2626"></span><span>Uncovered OD reference path</span></div>
        <div class="legend-row"><span class="swatch-dot" style="background: var(--od-fill); border-color: var(--od-stroke)"></span><span>OD node</span></div>
      </div>
      <div class="details" id="odDetails">
        <h2>Selected OD</h2>
        <div class="muted">Choose an OD from the selector or click an OD path on the map.</div>
      </div>
      <div class="details" id="details">
        <h2>Hover Details</h2>
        <div class="muted">Move the cursor over an edge or node to inspect it.</div>
      </div>
    </aside>
    <main class="main">
      <div class="canvas-wrap">
        <canvas id="canvas"></canvas>
        <div class="tooltip" id="tooltip"></div>
      </div>
    </main>
  </div>
  <script>
    const DATA = __DATA_JSON__;
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const details = document.getElementById("details");
    const odDetails = document.getElementById("odDetails");
    const stats = document.getElementById("stats");
    const odSelect = document.getElementById("odSelect");
    const odSummary = document.getElementById("odSummary");
    const ALL_ODS = "__ALL__";
    const controls = {
      boundary: document.getElementById("showBoundary"),
      taz: document.getElementById("showTAZ"),
      tazCenters: document.getElementById("showTAZCenters"),
      snapping: document.getElementById("showSnapping"),
      road: document.getElementById("showRoad"),
      connectors: document.getElementById("showConnectors"),
      bike: document.getElementById("showBike"),
      coveredPaths: document.getElementById("showCoveredPaths"),
      relaxedPaths: document.getElementById("showRelaxedPaths"),
      roadNodes: document.getElementById("showRoadNodes"),
      odNodes: document.getElementById("showODNodes"),
    };
    const pathMap = Object.fromEntries((DATA.paths || []).map((path) => [path.id, path]));

    let deviceScale = window.devicePixelRatio || 1;
    let viewport = { scale: 1, offsetX: 0, offsetY: 0 };
    let dragging = false;
    let dragStart = null;
    let hoverItem = null;

    function renderStats() {
      const s = DATA.summary;
      const entries = [
        ["Status", s.status_name],
        ["Objective", s.objective.toFixed(2)],
        ["Chosen Links", String(s.chosen_links)],
        ["Runtime (s)", s.runtime.toFixed(2)],
        ["Gap", (s.gap * 100).toFixed(2) + "%"],
        ["OD Count", String(s.num_ods)],
        ["Covered ODs", String(s.covered_od_count)],
        ["Uncovered OD references", String(s.relaxed_od_count)],
        ["TAZ Areas", String((DATA.zones || []).length)],
      ];
      if (Number.isFinite(s.target_share)) {
        entries.push(["Target Share", (s.target_share * 100).toFixed(1) + "%"]);
      }
      if (Number.isFinite(s.covered_share)) {
        entries.push(["Covered Share", (s.covered_share * 100).toFixed(1) + "%"]);
      }
      stats.innerHTML = entries.map(([k, v]) => `<div class="stat"><div class="label">${k}</div><div class="value">${v}</div></div>`).join("");
    }

    function renderOdSelector() {
      const options = [`<option value="${ALL_ODS}">All visible ODs</option>`];
      for (const path of DATA.paths || []) {
        const tag = path.coverage_required ? "[Covered]" : "[Uncovered]";
        const coverText = Number.isFinite(path.cover_ratio) ? ` | cover=${(path.cover_ratio * 100).toFixed(1)}%` : "";
        const lenRatioText = Number.isFinite(path.length_to_shortest_ratio) ? ` | len/K=${path.length_to_shortest_ratio.toFixed(3)}` : "";
        options.push(`<option value="${path.id}">${tag} ${path.id} | demand=${path.demand.toFixed(3)}${coverText}${lenRatioText}</option>`);
      }
      odSelect.innerHTML = options.join("");
      const s = DATA.summary;
      odSummary.textContent = `${s.covered_od_count} covered ODs and ${s.relaxed_od_count} uncovered OD references are available. Select one OD to isolate its path.`;
    }

    function pathNodeSequence(path) {
      if (!path || !path.segments || path.segments.length === 0) return [];
      const nodes = [path.segments[0][0]];
      for (const segment of path.segments) nodes.push(segment[1]);
      return nodes;
    }

    function renderSelectedOd() {
      const selected = selectedOdId();
      if (selected === ALL_ODS || !pathMap[selected]) {
        odDetails.innerHTML = '<h2>Selected OD</h2><div class="muted">Choose an OD from the selector or click an OD path on the map.</div>';
        return;
      }
      const path = pathMap[selected];
      const nodeSeq = pathNodeSequence(path);
      const numericSeq = nodeSeq.join(" -> ");
      const originalSeq = nodeSeq.map((nodeId) => {
        const node = DATA.nodeMap[nodeId];
        return node ? node.original_id : String(nodeId);
      }).join(" -> ");
      const kind = path.coverage_required ? "Covered OD" : "Uncovered OD reference";
      const coverPct = Number.isFinite(path.cover_ratio) ? (path.cover_ratio * 100).toFixed(2) + "%" : "N/A";
      const uncoveredPct = Number.isFinite(path.uncovered_ratio) ? (path.uncovered_ratio * 100).toFixed(2) + "%" : "N/A";
      const lengthRatio = Number.isFinite(path.length_to_shortest_ratio) ? path.length_to_shortest_ratio.toFixed(4) : "N/A";
      const sourceText = path.reference_path_used ? "minimum-uncovered reference path within detour tolerance" : "optimized covered path";
      odDetails.innerHTML = `
        <h2>Selected OD</h2>
        <strong>${path.id}</strong><br>
        Status: ${kind}<br>
        Path source: ${sourceText}<br>
        Numeric OD: ${path.origin} -> ${path.destination}<br>
        Original OD: ${path.original_origin} -> ${path.original_destination}<br>
        Demand: ${path.demand.toFixed(6)}<br>
        Path length: ${path.path_length.toFixed(2)}<br>
        Shortest length: ${path.shortest_length.toFixed(2)}<br>
        Length / shortest-path ratio: ${lengthRatio}<br>
        Covered length: ${path.covered_length.toFixed(2)}<br>
        Uncovered length: ${path.uncovered_length.toFixed(2)}<br>
        Cover ratio: ${coverPct}<br>
        Uncovered ratio: ${uncoveredPct}<br>
        Detour cap: ${Number.isFinite(path.detour_cap) ? path.detour_cap.toFixed(2) : "N/A"}<br>
        Uncovered cap if required: ${Number.isFinite(path.uncovered_cap_if_required) ? path.uncovered_cap_if_required.toFixed(2) : "N/A"}<br>
        Arc count: ${path.segments.length}<br><br>
        <strong>Node sequence (numeric)</strong>
        <div class="path-box">${numericSeq}</div>
        <br>
        <strong>Node sequence (original ids)</strong>
        <div class="path-box">${originalSeq}</div>
      `;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      deviceScale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * deviceScale));
      canvas.height = Math.max(1, Math.floor(rect.height * deviceScale));
      ctx.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);
      draw();
    }

    function fitToView() {
      const bounds = DATA.bounds;
      const width = canvas.clientWidth || 1;
      const height = canvas.clientHeight || 1;
      const worldWidth = Math.max(bounds.maxX - bounds.minX, 1e-9);
      const worldHeight = Math.max(bounds.maxY - bounds.minY, 1e-9);
      const scale = 0.9 * Math.min(width / worldWidth, height / worldHeight);
      viewport.scale = scale;
      viewport.offsetX = (width - worldWidth * scale) / 2 - bounds.minX * scale;
      viewport.offsetY = (height - worldHeight * scale) / 2 + bounds.maxY * scale;
      draw();
    }

    function worldToScreen(x, y) {
      return [x * viewport.scale + viewport.offsetX, viewport.offsetY - y * viewport.scale];
    }

    function screenToWorld(x, y) {
      return [(x - viewport.offsetX) / viewport.scale, (viewport.offsetY - y) / viewport.scale];
    }

    function makePath(points, close) {
      if (!points || points.length === 0) return false;
      const [x0, y0] = worldToScreen(points[0][0], points[0][1]);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      for (const point of points.slice(1)) {
        const [x, y] = worldToScreen(point[0], point[1]);
        ctx.lineTo(x, y);
      }
      if (close) ctx.closePath();
      return true;
    }

    function drawTAZLayers() {
      if (controls.taz.checked) {
        ctx.save();
        for (const zone of DATA.zones || []) {
          if (!makePath(zone.polygon, true)) continue;
          ctx.fillStyle = "rgba(251, 191, 36, 0.12)";
          ctx.fill();
        }
        ctx.restore();
      }

      if (controls.snapping.checked) {
        ctx.save();
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.58)";
        ctx.lineWidth = 1;
        for (const zone of DATA.zones || []) {
          if (!zone.centroid || !zone.snapped) continue;
          if (!makePath([zone.centroid, zone.snapped], false)) continue;
          ctx.stroke();
        }
        ctx.restore();
      }

      if (controls.tazCenters.checked) {
        ctx.save();
        for (const zone of DATA.zones || []) {
          if (!zone.centroid) continue;
          const [x, y] = worldToScreen(zone.centroid[0], zone.centroid[1]);
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#ef4444";
          ctx.fill();
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = "#7f1d1d";
          ctx.stroke();
          ctx.fillStyle = "#7f1d1d";
          ctx.font = "10px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(String(zone.original_id), x, y - 6);
        }
        ctx.restore();
      }
    }

    function drawTAZBoundaries() {
      if (!controls.taz.checked) return;
      ctx.save();
      for (const zone of DATA.zones || []) {
        if (!makePath(zone.polygon, true)) continue;
        ctx.setLineDash([]);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
        ctx.lineWidth = 3.0;
        ctx.stroke();
        if (!makePath(zone.polygon, true)) continue;
        ctx.strokeStyle = "rgba(146, 64, 14, 1)";
        ctx.lineWidth = 1.35;
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawBoundary() {
      const boundary = DATA.boundary || [];
      if (!controls.boundary.checked || boundary.length < 3) return;
      ctx.save();
      if (makePath(boundary, true)) {
        ctx.setLineDash([]);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
        ctx.lineWidth = 4.5;
        ctx.stroke();
      }
      if (makePath(boundary, true)) {
        ctx.setLineDash([12, 7]);
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 2.1;
        ctx.stroke();
      }
      ctx.restore();
    }

    function selectedOdId() {
      return odSelect.value || ALL_ODS;
    }

    function isPathVisible(path) {
      if (path.coverage_required && !controls.coveredPaths.checked) return false;
      if (!path.coverage_required && !controls.relaxedPaths.checked) return false;
      const selected = selectedOdId();
      return selected === ALL_ODS || selected === path.id;
    }

    function drawEdge(edge, forceHighlight, mode = "base") {
      const isBikeOverlay = mode === "bike";
      if (isBikeOverlay) {
        if (!edge.chosen || !controls.bike.checked) return;
        if (edge.kind === "connector" && !controls.connectors.checked) return;
      } else {
        if (edge.kind === "road" && !controls.road.checked) return;
        if (edge.kind === "connector" && !controls.connectors.checked) return;
      }
      const u = DATA.nodeMap[edge.u];
      const v = DATA.nodeMap[edge.v];
      if (!u || !v) return;
      const [x0, y0] = worldToScreen(u.x, u.y);
      const [x1, y1] = worldToScreen(v.x, v.y);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.setLineDash([]);
      if (isBikeOverlay && edge.kind === "connector") {
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth = forceHighlight ? 6 : 4;
        ctx.setLineDash(forceHighlight ? [] : [7, 6]);
        ctx.globalAlpha = 0.95;
      } else if (edge.kind === "connector") {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = forceHighlight ? 2.4 : 1.4;
        ctx.setLineDash(forceHighlight ? [] : [7, 6]);
        ctx.globalAlpha = 0.7;
      } else if (isBikeOverlay) {
        ctx.strokeStyle = "#16a34a";
        ctx.lineWidth = forceHighlight ? 8 : 5.8;
        ctx.globalAlpha = 0.96;
      } else {
        ctx.strokeStyle = "#cbd5e1";
        ctx.lineWidth = forceHighlight ? 2.4 : 1;
        ctx.globalAlpha = 0.85;
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    function drawPath(path, forceHighlight) {
      if (!isPathVisible(path) || !path.segments || path.segments.length === 0) return;
      const selected = selectedOdId() === path.id;
      ctx.beginPath();
      let started = false;
      for (const segment of path.segments) {
        const u = DATA.nodeMap[segment[0]];
        const v = DATA.nodeMap[segment[1]];
        if (!u || !v) continue;
        const [x0, y0] = worldToScreen(u.x, u.y);
        const [x1, y1] = worldToScreen(v.x, v.y);
        if (!started) {
          ctx.moveTo(x0, y0);
          started = true;
        }
        ctx.lineTo(x1, y1);
      }
      if (!started) return;
      const baseColor = path.coverage_required ? "#2563eb" : "#dc2626";
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = selected || forceHighlight ? 4.4 : 2.0;
      ctx.globalAlpha = selected || forceHighlight ? 0.95 : (path.coverage_required ? 0.45 : 0.35);
      ctx.setLineDash(path.coverage_required ? [] : [8, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    function drawNode(node, forceHighlight) {
      if (node.kind === "od") {
        if (!controls.odNodes.checked) return;
        const [x, y] = worldToScreen(node.x, node.y);
        ctx.beginPath();
        ctx.arc(x, y, forceHighlight ? 8 : 6, 0, Math.PI * 2);
        ctx.fillStyle = "#f59e0b";
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#7c2d12";
        ctx.stroke();
        ctx.fillStyle = "#7c2d12";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(node.id), x, y);
      } else if (controls.roadNodes.checked) {
        const [x, y] = worldToScreen(node.x, node.y);
        ctx.beginPath();
        ctx.arc(x, y, forceHighlight ? 2.8 : 1.6, 0, Math.PI * 2);
        ctx.fillStyle = "#94a3b8";
        ctx.fill();
      }
    }

    function draw() {
      const w = canvas.clientWidth || 1;
      const h = canvas.clientHeight || 1;
      ctx.clearRect(0, 0, w, h);

      drawTAZLayers();
      for (const edge of DATA.edges) {
        drawEdge(edge, false, "base");
      }
      for (const path of DATA.paths || []) {
        if (selectedOdId() !== path.id) drawPath(path, hoverItem && hoverItem.type === "path" && hoverItem.data.id === path.id);
      }
      for (const edge of DATA.edges) {
        drawEdge(edge, hoverItem && hoverItem.type === "edge" && hoverItem.data.id === edge.id, "bike");
      }
      const selectedPath = pathMap[selectedOdId()];
      if (selectedPath) {
        drawPath(selectedPath, true);
      }
      drawBoundary();
      for (const node of DATA.nodes) {
        drawNode(node, hoverItem && hoverItem.type === "node" && hoverItem.data.id === node.id);
      }
      drawTAZBoundaries();
    }

    function distToSegment(px, py, ax, ay, bx, by) {
      const dx = bx - ax;
      const dy = by - ay;
      const len2 = dx * dx + dy * dy;
      if (len2 === 0) return Math.hypot(px - ax, py - ay);
      let t = ((px - ax) * dx + (py - ay) * dy) / len2;
      t = Math.max(0, Math.min(1, t));
      const qx = ax + t * dx;
      const qy = ay + t * dy;
      return Math.hypot(px - qx, py - qy);
    }

    function distToPath(px, py, path) {
      let best = Infinity;
      for (const segment of path.segments || []) {
        const u = DATA.nodeMap[segment[0]];
        const v = DATA.nodeMap[segment[1]];
        if (!u || !v) continue;
        const [x0, y0] = worldToScreen(u.x, u.y);
        const [x1, y1] = worldToScreen(v.x, v.y);
        best = Math.min(best, distToSegment(px, py, x0, y0, x1, y1));
      }
      return best;
    }

    function findHover(mx, my) {
      let bestNode = null;
      let bestNodeDist = Infinity;
      for (const node of DATA.nodes) {
        if (node.kind === "od" && !controls.odNodes.checked) continue;
        if (node.kind === "through" && !controls.roadNodes.checked) continue;
        const [sx, sy] = worldToScreen(node.x, node.y);
        const d = Math.hypot(mx - sx, my - sy);
        const threshold = node.kind === "od" ? 12 : 6;
        if (d < threshold && d < bestNodeDist) {
          bestNodeDist = d;
          bestNode = node;
        }
      }
      if (bestNode) return { type: "node", data: bestNode };

      let bestPath = null;
      let bestPathDist = Infinity;
      for (const path of DATA.paths || []) {
        if (!isPathVisible(path)) continue;
        const d = distToPath(mx, my, path);
        const threshold = selectedOdId() === path.id ? 10 : 7;
        if (d < threshold && d < bestPathDist) {
          bestPathDist = d;
          bestPath = path;
        }
      }
      if (bestPath) return { type: "path", data: bestPath };

      let bestEdge = null;
      let bestEdgeDist = Infinity;
      for (const edge of DATA.edges) {
        const baseVisible = (
          (edge.kind === "road" && controls.road.checked) ||
          (edge.kind === "connector" && controls.connectors.checked)
        );
        const bikeVisible = edge.chosen && controls.bike.checked;
        if (!baseVisible && !bikeVisible) continue;
        const u = DATA.nodeMap[edge.u];
        const v = DATA.nodeMap[edge.v];
        const [x0, y0] = worldToScreen(u.x, u.y);
        const [x1, y1] = worldToScreen(v.x, v.y);
        const d = distToSegment(mx, my, x0, y0, x1, y1);
        const threshold = bikeVisible ? 9 : 6;
        if (d < threshold && d < bestEdgeDist) {
          bestEdgeDist = d;
          bestEdge = edge;
        }
      }
      return bestEdge ? { type: "edge", data: bestEdge } : null;
    }

    function showHover(item, clientX, clientY) {
      hoverItem = item;
      if (!item) {
        tooltip.style.display = "none";
        details.innerHTML = '<h2>Hover Details</h2><div class="muted">Move the cursor over an OD path, edge, or node to inspect it.</div>';
        draw();
        return;
      }

      if (item.type === "node") {
        const node = item.data;
        tooltip.innerHTML = `<strong>Node ${node.id}</strong><br>Original id: ${node.original_id}<br>Type: ${node.kind === "od" ? "OD node" : "road node"}${node.kind === "od" ? `<br>OD role: ${node.od_role || "origin/destination"}` : ""}`;
        details.innerHTML = `<h2>Hover Details</h2><strong>Node ${node.id}</strong><br>Original id: ${node.original_id}<br>Type: ${node.kind === "od" ? "OD node" : "road node"}${node.kind === "od" ? `<br>OD role: ${node.od_role || "origin/destination"}` : ""}<br>X: ${node.x.toFixed(2)}<br>Y: ${node.y.toFixed(2)}`;
      } else if (item.type === "path") {
        const path = item.data;
        const kind = path.coverage_required ? "covered OD" : "uncovered OD reference";
        const coverPct = Number.isFinite(path.cover_ratio) ? (path.cover_ratio * 100).toFixed(2) + "%" : "N/A";
        const lengthRatio = Number.isFinite(path.length_to_shortest_ratio) ? path.length_to_shortest_ratio.toFixed(4) : "N/A";
        const sourceText = path.reference_path_used ? "minimum-uncovered reference path" : "optimized covered path";
        tooltip.innerHTML = `<strong>${path.id}</strong><br>${kind}<br>${sourceText}<br>Demand: ${path.demand.toFixed(4)}<br>Cover ratio: ${coverPct}<br>Length/K: ${lengthRatio}`;
        details.innerHTML = `<h2>Hover Details</h2><strong>OD ${path.id}</strong><br>Status: ${kind}<br>Path source: ${sourceText}<br>Demand: ${path.demand.toFixed(6)}<br>Path length: ${path.path_length.toFixed(2)}<br>Shortest path: ${path.shortest_length.toFixed(2)}<br>Length / shortest-path ratio: ${lengthRatio}<br>Covered length: ${path.covered_length.toFixed(2)}<br>Uncovered length: ${path.uncovered_length.toFixed(2)}<br>Cover ratio: ${coverPct}<br>Arcs on displayed path: ${path.segments.length}`;
      } else {
        const edge = item.data;
        tooltip.innerHTML = `<strong>${edge.kind === "connector" ? "Connector" : edge.chosen ? "Chosen bike lane" : "Road edge"}</strong><br>${edge.u} → ${edge.v}<br>Original: ${edge.original_u} → ${edge.original_v}<br>Length: ${edge.length.toFixed(2)}`;
        details.innerHTML = `<h2>Hover Details</h2><strong>${edge.kind === "connector" ? "Connector" : edge.chosen ? "Chosen bicycle lane" : "Road edge"}</strong><br>Numeric: ${edge.u} → ${edge.v}<br>Original: ${edge.original_u} → ${edge.original_v}<br>Length: ${edge.length.toFixed(2)}<br>Selected in design: ${edge.chosen ? "yes" : "no"}`;
      }

      tooltip.style.display = "block";
      tooltip.style.left = `${clientX + 14}px`;
      tooltip.style.top = `${clientY + 14}px`;
      draw();
    }

    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      if (dragging) {
        viewport.offsetX = dragStart.offsetX + (mx - dragStart.x);
        viewport.offsetY = dragStart.offsetY + (my - dragStart.y);
        draw();
        return;
      }
      showHover(findHover(mx, my), event.clientX, event.clientY);
    });

    canvas.addEventListener("mouseleave", () => {
      if (!dragging) showHover(null, 0, 0);
    });

    canvas.addEventListener("mousedown", (event) => {
      const rect = canvas.getBoundingClientRect();
      dragging = true;
      dragStart = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        offsetX: viewport.offsetX,
        offsetY: viewport.offsetY,
      };
    });

    window.addEventListener("mouseup", () => {
      dragging = false;
      dragStart = null;
    });

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const [wx, wy] = screenToWorld(mx, my);
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      viewport.scale = Math.max(0.02, Math.min(500000, viewport.scale * factor));
      viewport.offsetX = mx - wx * viewport.scale;
      viewport.offsetY = my + wy * viewport.scale;
      draw();
    }, { passive: false });

    document.getElementById("fitBtn").addEventListener("click", fitToView);
    Object.values(controls).forEach((el) => el.addEventListener("change", () => showHover(null, 0, 0)));
    odSelect.addEventListener("change", () => {
      renderSelectedOd();
      showHover(null, 0, 0);
    });
    canvas.addEventListener("click", () => {
      if (hoverItem && hoverItem.type === "path") {
        odSelect.value = hoverItem.data.id;
        renderSelectedOd();
        draw();
      }
    });
    window.addEventListener("resize", resizeCanvas);

    renderStats();
    renderOdSelector();
    renderSelectedOd();
    resizeCanvas();
    fitToView();
  </script>
</body>
</html>
"""


def build_interactive_payload(
    data: TNTPNetworkData,
    result: Dict[str, object],
    zone_count: int,
    original_id_map: Dict[int, str],
    zone_geometries: Dict[str, Dict[str, object]],
    zone_centers: Dict[str, Tuple[float, float]],
    boundary_polygon: List[Tuple[float, float]],
) -> Dict[str, object]:
    pos = compute_visual_positions(data, zone_count)
    chosen_links = {tuple(link) for link in result.get("chosen_links", [])}
    od_paths = result.get("od_paths", {})
    od_stats = result.get("od_stats", {})
    od_origin_nodes = set(result.get("od_origin_nodes", []))
    od_destination_nodes = set(result.get("od_destination_nodes", []))
    if not od_origin_nodes and not od_destination_nodes:
        od_origin_nodes = {od[0] for od in od_paths.keys()}
        od_destination_nodes = {od[1] for od in od_paths.keys()}
    od_nodes = od_origin_nodes | od_destination_nodes

    nodes = []
    for node in sorted(data.nodes):
        if node not in pos:
            continue
        neighbor_count = 0
        if node <= zone_count:
            neighbors = set()
            for arc_index in data.out_arcs.get(node, []):
                _, v = data.idx_to_arc[arc_index]
                if v > zone_count:
                    neighbors.add(v)
            for arc_index in data.in_arcs.get(node, []):
                u, _ = data.idx_to_arc[arc_index]
                if u > zone_count:
                    neighbors.add(u)
            neighbor_count = len(neighbors)
        if node in od_origin_nodes and node in od_destination_nodes:
            od_role = "origin and destination"
        elif node in od_origin_nodes:
            od_role = "origin"
        elif node in od_destination_nodes:
            od_role = "destination"
        else:
            od_role = ""
        nodes.append(
            {
                "id": node,
                "original_id": original_id_map.get(node, str(node)),
                "kind": "od" if node in od_nodes else "through",
                "x": pos[node][0],
                "y": pos[node][1],
                "neighbor_count": neighbor_count,
                "od_role": od_role,
            }
        )

    node_map = {node["id"]: node for node in nodes}

    zones = []
    for zone in range(1, zone_count + 1):
        original_id = original_id_map.get(zone, str(zone))
        geometry = zone_geometries.get(original_id)
        if not geometry:
            continue
        polygon = geometry.get("polygon", [])
        centroid = zone_centers.get(original_id) or fallback_polygon_centroid(polygon)  # type: ignore[arg-type]
        zones.append(
            {
                "id": zone,
                "original_id": original_id,
                "name": geometry.get("name", ""),
                "polygon": polygon,
                "centroid": centroid,
                "snapped": pos.get(zone),
            }
        )

    edges = []
    for idx, (u, v) in enumerate(data.arcs):
        if u not in node_map or v not in node_map:
            continue
        edge_kind = "connector" if data.arc_type.get(idx, 1) != 1 else "road"
        edges.append(
            {
                "id": idx,
                "u": u,
                "v": v,
                "original_u": original_id_map.get(u, str(u)),
                "original_v": original_id_map.get(v, str(v)),
                "length": data.arc_len[idx],
                "kind": edge_kind,
                "chosen": (u, v) in chosen_links,
            }
        )

    paths = []
    for od in sorted(od_paths.keys()):
        o, d = od
        segments = []
        for u, v in od_paths[od]:
            if u in node_map and v in node_map:
                segments.append([u, v])
        stats = od_stats.get(od, {})
        paths.append(
            {
                "id": f"{o}->{d}",
                "origin": o,
                "destination": d,
                "original_origin": original_id_map.get(o, str(o)),
                "original_destination": original_id_map.get(d, str(d)),
                "coverage_required": bool(stats.get("coverage_required", False)),
                "path_source": str(stats.get("path_source", "")),
                "reference_path_used": bool(stats.get("reference_path_used", False)),
                "demand": float(stats.get("demand", 0.0)),
                "path_length": float(stats.get("path_length", 0.0)),
                "shortest_length": float(stats.get("shortest_length", 0.0)),
                "length_to_shortest_ratio": float(stats.get("length_to_shortest_ratio", math.nan)),
                "detour_cap": float(stats.get("detour_cap", math.nan)),
                "uncovered_length": float(stats.get("uncovered_length", 0.0)),
                "model_uncovered_length": float(stats.get("model_uncovered_length", math.nan)),
                "covered_length": float(stats.get("covered_length", 0.0)),
                "cover_ratio": float(stats.get("cover_ratio", 0.0)),
                "uncovered_ratio": float(stats.get("uncovered_ratio", 0.0)),
                "uncovered_cap_active": float(stats.get("uncovered_cap_active", 0.0)),
                "uncovered_cap_if_required": float(stats.get("uncovered_cap_if_required", math.nan)),
                "segments": segments,
            }
        )
    paths.sort(key=lambda path: (not path["coverage_required"], path["origin"], path["destination"]))

    overlay_points: List[Tuple[float, float]] = []
    for zone in zones:
        overlay_points.extend(zone.get("polygon", []))  # type: ignore[arg-type]
        centroid = zone.get("centroid")
        snapped = zone.get("snapped")
        if centroid:
            overlay_points.append(centroid)  # type: ignore[arg-type]
        if snapped:
            overlay_points.append(snapped)  # type: ignore[arg-type]
    overlay_points.extend(boundary_polygon)

    xs = [node["x"] for node in nodes] + [point[0] for point in overlay_points]
    ys = [node["y"] for node in nodes] + [point[1] for point in overlay_points]
    covered_od_count = int(result.get("covered_od_count", 0))
    total_od_count = int(result.get("num_ods", 0))
    payload = {
        "summary": {
            "status_name": result.get("status_name", "UNKNOWN"),
            "objective": float(result.get("objective", math.inf)),
            "runtime": float(result.get("runtime", math.nan)),
            "gap": float(result.get("gap", math.inf)),
            "chosen_links": len(chosen_links),
            "num_ods": total_od_count,
            "covered_od_count": covered_od_count,
            "relaxed_od_count": max(0, total_od_count - covered_od_count),
            "target_share": float(result.get("demand_target_share", math.nan)),
            "covered_share": float(result.get("covered_demand_share", math.nan)),
            "uncovered_reference_path_count": int(result.get("uncovered_reference_path_count", 0)),
            "uncovered_reference_path_failure_count": int(result.get("uncovered_reference_path_failure_count", 0)),
        },
        "bounds": {
            "minX": min(xs),
            "maxX": max(xs),
            "minY": min(ys),
            "maxY": max(ys),
        },
        "nodes": nodes,
        "nodeMap": node_map,
        "edges": edges,
        "zones": zones,
        "boundary": boundary_polygon,
        "paths": paths,
    }
    return payload


def write_interactive_html(
    data: TNTPNetworkData,
    result: Dict[str, object],
    zone_count: int,
    original_id_map: Dict[int, str],
    zone_geometries: Dict[str, Dict[str, object]],
    zone_centers: Dict[str, Tuple[float, float]],
    boundary_polygon: List[Tuple[float, float]],
    out_path: str,
) -> None:
    payload = build_interactive_payload(
        data,
        result,
        zone_count,
        original_id_map,
        zone_geometries,
        zone_centers,
        boundary_polygon,
    )
    json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = INTERACTIVE_HTML_TEMPLATE.replace("__DATA_JSON__", json_text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_and_solve_direct_milp_with_warm_start(
    input_dir: str,
    epsilon: float = 0.2,
    t: float = 0.2,
    covered_demand_share: float = 1.0,
    od_limit: int = 100,
    seed: int = 42,
    time_limit: float = 3600.0,
    mip_gap: float = 1e-4,
    output_flag: int = 1,
    threads: int | None = None,
    use_od_arc_pruning: bool = True,
    use_node_reachability_pruning: bool = True,
    use_source_sink_variable_pruning: bool = True,
    use_directed_subgraph_trimming: bool = True,
    use_unused_design_unit_pruning: bool = True,
    drop_long_uncovered_arcs: bool = False,
    add_origin_destination_strengthening: bool = True,
    add_opposite_direction_x_cut: bool = True,
    add_single_arc_implied_cut: bool = True,
    add_minimum_covered_length_cut: bool = True,
    add_single_arc_detour_implied_cut: bool = True,
    balanced_od_sampling: bool = True,
) -> Dict[str, object]:
    if nx is None:
        raise RuntimeError("The Gurobi solver requires networkx. Install it with: pip install networkx")
    if gp is None or GRB is None:
        raise RuntimeError(
            "The Gurobi solver requires gurobipy. Install gurobipy and configure "
            "a valid Gurobi license before running this script."
        )

    data = TNTPNetworkData(input_dir)
    od_demand, od_sampling_stats = select_ods_with_balanced_io(
        data.od_demand,
        od_limit,
        seed=seed,
        require_balanced_io=balanced_od_sampling,
    )
    ods = sorted(od_demand.keys())
    num_arcs = len(data.arcs)
    target_share = normalize_demand_share(covered_demand_share)
    total_demand = sum(od_demand.values())
    target_demand = target_share * total_demand

    K: Dict[OD, float] = {}
    B: Dict[OD, float] = {}
    uncovered_cap_if_required: Dict[OD, float] = {}
    warm_path_arcs: Dict[OD, List[int]] = {}
    warm_covered_ods: List[OD] = []
    covered_demand_so_far = 0.0
    for od in sorted(ods, key=lambda cur: (-od_demand[cur], cur)):
        if covered_demand_so_far + 1e-9 >= target_demand:
            break
        warm_covered_ods.append(od)
        covered_demand_so_far += od_demand[od]
    warm_covered_set = set(warm_covered_ods)
    warm_design_arcs = set()
    for od in ods:
        K[od] = shortest_path_length(data, od)
        B[od] = (1.0 + epsilon) * K[od]
        uncovered_cap_if_required[od] = t * K[od]
        path_arcs, _ = shortest_path_arc_indices(data, od)
        warm_path_arcs[od] = path_arcs
        if od in warm_covered_set:
            warm_design_arcs.update(path_arcs)

    design_units, arc_to_design_unit, design_unit_arcs, design_unit_cost = build_undirected_design_units(data)
    num_design_units = len(design_units)

    feasible_arcs, pruning_stats = compute_od_feasible_arcs(
        data,
        ods,
        B,
        uncovered_cap_if_required,
        use_detour_pruning=use_od_arc_pruning,
        use_node_reachability_pruning=use_node_reachability_pruning,
        use_source_sink_variable_pruning=use_source_sink_variable_pruning,
        use_directed_subgraph_trimming=use_directed_subgraph_trimming,
        drop_long_uncovered_arcs=drop_long_uncovered_arcs,
    )
    feasible_arc_sets = {od: set(arcs) for od, arcs in feasible_arcs.items()}
    total_possible_od_arcs = len(ods) * num_arcs
    total_kept_od_arcs = sum(len(arcs) for arcs in feasible_arcs.values())
    removed_by_source_sink = sum(stats["removed_by_source_sink"] for stats in pruning_stats.values())
    removed_by_node_reachability = sum(stats["removed_by_node_reachability"] for stats in pruning_stats.values())
    removed_by_detour = sum(stats["removed_by_detour"] for stats in pruning_stats.values())
    removed_by_directed_subgraph_trimming = sum(
        stats["removed_by_directed_subgraph_trimming"] for stats in pruning_stats.values()
    )
    removed_by_long_uncovered = sum(stats["removed_by_long_uncovered"] for stats in pruning_stats.values())
    removed_unreachable = sum(stats["removed_unreachable"] for stats in pruning_stats.values())

    num_design_units_full = len(design_units)
    if use_unused_design_unit_pruning:
        active_design_unit_indices = sorted(
            {
                arc_to_design_unit[a]
                for arcs in feasible_arcs.values()
                for a in arcs
            }
        )
    else:
        active_design_unit_indices = list(range(num_design_units_full))
    active_design_unit_set = set(active_design_unit_indices)
    num_design_units_after_pruning = len(active_design_unit_indices)
    design_units_removed_unused = num_design_units_full - num_design_units_after_pruning
    num_design_units = num_design_units_after_pruning

    # Keep the warm start consistent with the sparse OD-arc variable set. If the
    # shortest path of a warm-covered OD contains an arc removed by pruning, do not
    # force that OD to be covered in the MIP start.
    feasible_warm_covered_set = {
        od
        for od in warm_covered_set
        if all(a in feasible_arc_sets[od] for a in warm_path_arcs[od])
    }
    warm_design_arcs = {
        a
        for od in feasible_warm_covered_set
        for a in warm_path_arcs[od]
        if a in feasible_arc_sets[od]
    }
    warm_design_units = {arc_to_design_unit[a] for a in warm_design_arcs}

    m = gp.Model("scaled_no_big_m_link_based_bikelane_undirected_design_pruned_exact_5cuts")
    m.Params.OutputFlag = int(output_flag)
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)
    if threads is not None:
        m.Params.Threads = int(threads)

    # One binary construction variable is defined for each undirected physical link.
    # If arcs (i, j) and (j, i) both exist, they share the same delta variable.
    delta = {
        q: m.addVar(vtype=GRB.BINARY, name=f"delta_undirected[{design_units[q][0]},{design_units[q][1]}]")
        for q in active_design_unit_indices
    }
    x = {
        (od, a): m.addVar(vtype=GRB.BINARY, name=f"x[{od[0]},{od[1]},{a}]")
        for od in ods
        for a in feasible_arcs[od]
    }
    c = {
        (od, a): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"c[{od[0]},{od[1]},{a}]")
        for od in ods
        for a in feasible_arcs[od]
    }
    # z is declared continuous. With binary OD-link variables x and scaled flow balance,
    # z is implicitly driven to 0/1 in integer feasible solutions.
    z = {od: m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z[{od[0]},{od[1]}]") for od in ods}

    # The construction cost is counted once per undirected physical link, not once
    # per directed arc. If the two directions have different arc costs, the design
    # unit cost uses the conservative maximum cost across its represented arcs.
    m.setObjective(
        gp.quicksum(design_unit_cost[q] * delta[q] for q in active_design_unit_indices),
        GRB.MINIMIZE,
    )

    if ods:
        m.addConstr(
            gp.quicksum(od_demand[od] * z[od] for od in ods) >= target_demand,
            name="covered_demand_requirement",
        )

    origin_destination_strengthening_count = 0
    opposite_direction_x_cut_count = 0
    single_arc_uncovered_implied_cut_count = 0
    minimum_covered_length_cut_count = 0
    single_arc_detour_implied_cut_count = 0

    for od in ods:
        o, d = od
        feasible_set = feasible_arc_sets[od]
        for i in data.nodes:
            out_expr = gp.quicksum(
                x[(od, a)]
                for a in data.out_arcs.get(i, [])
                if a in feasible_set
            )
            in_expr = gp.quicksum(
                x[(od, a)]
                for a in data.in_arcs.get(i, [])
                if a in feasible_set
            )
            b = 1 if i == o else (-1 if i == d else 0)
            # Scaled no-Big-M flow balance. If z[od] = 0, no path is constructed;
            # if z[od] = 1, one unit of OD flow is routed from origin to destination.
            # The sums are restricted to the OD-specific feasible arc set.
            m.addConstr(out_expr - in_expr == b * z[od], name=f"flow_scaled[{od[0]},{od[1]},{i}]")
            m.addConstr(out_expr <= z[od], name=f"outdeg_scaled[{od[0]},{od[1]},{i}]")
            m.addConstr(in_expr <= z[od], name=f"indeg_scaled[{od[0]},{od[1]},{i}]")

        # Optional source/sink strengthening. These equations are implied by
        # the scaled flow balance and degree constraints, but writing them explicitly
        # can tighten the root relaxation and reduce source/sink fractional cycles.
        if add_origin_destination_strengthening:
            out_o = gp.quicksum(
                x[(od, a)]
                for a in data.out_arcs.get(o, [])
                if a in feasible_set
            )
            in_o = gp.quicksum(
                x[(od, a)]
                for a in data.in_arcs.get(o, [])
                if a in feasible_set
            )
            out_d = gp.quicksum(
                x[(od, a)]
                for a in data.out_arcs.get(d, [])
                if a in feasible_set
            )
            in_d = gp.quicksum(
                x[(od, a)]
                for a in data.in_arcs.get(d, [])
                if a in feasible_set
            )
            m.addConstr(in_o == 0.0, name=f"origin_no_inflow[{od[0]},{od[1]}]")
            m.addConstr(out_o == z[od], name=f"origin_one_outflow[{od[0]},{od[1]}]")
            m.addConstr(out_d == 0.0, name=f"destination_no_outflow[{od[0]},{od[1]}]")
            m.addConstr(in_d == z[od], name=f"destination_one_inflow[{od[0]},{od[1]}]")
            origin_destination_strengthening_count += 4

        # Optional opposite-direction cut. Since construction is undirected but
        # paths are directed, this prevents a single OD path from selecting both
        # directions of the same physical link.
        if add_opposite_direction_x_cut:
            for q, arcs_in_unit in design_unit_arcs.items():
                unit_od_arcs = [a for a in arcs_in_unit if a in feasible_set]
                if len(unit_od_arcs) <= 1:
                    continue
                m.addConstr(
                    gp.quicksum(x[(od, a)] for a in unit_od_arcs) <= z[od],
                    name=f"opposite_direction_x_cut[{od[0]},{od[1]},{q}]",
                )
                opposite_direction_x_cut_count += 1

        # Scaled detour and uncovered-length constraints. These replace the old Big-M
        # uncovered-cap formulation. If z[od] = 0, both sides are zero and all OD flow
        # is shut down by the scaled flow constraints.
        m.addConstr(
            gp.quicksum(data.arc_len[a] * x[(od, a)] for a in feasible_arcs[od]) <= B[od] * z[od],
            name=f"detour_scaled[{od[0]},{od[1]}]",
        )
        m.addConstr(
            gp.quicksum(c[(od, a)] for a in feasible_arcs[od]) <= uncovered_cap_if_required[od] * z[od],
            name=f"uncovered_cap_scaled[{od[0]},{od[1]}]",
        )

        # Optional minimum covered-length cut:
        # total path length minus uncovered length must include at least
        # K_od - t*K_od units of bicycle-lane-supported length when z_od = 1.
        if add_minimum_covered_length_cut:
            min_covered_len = max(0.0, K[od] - uncovered_cap_if_required[od])
            if min_covered_len > 1e-9:
                m.addConstr(
                    gp.quicksum(
                        data.arc_len[a] * x[(od, a)] - c[(od, a)]
                        for a in feasible_arcs[od]
                    ) >= min_covered_len * z[od],
                    name=f"minimum_covered_length[{od[0]},{od[1]}]",
                )
                minimum_covered_length_cut_count += 1

    for od in ods:
        C = uncovered_cap_if_required[od]
        for a in feasible_arcs[od]:
            ell = data.arc_len[a]
            q = arc_to_design_unit[a]
            m.addConstr(c[(od, a)] <= ell * x[(od, a)], name=f"c_ub_x[{od[0]},{od[1]},{a}]")
            m.addConstr(c[(od, a)] <= ell * (1 - delta[q]), name=f"c_ub_d[{od[0]},{od[1]},{a}]")
            m.addConstr(c[(od, a)] >= ell * (x[(od, a)] - delta[q]), name=f"c_lb[{od[0]},{od[1]},{a}]")

            # Single-arc uncovered implied cut:
            # If this OD uses arc a while the corresponding undirected design unit
            # is not built, the arc length alone must fit within the OD's allowed
            # uncovered-length budget C = t * K_od. This is implied by the total
            # uncovered-cap constraint and c_lb, but it tightens the LP relaxation.
            if add_single_arc_implied_cut:
                m.addConstr(
                    ell * (x[(od, a)] - delta[q]) <= C * z[od],
                    name=f"single_arc_uncovered_implied[{od[0]},{od[1]},{a}]",
                )
                single_arc_uncovered_implied_cut_count += 1

            # Single-arc detour implied cut. This is implied by the aggregate
            # detour constraint and nonnegativity, but can strengthen LP relaxation
            # locally for each OD-arc variable.
            if add_single_arc_detour_implied_cut:
                m.addConstr(
                    ell * x[(od, a)] <= B[od] * z[od],
                    name=f"single_arc_detour_implied[{od[0]},{od[1]},{a}]",
                )
                single_arc_detour_implied_cut_count += 1

    for q in active_design_unit_indices:
        delta[q].Start = 1.0 if q in warm_design_units else 0.0

    for od in ods:
        z[od].Start = 1.0 if od in feasible_warm_covered_set else 0.0
        path_arc_set = set(warm_path_arcs[od]) if od in feasible_warm_covered_set else set()
        for a in feasible_arcs[od]:
            x[(od, a)].Start = 1.0 if a in path_arc_set else 0.0
            if a not in path_arc_set:
                c[(od, a)].Start = 0.0
            elif arc_to_design_unit[a] in warm_design_units:
                c[(od, a)].Start = 0.0
            else:
                c[(od, a)].Start = data.arc_len[a]

    start_time = time.time()
    m.optimize()
    elapsed = time.time() - start_time

    od_origin_nodes = sorted({od[0] for od in ods})
    od_destination_nodes = sorted({od[1] for od in ods})
    od_nodes = sorted(set(od_origin_nodes) | set(od_destination_nodes))

    result: Dict[str, object] = {
        "status": m.Status,
        "status_name": {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status)),
        "runtime": elapsed,
        "objective": math.inf,
        "best_bound": getattr(m, "ObjBound", math.inf),
        "gap": math.inf,
        "num_ods": len(ods),
        "num_arcs": num_arcs,
        "num_nodes": len(data.nodes),
        "total_demand": total_demand,
        "demand_target": target_demand,
        "demand_target_share": target_share,
        "warm_start_covered_ods": sorted(feasible_warm_covered_set),
        "original_warm_start_covered_ods": warm_covered_ods,
        "num_design_units": num_design_units,
        "num_design_units_full": num_design_units_full,
        "num_design_units_after_pruning": num_design_units_after_pruning,
        "design_units_removed_unused": design_units_removed_unused,
        "num_od_arc_variables_full": total_possible_od_arcs,
        "num_od_arc_variables_after_pruning": total_kept_od_arcs,
        "od_arc_pruning_removed_by_source_sink": removed_by_source_sink,
        "od_arc_pruning_removed_by_node_reachability": removed_by_node_reachability,
        "od_arc_pruning_removed_by_detour": removed_by_detour,
        "od_arc_pruning_removed_by_directed_subgraph_trimming": removed_by_directed_subgraph_trimming,
        "od_arc_pruning_removed_by_long_uncovered": removed_by_long_uncovered,
        "od_arc_pruning_removed_unreachable": removed_unreachable,
        "use_od_arc_pruning": use_od_arc_pruning,
        "use_source_sink_variable_pruning": use_source_sink_variable_pruning,
        "use_node_reachability_pruning": use_node_reachability_pruning,
        "use_directed_subgraph_trimming": use_directed_subgraph_trimming,
        "use_unused_design_unit_pruning": use_unused_design_unit_pruning,
        "drop_long_uncovered_arcs": drop_long_uncovered_arcs,
        "add_origin_destination_strengthening": add_origin_destination_strengthening,
        "add_opposite_direction_x_cut": add_opposite_direction_x_cut,
        "add_single_arc_uncovered_implied_cut": add_single_arc_implied_cut,
        "add_minimum_covered_length_cut": add_minimum_covered_length_cut,
        "add_single_arc_detour_implied_cut": add_single_arc_detour_implied_cut,
        "balanced_od_sampling": balanced_od_sampling,
        "od_sampling_stats": od_sampling_stats,
        "od_nodes": od_nodes,
        "od_origin_nodes": od_origin_nodes,
        "od_destination_nodes": od_destination_nodes,
        "origin_destination_strengthening_count": origin_destination_strengthening_count,
        "opposite_direction_x_cut_count": opposite_direction_x_cut_count,
        "single_arc_uncovered_implied_cut_count": single_arc_uncovered_implied_cut_count,
        "minimum_covered_length_cut_count": minimum_covered_length_cut_count,
        "single_arc_detour_implied_cut_count": single_arc_detour_implied_cut_count,
        "warm_start_design_links": [data.idx_to_arc[a] for a in sorted(warm_design_arcs)],
        "warm_start_design_units": [design_units[q] for q in sorted(warm_design_units)],
        "model_type": "scaled_no_big_m_link_based_undirected_design_variable_with_node_reachability_pruning_od_arc_pruning_and_5cuts",
        "construction_variable_type": "one_binary_per_undirected_physical_link",
        "z_variable_type": "continuous_implicit_binary_by_scaled_flow",
    }

    if m.SolCount > 0:
        result["objective"] = m.ObjVal
        result["best_bound"] = m.ObjBound
        result["gap"] = abs(m.ObjVal - m.ObjBound) / max(1.0, abs(m.ObjVal))
        chosen_design_unit_indices = [q for q in active_design_unit_indices if delta[q].X > 0.5]
        chosen_design_unit_set = set(chosen_design_unit_indices)
        result["chosen_design_units"] = [design_units[q] for q in chosen_design_unit_indices]
        result["chosen_design_unit_count"] = len(chosen_design_unit_indices)
        # Expand selected undirected units back to directed arcs for path accounting and visualization.
        result["chosen_links"] = [
            data.idx_to_arc[a]
            for a in range(num_arcs)
            if arc_to_design_unit[a] in chosen_design_unit_set
        ]
        result["covered_ods"] = [od for od in ods if z[od].X > 0.5]
        result["relaxed_ods"] = [od for od in ods if z[od].X <= 0.5]
        result["covered_od_count"] = len(result["covered_ods"])
        result["covered_demand"] = sum(od_demand[od] for od in result["covered_ods"])
        result["covered_demand_share"] = (
            result["covered_demand"] / total_demand if total_demand > 0.0 else 0.0
        )
        result["od_paths"] = {}
        result["od_stats"] = {}
        reference_path_count = 0
        reference_path_failure_count = 0
        for od in ods:
            z_val = z[od].X
            is_covered = z_val > 0.5
            if is_covered:
                path_arc_idx = extract_path_for_od_sparse(data, x, od, feasible_arc_sets[od])
                path_source = "optimized_covered_path"
                model_uncovered = sum(c[(od, a)].X for a in feasible_arcs[od])
            else:
                # For an uncovered OD, display a reference path that minimizes
                # non-bicycle-lane distance under the same detour tolerance
                # length <= B_od. This path is not part of the optimization model.
                path_arc_idx = find_min_uncovered_reference_path(
                    data,
                    od,
                    chosen_design_unit_set,
                    arc_to_design_unit,
                    B[od],
                )
                path_source = "min_uncovered_reference_path_within_detour_tolerance"
                model_uncovered = 0.0
                if path_arc_idx:
                    reference_path_count += 1
                else:
                    reference_path_failure_count += 1
            path_arcs = [data.idx_to_arc[a] for a in path_arc_idx]
            metrics = compute_path_design_metrics(
                data,
                path_arc_idx,
                chosen_design_unit_set,
                arc_to_design_unit,
                K[od],
            )
            result["od_paths"][od] = path_arcs
            result["od_stats"][od] = {
                "demand": od_demand[od],
                "z_value": z_val,
                "coverage_required": bool(is_covered),
                "path_source": path_source,
                "reference_path_used": bool(not is_covered),
                "path_length": metrics["path_length"],
                "shortest_length": K[od],
                "length_to_shortest_ratio": metrics["length_to_shortest_ratio"],
                "detour_cap": B[od],
                "detour_cap_ratio": B[od] / K[od] if K[od] > 1e-12 else math.nan,
                "uncovered_length": metrics["uncovered_length"],
                "model_uncovered_length": model_uncovered,
                "covered_length": metrics["covered_length"],
                "cover_ratio": metrics["cover_ratio"],
                "uncovered_ratio": 1.0 - metrics["cover_ratio"] if metrics["path_length"] > 1e-12 else 0.0,
                "uncovered_cap_if_required": uncovered_cap_if_required[od],
                "uncovered_cap_active": uncovered_cap_if_required[od] * z_val,
                "uncovered_cap_ratio": uncovered_cap_if_required[od] / K[od] if K[od] > 1e-12 else math.nan,
            }
        result["uncovered_reference_path_count"] = reference_path_count
        result["uncovered_reference_path_failure_count"] = reference_path_failure_count

    print("===== Scaled no-Big-M link-based MILP summary (partial demand coverage, no cycle constraints) =====")
    print(f"Status       : {result['status_name']}")
    print(f"Runtime (s)  : {result['runtime']:.2f}")
    print(f"Demand target: {target_share:.2%} ({target_demand:.6f} / {total_demand:.6f})")
    print(
        "OD sampling: "
        f"balanced={'on' if balanced_od_sampling else 'off'}, "
        f"selected={len(ods)}, "
        f"balanced_nodes={len(od_sampling_stats.get('balanced_od_nodes', []))}, "
        f"warning={od_sampling_stats.get('warning', '') or 'none'}"
    )
    print("Objective: construction cost only")
    print(
        "OD-arc variables: "
        f"{total_kept_od_arcs} kept / {total_possible_od_arcs} full "
        f"(source/sink-pruned={removed_by_source_sink}, "
        f"node-reachability-pruned={removed_by_node_reachability}, "
        f"detour-pruned={removed_by_detour}, "
        f"subgraph-trimmed={removed_by_directed_subgraph_trimming}, "
        f"long-uncovered-pruned={removed_by_long_uncovered}, "
        f"unreachable-pruned={removed_unreachable})"
    )
    print(
        "Design units: "
        f"{num_design_units_after_pruning} active / {num_design_units_full} full "
        f"(unused-pruned={design_units_removed_unused})"
    )
    print(
        "Cuts: "
        f"origin/destination={'on' if add_origin_destination_strengthening else 'off'} "
        f"({origin_destination_strengthening_count}), "
        f"opposite-direction={'on' if add_opposite_direction_x_cut else 'off'} "
        f"({opposite_direction_x_cut_count}), "
        f"single-arc-uncovered={'on' if add_single_arc_implied_cut else 'off'} "
        f"({single_arc_uncovered_implied_cut_count}), "
        f"minimum-covered-length={'on' if add_minimum_covered_length_cut else 'off'} "
        f"({minimum_covered_length_cut_count}), "
        f"single-arc-detour={'on' if add_single_arc_detour_implied_cut else 'off'} "
        f"({single_arc_detour_implied_cut_count})"
    )
    if m.SolCount > 0:
        print(f"Incumbent UB : {result['objective']:.6f}")
        print(f"Best bound LB: {result['best_bound']:.6f}")
        print(f"Relative gap : {result['gap']:.6%}")
        print(f"Chosen design units: {result.get('chosen_design_unit_count', 0)}")
        print(f"Chosen directed links shown: {len(result['chosen_links'])}")
        print(
            f"Covered demand: {result['covered_demand']:.6f} "
            f"({result['covered_demand_share']:.2%})"
        )
        print(f"Covered ODs  : {result['covered_od_count']} / {len(ods)}")
    else:
        print("No feasible incumbent found.")

    return result


def write_summary_html(
    result: Dict[str, object],
    svg_name: str,
    out_path: str,
) -> None:
    chosen_count = len(result.get("chosen_links", []))
    objective = result.get("objective", math.inf)
    runtime = result.get("runtime", math.nan)
    gap = result.get("gap", math.inf)
    status_name = result.get("status_name", "UNKNOWN")
    target_share = result.get("demand_target_share", math.nan)
    covered_share = result.get("covered_demand_share", math.nan)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Direct Bicycle-Lane Solution</title>
  <style>
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: #f4f7fb;
      color: #1f2937;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: white;
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      padding: 20px 24px;
      margin-bottom: 20px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .stat {{
      padding: 14px 16px;
      border-radius: 12px;
      background: #eef4ff;
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #64748b;
    }}
    .value {{
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
      color: #0f172a;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      margin-top: 10px;
      font-size: 14px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .line {{
      width: 28px;
      height: 0;
      border-top-width: 4px;
      border-top-style: solid;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      background: white;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Scaled No-Big-M Link-Based MILP Solution with Undirected Design Variables</h1>
      <div class="stats">
        <div class="stat"><div class="label">Status</div><div class="value">{status_name}</div></div>
        <div class="stat"><div class="label">Objective</div><div class="value">{objective:.2f}</div></div>
        <div class="stat"><div class="label">Chosen Links</div><div class="value">{chosen_count}</div></div>
        <div class="stat"><div class="label">Runtime (s)</div><div class="value">{runtime:.2f}</div></div>
        <div class="stat"><div class="label">Gap</div><div class="value">{gap:.4%}</div></div>
        <div class="stat"><div class="label">Target Share</div><div class="value">{target_share:.1%}</div></div>
        <div class="stat"><div class="label">Covered Share</div><div class="value">{covered_share:.1%}</div></div>
      </div>
      <div class="legend">
        <span class="chip"><span class="line" style="border-color:#cbd5e1"></span>Road network</span>
        <span class="chip"><span class="line" style="border-color:#16a34a"></span>Chosen bicycle lanes</span>
        <span class="chip"><span class="line" style="border-color:#fb923c"></span>Connectors</span>
      </div>
    </div>
    <div class="card">
      <img src="{svg_name}" alt="Direct bicycle lane solution">
    </div>
  </div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def draw_solution(
    data: TNTPNetworkData,
    result: Dict[str, object],
    zone_count: int,
    out_svg: str,
    out_png: str,
    out_summary_html: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to write SVG/PNG visualizations.")

    pos = compute_visual_positions(data, zone_count)
    through_nodes = {node for node in data.nodes if node > zone_count}
    chosen_links = {tuple(link) for link in result.get("chosen_links", [])}

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    pad_x = 0.04 * width
    pad_y = 0.04 * height
    aspect = height / width

    fig_width = 12.0
    fig_height = max(8.0, fig_width * aspect)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.set_facecolor("#f8fafc")

    for idx, arc in enumerate(data.arcs):
        u, v = arc
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if data.arc_type.get(idx, 1) == 1:
            ax.plot([x0, x1], [y0, y1], color="#cbd5e1", linewidth=0.8, alpha=0.85, zorder=1)
        else:
            ax.plot([x0, x1], [y0, y1], color="#fb923c", linewidth=0.9, alpha=0.55, linestyle="--", zorder=2)

    for u, v in chosen_links:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        arc_idx = data.arc_to_idx.get((u, v))
        color = "#16a34a" if arc_idx is not None and data.arc_type.get(arc_idx, 1) == 1 else "#f97316"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=3.0, alpha=0.95, zorder=4)

    zone_x = [pos[node][0] for node in range(1, zone_count + 1) if node in pos]
    zone_y = [pos[node][1] for node in range(1, zone_count + 1) if node in pos]
    ax.scatter(zone_x, zone_y, s=42, color="#f59e0b", edgecolors="#7c2d12", linewidths=0.6, zorder=5)
    for node in range(1, zone_count + 1):
        if node not in pos:
            continue
        x, y = pos[node]
        ax.text(x, y, str(node), fontsize=7, color="#7c2d12", ha="center", va="center", zorder=6)

    title = (
        f"Scaled No-Big-M Link-Based MILP | status={result.get('status_name', 'UNKNOWN')} | "
        f"obj={result.get('objective', math.inf):.2f} | "
        f"chosen_links={len(chosen_links)} | "
        f"runtime={result.get('runtime', math.nan):.2f}s"
    )
    ax.set_title(title, fontsize=12, pad=16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.axis("off")

    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    write_summary_html(result, os.path.basename(out_svg), out_summary_html)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the scaled no-Big-M direct link-based bicycle-lane MILP on the current polygon TNTP data "
            "without MTZ cycle-elimination variables/constraints. OD service is modeled through scaled flow, "
            "detour, and uncovered-length constraints instead of a Big-M uncovered-cap constraint. "
            "Construction decisions use one binary variable per undirected physical link, so opposite directions "
            "share the same design variable rather than being linked by equality constraints. "
            "OD nodes are relocated to the centroid of the road nodes they connect to. The default version also adds five strengthening cut families."
        )
    )
    parser.add_argument(
        "--network-dir",
        default=DEFAULT_NETWORK_DIR,
        help="Directory containing net.tntp, trips.tntp, node.tntp, and n2024_polygon_id_map.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated JSON, HTML, SVG, and PNG outputs.",
    )
    parser.add_argument(
        "--zone-geometries",
        default=DEFAULT_ZONE_GEOMETRIES_PATH,
        help="CSV with TAZ polygon WKT geometries. Use an empty string to disable TAZ overlays.",
    )
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument(
        "--covered-demand-share",
        type=float,
        default=0.8 ,
        help="Fraction or percentage of total demand that must satisfy the coverage cap, e.g. 0.8 or 80.",
    )
    parser.add_argument("--od-limit", type=int, default=0, help="0 means all OD pairs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=72000.0)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument(
        "--disable-od-arc-pruning",
        action="store_true",
        help="Disable OD-specific detour-feasible arc pruning and create OD-arc variables on all arcs.",
    )
    parser.add_argument(
        "--disable-node-reachability-pruning",
        action="store_true",
        help=(
            "Disable exact node-based reachability pruning. By default, for each OD, nodes satisfying "
            "dist(o,i)+dist(i,d) > (1+epsilon)*K_od are removed from that OD's arc-variable corridor."
        ),
    )
    parser.add_argument(
        "--disable-source-sink-variable-pruning",
        action="store_true",
        help=(
            "Disable exact source/sink variable pruning. By default, for each OD, arcs entering "
            "the origin and arcs leaving the destination are not defined as OD-arc variables."
        ),
    )
    parser.add_argument(
        "--disable-directed-subgraph-trimming",
        action="store_true",
        help=(
            "Disable exact directed subgraph trimming. By default, after OD-specific arc pruning, "
            "only arcs that remain on a directed origin-to-destination path in the pruned subgraph "
            "are defined as OD-arc variables."
        ),
    )
    parser.add_argument(
        "--disable-unused-design-unit-pruning",
        action="store_true",
        help=(
            "Disable unused design-unit pruning. By default, construction variables are created "
            "only for undirected physical links that appear in at least one retained OD-arc set."
        ),
    )
    parser.add_argument(
        "--drop-long-uncovered-arcs",
        action="store_true",
        help=(
            "Optional non-exact acceleration: for each OD, remove arcs whose individual length "
            "exceeds t*K_od from that OD path-variable set. This is off by default because "
            "a long arc may still be feasible when the corresponding design unit is constructed."
        ),
    )
    parser.add_argument(
        "--disable-origin-destination-strengthening",
        action="store_true",
        help="Disable explicit source/sink strengthening constraints for each OD.",
    )
    parser.add_argument(
        "--disable-opposite-direction-x-cut",
        action="store_true",
        help="Disable x_od,ij + x_od,ji <= z_od cuts for each undirected design unit.",
    )
    parser.add_argument(
        "--disable-single-arc-implied-cut",
        default=False,
        action="store_true",
        help="Disable the valid inequality ell_a * (x_od,a - delta_e(a)) <= t*K_od*z_od.",
    )
    parser.add_argument(
        "--disable-minimum-covered-length-cut",
        action="store_true",
        help="Disable the aggregate minimum bicycle-lane-supported path-length cut.",
    )
    parser.add_argument(
        "--disable-single-arc-detour-implied-cut",
        default=False,
        action="store_true",
        help="Disable the valid inequality ell_a * x_od,a <= (1+epsilon)*K_od*z_od.",
    )
    parser.add_argument(
        "--disable-balanced-od-sampling",
        action="store_true",
        help=(
            "Disable balanced OD sampling. By default, random OD selection tries to ensure "
            "each selected OD node has at least one outgoing and one incoming selected demand."
        ),
    )
    parser.add_argument("--prefix", default="scaled_no_big_m_undirected_design_pruned_exact_5cuts_odpaths_solution")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    network_dir = resolve_project_path(args.network_dir)
    output_dir = resolve_project_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    net_path = os.path.join(network_dir, "net.tntp")
    node_path = os.path.join(network_dir, "node.tntp")
    id_map_path = os.path.join(network_dir, "n2024_polygon_id_map.csv")
    zone_geometries_path = resolve_project_path(args.zone_geometries) if args.zone_geometries else ""

    zone_count = read_zone_count(net_path)
    original_id_map = load_original_id_map(id_map_path)
    zone_geometries = load_zone_geometries(zone_geometries_path)
    zone_centers = load_zone_centers_by_original_id(network_dir)
    boundary_polygon = load_boundary_polygon(network_dir)
    result = build_and_solve_direct_milp_with_warm_start(
        input_dir=network_dir,
        epsilon=args.epsilon,
        t=args.t,
        covered_demand_share=args.covered_demand_share,
        od_limit=args.od_limit,
        seed=args.seed,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        output_flag=args.output_flag,
        threads=args.threads,
        use_od_arc_pruning=not args.disable_od_arc_pruning,
        use_node_reachability_pruning=not args.disable_node_reachability_pruning,
        use_source_sink_variable_pruning=not args.disable_source_sink_variable_pruning,
        use_directed_subgraph_trimming=not args.disable_directed_subgraph_trimming,
        use_unused_design_unit_pruning=not args.disable_unused_design_unit_pruning,
        drop_long_uncovered_arcs=args.drop_long_uncovered_arcs,
        add_origin_destination_strengthening=not args.disable_origin_destination_strengthening,
        add_opposite_direction_x_cut=not args.disable_opposite_direction_x_cut,
        add_single_arc_implied_cut=not args.disable_single_arc_implied_cut,
        add_minimum_covered_length_cut=not args.disable_minimum_covered_length_cut,
        add_single_arc_detour_implied_cut=not args.disable_single_arc_detour_implied_cut,
        balanced_od_sampling=not args.disable_balanced_od_sampling,
    )
    if "chosen_links" not in result:
        raise SystemExit("No feasible incumbent was found, so there is nothing to visualize.")

    data = TNTPNetworkData(network_dir)
    out_svg = os.path.join(output_dir, f"{args.prefix}.svg")
    out_png = os.path.join(output_dir, f"{args.prefix}.png")
    out_html = os.path.join(output_dir, f"{args.prefix}.html")
    out_summary_html = os.path.join(output_dir, f"{args.prefix}_summary.html")
    out_json = os.path.join(output_dir, f"{args.prefix}.json")

    draw_solution(data, result, zone_count, out_svg, out_png, out_summary_html)
    write_interactive_html(
        data,
        result,
        zone_count,
        original_id_map,
        zone_geometries,
        zone_centers,
        boundary_polygon,
        out_html,
    )
    clean_result = sanitize_result(result, original_id_map)
    clean_result["node_file"] = node_path
    clean_result["zone_count"] = zone_count
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_result, f, ensure_ascii=False, indent=2)

    print(f"Wrote interactive HTML   : {out_html}")
    print(f"Wrote summary HTML       : {out_summary_html}")
    print(f"Wrote visualization SVG : {out_svg}")
    print(f"Wrote visualization PNG : {out_png}")
    print(f"Wrote solution JSON     : {out_json}")


if __name__ == "__main__":
    main()
