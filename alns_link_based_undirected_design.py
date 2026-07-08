#!/usr/bin/env python3
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Arc = Tuple[int, int]
OD = Tuple[int, int]

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_NETWORK_DIR = Path("data") / "full_network"
DEFAULT_OUTPUT_DIR = Path("outputs") / "alns_full_network"


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return PACKAGE_ROOT / resolved


@dataclass(frozen=True)
class CandidatePath:
    arcs: Tuple[int, ...]
    units: Tuple[int, ...]
    unit_lengths: Tuple[Tuple[int, float], ...]
    length: float


@dataclass
class SolutionState:
    chosen_units: Set[int]
    cost: float
    covered_demand: float
    covered_share: float
    objective: float
    covered_ods: Set[OD]
    od_best_path: Dict[OD, int]
    od_uncovered_length: Dict[OD, float]


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


class TNTPNetworkData:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.net_path = self.input_dir / "net.tntp"
        self.trips_path = self.input_dir / "trips.tntp"
        self.node_path = self.input_dir / "node.tntp"
        self.id_map_path = self.input_dir / "n2024_polygon_id_map.csv"

        self.zone_count = self._read_zone_count()
        self.nodes: List[int] = []
        self.arcs: List[Arc] = []
        self.arc_len: Dict[int, float] = {}
        self.arc_cost: Dict[int, float] = {}
        self.arc_type: Dict[int, int] = {}
        self.arc_to_idx: Dict[Arc, int] = {}
        self.idx_to_arc: Dict[int, Arc] = {}
        self.out_arcs: Dict[int, List[int]] = {}
        self.in_arcs: Dict[int, List[int]] = {}
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.original_id: Dict[int, str] = {}
        self.od_demand: Dict[OD, float] = {}

        self._parse_network()
        self._parse_nodes()
        self._parse_id_map()
        self._parse_trips()

    def _read_zone_count(self) -> int:
        for line in self.net_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.match(r"\s*<NUMBER OF ZONES>\s+(\d+)", line)
            if match:
                return int(match.group(1))
        raise ValueError(f"Cannot read <NUMBER OF ZONES> from {self.net_path}")

    def _parse_network(self) -> None:
        nodes_set: Set[int] = set()
        for line in self.net_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip().rstrip(";").strip()
            if not raw or raw.startswith("<") or raw.startswith("~"):
                continue
            parts = raw.split()
            if len(parts) < 10:
                continue
            u = int(parts[0])
            v = int(parts[1])
            length = float(parts[3])
            link_type = int(float(parts[9]))
            idx = len(self.arcs)
            self.arcs.append((u, v))
            self.idx_to_arc[idx] = (u, v)
            self.arc_to_idx[(u, v)] = idx
            self.arc_len[idx] = length
            self.arc_cost[idx] = length
            self.arc_type[idx] = link_type
            nodes_set.add(u)
            nodes_set.add(v)

        self.nodes = sorted(nodes_set)
        self.out_arcs = {node: [] for node in self.nodes}
        self.in_arcs = {node: [] for node in self.nodes}
        for idx, (u, v) in enumerate(self.arcs):
            self.out_arcs[u].append(idx)
            self.in_arcs[v].append(idx)

    def _parse_nodes(self) -> None:
        for line in self.node_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip().rstrip(";").strip()
            if not raw or raw.lower().startswith("node"):
                continue
            parts = raw.split()
            if len(parts) >= 3:
                self.pos[int(parts[0])] = (float(parts[1]), float(parts[2]))

    def _parse_id_map(self) -> None:
        if not self.id_map_path.exists():
            self.original_id = {node: str(node) for node in self.nodes}
            return
        with self.id_map_path.open(newline="", encoding="utf-8") as f:
            self.original_id = {int(row["numeric_id"]): row["original_id"] for row in csv.DictReader(f)}

    def _parse_trips(self) -> None:
        origin: Optional[int] = None
        for line in self.trips_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("<"):
                continue
            match = re.match(r"Origin\s+(\d+)", raw)
            if match:
                origin = int(match.group(1))
                continue
            if origin is None:
                continue
            for dest_str, demand_str in re.findall(r"(\d+)\s*:\s*([0-9.]+)", raw):
                dest = int(dest_str)
                demand = float(demand_str)
                if origin != dest and demand > 1e-9:
                    self.od_demand[(origin, dest)] = demand


def build_undirected_design_units(
    data: TNTPNetworkData,
) -> Tuple[List[Arc], Dict[int, int], Dict[int, List[int]], Dict[int, float]]:
    unit_to_arcs: Dict[Arc, List[int]] = {}
    arc_to_key: Dict[int, Arc] = {}
    for arc_idx, (u, v) in enumerate(data.arcs):
        if data.arc_type.get(arc_idx, 1) != 1:
            continue
        key = (u, v) if u <= v else (v, u)
        unit_to_arcs.setdefault(key, []).append(arc_idx)
        arc_to_key[arc_idx] = key

    units = sorted(unit_to_arcs)
    key_to_unit = {key: idx for idx, key in enumerate(units)}
    arc_to_unit = {arc_idx: key_to_unit[key] for arc_idx, key in arc_to_key.items()}
    unit_arcs = {key_to_unit[key]: sorted(arcs) for key, arcs in unit_to_arcs.items()}
    unit_cost = {
        unit_idx: max(data.arc_cost[arc_idx] for arc_idx in arcs)
        for unit_idx, arcs in unit_arcs.items()
    }
    return units, arc_to_unit, unit_arcs, unit_cost


def select_ods(od_demand: Dict[OD, float], od_limit: int, seed: int) -> Dict[OD, float]:
    ods = sorted(od_demand)
    if od_limit <= 0 or od_limit >= len(ods):
        return {od: od_demand[od] for od in ods}
    rng = random.Random(seed)
    chosen = sorted(rng.sample(ods, od_limit))
    return {od: od_demand[od] for od in chosen}


def dijkstra_path(
    data: TNTPNetworkData,
    origin: int,
    destination: int,
    weight_multiplier: Optional[Dict[int, float]] = None,
) -> Tuple[float, List[int]]:
    dist = {origin: 0.0}
    pred: Dict[int, Tuple[int, int]] = {}
    heap = [(0.0, origin)]
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist != dist[node]:
            continue
        if node == destination:
            break
        for arc_idx in data.out_arcs.get(node, []):
            _u, v = data.idx_to_arc[arc_idx]
            mult = 1.0 if weight_multiplier is None else weight_multiplier.get(arc_idx, 1.0)
            new_dist = cur_dist + data.arc_len[arc_idx] * mult
            if new_dist < dist.get(v, math.inf):
                dist[v] = new_dist
                pred[v] = (node, arc_idx)
                heapq.heappush(heap, (new_dist, v))

    if destination not in dist:
        raise ValueError(f"OD {origin}->{destination} is unreachable")

    arcs: List[int] = []
    node = destination
    while node != origin:
        prev, arc_idx = pred[node]
        arcs.append(arc_idx)
        node = prev
    arcs.reverse()
    true_length = sum(data.arc_len[arc_idx] for arc_idx in arcs)
    return true_length, arcs


def make_candidate_path(
    data: TNTPNetworkData,
    arcs: Sequence[int],
    arc_to_unit: Dict[int, int],
) -> CandidatePath:
    unit_lengths: Dict[int, float] = {}
    units: Set[int] = set()
    total = 0.0
    for arc_idx in arcs:
        if arc_idx not in arc_to_unit:
            continue
        unit = arc_to_unit[arc_idx]
        length = float(data.arc_len[arc_idx])
        unit_lengths[unit] = unit_lengths.get(unit, 0.0) + length
        units.add(unit)
        total += length
    return CandidatePath(
        arcs=tuple(arcs),
        units=tuple(sorted(units)),
        unit_lengths=tuple(sorted(unit_lengths.items())),
        length=total,
    )


def build_path_pool(
    data: TNTPNetworkData,
    ods: Sequence[OD],
    arc_to_unit: Dict[int, int],
    epsilon: float,
    path_pool_size: int,
    seed: int,
    penalty_growth: float,
    random_noise: float,
) -> Tuple[Dict[OD, List[CandidatePath]], Dict[OD, float]]:
    rng = random.Random(seed)
    path_pool: Dict[OD, List[CandidatePath]] = {}
    shortest: Dict[OD, float] = {}

    for od_idx, od in enumerate(ods, start=1):
        origin, destination = od
        base_length, base_arcs = dijkstra_path(data, origin, destination)
        shortest[od] = base_length
        cap = (1.0 + epsilon) * base_length
        seen_paths = {tuple(base_arcs)}
        paths = [make_candidate_path(data, base_arcs, arc_to_unit)]
        arc_penalty: Dict[int, float] = {arc_idx: 1.0 for arc_idx in base_arcs}

        attempts = 0
        while len(paths) < path_pool_size and attempts < max(25, path_pool_size * 8):
            attempts += 1
            multipliers: Dict[int, float] = {}
            for arc_idx in range(len(data.arcs)):
                penalty = arc_penalty.get(arc_idx, 0.0)
                if penalty <= 0.0 and random_noise <= 0.0:
                    continue
                multipliers[arc_idx] = 1.0 + penalty + rng.random() * random_noise

            try:
                true_length, arcs = dijkstra_path(data, origin, destination, multipliers)
            except ValueError:
                continue
            key = tuple(arcs)
            for arc_idx in arcs:
                arc_penalty[arc_idx] = arc_penalty.get(arc_idx, 0.0) + penalty_growth * rng.uniform(0.65, 1.35)
            if true_length <= cap + 1e-9 and key not in seen_paths:
                seen_paths.add(key)
                paths.append(make_candidate_path(data, arcs, arc_to_unit))

        path_pool[od] = sorted(paths, key=lambda path: (path.length, len(path.arcs)))
        if od_idx % 100 == 0:
            print(f"Built path pools for {od_idx}/{len(ods)} ODs")
    return path_pool, shortest


def uncovered_length(path: CandidatePath, chosen_units: Set[int]) -> float:
    covered = 0.0
    for unit, length in path.unit_lengths:
        if unit in chosen_units:
            covered += length
    return max(0.0, path.length - covered)


def evaluate_solution(
    chosen_units: Set[int],
    path_pool: Dict[OD, List[CandidatePath]],
    od_demand: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t: float,
    target_demand: float,
    penalty_factor: float,
) -> SolutionState:
    cost = sum(unit_cost[unit] for unit in chosen_units)
    covered_demand = 0.0
    covered_ods: Set[OD] = set()
    od_best_path: Dict[OD, int] = {}
    od_uncovered: Dict[OD, float] = {}

    for od, paths in path_pool.items():
        best_idx = 0
        best_uncovered = math.inf
        best_len = math.inf
        for idx, path in enumerate(paths):
            unc = uncovered_length(path, chosen_units)
            if (unc, path.length) < (best_uncovered, best_len):
                best_idx = idx
                best_uncovered = unc
                best_len = path.length
        od_best_path[od] = best_idx
        od_uncovered[od] = best_uncovered
        if best_uncovered <= t * shortest[od] + 1e-8:
            covered_ods.add(od)
            covered_demand += od_demand[od]

    shortfall = max(0.0, target_demand - covered_demand)
    objective = cost + penalty_factor * shortfall
    total_demand = sum(od_demand.values())
    return SolutionState(
        chosen_units=set(chosen_units),
        cost=cost,
        covered_demand=covered_demand,
        covered_share=covered_demand / total_demand if total_demand else 0.0,
        objective=objective,
        covered_ods=covered_ods,
        od_best_path=od_best_path,
        od_uncovered_length=od_uncovered,
    )


def score_candidate_units(
    state: SolutionState,
    path_pool: Dict[OD, List[CandidatePath]],
    od_demand: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t: float,
    max_paths_per_od: int,
) -> List[Tuple[float, int]]:
    scores: Dict[int, float] = {}
    for od, demand in od_demand.items():
        cap = t * shortest[od]
        best_deficit = max(0.0, state.od_uncovered_length[od] - cap)
        if best_deficit <= 1e-8:
            continue
        paths = path_pool[od][:max_paths_per_od]
        for path in paths:
            path_unc = uncovered_length(path, state.chosen_units)
            deficit = max(0.0, path_unc - cap)
            if deficit <= 1e-8:
                continue
            for unit, length in path.unit_lengths:
                if unit in state.chosen_units:
                    continue
                improvement = min(length, deficit)
                scores[unit] = scores.get(unit, 0.0) + demand * improvement / max(deficit, 1.0)

    ranked = []
    for unit, score in scores.items():
        ranked.append((score / max(unit_cost[unit], 1e-9), unit))
    ranked.sort(reverse=True)
    return ranked


def greedy_repair(
    chosen: Set[int],
    current_state: SolutionState,
    evaluator,
    path_pool: Dict[OD, List[CandidatePath]],
    od_demand: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    target_share: float,
    t: float,
    rng: random.Random,
    candidate_top_k: int,
    max_additions: int,
    max_paths_per_od: int,
) -> SolutionState:
    state = current_state
    additions = 0
    total_demand = sum(od_demand.values())
    while state.covered_share + 1e-12 < target_share and additions < max_additions:
        ranked = score_candidate_units(state, path_pool, od_demand, shortest, unit_cost, t, max_paths_per_od)
        ranked = [(score, unit) for score, unit in ranked if unit not in chosen]
        if not ranked:
            break
        candidates = [unit for _score, unit in ranked[:candidate_top_k]]
        best_state = None
        best_unit = None
        for unit in candidates:
            trial = set(chosen)
            trial.add(unit)
            trial_state = evaluator(trial)
            if best_state is None or (
                trial_state.objective,
                trial_state.cost,
                -trial_state.covered_demand,
            ) < (
                best_state.objective,
                best_state.cost,
                -best_state.covered_demand,
            ):
                best_state = trial_state
                best_unit = unit
        if best_state is None or best_unit is None:
            break
        chosen.add(best_unit)
        state = best_state
        additions += 1
        if state.covered_demand >= target_share * total_demand - 1e-9:
            break
    return state


def random_repair(
    chosen: Set[int],
    current_state: SolutionState,
    evaluator,
    path_pool: Dict[OD, List[CandidatePath]],
    od_demand: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    target_share: float,
    t: float,
    rng: random.Random,
    candidate_top_k: int,
    max_additions: int,
    max_paths_per_od: int,
) -> SolutionState:
    state = current_state
    additions = 0
    while state.covered_share + 1e-12 < target_share and additions < max_additions:
        ranked = score_candidate_units(state, path_pool, od_demand, shortest, unit_cost, t, max_paths_per_od)
        ranked = [(score, unit) for score, unit in ranked if unit not in chosen][: max(candidate_top_k, 1)]
        if not ranked:
            break
        weights = [max(score, 1e-9) for score, _unit in ranked]
        total = sum(weights)
        pick = rng.random() * total
        acc = 0.0
        selected = ranked[-1][1]
        for weight, (_score, unit) in zip(weights, ranked):
            acc += weight
            if acc >= pick:
                selected = unit
                break
        chosen.add(selected)
        state = evaluator(chosen)
        additions += 1
    return state


def add_to_improve_repair(
    chosen: Set[int],
    current_state: SolutionState,
    evaluator,
    path_pool: Dict[OD, List[CandidatePath]],
    od_demand: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    target_share: float,
    t: float,
    rng: random.Random,
    candidate_top_k: int,
    max_additions: int,
    max_paths_per_od: int,
) -> SolutionState:
    state = current_state
    ranked = score_candidate_units(state, path_pool, od_demand, shortest, unit_cost, t, max_paths_per_od)
    additions = 0
    for _score, unit in ranked[: candidate_top_k * 2]:
        if additions >= max_additions:
            break
        if unit in chosen:
            continue
        trial = set(chosen)
        trial.add(unit)
        trial_state = evaluator(trial)
        if trial_state.objective <= state.objective or trial_state.covered_demand > state.covered_demand:
            chosen.add(unit)
            state = trial_state
            additions += 1
        if state.covered_share + 1e-12 >= target_share and additions >= max(1, max_additions // 3):
            break
    if state.covered_share < target_share:
        state = greedy_repair(
            chosen, state, evaluator, path_pool, od_demand, shortest, unit_cost, target_share,
            t, rng, candidate_top_k, max_additions, max_paths_per_od,
        )
    return state


def destroy_random(chosen: Set[int], rng: random.Random, fraction: float, **_kwargs) -> Set[int]:
    if not chosen:
        return set()
    remove_count = max(1, int(round(len(chosen) * fraction)))
    remove = set(rng.sample(sorted(chosen), min(remove_count, len(chosen))))
    return set(chosen - remove)


def destroy_costly(chosen: Set[int], rng: random.Random, fraction: float, unit_cost: Dict[int, float], **_kwargs) -> Set[int]:
    if not chosen:
        return set()
    remove_count = max(1, int(round(len(chosen) * fraction)))
    ordered = sorted(chosen, key=lambda unit: unit_cost[unit], reverse=True)
    return set(chosen - set(ordered[:remove_count]))


def destroy_low_impact(
    chosen: Set[int],
    rng: random.Random,
    fraction: float,
    evaluator,
    current_state: SolutionState,
    unit_cost: Dict[int, float],
    sample_size: int,
    **_kwargs,
) -> Set[int]:
    if not chosen:
        return set()
    remove_count = max(1, int(round(len(chosen) * fraction)))
    sample = sorted(chosen)
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)
    scored = []
    for unit in sample:
        trial = set(chosen)
        trial.remove(unit)
        state = evaluator(trial)
        coverage_loss = max(0.0, current_state.covered_demand - state.covered_demand)
        scored.append((coverage_loss / max(unit_cost[unit], 1e-9), unit))
    scored.sort()
    remove = {unit for _score, unit in scored[:remove_count]}
    return set(chosen - remove)


def destroy_path_based(
    chosen: Set[int],
    rng: random.Random,
    fraction: float,
    current_state: SolutionState,
    path_pool: Dict[OD, List[CandidatePath]],
    **_kwargs,
) -> Set[int]:
    if not chosen or not current_state.covered_ods:
        return destroy_random(chosen, rng, fraction)
    remove_count = max(1, int(round(len(chosen) * fraction)))
    covered = sorted(current_state.covered_ods)
    rng.shuffle(covered)
    remove: Set[int] = set()
    for od in covered:
        path = path_pool[od][current_state.od_best_path[od]]
        candidates = [unit for unit in path.units if unit in chosen]
        rng.shuffle(candidates)
        for unit in candidates:
            remove.add(unit)
            if len(remove) >= remove_count:
                break
        if len(remove) >= remove_count:
            break
    if not remove:
        return destroy_random(chosen, rng, fraction)
    return set(chosen - remove)


def adaptive_pick(weights: Dict[str, float], rng: random.Random) -> str:
    total = sum(max(weight, 1e-9) for weight in weights.values())
    pick = rng.random() * total
    acc = 0.0
    last = next(iter(weights))
    for name, weight in weights.items():
        last = name
        acc += max(weight, 1e-9)
        if acc >= pick:
            return name
    return last


def update_weights(
    weights: Dict[str, float],
    scores: Dict[str, float],
    counts: Dict[str, int],
    reaction: float,
) -> None:
    for name in weights:
        if counts.get(name, 0) > 0:
            avg = scores[name] / counts[name]
            weights[name] = (1.0 - reaction) * weights[name] + reaction * max(avg, 0.05)
    scores.clear()
    counts.clear()


def prune_solution(
    state: SolutionState,
    evaluator,
    target_share: float,
    unit_cost: Dict[int, float],
    rng: random.Random,
    passes: int,
) -> SolutionState:
    best = state
    for _ in range(passes):
        units = sorted(best.chosen_units, key=lambda unit: unit_cost[unit], reverse=True)
        # Mix in a small randomization so repeated passes are not identical.
        chunks = [units[i : i + 25] for i in range(0, len(units), 25)]
        for chunk in chunks:
            rng.shuffle(chunk)
        units = [unit for chunk in chunks for unit in chunk]
        changed = False
        for unit in units:
            trial = set(best.chosen_units)
            trial.remove(unit)
            state = evaluator(trial)
            if state.covered_share + 1e-12 >= target_share and state.cost <= best.cost + 1e-9:
                best = state
                changed = True
        if not changed:
            break
    return best


def run_alns(args: argparse.Namespace) -> Tuple[TNTPNetworkData, Dict[str, object], Dict[OD, List[CandidatePath]], Dict[int, int], List[Arc]]:
    start_time = time.time()
    rng = random.Random(args.seed)
    target_share = normalize_demand_share(args.covered_demand_share)
    data = TNTPNetworkData(args.network_dir)
    design_units, arc_to_unit, _unit_arcs, unit_cost = build_undirected_design_units(data)
    od_demand = select_ods(data.od_demand, args.od_limit, args.seed)
    ods = sorted(od_demand)
    total_demand = sum(od_demand.values())
    target_demand = target_share * total_demand
    penalty_factor = args.penalty_factor
    if penalty_factor <= 0:
        penalty_factor = max(unit_cost.values()) * 100.0 if unit_cost else 1e6

    print(f"Network: {len(data.nodes)} nodes, {len(data.arcs)} directed arcs, {len(design_units)} design units")
    print(f"ODs: {len(ods)} positive OD pairs, total demand={total_demand:.6f}, target={target_demand:.6f}")
    path_pool, shortest = build_path_pool(
        data,
        ods,
        arc_to_unit,
        args.epsilon,
        args.path_pool_size,
        args.seed,
        args.path_penalty_growth,
        args.path_random_noise,
    )
    avg_paths = sum(len(paths) for paths in path_pool.values()) / max(len(path_pool), 1)
    print(f"Candidate path pool ready: avg {avg_paths:.2f} paths/OD")

    def evaluator(chosen_units: Set[int]) -> SolutionState:
        return evaluate_solution(
            chosen_units,
            path_pool,
            od_demand,
            shortest,
            unit_cost,
            args.t,
            target_demand,
            penalty_factor,
        )

    empty_state = evaluator(set())
    current = greedy_repair(
        set(),
        empty_state,
        evaluator,
        path_pool,
        od_demand,
        shortest,
        unit_cost,
        target_share,
        args.t,
        rng,
        args.candidate_top_k,
        args.initial_max_additions,
        args.max_paths_per_od_scoring,
    )
    best = current
    best_feasible = current if current.covered_share + 1e-12 >= target_share else None
    print(f"Initial: cost={current.cost:.3f}, covered={current.covered_share:.4f}, units={len(current.chosen_units)}")

    destroy_ops = {
        "random": destroy_random,
        "costly": destroy_costly,
        "low_impact": destroy_low_impact,
        "path_based": destroy_path_based,
    }
    repair_ops = {
        "greedy": greedy_repair,
        "random": random_repair,
        "improve": add_to_improve_repair,
    }
    destroy_weights = {name: 1.0 for name in destroy_ops}
    repair_weights = {name: 1.0 for name in repair_ops}
    destroy_scores: Dict[str, float] = {}
    repair_scores: Dict[str, float] = {}
    destroy_counts: Dict[str, int] = {}
    repair_counts: Dict[str, int] = {}

    temperature = max(args.initial_temperature, 0.0)
    last_report = time.time()

    for iteration in range(1, args.iterations + 1):
        if args.time_limit > 0 and time.time() - start_time >= args.time_limit:
            break

        destroy_name = adaptive_pick(destroy_weights, rng)
        repair_name = adaptive_pick(repair_weights, rng)
        fraction = rng.uniform(args.destroy_fraction_min, args.destroy_fraction_max)
        destroyed = destroy_ops[destroy_name](
            current.chosen_units,
            rng=rng,
            fraction=fraction,
            evaluator=evaluator,
            current_state=current,
            unit_cost=unit_cost,
            sample_size=args.low_impact_sample_size,
            path_pool=path_pool,
        )
        destroyed_state = evaluator(destroyed)
        repaired = repair_ops[repair_name](
            set(destroyed),
            destroyed_state,
            evaluator,
            path_pool,
            od_demand,
            shortest,
            unit_cost,
            target_share,
            args.t,
            rng,
            args.candidate_top_k,
            args.repair_max_additions,
            args.max_paths_per_od_scoring,
        )

        delta = repaired.objective - current.objective
        accepted = delta <= 0.0 or (temperature > 1e-12 and rng.random() < math.exp(-delta / temperature))
        score = 0.0
        if accepted:
            current = repaired
            score = 1.0
            if repaired.objective < best.objective - 1e-9:
                best = repaired
                score = 4.0
            if repaired.covered_share + 1e-12 >= target_share:
                if best_feasible is None or repaired.cost < best_feasible.cost - 1e-9:
                    best_feasible = repaired
                    score = 8.0

        destroy_scores[destroy_name] = destroy_scores.get(destroy_name, 0.0) + score
        repair_scores[repair_name] = repair_scores.get(repair_name, 0.0) + score
        destroy_counts[destroy_name] = destroy_counts.get(destroy_name, 0) + 1
        repair_counts[repair_name] = repair_counts.get(repair_name, 0) + 1

        temperature *= args.cooling_rate
        if iteration % args.adaptation_interval == 0:
            update_weights(destroy_weights, destroy_scores, destroy_counts, args.reaction)
            update_weights(repair_weights, repair_scores, repair_counts, args.reaction)

        now = time.time()
        if args.report_interval > 0 and (iteration == 1 or now - last_report >= args.report_interval):
            incumbent = best_feasible if best_feasible is not None else best
            print(
                f"iter={iteration} current(cost={current.cost:.2f},cov={current.covered_share:.4f},units={len(current.chosen_units)}) "
                f"best(cost={incumbent.cost:.2f},cov={incumbent.covered_share:.4f},units={len(incumbent.chosen_units)}) "
                f"temp={temperature:.3f}"
            )
            last_report = now

    final_state = best_feasible if best_feasible is not None else best
    if final_state.covered_share + 1e-12 >= target_share:
        final_state = prune_solution(final_state, evaluator, target_share, unit_cost, rng, args.prune_passes)

    runtime = time.time() - start_time
    chosen_links = sorted(
        {
            data.idx_to_arc[arc_idx]
            for arc_idx, unit in arc_to_unit.items()
            if unit in final_state.chosen_units
        }
    )
    chosen_design_units = sorted(design_units[unit] for unit in final_state.chosen_units)

    od_paths = {}
    od_stats = {}
    for od in ods:
        path = path_pool[od][final_state.od_best_path[od]]
        path_arcs = [data.idx_to_arc[arc_idx] for arc_idx in path.arcs]
        unc = final_state.od_uncovered_length[od]
        covered = od in final_state.covered_ods
        od_paths[od] = path_arcs
        od_stats[od] = {
            "demand": od_demand[od],
            "coverage_required": covered,
            "path_source": "alns_candidate_path_pool",
            "reference_path_used": not covered,
            "path_length": path.length,
            "shortest_length": shortest[od],
            "length_to_shortest_ratio": path.length / shortest[od] if shortest[od] > 1e-12 else math.nan,
            "detour_cap": (1.0 + args.epsilon) * shortest[od],
            "uncovered_length": unc,
            "covered_length": max(0.0, path.length - unc),
            "cover_ratio": max(0.0, path.length - unc) / path.length if path.length > 1e-12 else 0.0,
            "uncovered_ratio": unc / path.length if path.length > 1e-12 else 0.0,
            "uncovered_cap_if_required": args.t * shortest[od],
            "uncovered_cap_active": args.t * shortest[od] if covered else 0.0,
        }

    result = {
        "algorithm": "ALNS",
        "status_name": "FEASIBLE" if final_state.covered_share + 1e-12 >= target_share else "INFEASIBLE_HEURISTIC",
        "runtime": runtime,
        "objective": final_state.cost,
        "best_bound": math.nan,
        "gap": math.nan,
        "num_ods": len(ods),
        "num_arcs": len(data.arcs),
        "num_nodes": len(data.nodes),
        "num_design_units": len(design_units),
        "chosen_design_unit_count": len(final_state.chosen_units),
        "chosen_links": chosen_links,
        "chosen_design_units": chosen_design_units,
        "total_demand": total_demand,
        "demand_target": target_demand,
        "demand_target_share": target_share,
        "covered_demand": final_state.covered_demand,
        "covered_demand_share": final_state.covered_share,
        "covered_od_count": len(final_state.covered_ods),
        "covered_ods": sorted(final_state.covered_ods),
        "relaxed_ods": sorted(set(ods) - final_state.covered_ods),
        "od_paths": od_paths,
        "od_stats": od_stats,
        "path_pool_size_requested": args.path_pool_size,
        "avg_candidate_paths_per_od": avg_paths,
        "epsilon": args.epsilon,
        "t": args.t,
        "iterations_requested": args.iterations,
        "destroy_weights": destroy_weights,
        "repair_weights": repair_weights,
    }
    return data, result, path_pool, arc_to_unit, design_units


def json_safe_result(result: Dict[str, object], original_id: Dict[int, str]) -> Dict[str, object]:
    clean = dict(result)
    clean["chosen_links"] = [
        {
            "numeric_u": u,
            "numeric_v": v,
            "original_u": original_id.get(u, str(u)),
            "original_v": original_id.get(v, str(v)),
        }
        for u, v in result["chosen_links"]  # type: ignore[index]
    ]
    clean["chosen_design_units"] = [
        {
            "numeric_u": u,
            "numeric_v": v,
            "original_u": original_id.get(u, str(u)),
            "original_v": original_id.get(v, str(v)),
        }
        for u, v in result["chosen_design_units"]  # type: ignore[index]
    ]
    clean["covered_ods"] = [f"{o}->{d}" for o, d in result["covered_ods"]]  # type: ignore[index]
    clean["relaxed_ods"] = [f"{o}->{d}" for o, d in result["relaxed_ods"]]  # type: ignore[index]
    clean["od_paths"] = {
        f"{od[0]}->{od[1]}": [
            {
                "numeric_u": u,
                "numeric_v": v,
                "original_u": original_id.get(u, str(u)),
                "original_v": original_id.get(v, str(v)),
            }
            for u, v in path
        ]
        for od, path in result["od_paths"].items()  # type: ignore[union-attr]
    }
    clean["od_stats"] = {f"{od[0]}->{od[1]}": stats for od, stats in result["od_stats"].items()}  # type: ignore[union-attr]
    return clean


def write_solution_csv(data: TNTPNetworkData, result: Dict[str, object], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v", "original_u", "original_v"])
        for u, v in result["chosen_links"]:  # type: ignore[index]
            writer.writerow([u, v, data.original_id.get(u, str(u)), data.original_id.get(v, str(v))])


def parse_polygon_wkt(wkt: str) -> List[List[List[float]]]:
    text = wkt.strip()
    if not text.upper().startswith("POLYGON"):
        return []
    body = text[text.find("(") :].strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
    rings: List[List[List[float]]] = []
    for ring_text in re.findall(r"\(([^()]*)\)", body):
        coords: List[List[float]] = []
        for item in ring_text.split(","):
            parts = item.strip().split()
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])
        if coords:
            rings.append(coords)
    return rings


def read_zone_polygons(data: TNTPNetworkData) -> List[Dict[str, object]]:
    candidates = [
        data.input_dir / "OD_geometries.csv",
        data.input_dir.parent / "OD_geometries.csv",
    ]
    geometry_path = next((path for path in candidates if path.exists()), None)
    if geometry_path is None:
        return []

    zone_ids = {
        data.original_id.get(node, str(node))
        for node in data.nodes
        if node <= data.zone_count
    }
    polygons: List[Dict[str, object]] = []
    with geometry_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            zone_id = str(row.get("id", "")).strip()
            if zone_id not in zone_ids:
                continue
            rings = parse_polygon_wkt(row.get("geometry", ""))
            if not rings:
                continue
            xs = [xy[0] for ring in rings for xy in ring]
            ys = [xy[1] for ring in rings for xy in ring]
            polygons.append(
                {
                    "id": zone_id,
                    "name": str(row.get("name", "")).strip(),
                    "rings": rings,
                    "center": [sum(xs) / len(xs), sum(ys) / len(ys)],
                }
            )
    return polygons


def write_html(data: TNTPNetworkData, result: Dict[str, object], out_path: Path) -> None:
    chosen = {tuple(edge) for edge in result["chosen_links"]}  # type: ignore[arg-type]
    od_paths = result["od_paths"]  # type: ignore[assignment]
    od_stats = result["od_stats"]  # type: ignore[assignment]
    zone_polygons = read_zone_polygons(data)
    nodes = [
        {
            "id": node,
            "original_id": data.original_id.get(node, str(node)),
            "x": data.pos[node][0],
            "y": data.pos[node][1],
            "kind": "od" if node <= data.zone_count else "through",
        }
        for node in data.nodes
        if node in data.pos
    ]
    edges = [
        {
            "u": u,
            "v": v,
            "chosen": (u, v) in chosen,
            "length": data.arc_len[idx],
        }
        for idx, (u, v) in enumerate(data.arcs)
        if u in data.pos and v in data.pos
    ]
    paths = []
    for od, arcs in od_paths.items():  # type: ignore[union-attr]
        stats = od_stats[od]  # type: ignore[index]
        paths.append(
            {
                "id": f"{od[0]}->{od[1]}",
                "origin": od[0],
                "destination": od[1],
                "original_origin": data.original_id.get(od[0], str(od[0])),
                "original_destination": data.original_id.get(od[1], str(od[1])),
                "coverage_required": bool(stats["coverage_required"]),
                "demand": float(stats["demand"]),
                "cover_ratio": float(stats["cover_ratio"]),
                "path_length": float(stats["path_length"]),
                "shortest_length": float(stats["shortest_length"]),
                "uncovered_length": float(stats["uncovered_length"]),
                "segments": [[u, v] for u, v in arcs],
            }
        )
    xs = [node["x"] for node in nodes]
    ys = [node["y"] for node in nodes]
    for zone in zone_polygons:
        for ring in zone["rings"]:  # type: ignore[index]
            for x, y in ring:
                xs.append(float(x))
                ys.append(float(y))
    payload = {
        "summary": {
            "algorithm": "ALNS",
            "status_name": result["status_name"],
            "objective": result["objective"],
            "runtime": result["runtime"],
            "chosen_links": len(result["chosen_links"]),  # type: ignore[arg-type]
            "covered_share": result["covered_demand_share"],
            "covered_demand": result["covered_demand"],
            "target_share": result["demand_target_share"],
            "covered_od_count": result["covered_od_count"],
            "num_ods": result["num_ods"],
        },
        "bounds": {"minX": min(xs), "maxX": max(xs), "minY": min(ys), "maxY": max(ys)},
        "nodes": nodes,
        "edges": edges,
        "paths": paths,
        "zones": zone_polygons,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ALNS Bicycle-Lane Solution</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #0f172a; background: #eef2f7; }}
    .layout {{ display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: 100vh; }}
    aside {{ padding: 18px; background: rgba(255,255,255,.95); border-right: 1px solid #d8dee8; overflow: auto; }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    .sub {{ color: #64748b; font-size: 13px; line-height: 1.45; margin-bottom: 14px; }}
    .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .stat {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 9px; background: #f8fbff; }}
    .stat span {{ display: block; color: #64748b; font-size: 11px; margin-bottom: 4px; }}
    .stat strong {{ font-size: 16px; }}
    .legend {{ margin-top: 14px; display: grid; gap: 7px; font-size: 12px; color: #475569; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 24px; height: 4px; border-radius: 2px; background: #94a3b8; }}
    .swatch.zone {{ height: 14px; border: 1px solid #7c3aed; background: rgba(124,58,237,.08); }}
    .swatch.bike {{ background: #16a34a; height: 6px; }}
    .swatch.path {{ background: #2563eb; }}
    label {{ display: flex; gap: 8px; align-items: center; margin-top: 12px; font-size: 13px; }}
    main {{ min-width: 0; }}
    canvas {{ display: block; width: 100%; height: 100vh; background: white; cursor: grab; }}
    canvas.dragging {{ cursor: grabbing; }}
    button {{ width: 100%; height: 36px; margin-top: 14px; border: 0; border-radius: 8px; background: #0f172a; color: white; font-weight: 700; }}
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <h1>ALNS Solution</h1>
    <div class="sub">Green lines are bike lanes selected by ALNS, gray lines are the full road network, and purple outlines are OD zone boundaries.</div>
    <div id="stats" class="stats"></div>
    <div class="legend">
      <div class="legend-row"><span class="swatch zone"></span><span>OD zone boundaries</span></div>
      <div class="legend-row"><span class="swatch"></span><span>Full road network links</span></div>
      <div class="legend-row"><span class="swatch bike"></span><span>Selected bike lanes</span></div>
      <div class="legend-row"><span class="swatch path"></span><span>OD paths</span></div>
    </div>
    <label><input id="showZones" type="checkbox" checked>Show zone boundaries</label>
    <label><input id="showPaths" type="checkbox" checked>Show OD paths</label>
    <label><input id="showRoad" type="checkbox" checked>Show road network</label>
    <label><input id="showBike" type="checkbox" checked>Show selected bike lanes</label>
    <label><input id="showNodes" type="checkbox" checked>Show OD nodes</label>
    <label><input id="showZoneLabels" type="checkbox" checked>Show zone labels</label>
    <button id="fit">Fit</button>
  </aside>
  <main><canvas id="map"></canvas></main>
</div>
<script>
const DATA = {data_json};
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const controls = {{
  zones: document.getElementById("showZones"),
  paths: document.getElementById("showPaths"),
  road: document.getElementById("showRoad"),
  bike: document.getElementById("showBike"),
  nodes: document.getElementById("showNodes"),
  zoneLabels: document.getElementById("showZoneLabels"),
}};
const nodeMap = new Map(DATA.nodes.map(n => [n.id, n]));
const state = {{ scale: 1, ox: 0, oy: 0, dragging: false, lastX: 0, lastY: 0 }};
document.getElementById("stats").innerHTML = Object.entries(DATA.summary).map(([k,v]) =>
  `<div class="stat"><span>${{k}}</span><strong>${{typeof v === "number" ? Number(v).toFixed(3).replace(/\\.000$/,"") : v}}</strong></div>`
).join("");
function resize() {{
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(r.width * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fit();
}}
function fit() {{
  const b = DATA.bounds, w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
  const sx = Math.max(b.maxX-b.minX, 1e-9), sy = Math.max(b.maxY-b.minY, 1e-9);
  state.scale = 0.9 * Math.min(w/sx, h/sy);
  state.ox = (w - sx*state.scale)/2 - b.minX*state.scale;
  state.oy = (h - sy*state.scale)/2 + b.maxY*state.scale;
  draw();
}}
function project(x,y) {{ return [x*state.scale+state.ox, state.oy-y*state.scale]; }}
function drawSegment(uId, vId, color, width, alpha=1, dash=[]) {{
  const u=nodeMap.get(uId), v=nodeMap.get(vId); if(!u||!v) return;
  const [x0,y0]=project(u.x,u.y), [x1,y1]=project(v.x,v.y);
  ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
  ctx.globalAlpha=1; ctx.setLineDash([]);
}}
function drawZone(zone, fillAlpha, strokeAlpha, width) {{
  ctx.beginPath();
  for(const ring of zone.rings) {{
    ring.forEach(([x,y], i) => {{
      const [px,py]=project(x,y);
      if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
    }});
    ctx.closePath();
  }}
  if(fillAlpha > 0) {{
    ctx.globalAlpha=fillAlpha; ctx.fillStyle="#7c3aed"; ctx.fill("evenodd");
  }}
  ctx.globalAlpha=strokeAlpha; ctx.strokeStyle="#6d28d9"; ctx.lineWidth=width; ctx.setLineDash([]);
  ctx.stroke();
  ctx.globalAlpha=1;
}}
function drawZoneLabel(zone) {{
  const [x,y]=project(zone.center[0], zone.center[1]);
  ctx.font="11px sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle";
  ctx.lineWidth=3; ctx.strokeStyle="rgba(255,255,255,.9)"; ctx.strokeText(zone.id,x,y);
  ctx.fillStyle="#4c1d95"; ctx.fillText(zone.id,x,y);
}}
function draw() {{
  const w=canvas.clientWidth||1, h=canvas.clientHeight||1;
  ctx.clearRect(0,0,w,h);
  if(controls.zones.checked) for(const z of DATA.zones || []) drawZone(z,0.055,0.18,0.9);
  if(controls.road.checked) for(const e of DATA.edges) drawSegment(e.u,e.v,"#94a3b8",0.9,0.62);
  if(controls.paths.checked) for(const p of DATA.paths) {{
    const color = p.coverage_required ? "#2563eb" : "#dc2626";
    const dash = p.coverage_required ? [] : [8,5];
    const alpha = p.coverage_required ? 0.28 : 0.22;
    for(const s of p.segments) drawSegment(s[0],s[1],color,1.6,alpha,dash);
  }}
  if(controls.bike.checked) for(const e of DATA.edges) if(e.chosen) drawSegment(e.u,e.v,"#16a34a",5.8,0.96);
  if(controls.nodes.checked) for(const n of DATA.nodes) {{
    if(n.kind !== "od") continue;
    const [x,y]=project(n.x,n.y);
    ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2);
    ctx.fillStyle="#f59e0b"; ctx.fill();
    ctx.strokeStyle="#7c2d12"; ctx.lineWidth=1.2; ctx.stroke();
    ctx.fillStyle="#7c2d12"; ctx.font="10px sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText(String(n.original_id),x,y);
  }}
  if(controls.zones.checked) for(const z of DATA.zones || []) drawZone(z,0,0.95,1.15);
  if(controls.zones.checked && controls.zoneLabels.checked) for(const z of DATA.zones || []) drawZoneLabel(z);
}}
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const wx=(mx-state.ox)/state.scale, wy=(state.oy-my)/state.scale;
  state.scale=Math.max(0.02,Math.min(500000,state.scale*(e.deltaY<0?1.12:1/1.12)));
  state.ox=mx-wx*state.scale; state.oy=my+wy*state.scale; draw();
}}, {{ passive:false }});
canvas.addEventListener("mousedown", e => {{ state.dragging=true; state.lastX=e.clientX; state.lastY=e.clientY; canvas.classList.add("dragging"); }});
window.addEventListener("mouseup", () => {{ state.dragging=false; canvas.classList.remove("dragging"); }});
canvas.addEventListener("mousemove", e => {{
  if(!state.dragging) return;
  state.ox += e.clientX-state.lastX; state.oy += e.clientY-state.lastY;
  state.lastX=e.clientX; state.lastY=e.clientY; draw();
}});
Object.values(controls).forEach(el => el.addEventListener("change", draw));
document.getElementById("fit").addEventListener("click", fit);
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive Large Neighborhood Search heuristic for the link-based undirected bicycle-lane design problem. "
            "This is a standalone non-Gurobi solver using candidate OD paths and ALNS destroy/repair operators."
        )
    )
    parser.add_argument("--network-dir", default=str(DEFAULT_NETWORK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="alns_undirected_design_solution")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--covered-demand-share", type=float, default=0.8)
    parser.add_argument("--od-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--time-limit", type=float, default=36000.0)
    parser.add_argument("--path-pool-size", type=int, default=12)
    parser.add_argument("--path-penalty-growth", type=float, default=0.35)
    parser.add_argument("--path-random-noise", type=float, default=0.12)
    parser.add_argument("--candidate-top-k", type=int, default=30)
    parser.add_argument("--max-paths-per-od-scoring", type=int, default=4)
    parser.add_argument("--initial-max-additions", type=int, default=2000)
    parser.add_argument("--repair-max-additions", type=int, default=80)
    parser.add_argument("--destroy-fraction-min", type=float, default=0.04)
    parser.add_argument("--destroy-fraction-max", type=float, default=0.22)
    parser.add_argument("--low-impact-sample-size", type=int, default=80)
    parser.add_argument("--initial-temperature", type=float, default=500.0)
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--adaptation-interval", type=int, default=25)
    parser.add_argument("--reaction", type=float, default=0.25)
    parser.add_argument("--penalty-factor", type=float, default=0.0)
    parser.add_argument("--prune-passes", type=int, default=3)
    parser.add_argument("--report-interval", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.network_dir = str(resolve_project_path(args.network_dir))
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else Path(args.network_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, result, _path_pool, _arc_to_unit, _design_units = run_alns(args)

    json_path = output_dir / f"{args.prefix}.json"
    csv_path = output_dir / f"{args.prefix}_chosen_links.csv"
    html_path = output_dir / f"{args.prefix}.html"
    clean = json_safe_result(result, data.original_id)
    clean["node_file"] = str(data.node_path)
    clean["zone_count"] = data.zone_count
    json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    write_solution_csv(data, result, csv_path)
    write_html(data, result, html_path)

    print("===== ALNS summary =====")
    print(f"Status              : {result['status_name']}")
    print(f"Runtime (s)         : {result['runtime']:.2f}")
    print(f"Objective/cost      : {result['objective']:.6f}")
    print(f"Covered demand      : {result['covered_demand']:.6f} ({result['covered_demand_share']:.2%})")
    print(f"Target demand share : {result['demand_target_share']:.2%}")
    print(f"Covered ODs         : {result['covered_od_count']} / {result['num_ods']}")
    print(f"Chosen design units : {result['chosen_design_unit_count']}")
    print(f"Chosen directed links: {len(result['chosen_links'])}")
    print(f"Wrote JSON          : {json_path}")
    print(f"Wrote chosen links  : {csv_path}")
    print(f"Wrote HTML          : {html_path}")


if __name__ == "__main__":
    main()
