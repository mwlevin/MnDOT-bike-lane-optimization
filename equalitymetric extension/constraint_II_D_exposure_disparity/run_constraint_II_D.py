#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_SRC = SCRIPT_DIR / "legacy_src"
DEFAULT_TARGETED_EQUITY_FILE = "od_equity_shares_targeted_base_uncovered.csv"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
(SCRIPT_DIR / ".mplconfig").mkdir(exist_ok=True)
(SCRIPT_DIR / ".cache").mkdir(exist_ok=True)
sys.path.insert(0, str(LEGACY_SRC))

import alns_link_based_undirected_design as legacy_alns  # noqa: E402
from solve_constraint_II_D import (  # noqa: E402
    arc_list_to_rows,
    load_equity_shares,
    load_id_map,
    normalize_demand_share,
    od_text,
    solve_constraint_ii_d,
    unit_to_row,
    write_outputs,
)


OD = Tuple[int, int]
Arc = Tuple[int, int]


@dataclass
class ConstraintState:
    chosen_units: Set[int]
    cost: float
    protected_burden_sum: float
    unprotected_burden_sum: float
    protected_uncovered_exposure: float
    unprotected_uncovered_exposure: float
    exposure_gap: float
    covered_demand: float
    covered_share: float
    protected_covered_demand: float
    unprotected_covered_demand: float
    covered_ods: Set[OD]
    od_best_path: Dict[OD, int]
    od_u: Dict[OD, float]
    od_uncovered_length: Dict[OD, float]
    coverage_violation: float
    disparity_violation: float
    feasible: bool

    @property
    def violation(self) -> float:
        return self.coverage_violation + self.disparity_violation


def build_exact_namespace(
    network_dir: Path,
    output_dir: Path,
    prefix: str,
    args: argparse.Namespace,
    od_limit: int,
) -> argparse.Namespace:
    equity_file = Path(args.small_equity_file).expanduser().resolve() if args.small_equity_file else network_dir / args.equity_file_name
    return argparse.Namespace(
        network_dir=str(network_dir),
        equity_file=str(equity_file),
        output_dir=str(output_dir),
        prefix=prefix,
        coverage_share=args.coverage_share,
        eta_exposure=args.eta_exposure,
        epsilon=args.epsilon,
        t=args.t,
        od_limit=od_limit,
        seed=args.seed,
        time_limit=args.small_time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        output_flag=args.output_flag,
        disable_balanced_od_sampling=args.disable_balanced_od_sampling,
        disable_od_arc_pruning=False,
        disable_node_reachability_pruning=False,
        disable_source_sink_variable_pruning=False,
        disable_directed_subgraph_trimming=False,
        disable_origin_destination_strengthening=False,
        disable_opposite_direction_x_cut=False,
        disable_single_arc_uncovered_implied_cut=False,
        disable_minimum_covered_length_cut=False,
        disable_single_arc_detour_implied_cut=False,
    )


def evaluate_constraint_solution(
    chosen_units: Set[int],
    *,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t_value: float,
    coverage_share: float,
    eta_exposure: float,
) -> ConstraintState:
    chosen = set(chosen_units)
    cost = sum(unit_cost[unit] for unit in chosen)
    protected_burden = 0.0
    unprotected_burden = 0.0
    covered_demand = 0.0
    protected_covered = 0.0
    unprotected_covered = 0.0
    covered_ods: Set[OD] = set()
    od_best_path: Dict[OD, int] = {}
    od_u: Dict[OD, float] = {}
    od_uncovered: Dict[OD, float] = {}

    for od, paths in path_pool.items():
        cap = t_value * shortest[od]
        best_idx = 0
        best_u = 1.0
        best_uncovered = math.inf
        best_len = math.inf
        best_covered = False
        for idx, path in enumerate(paths):
            unc = legacy_alns.uncovered_length(path, chosen)
            covered = unc <= cap + 1e-8
            candidate_u = unc / shortest[od] if covered and shortest[od] > 1e-12 else 1.0
            candidate_key = (candidate_u, 0 if covered else 1, unc, path.length)
            best_key = (best_u, 0 if best_covered else 1, best_uncovered, best_len)
            if candidate_key < best_key:
                best_idx = idx
                best_u = candidate_u
                best_uncovered = unc
                best_len = path.length
                best_covered = covered

        od_best_path[od] = best_idx
        od_u[od] = best_u
        od_uncovered[od] = best_uncovered
        protected_burden += q_protected[od] * best_u
        unprotected_burden += q_unprotected[od] * best_u
        if best_covered:
            covered_ods.add(od)
            covered_demand += od_demand[od]
            protected_covered += q_protected[od]
            unprotected_covered += q_unprotected[od]

    total_demand = sum(od_demand.values())
    total_protected = sum(q_protected.values())
    total_unprotected = sum(q_unprotected.values())
    ep = protected_burden / total_protected if total_protected > 0 else math.nan
    eu = unprotected_burden / total_unprotected if total_unprotected > 0 else math.nan
    gap = ep - eu if math.isfinite(ep) and math.isfinite(eu) else math.inf
    covered_share = covered_demand / total_demand if total_demand > 0 else 0.0
    coverage_violation = max(0.0, coverage_share - covered_share)
    disparity_violation = max(0.0, gap - eta_exposure)

    return ConstraintState(
        chosen_units=chosen,
        cost=cost,
        protected_burden_sum=protected_burden,
        unprotected_burden_sum=unprotected_burden,
        protected_uncovered_exposure=ep,
        unprotected_uncovered_exposure=eu,
        exposure_gap=gap,
        covered_demand=covered_demand,
        covered_share=covered_share,
        protected_covered_demand=protected_covered,
        unprotected_covered_demand=unprotected_covered,
        covered_ods=covered_ods,
        od_best_path=od_best_path,
        od_u=od_u,
        od_uncovered_length=od_uncovered,
        coverage_violation=coverage_violation,
        disparity_violation=disparity_violation,
        feasible=coverage_violation <= 1e-8 and disparity_violation <= 1e-8,
    )


def equity_gap_reduction_weight(
    od: OD,
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    total_protected: float,
    total_unprotected: float,
) -> float:
    return q_protected[od] / max(total_protected, 1e-9) - q_unprotected[od] / max(total_unprotected, 1e-9)


def rank_cover_bundles(
    state: ConstraintState,
    *,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t_value: float,
    max_paths_per_od: int,
) -> List[Tuple[float, Tuple[int, ...]]]:
    total_demand = sum(od_demand.values())
    total_protected = sum(q_protected.values())
    total_unprotected = sum(q_unprotected.values())
    bundles: List[Tuple[float, Tuple[int, ...]]] = []

    for od, paths in path_pool.items():
        cap = t_value * shortest[od]
        equity_weight = max(0.0, equity_gap_reduction_weight(od, q_protected, q_unprotected, total_protected, total_unprotected))
        for path in paths[:max_paths_per_od]:
            current_unc = legacy_alns.uncovered_length(path, state.chosen_units)
            if current_unc <= cap + 1e-8:
                continue
            missing = [
                (unit, length, unit_cost[unit])
                for unit, length in path.unit_lengths
                if unit not in state.chosen_units
            ]
            if not missing:
                continue

            selected: List[int] = []
            selected_cost = 0.0
            reduced = 0.0
            need_reduce = current_unc - cap
            for unit, length, cost in sorted(missing, key=lambda item: item[1] / max(item[2], 1e-9), reverse=True):
                selected.append(unit)
                selected_cost += cost
                reduced += length
                if reduced >= need_reduce - 1e-9:
                    break
            if reduced < need_reduce - 1e-9 or selected_cost <= 1e-12:
                continue

            new_unc = max(0.0, current_unc - reduced)
            new_u = new_unc / max(shortest[od], 1e-9)
            coverage_benefit = od_demand[od] / max(total_demand, 1e-9) if od not in state.covered_ods else 0.0
            exposure_benefit = equity_weight * max(0.0, state.od_u[od] - new_u)
            score = coverage_benefit + 3.0 * exposure_benefit
            if score > 1e-12:
                bundles.append((score / selected_cost, tuple(sorted(set(selected)))))

    best_by_units: Dict[Tuple[int, ...], float] = {}
    for score, units in bundles:
        best_by_units[units] = max(score, best_by_units.get(units, 0.0))
    ranked = [(score, units) for units, score in best_by_units.items()]
    ranked.sort(reverse=True)
    return ranked


def rank_single_unit_additions(
    state: ConstraintState,
    *,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t_value: float,
    max_paths_per_od: int,
) -> List[Tuple[float, int]]:
    total_demand = sum(od_demand.values())
    total_protected = sum(q_protected.values())
    total_unprotected = sum(q_unprotected.values())
    scores: Dict[int, float] = {}

    for od, paths in path_pool.items():
        cap = t_value * shortest[od]
        equity_weight = max(0.0, equity_gap_reduction_weight(od, q_protected, q_unprotected, total_protected, total_unprotected))
        for path in paths[:max_paths_per_od]:
            path_unc = legacy_alns.uncovered_length(path, state.chosen_units)
            deficit = max(0.0, path_unc - cap)
            for unit, length in path.unit_lengths:
                if unit in state.chosen_units:
                    continue
                coverage_benefit = 0.0
                if od not in state.covered_ods and deficit > 1e-8:
                    coverage_benefit = (od_demand[od] / max(total_demand, 1e-9)) * min(length, deficit) / max(deficit, 1e-9)
                exposure_benefit = 0.0
                if equity_weight > 0.0 and state.od_u[od] > 1e-9:
                    exposure_benefit = equity_weight * min(length / max(shortest[od], 1e-9), state.od_u[od])
                benefit = coverage_benefit + 3.0 * exposure_benefit
                if benefit > 1e-12:
                    scores[unit] = scores.get(unit, 0.0) + benefit

    ranked = [
        (score / max(unit_cost[unit], 1e-9), unit)
        for unit, score in scores.items()
        if score > 1e-12
    ]
    ranked.sort(reverse=True)
    return ranked


def repair_key(state: ConstraintState) -> Tuple[float, float, float]:
    if state.coverage_violation > 1e-8:
        return (state.coverage_violation, state.disparity_violation, state.cost)
    return (state.disparity_violation, 0.0, state.cost)


def greedy_constraint_repair(
    state: ConstraintState,
    *,
    evaluator,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t_value: float,
    candidate_top_k: int,
    max_additions: int,
    max_paths_per_od: int,
) -> ConstraintState:
    current = state
    for _ in range(max_additions):
        if current.feasible:
            break

        trial_states: List[ConstraintState] = []
        for _score, units in rank_cover_bundles(
            current,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            q_unprotected=q_unprotected,
            shortest=shortest,
            unit_cost=unit_cost,
            t_value=t_value,
            max_paths_per_od=max_paths_per_od,
        )[:candidate_top_k]:
            trial_units = set(current.chosen_units)
            trial_units.update(units)
            trial_states.append(evaluator(trial_units))

        for _score, unit in rank_single_unit_additions(
            current,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            q_unprotected=q_unprotected,
            shortest=shortest,
            unit_cost=unit_cost,
            t_value=t_value,
            max_paths_per_od=max_paths_per_od,
        )[:candidate_top_k]:
            trial_units = set(current.chosen_units)
            trial_units.add(unit)
            trial_states.append(evaluator(trial_units))

        if not trial_states:
            break

        best_trial = min(trial_states, key=repair_key)
        if repair_key(best_trial) < repair_key(current):
            current = best_trial
            continue

        # Some coverage repairs are only useful after several additions. Fall
        # back to the best scored candidate if the formal violation key stalls.
        best_by_score = min(trial_states, key=lambda item: (item.violation, -item.covered_share, item.cost))
        non_worsening = best_by_score.violation <= current.violation + 1e-9
        coverage_non_worsening = (
            current.coverage_violation > 1e-8
            and best_by_score.covered_share >= current.covered_share - 1e-9
            and best_by_score.disparity_violation <= current.disparity_violation + 1e-9
        )
        if (
            best_by_score.cost > current.cost + 1e-9
            and best_by_score.chosen_units != current.chosen_units
            and (non_worsening or coverage_non_worsening)
        ):
            current = best_by_score
        else:
            break

    return current


def prune_cost_preserving_feasibility(
    state: ConstraintState,
    *,
    evaluator,
    unit_cost: Dict[int, float],
    max_passes: int,
) -> ConstraintState:
    current = state
    for _ in range(max_passes):
        removed_any = False
        for unit in sorted(current.chosen_units, key=lambda item: unit_cost[item], reverse=True):
            trial_units = set(current.chosen_units)
            trial_units.remove(unit)
            trial = evaluator(trial_units)
            if trial.feasible and trial.cost < current.cost - 1e-9:
                current = trial
                removed_any = True
        if not removed_any:
            break
    return current


def destroy_random_units(state: ConstraintState, rng: random.Random, fraction: float) -> Set[int]:
    if not state.chosen_units:
        return set()
    count = max(1, int(round(len(state.chosen_units) * fraction)))
    remove = set(rng.sample(sorted(state.chosen_units), min(count, len(state.chosen_units))))
    return set(state.chosen_units - remove)


def destroy_costly_units(state: ConstraintState, unit_cost: Dict[int, float], fraction: float) -> Set[int]:
    if not state.chosen_units:
        return set()
    count = max(1, int(round(len(state.chosen_units) * fraction)))
    ordered = sorted(state.chosen_units, key=lambda unit: unit_cost[unit], reverse=True)
    return set(state.chosen_units - set(ordered[:count]))


def destroy_low_impact_units(
    state: ConstraintState,
    *,
    evaluator,
    rng: random.Random,
    unit_cost: Dict[int, float],
    fraction: float,
    sample_size: int,
) -> Set[int]:
    if not state.chosen_units:
        return set()
    count = max(1, int(round(len(state.chosen_units) * fraction)))
    sample = sorted(state.chosen_units)
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)
    scored = []
    for unit in sample:
        trial_units = set(state.chosen_units)
        trial_units.remove(unit)
        trial = evaluator(trial_units)
        scored.append((trial.violation, -max(0.0, state.cost - trial.cost), unit))
    scored.sort()
    return set(state.chosen_units - {unit for _violation, _saving, unit in scored[:count]})


def state_to_result(
    *,
    best: ConstraintState,
    status_name: str,
    start_time: float,
    config: Dict[str, object],
    data: legacy_alns.TNTPNetworkData,
    design_units: Sequence[Arc],
    arc_to_unit: Dict[int, int],
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    pi: Dict[OD, float],
    shortest: Dict[OD, float],
    id_map: Dict[int, Dict[str, str]],
    iterations_completed: int,
) -> Dict[str, object]:
    network_dir = Path(str(config["network_dir"])).resolve()
    chosen_links = sorted(
        {
            data.idx_to_arc[arc_idx]
            for arc_idx, unit in arc_to_unit.items()
            if unit in best.chosen_units
        }
    )
    chosen_design_units = sorted(design_units[unit] for unit in best.chosen_units)
    total_demand = sum(od_demand.values())
    total_protected = sum(q_protected.values())
    total_unprotected = sum(q_unprotected.values())
    protected_coverage = best.protected_covered_demand / total_protected if total_protected > 0 else math.nan
    unprotected_coverage = best.unprotected_covered_demand / total_unprotected if total_unprotected > 0 else math.nan

    od_stats: Dict[str, Dict[str, object]] = {}
    od_paths: Dict[str, List[Dict[str, object]]] = {}
    for od in sorted(od_demand):
        path = path_pool[od][best.od_best_path[od]]
        path_arcs = [data.idx_to_arc[arc_idx] for arc_idx in path.arcs]
        key = od_text(od)
        covered = od in best.covered_ods
        od_paths[key] = arc_list_to_rows(path_arcs, id_map)
        od_stats[key] = {
            "od": key,
            "origin": od[0],
            "destination": od[1],
            "original_origin": id_map.get(od[0], {}).get("original_id", str(od[0])),
            "original_destination": id_map.get(od[1], {}).get("original_id", str(od[1])),
            "demand": od_demand[od],
            "pi_w": pi[od],
            "protected_demand": q_protected[od],
            "unprotected_demand": q_unprotected[od],
            "z_value": 1.0 if covered else 0.0,
            "covered": covered,
            "u_w": best.od_u[od],
            "service_capacity_ratio": 1.0 - min(max(best.od_u[od], 0.0), 1.0),
            "path_length": path.length,
            "shortest_length": shortest[od],
            "detour_cap": (1.0 + float(config["epsilon"])) * shortest[od],
            "model_uncovered_length": best.od_uncovered_length[od],
            "uncovered_cap_if_required": float(config["t"]) * shortest[od],
            "reference_path_used": not covered,
        }

    coverage_share = float(config["coverage_share"])
    eta_exposure = float(config["eta_exposure"])
    return {
        "model_name": "Constraint II-D large-network cost-min heuristic",
        "algorithm": "constraint_repair_alns_cost_min",
        "status_name": status_name,
        "runtime": time.time() - start_time,
        "network_dir": str(network_dir),
        "equity_file": str(config["equity_file"]),
        "epsilon": float(config["epsilon"]),
        "t": float(config["t"]),
        "coverage_target_share": coverage_share,
        "coverage_target_demand": coverage_share * total_demand,
        "eta_exposure": eta_exposure,
        "service_interpretation": "At least coverage_target_share of total demand is covered; every covered OD has uncovered path length <= t times its shortest-path length.",
        "od_limit": int(config["od_limit"]),
        "seed": int(config["seed"]),
        "num_ods": len(od_demand),
        "num_nodes": len(data.nodes),
        "num_arcs": len(data.arcs),
        "num_design_units": len(design_units),
        "total_demand": total_demand,
        "total_protected_demand": total_protected,
        "total_unprotected_demand": total_unprotected,
        "protected_population_demand_share": total_protected / total_demand if total_demand > 0 else math.nan,
        "objective": best.cost,
        "construction_cost": best.cost,
        "protected_uncovered_exposure_burden_sum": best.protected_burden_sum,
        "unprotected_uncovered_exposure_burden_sum": best.unprotected_burden_sum,
        "protected_uncovered_exposure_burden": best.protected_uncovered_exposure,
        "unprotected_uncovered_exposure_burden": best.unprotected_uncovered_exposure,
        "exposure_burden_gap_Ep_minus_Eu": best.exposure_gap,
        "covered_demand": best.covered_demand,
        "covered_demand_share": best.covered_share,
        "coverage_requirement_satisfied": best.coverage_violation <= 1e-8,
        "exposure_disparity_satisfied": best.disparity_violation <= 1e-8,
        "service_80_80_satisfied": best.covered_share >= 0.8 - 1e-8 and float(config["t"]) <= 0.2 + 1e-9,
        "coverage_violation": best.coverage_violation,
        "disparity_violation": best.disparity_violation,
        "protected_covered_demand": best.protected_covered_demand,
        "protected_coverage_ratio": protected_coverage,
        "unprotected_covered_demand": best.unprotected_covered_demand,
        "unprotected_coverage_ratio": unprotected_coverage,
        "covered_od_count": len(best.covered_ods),
        "chosen_design_unit_count": len(best.chosen_units),
        "chosen_design_units": [unit_to_row(unit, id_map) for unit in chosen_design_units],
        "chosen_links": arc_list_to_rows(chosen_links, id_map),
        "od_stats": od_stats,
        "od_paths": od_paths,
        "path_pool_size_requested": int(config["path_pool_size"]),
        "iterations_completed": iterations_completed,
    }


def solve_large_alns_constraint_ii_d(config: Dict[str, object]) -> Dict[str, object]:
    start_time = time.time()
    network_dir = Path(str(config["network_dir"])).resolve()
    equity_file = Path(str(config["equity_file"])).resolve()
    rng = random.Random(int(config["seed"]))

    data = legacy_alns.TNTPNetworkData(str(network_dir))
    design_units, arc_to_unit, _unit_arcs, unit_cost = legacy_alns.build_undirected_design_units(data)
    od_demand = legacy_alns.select_ods(data.od_demand, int(config["od_limit"]), int(config["seed"]))
    ods = sorted(od_demand)
    pi = load_equity_shares(equity_file, od_demand)
    q_protected = {od: od_demand[od] * pi[od] for od in ods}
    q_unprotected = {od: od_demand[od] * (1.0 - pi[od]) for od in ods}
    coverage_share = normalize_demand_share(float(config["coverage_share"]))
    eta_exposure = float(config["eta_exposure"])

    path_pool, shortest = legacy_alns.build_path_pool(
        data,
        ods,
        arc_to_unit,
        float(config["epsilon"]),
        int(config["path_pool_size"]),
        int(config["seed"]),
        float(config["path_penalty_growth"]),
        float(config["path_random_noise"]),
    )

    def evaluator(chosen_units: Set[int]) -> ConstraintState:
        return evaluate_constraint_solution(
            chosen_units,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            q_unprotected=q_unprotected,
            shortest=shortest,
            unit_cost=unit_cost,
            t_value=float(config["t"]),
            coverage_share=coverage_share,
            eta_exposure=eta_exposure,
        )

    current = evaluator(set())
    current = greedy_constraint_repair(
        current,
        evaluator=evaluator,
        path_pool=path_pool,
        od_demand=od_demand,
        q_protected=q_protected,
        q_unprotected=q_unprotected,
        shortest=shortest,
        unit_cost=unit_cost,
        t_value=float(config["t"]),
        candidate_top_k=int(config["candidate_top_k"]),
        max_additions=int(config["initial_max_additions"]),
        max_paths_per_od=int(config["max_paths_per_od_scoring"]),
    )
    if current.feasible:
        current = prune_cost_preserving_feasibility(
            current,
            evaluator=evaluator,
            unit_cost=unit_cost,
            max_passes=int(config["prune_passes"]),
        )
    best = current if current.feasible else None
    temperature = float(config["initial_temperature"])
    iterations = int(config["iterations"])
    time_limit = float(config["time_limit"])
    last_report = time.time()
    iterations_completed = 0

    for iteration in range(1, iterations + 1):
        if time_limit > 0 and time.time() - start_time >= time_limit:
            break
        iterations_completed = iteration
        base = current if current.chosen_units else (best if best is not None else current)
        fraction = rng.uniform(float(config["destroy_fraction_min"]), float(config["destroy_fraction_max"]))
        destroy_choice = rng.choice(["random", "costly", "low_impact"])
        if destroy_choice == "random":
            destroyed_units = destroy_random_units(base, rng, fraction)
        elif destroy_choice == "costly":
            destroyed_units = destroy_costly_units(base, unit_cost, fraction)
        else:
            destroyed_units = destroy_low_impact_units(
                base,
                evaluator=evaluator,
                rng=rng,
                unit_cost=unit_cost,
                fraction=fraction,
                sample_size=int(config["low_impact_sample_size"]),
            )

        repaired = greedy_constraint_repair(
            evaluator(destroyed_units),
            evaluator=evaluator,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            q_unprotected=q_unprotected,
            shortest=shortest,
            unit_cost=unit_cost,
            t_value=float(config["t"]),
            candidate_top_k=int(config["candidate_top_k"]),
            max_additions=int(config["repair_max_additions"]),
            max_paths_per_od=int(config["max_paths_per_od_scoring"]),
        )
        if repaired.feasible:
            repaired = prune_cost_preserving_feasibility(
                repaired,
                evaluator=evaluator,
                unit_cost=unit_cost,
                max_passes=int(config["prune_passes"]),
            )
            if best is None or repaired.cost < best.cost - 1e-9:
                best = repaired

        if current.feasible and repaired.feasible:
            delta = repaired.cost - current.cost
            accepted = delta <= 0.0 or (temperature > 1e-12 and rng.random() < math.exp(-delta / temperature))
            if accepted:
                current = repaired
        elif not current.feasible and repair_key(repaired) <= repair_key(current):
            current = repaired
        elif repaired.feasible:
            current = repaired
        temperature *= float(config["cooling_rate"])

        report_interval = float(config["report_interval"])
        if report_interval > 0 and time.time() - last_report >= report_interval:
            ref = best if best is not None else current
            print(
                f"[large II-D seed={config['seed']}] iter={iteration} "
                f"cost={ref.cost:.2f} coverage={ref.covered_share:.2%} "
                f"gap={ref.exposure_gap:.4f} feasible={ref.feasible}"
            )
            last_report = time.time()

    final_state = best if best is not None else current
    status_name = "FEASIBLE_HEURISTIC" if final_state.feasible else "NO_FEASIBLE_HEURISTIC_SOLUTION"
    id_map = load_id_map(network_dir / "n2024_polygon_id_map.csv")
    return state_to_result(
        best=final_state,
        status_name=status_name,
        start_time=start_time,
        config=config,
        data=data,
        design_units=design_units,
        arc_to_unit=arc_to_unit,
        path_pool=path_pool,
        od_demand=od_demand,
        q_protected=q_protected,
        q_unprotected=q_unprotected,
        pi=pi,
        shortest=shortest,
        id_map=id_map,
        iterations_completed=iterations_completed,
    )


def large_config_from_args(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    network_dir = SCRIPT_DIR / "data" / "full_network"
    equity_file = Path(args.large_equity_file).expanduser().resolve() if args.large_equity_file else network_dir / args.equity_file_name
    return {
        "network_dir": str(network_dir),
        "equity_file": str(equity_file),
        "coverage_share": args.coverage_share,
        "eta_exposure": args.eta_exposure,
        "equity_file_name": args.equity_file_name,
        "epsilon": args.epsilon,
        "t": args.t,
        "od_limit": args.large_od_limit,
        "seed": seed,
        "iterations": args.large_iterations,
        "time_limit": args.large_time_limit,
        "path_pool_size": args.path_pool_size,
        "path_penalty_growth": args.path_penalty_growth,
        "path_random_noise": args.path_random_noise,
        "candidate_top_k": args.candidate_top_k,
        "max_paths_per_od_scoring": args.max_paths_per_od_scoring,
        "initial_max_additions": args.initial_max_additions,
        "repair_max_additions": args.repair_max_additions,
        "prune_passes": args.prune_passes,
        "destroy_fraction_min": args.destroy_fraction_min,
        "destroy_fraction_max": args.destroy_fraction_max,
        "low_impact_sample_size": args.low_impact_sample_size,
        "initial_temperature": args.initial_temperature,
        "cooling_rate": args.cooling_rate,
        "report_interval": args.report_interval,
    }


def run(args: argparse.Namespace | None = None) -> Dict[str, object]:
    if args is None:
        args = build_parser().parse_args()

    args.coverage_share = normalize_demand_share(args.coverage_share)
    base_results = Path(args.output_dir).expanduser().resolve()
    base_results.mkdir(parents=True, exist_ok=True)

    small_dir = SCRIPT_DIR / "data" / "cropped_subgraph"
    large_dir = SCRIPT_DIR / "data" / "full_network"
    small_out = base_results / "small_exact_gurobi"
    large_out = base_results / "large_alns"

    print("===== Running small network: exact Gurobi Constraint II-D =====")
    small_ns = build_exact_namespace(
        small_dir,
        small_out,
        "small_constraint_II_D_exact",
        args,
        args.small_od_limit,
    )
    small_result = solve_constraint_ii_d(small_ns)
    small_paths = write_outputs(
        small_result,
        small_out,
        "small_constraint_II_D_exact",
        load_id_map(small_dir / "n2024_polygon_id_map.csv"),
    )
    small_result["output_paths"] = small_paths
    Path(small_paths["json"]).write_text(json.dumps(small_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Running large network: multi-start ALNS Constraint II-D =====")
    seeds = [args.seed + i for i in range(max(1, args.large_workers))]
    configs = [large_config_from_args(args, seed) for seed in seeds]
    if args.large_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.large_workers) as pool:
            large_candidates = list(pool.map(solve_large_alns_constraint_ii_d, configs))
    else:
        large_candidates = [solve_large_alns_constraint_ii_d(configs[0])]

    def candidate_key(result: Dict[str, object]) -> Tuple[int, float, float, float]:
        feasible = bool(result.get("coverage_requirement_satisfied")) and bool(result.get("exposure_disparity_satisfied"))
        return (
            0 if feasible else 1,
            float(result.get("coverage_violation", math.inf)) + float(result.get("disparity_violation", math.inf)),
            float(result.get("construction_cost", math.inf)),
            float(result.get("runtime", math.inf)),
        )

    large_result = min(large_candidates, key=candidate_key)
    large_result["multistart_worker_count"] = len(large_candidates)
    large_result["multistart_candidate_summaries"] = [
        {
            "seed": item.get("seed"),
            "status_name": item.get("status_name"),
            "construction_cost": item.get("construction_cost"),
            "covered_demand_share": item.get("covered_demand_share"),
            "protected_uncovered_exposure_burden": item.get("protected_uncovered_exposure_burden"),
            "unprotected_uncovered_exposure_burden": item.get("unprotected_uncovered_exposure_burden"),
            "exposure_burden_gap_Ep_minus_Eu": item.get("exposure_burden_gap_Ep_minus_Eu"),
            "coverage_violation": item.get("coverage_violation"),
            "disparity_violation": item.get("disparity_violation"),
            "runtime": item.get("runtime"),
        }
        for item in large_candidates
    ]
    large_paths = write_outputs(
        large_result,
        large_out,
        "large_constraint_II_D_alns",
        load_id_map(large_dir / "n2024_polygon_id_map.csv"),
    )
    large_result["output_paths"] = large_paths
    Path(large_paths["json"]).write_text(json.dumps(large_result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "experiment": "Constraint II-D uncovered-exposure disparity with 80/80 service",
        "coverage_target_share": args.coverage_share,
        "service_uncovered_cap_t": args.t,
        "eta_exposure": args.eta_exposure,
        "small_network": {
            "algorithm": "exact_gurobi",
            "status_name": small_result.get("status_name"),
            "construction_cost": small_result.get("construction_cost"),
            "covered_demand_share": small_result.get("covered_demand_share"),
            "protected_uncovered_exposure_burden": small_result.get("protected_uncovered_exposure_burden"),
            "unprotected_uncovered_exposure_burden": small_result.get("unprotected_uncovered_exposure_burden"),
            "exposure_burden_gap_Ep_minus_Eu": small_result.get("exposure_burden_gap_Ep_minus_Eu"),
            "coverage_requirement_satisfied": small_result.get("coverage_requirement_satisfied"),
            "exposure_disparity_satisfied": small_result.get("exposure_disparity_satisfied"),
            "equity_file": small_result.get("equity_file"),
            "json": small_paths["json"],
        },
        "large_network": {
            "algorithm": "constraint_repair_alns",
            "status_name": large_result.get("status_name"),
            "construction_cost": large_result.get("construction_cost"),
            "covered_demand_share": large_result.get("covered_demand_share"),
            "protected_uncovered_exposure_burden": large_result.get("protected_uncovered_exposure_burden"),
            "unprotected_uncovered_exposure_burden": large_result.get("unprotected_uncovered_exposure_burden"),
            "exposure_burden_gap_Ep_minus_Eu": large_result.get("exposure_burden_gap_Ep_minus_Eu"),
            "coverage_requirement_satisfied": large_result.get("coverage_requirement_satisfied"),
            "exposure_disparity_satisfied": large_result.get("exposure_disparity_satisfied"),
            "equity_file": large_result.get("equity_file"),
            "json": large_paths["json"],
        },
    }
    summary_path = base_results / "constraint_II_D_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote combined summary: {summary_path}")
    return {"small": small_result, "large": large_result, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Constraint II-D sequentially: small exact Gurobi, then large multi-start ALNS."
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument(
        "--equity-file-name",
        default=DEFAULT_TARGETED_EQUITY_FILE,
        help="Default equity CSV filename inside each network data directory.",
    )
    parser.add_argument("--small-equity-file", default="", help="Optional explicit small-network equity CSV.")
    parser.add_argument("--large-equity-file", default="", help="Optional explicit large-network equity CSV.")
    parser.add_argument("--coverage-share", type=float, default=0.8, help="0.8 means 80% demand must be covered.")
    parser.add_argument("--eta-exposure", type=float, default=0.05, help="Allowed gap E^P - E^U.")
    parser.add_argument("--small-od-limit", type=int, default=0, help="0 means all small-network ODs.")
    parser.add_argument("--large-od-limit", type=int, default=0, help="0 means all large-network ODs.")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.2, help="0.2 means covered OD paths are at least 80% served.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-time-limit", type=float, default=7200.0)
    parser.add_argument("--large-time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=None, help="Gurobi threads for the small exact model.")
    parser.add_argument("--large-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--disable-balanced-od-sampling", action="store_true")

    parser.add_argument("--large-iterations", type=int, default=1200)
    parser.add_argument("--path-pool-size", type=int, default=12)
    parser.add_argument("--path-penalty-growth", type=float, default=0.35)
    parser.add_argument("--path-random-noise", type=float, default=0.12)
    parser.add_argument("--candidate-top-k", type=int, default=30)
    parser.add_argument("--max-paths-per-od-scoring", type=int, default=5)
    parser.add_argument("--initial-max-additions", type=int, default=900)
    parser.add_argument("--repair-max-additions", type=int, default=140)
    parser.add_argument("--prune-passes", type=int, default=2)
    parser.add_argument("--destroy-fraction-min", type=float, default=0.04)
    parser.add_argument("--destroy-fraction-max", type=float, default=0.18)
    parser.add_argument("--low-impact-sample-size", type=int, default=80)
    parser.add_argument("--initial-temperature", type=float, default=150.0)
    parser.add_argument("--cooling-rate", type=float, default=0.996)
    parser.add_argument("--report-interval", type=float, default=30.0)
    return parser


if __name__ == "__main__":
    run()
