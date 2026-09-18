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
from solve_objective_I_E_refine import (  # noqa: E402
    arc_list_to_rows,
    load_equity_shares,
    load_id_map,
    normalize_demand_share,
    od_text,
    solve_objective_i_e_refine,
    unit_to_row,
    write_outputs,
)


OD = Tuple[int, int]
Arc = Tuple[int, int]


@dataclass
class ExposureState:
    chosen_units: Set[int]
    cost: float
    protected_burden_sum: float
    unprotected_burden_sum: float
    covered_demand: float
    protected_covered_demand: float
    unprotected_covered_demand: float
    covered_ods: Set[OD]
    od_best_path: Dict[OD, int]
    od_u: Dict[OD, float]
    od_uncovered_length: Dict[OD, float]


def build_exact_namespace(
    network_dir: Path,
    output_dir: Path,
    prefix: str,
    budget: float,
    args: argparse.Namespace,
    od_limit: int,
) -> argparse.Namespace:
    equity_file = Path(args.small_equity_file).expanduser().resolve() if args.small_equity_file else network_dir / args.equity_file_name
    return argparse.Namespace(
        network_dir=str(network_dir),
        equity_file=str(equity_file),
        output_dir=str(output_dir),
        prefix=prefix,
        budget=budget,
        coverage_share=args.coverage_share,
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
        disable_lexicographic_cost_tiebreak=False,
    )


def evaluate_exposure_solution(
    chosen_units: Set[int],
    *,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    q_unprotected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    t_value: float,
) -> ExposureState:
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
        for idx, path in enumerate(paths):
            unc = legacy_alns.uncovered_length(path, chosen)
            if unc <= cap + 1e-8:
                candidate_u = unc / shortest[od] if shortest[od] > 1e-12 else 1.0
            else:
                candidate_u = 1.0
            if (candidate_u, unc, path.length) < (best_u, best_uncovered, best_len):
                best_idx = idx
                best_u = candidate_u
                best_uncovered = unc
                best_len = path.length

        od_best_path[od] = best_idx
        od_u[od] = best_u
        od_uncovered[od] = best_uncovered
        protected_burden += q_protected[od] * best_u
        unprotected_burden += q_unprotected[od] * best_u
        if best_u < 1.0 - 1e-8:
            covered_ods.add(od)
            covered_demand += od_demand[od]
            protected_covered += q_protected[od]
            unprotected_covered += q_unprotected[od]

    return ExposureState(
        chosen_units=chosen,
        cost=cost,
        protected_burden_sum=protected_burden,
        unprotected_burden_sum=unprotected_burden,
        covered_demand=covered_demand,
        protected_covered_demand=protected_covered,
        unprotected_covered_demand=unprotected_covered,
        covered_ods=covered_ods,
        od_best_path=od_best_path,
        od_u=od_u,
        od_uncovered_length=od_uncovered,
    )


def coverage_shortfall(state: ExposureState, target_demand: float) -> float:
    return max(0.0, target_demand - state.covered_demand)


def refined_state_key(state: ExposureState, target_demand: float) -> Tuple[float, float, float]:
    return (coverage_shortfall(state, target_demand), state.protected_burden_sum, state.cost)


def rank_add_candidates(
    state: ExposureState,
    *,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    budget_remaining: float,
    target_demand: float,
    max_paths_per_od: int,
) -> List[Tuple[float, int]]:
    scores: Dict[int, float] = {}
    for od, paths in path_pool.items():
        if state.od_u[od] <= 1e-9:
            continue
        for path in paths[:max_paths_per_od]:
            path_unc = legacy_alns.uncovered_length(path, state.chosen_units)
            for unit, length in path.unit_lengths:
                if unit in state.chosen_units or unit_cost[unit] > budget_remaining + 1e-9:
                    continue
                exposure_benefit = q_protected[od] * min(length / max(shortest[od], 1e-9), state.od_u[od])
                coverage_benefit = 0.0
                if state.covered_demand < target_demand - 1e-8 and od not in state.covered_ods:
                    coverage_benefit = od_demand[od] * min(length / max(shortest[od], 1e-9), 1.0)
                benefit = exposure_benefit + coverage_benefit
                if path_unc <= 1e-9:
                    benefit *= 0.25
                scores[unit] = scores.get(unit, 0.0) + benefit

    ranked = [
        (score / max(unit_cost[unit], 1e-9), unit)
        for unit, score in scores.items()
        if score > 1e-12
    ]
    ranked.sort(reverse=True)
    return ranked


def rank_cover_bundles(
    state: ExposureState,
    *,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    budget_remaining: float,
    target_demand: float,
    t_value: float,
    max_paths_per_od: int,
) -> List[Tuple[float, Tuple[int, ...]]]:
    bundles: List[Tuple[float, Tuple[int, ...]]] = []
    for od, paths in path_pool.items():
        current_u = state.od_u[od]
        if current_u <= 1e-9:
            continue
        cap = t_value * shortest[od]
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
            # Add the most exposure-reducing units on this path until the path
            # becomes coverable under the uncovered-length cap.
            need_reduce = current_unc - cap
            selected: List[int] = []
            selected_cost = 0.0
            reduced = 0.0
            for unit, length, cost in sorted(missing, key=lambda item: item[1] / max(item[2], 1e-9), reverse=True):
                if selected_cost + cost > budget_remaining + 1e-9:
                    continue
                selected.append(unit)
                selected_cost += cost
                reduced += length
                if reduced >= need_reduce - 1e-9:
                    break
            if reduced < need_reduce - 1e-9 or selected_cost <= 1e-12:
                continue
            new_unc = max(0.0, current_unc - reduced)
            new_u = new_unc / max(shortest[od], 1e-9)
            exposure_benefit = q_protected[od] * max(0.0, current_u - new_u)
            coverage_benefit = od_demand[od] if state.covered_demand < target_demand - 1e-8 and od not in state.covered_ods else 0.0
            benefit = exposure_benefit + coverage_benefit
            if benefit <= 1e-12:
                continue
            bundles.append((benefit / selected_cost, tuple(sorted(set(selected)))))

    # Remove duplicate bundles and keep the best score for each unit set.
    best_by_units: Dict[Tuple[int, ...], float] = {}
    for score, units in bundles:
        best_by_units[units] = max(score, best_by_units.get(units, 0.0))
    ranked = [(score, units) for units, score in best_by_units.items()]
    ranked.sort(reverse=True)
    return ranked


def greedy_budget_repair(
    state: ExposureState,
    *,
    evaluator,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    budget: float,
    target_demand: float,
    t_value: float,
    candidate_top_k: int,
    max_additions: int,
    max_paths_per_od: int,
) -> ExposureState:
    current = state
    additions = 0
    while additions < max_additions:
        remaining = budget - current.cost
        if remaining <= 1e-9:
            break
        bundles = rank_cover_bundles(
            current,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            shortest=shortest,
            unit_cost=unit_cost,
            budget_remaining=remaining,
            target_demand=target_demand,
            t_value=t_value,
            max_paths_per_od=max_paths_per_od,
        )
        best_state = None
        for _score, units in bundles[:candidate_top_k]:
            trial = set(current.chosen_units)
            trial.update(units)
            trial_state = evaluator(trial)
            if trial_state.cost > budget + 1e-9:
                continue
            if best_state is None or refined_state_key(trial_state, target_demand) < refined_state_key(best_state, target_demand):
                best_state = trial_state
        if best_state is not None and refined_state_key(best_state, target_demand) < refined_state_key(current, target_demand):
            current = best_state
            additions += 1
            continue

        ranked = rank_add_candidates(
            current,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            shortest=shortest,
            unit_cost=unit_cost,
            budget_remaining=remaining,
            target_demand=target_demand,
            max_paths_per_od=max_paths_per_od,
        )
        if not ranked:
            break

        best_state = None
        for _score, unit in ranked[:candidate_top_k]:
            if unit in current.chosen_units or unit_cost[unit] > remaining + 1e-9:
                continue
            trial = set(current.chosen_units)
            trial.add(unit)
            trial_state = evaluator(trial)
            if trial_state.cost > budget + 1e-9:
                continue
            if best_state is None or refined_state_key(trial_state, target_demand) < refined_state_key(best_state, target_demand):
                best_state = trial_state

        if best_state is None or refined_state_key(best_state, target_demand) >= refined_state_key(current, target_demand):
            break
        current = best_state
        additions += 1
    return current


def random_budget_repair(
    state: ExposureState,
    *,
    evaluator,
    rng: random.Random,
    path_pool: Dict[OD, List[legacy_alns.CandidatePath]],
    od_demand: Dict[OD, float],
    q_protected: Dict[OD, float],
    shortest: Dict[OD, float],
    unit_cost: Dict[int, float],
    budget: float,
    target_demand: float,
    t_value: float,
    candidate_top_k: int,
    max_additions: int,
    max_paths_per_od: int,
) -> ExposureState:
    current = state
    for _ in range(max_additions):
        remaining = budget - current.cost
        bundles = rank_cover_bundles(
            current,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            shortest=shortest,
            unit_cost=unit_cost,
            budget_remaining=remaining,
            target_demand=target_demand,
            t_value=t_value,
            max_paths_per_od=max_paths_per_od,
        )[: max(candidate_top_k, 1)]
        if bundles and rng.random() < 0.8:
            weights = [max(score, 1e-9) for score, _units in bundles]
            pick = rng.random() * sum(weights)
            acc = 0.0
            selected_units = bundles[-1][1]
            for weight, (_score, units) in zip(weights, bundles):
                acc += weight
                if acc >= pick:
                    selected_units = units
                    break
            trial = set(current.chosen_units)
            trial.update(selected_units)
            trial_state = evaluator(trial)
            if trial_state.cost <= budget + 1e-9 and refined_state_key(trial_state, target_demand) <= refined_state_key(current, target_demand):
                current = trial_state
                continue

        ranked = rank_add_candidates(
            current,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            shortest=shortest,
            unit_cost=unit_cost,
            budget_remaining=remaining,
            target_demand=target_demand,
            max_paths_per_od=max_paths_per_od,
        )[: max(candidate_top_k, 1)]
        if not ranked:
            break
        weights = [max(score, 1e-9) for score, _unit in ranked]
        pick = rng.random() * sum(weights)
        acc = 0.0
        selected = ranked[-1][1]
        for weight, (_score, unit) in zip(weights, ranked):
            acc += weight
            if acc >= pick:
                selected = unit
                break
        trial = set(current.chosen_units)
        trial.add(selected)
        trial_state = evaluator(trial)
        if trial_state.cost <= budget + 1e-9 and refined_state_key(trial_state, target_demand) <= refined_state_key(current, target_demand):
            current = trial_state
    return current


def destroy_random_units(state: ExposureState, rng: random.Random, fraction: float) -> Set[int]:
    if not state.chosen_units:
        return set()
    count = max(1, int(round(len(state.chosen_units) * fraction)))
    remove = set(rng.sample(sorted(state.chosen_units), min(count, len(state.chosen_units))))
    return set(state.chosen_units - remove)


def destroy_costly_units(state: ExposureState, unit_cost: Dict[int, float], fraction: float) -> Set[int]:
    if not state.chosen_units:
        return set()
    count = max(1, int(round(len(state.chosen_units) * fraction)))
    ordered = sorted(state.chosen_units, key=lambda unit: unit_cost[unit], reverse=True)
    return set(state.chosen_units - set(ordered[:count]))


def destroy_low_impact_units(
    state: ExposureState,
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
        trial = set(state.chosen_units)
        trial.remove(unit)
        trial_state = evaluator(trial)
        burden_loss = max(0.0, trial_state.protected_burden_sum - state.protected_burden_sum)
        scored.append((burden_loss / max(unit_cost[unit], 1e-9), unit))
    scored.sort()
    return set(state.chosen_units - {unit for _score, unit in scored[:count]})


def solve_large_alns_objective_i_e(config: Dict[str, object]) -> Dict[str, object]:
    start_time = time.time()
    network_dir = Path(str(config["network_dir"])).resolve()
    equity_file = Path(str(config["equity_file"])).resolve()
    rng = random.Random(int(config["seed"]))
    budget = float(config["budget"])

    data = legacy_alns.TNTPNetworkData(str(network_dir))
    design_units, arc_to_unit, _unit_arcs, unit_cost = legacy_alns.build_undirected_design_units(data)
    od_demand = legacy_alns.select_ods(data.od_demand, int(config["od_limit"]), int(config["seed"]))
    ods = sorted(od_demand)
    pi = load_equity_shares(equity_file, od_demand)
    q_protected = {od: od_demand[od] * pi[od] for od in ods}
    q_unprotected = {od: od_demand[od] * (1.0 - pi[od]) for od in ods}
    total_demand = sum(od_demand.values())
    total_protected = sum(q_protected.values())
    total_unprotected = sum(q_unprotected.values())
    coverage_share = normalize_demand_share(float(config["coverage_share"]))
    target_demand = coverage_share * total_demand

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

    def evaluator(chosen_units: Set[int]) -> ExposureState:
        return evaluate_exposure_solution(
            chosen_units,
            path_pool=path_pool,
            od_demand=od_demand,
            q_protected=q_protected,
            q_unprotected=q_unprotected,
            shortest=shortest,
            unit_cost=unit_cost,
            t_value=float(config["t"]),
        )

    current = evaluator(set())
    current = greedy_budget_repair(
        current,
        evaluator=evaluator,
        path_pool=path_pool,
        od_demand=od_demand,
        q_protected=q_protected,
        shortest=shortest,
        unit_cost=unit_cost,
        budget=budget,
        target_demand=target_demand,
        t_value=float(config["t"]),
        candidate_top_k=int(config["candidate_top_k"]),
        max_additions=int(config["initial_max_additions"]),
        max_paths_per_od=int(config["max_paths_per_od_scoring"]),
    )
    best = current
    temperature = float(config["initial_temperature"])
    last_report = time.time()
    iterations = int(config["iterations"])
    time_limit = float(config["time_limit"])

    for iteration in range(1, iterations + 1):
        if time_limit > 0 and time.time() - start_time >= time_limit:
            break
        fraction = rng.uniform(float(config["destroy_fraction_min"]), float(config["destroy_fraction_max"]))
        destroy_choice = rng.choice(["random", "costly", "low_impact"])
        if destroy_choice == "random":
            destroyed_units = destroy_random_units(current, rng, fraction)
        elif destroy_choice == "costly":
            destroyed_units = destroy_costly_units(current, unit_cost, fraction)
        else:
            destroyed_units = destroy_low_impact_units(
                current,
                evaluator=evaluator,
                rng=rng,
                unit_cost=unit_cost,
                fraction=fraction,
                sample_size=int(config["low_impact_sample_size"]),
            )

        destroyed = evaluator(destroyed_units)
        if rng.random() < 0.75:
            repaired = greedy_budget_repair(
                destroyed,
                evaluator=evaluator,
                path_pool=path_pool,
                od_demand=od_demand,
                q_protected=q_protected,
                shortest=shortest,
                unit_cost=unit_cost,
                budget=budget,
                target_demand=target_demand,
                t_value=float(config["t"]),
                candidate_top_k=int(config["candidate_top_k"]),
                max_additions=int(config["repair_max_additions"]),
                max_paths_per_od=int(config["max_paths_per_od_scoring"]),
            )
        else:
            repaired = random_budget_repair(
                destroyed,
                evaluator=evaluator,
                rng=rng,
                path_pool=path_pool,
                od_demand=od_demand,
                q_protected=q_protected,
                shortest=shortest,
                unit_cost=unit_cost,
                budget=budget,
                target_demand=target_demand,
                t_value=float(config["t"]),
                candidate_top_k=int(config["candidate_top_k"]),
                max_additions=int(config["repair_max_additions"]),
                max_paths_per_od=int(config["max_paths_per_od_scoring"]),
            )

        current_key = refined_state_key(current, target_demand)
        repaired_key = refined_state_key(repaired, target_demand)
        delta = repaired.protected_burden_sum - current.protected_burden_sum
        accepted = repaired_key <= current_key or (
            coverage_shortfall(current, target_demand) <= 1e-8
            and coverage_shortfall(repaired, target_demand) <= 1e-8
            and temperature > 1e-12
            and rng.random() < math.exp(-delta / temperature)
        )
        if accepted:
            current = repaired
            if refined_state_key(current, target_demand) < refined_state_key(best, target_demand):
                best = current
        temperature *= float(config["cooling_rate"])

        report_interval = float(config["report_interval"])
        if report_interval > 0 and time.time() - last_report >= report_interval:
            print(
                f"[large ALNS seed={config['seed']}] iter={iteration} "
                f"coverage={current.covered_demand / max(total_demand, 1e-9):.2%} "
                f"current_Ep={current.protected_burden_sum / max(total_protected, 1e-9):.4f} "
                f"best_Ep={best.protected_burden_sum / max(total_protected, 1e-9):.4f} "
                f"cost={best.cost:.2f}"
            )
            last_report = time.time()

    id_map = load_id_map(network_dir / "n2024_polygon_id_map.csv")
    chosen_links = sorted(
        {
            data.idx_to_arc[arc_idx]
            for arc_idx, unit in arc_to_unit.items()
            if unit in best.chosen_units
        }
    )
    chosen_design_units = sorted(design_units[unit] for unit in best.chosen_units)
    protected_coverage = best.protected_covered_demand / total_protected if total_protected > 0 else math.nan
    unprotected_coverage = best.unprotected_covered_demand / total_unprotected if total_unprotected > 0 else math.nan
    ep = best.protected_burden_sum / total_protected if total_protected > 0 else math.nan
    eu = best.unprotected_burden_sum / total_unprotected if total_unprotected > 0 else math.nan

    od_stats: Dict[str, Dict[str, object]] = {}
    od_paths: Dict[str, List[Dict[str, object]]] = {}
    for od in ods:
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
            "path_length": path.length,
            "shortest_length": shortest[od],
            "detour_cap": (1.0 + float(config["epsilon"])) * shortest[od],
            "model_uncovered_length": best.od_uncovered_length[od] if covered else 0.0,
            "uncovered_cap_if_required": float(config["t"]) * shortest[od],
            "reference_path_used": not covered,
            "candidate_uncovered_length": best.od_uncovered_length[od],
        }

    covered_share = best.covered_demand / total_demand if total_demand > 0 else math.nan
    coverage_ok = best.covered_demand >= target_demand - 1e-8

    return {
        "model_name": "Objective I-E refine large-network ALNS heuristic",
        "algorithm": "budget_constrained_alns_protected_exposure_with_coverage",
        "status_name": "FEASIBLE_HEURISTIC" if coverage_ok else "NO_FEASIBLE_BUDGET_HEURISTIC_SOLUTION",
        "runtime": time.time() - start_time,
        "network_dir": str(network_dir),
        "equity_file": str(equity_file),
        "budget": budget,
        "epsilon": float(config["epsilon"]),
        "t": float(config["t"]),
        "coverage_target_share": coverage_share,
        "coverage_target_demand": target_demand,
        "service_interpretation": "Budget is a feasibility cap. The 80/80 service guarantee is enforced by coverage_target_share and t: covered_demand_share must reach the target, and every covered OD has uncovered length <= t*K.",
        "od_limit": int(config["od_limit"]),
        "seed": int(config["seed"]),
        "num_ods": len(ods),
        "num_nodes": len(data.nodes),
        "num_arcs": len(data.arcs),
        "num_design_units": len(design_units),
        "total_demand": total_demand,
        "total_protected_demand": total_protected,
        "total_unprotected_demand": total_unprotected,
        "protected_population_demand_share": total_protected / total_demand if total_demand > 0 else math.nan,
        "primary_objective_protected_burden_sum": best.protected_burden_sum,
        "protected_uncovered_exposure_burden": ep,
        "unprotected_uncovered_exposure_burden": eu,
        "exposure_burden_gap_Ep_minus_Eu": ep - eu if math.isfinite(ep) and math.isfinite(eu) else math.nan,
        "construction_cost": best.cost,
        "unused_budget": budget - best.cost,
        "covered_demand": best.covered_demand,
        "covered_demand_share": covered_share,
        "coverage_requirement_satisfied": coverage_ok,
        "service_80_80_satisfied": covered_share >= 0.8 - 1e-8 and float(config["t"]) <= 0.2 + 1e-9 if math.isfinite(covered_share) else False,
        "coverage_shortfall_demand": max(0.0, target_demand - best.covered_demand),
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
        "iterations": iterations,
    }


def large_config_from_args(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    network_dir = SCRIPT_DIR / "data" / "full_network"
    equity_file = Path(args.large_equity_file).expanduser().resolve() if args.large_equity_file else network_dir / args.equity_file_name
    return {
        "network_dir": str(network_dir),
        "equity_file": str(equity_file),
        "equity_file_name": args.equity_file_name,
        "budget": args.large_budget,
        "coverage_share": args.coverage_share,
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
    if args.small_budget is None:
        args.small_budget = args.budget_gamma * args.small_reference_cost
    if args.large_budget is None:
        args.large_budget = args.budget_gamma * args.large_reference_cost

    base_results = Path(args.output_dir).expanduser().resolve()
    base_results.mkdir(parents=True, exist_ok=True)

    small_dir = SCRIPT_DIR / "data" / "cropped_subgraph"
    small_out = base_results / "small_exact_gurobi"
    large_out = base_results / "large_alns"

    print("===== Running small network with exact Gurobi Objective I-E refine =====")
    small_ns = build_exact_namespace(
        small_dir,
        small_out,
        "small_objective_I_E_refine_exact",
        args.small_budget,
        args,
        args.small_od_limit,
    )
    small_result = solve_objective_i_e_refine(small_ns)
    small_paths = write_outputs(
        small_result,
        small_out,
        "small_objective_I_E_refine_exact",
        load_id_map(small_dir / "n2024_polygon_id_map.csv"),
    )
    small_result["output_paths"] = small_paths
    Path(small_paths["json"]).write_text(json.dumps(small_result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Running large network with budget-constrained ALNS Objective I-E refine =====")
    seeds = [args.seed + i for i in range(max(1, args.large_workers))]
    configs = [large_config_from_args(args, seed) for seed in seeds]
    if args.large_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.large_workers) as pool:
            large_candidates = list(pool.map(solve_large_alns_objective_i_e, configs))
    else:
        large_candidates = [solve_large_alns_objective_i_e(configs[0])]

    large_result = min(
        large_candidates,
        key=lambda result: (
            float(result.get("coverage_shortfall_demand", math.inf)),
            float(result.get("primary_objective_protected_burden_sum", math.inf)),
            float(result.get("construction_cost", math.inf)),
        ),
    )
    large_result["multistart_worker_count"] = len(large_candidates)
    large_result["multistart_candidate_summaries"] = [
        {
            "seed": item.get("seed"),
            "status_name": item.get("status_name"),
            "protected_uncovered_exposure_burden": item.get("protected_uncovered_exposure_burden"),
            "construction_cost": item.get("construction_cost"),
            "covered_demand_share": item.get("covered_demand_share"),
            "coverage_requirement_satisfied": item.get("coverage_requirement_satisfied"),
            "coverage_shortfall_demand": item.get("coverage_shortfall_demand"),
            "runtime": item.get("runtime"),
        }
        for item in large_candidates
    ]
    large_paths = write_outputs(
        large_result,
        large_out,
        "large_objective_I_E_refine_alns",
        load_id_map((SCRIPT_DIR / "data" / "full_network") / "n2024_polygon_id_map.csv"),
    )
    large_result["output_paths"] = large_paths
    Path(large_paths["json"]).write_text(json.dumps(large_result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "experiment": "Objective I-E refine protected uncovered-exposure burden with 80/80 service",
        "coverage_target_share": args.coverage_share,
        "service_uncovered_cap_t": args.t,
        "budget_rule": "B = gamma * C_80/80",
        "budget_gamma": args.budget_gamma,
        "small_reference_C_80_80_from_paper": args.small_reference_cost,
        "large_reference_C_80_80_from_paper": args.large_reference_cost,
        "small_network": {
            "algorithm": "exact_gurobi",
            "budget": args.small_budget,
            "status_name": small_result.get("status_name"),
            "protected_uncovered_exposure_burden": small_result.get("protected_uncovered_exposure_burden"),
            "unprotected_uncovered_exposure_burden": small_result.get("unprotected_uncovered_exposure_burden"),
            "construction_cost": small_result.get("construction_cost"),
            "covered_demand_share": small_result.get("covered_demand_share"),
            "coverage_requirement_satisfied": small_result.get("coverage_requirement_satisfied"),
            "service_80_80_satisfied": small_result.get("service_80_80_satisfied"),
            "equity_file": small_result.get("equity_file"),
            "json": small_paths["json"],
        },
        "large_network": {
            "algorithm": "alns_heuristic",
            "budget": args.large_budget,
            "status_name": large_result.get("status_name"),
            "protected_uncovered_exposure_burden": large_result.get("protected_uncovered_exposure_burden"),
            "unprotected_uncovered_exposure_burden": large_result.get("unprotected_uncovered_exposure_burden"),
            "construction_cost": large_result.get("construction_cost"),
            "covered_demand_share": large_result.get("covered_demand_share"),
            "coverage_requirement_satisfied": large_result.get("coverage_requirement_satisfied"),
            "service_80_80_satisfied": large_result.get("service_80_80_satisfied"),
            "coverage_shortfall_demand": large_result.get("coverage_shortfall_demand"),
            "equity_file": large_result.get("equity_file"),
            "json": large_paths["json"],
        },
    }
    summary_path = base_results / "objective_I_E_refine_run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote combined summary: {summary_path}")
    return {"small": small_result, "large": large_result, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Objective I-E refine on small exact and large ALNS networks sequentially.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument(
        "--equity-file-name",
        default=DEFAULT_TARGETED_EQUITY_FILE,
        help="Default equity CSV filename inside each network data directory.",
    )
    parser.add_argument("--small-equity-file", default="", help="Optional explicit small-network equity CSV.")
    parser.add_argument("--large-equity-file", default="", help="Optional explicit large-network equity CSV.")
    parser.add_argument("--budget-gamma", type=float, default=1.5, help="Budget multiplier applied to the paper's C_80/80 reference costs.")
    parser.add_argument("--small-reference-cost", type=float, default=5169.03, help="Paper base-case C_80/80 cost for the small network.")
    parser.add_argument("--large-reference-cost", type=float, default=19469.69, help="Paper base-case C_80/80 ALNS cost for the large network.")
    parser.add_argument("--small-budget", type=float, default=None, help="Optional override. Defaults to budget-gamma * small-reference-cost.")
    parser.add_argument("--large-budget", type=float, default=None, help="Optional override. Defaults to budget-gamma * large-reference-cost.")
    parser.add_argument("--coverage-share", type=float, default=0.8, help="Demand share that must be covered; 0.8 means 80%.")
    parser.add_argument("--small-od-limit", type=int, default=0, help="0 means all small-network ODs.")
    parser.add_argument("--large-od-limit", type=int, default=0, help="0 means all large-network ODs.")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-time-limit", type=float, default=7200.0)
    parser.add_argument("--large-time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=None, help="Gurobi threads for the small exact model.")
    parser.add_argument("--large-workers", type=int, default=1, help="Parallel ALNS multistart workers for the large network.")
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--disable-balanced-od-sampling", action="store_true")

    parser.add_argument("--large-iterations", type=int, default=2500)
    parser.add_argument("--path-pool-size", type=int, default=10)
    parser.add_argument("--path-penalty-growth", type=float, default=0.35)
    parser.add_argument("--path-random-noise", type=float, default=0.12)
    parser.add_argument("--candidate-top-k", type=int, default=25)
    parser.add_argument("--max-paths-per-od-scoring", type=int, default=4)
    parser.add_argument("--initial-max-additions", type=int, default=300)
    parser.add_argument("--repair-max-additions", type=int, default=35)
    parser.add_argument("--destroy-fraction-min", type=float, default=0.04)
    parser.add_argument("--destroy-fraction-max", type=float, default=0.18)
    parser.add_argument("--low-impact-sample-size", type=int, default=60)
    parser.add_argument("--initial-temperature", type=float, default=25.0)
    parser.add_argument("--cooling-rate", type=float, default=0.996)
    parser.add_argument("--report-interval", type=float, default=30.0)
    return parser


if __name__ == "__main__":
    run()
