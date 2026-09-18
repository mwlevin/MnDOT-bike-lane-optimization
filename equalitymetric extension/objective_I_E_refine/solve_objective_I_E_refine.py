#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_SRC = SCRIPT_DIR / "legacy_src"
DEFAULT_TARGETED_EQUITY_FILE = "od_equity_shares_targeted_base_uncovered.csv"
FALLBACK_SYNTHETIC_EQUITY_FILE = "od_equity_shares_synthetic.csv"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
(SCRIPT_DIR / ".mplconfig").mkdir(exist_ok=True)
(SCRIPT_DIR / ".cache").mkdir(exist_ok=True)
sys.path.insert(0, str(LEGACY_SRC))

from link_based_z_relax_undirected_design_pruned_exact_5cuts_odpaths_advanced_pruning import (  # noqa: E402
    TNTPNetworkData,
    build_undirected_design_units,
    compute_od_feasible_arcs,
    compute_path_design_metrics,
    extract_path_for_od_sparse,
    find_min_uncovered_reference_path,
    normalize_demand_share,
    select_ods_with_balanced_io,
    shortest_path_arc_indices,
    shortest_path_length,
)


Arc = Tuple[int, int]
OD = Tuple[int, int]


def load_id_map(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {int(row["numeric_id"]): row for row in csv.DictReader(f)}


def load_equity_shares(path: Path, od_demand: Dict[OD, float]) -> Dict[OD, float]:
    if not path.exists():
        raise FileNotFoundError(f"Equity file does not exist: {path}")

    shares: Dict[OD, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"origin", "destination", "pi_w"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Equity file is missing columns: {sorted(missing)}")
        for row in reader:
            od = (int(row["origin"]), int(row["destination"]))
            pi = float(row["pi_w"])
            if pi < -1e-12 or pi > 1.0 + 1e-12:
                raise ValueError(f"pi_w must be in [0, 1], got {pi} for OD {od}")
            shares[od] = min(max(pi, 0.0), 1.0)

    missing_ods = sorted(od for od in od_demand if od not in shares)
    if missing_ods:
        preview = ", ".join(f"{o}->{d}" for o, d in missing_ods[:10])
        raise ValueError(f"Equity file is missing {len(missing_ods)} selected ODs, including {preview}")
    return {od: shares[od] for od in od_demand}


def od_text(od: OD) -> str:
    return f"{od[0]}->{od[1]}"


def unit_to_row(unit: Arc, id_map: Dict[int, Dict[str, str]]) -> Dict[str, object]:
    u, v = unit
    return {
        "numeric_u": u,
        "numeric_v": v,
        "original_u": id_map.get(u, {}).get("original_id", str(u)),
        "original_v": id_map.get(v, {}).get("original_id", str(v)),
    }


def arc_list_to_rows(arcs: List[Arc], id_map: Dict[int, Dict[str, str]]) -> List[Dict[str, object]]:
    return [unit_to_row(arc, id_map) for arc in arcs]


def write_outputs(
    result: Dict[str, object],
    output_dir: Path,
    prefix: str,
    id_map: Dict[int, Dict[str, str]],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    od_csv_path = output_dir / f"{prefix}_od_metrics.csv"
    unit_csv_path = output_dir / f"{prefix}_chosen_design_units.csv"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    od_stats = result.get("od_stats", {})
    if isinstance(od_stats, dict):
        fieldnames = [
            "od",
            "origin",
            "destination",
            "original_origin",
            "original_destination",
            "demand",
            "pi_w",
            "protected_demand",
            "unprotected_demand",
            "z_value",
            "covered",
            "u_w",
            "path_length",
            "shortest_length",
            "detour_cap",
            "model_uncovered_length",
            "uncovered_cap_if_required",
        ]
        with od_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(od_stats):
                stats = od_stats[key]
                if isinstance(stats, dict):
                    writer.writerow({field: stats.get(field, "") for field in fieldnames})

    chosen_units = result.get("chosen_design_units", [])
    with unit_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["numeric_u", "numeric_v", "original_u", "original_v"])
        writer.writeheader()
        if isinstance(chosen_units, list):
            for item in chosen_units:
                if isinstance(item, dict):
                    writer.writerow(item)

    return {
        "json": str(json_path),
        "od_metrics_csv": str(od_csv_path),
        "chosen_design_units_csv": str(unit_csv_path),
    }


def solve_objective_i_e_refine(args: argparse.Namespace) -> Dict[str, object]:
    network_dir = Path(args.network_dir).expanduser().resolve()
    if args.equity_file:
        equity_file = Path(args.equity_file).expanduser().resolve()
    else:
        targeted_file = network_dir / DEFAULT_TARGETED_EQUITY_FILE
        equity_file = targeted_file if targeted_file.exists() else network_dir / FALLBACK_SYNTHETIC_EQUITY_FILE
    id_map = load_id_map(network_dir / "n2024_polygon_id_map.csv")
    coverage_share = normalize_demand_share(args.coverage_share)

    data = TNTPNetworkData(str(network_dir))
    od_demand, od_sampling_stats = select_ods_with_balanced_io(
        data.od_demand,
        args.od_limit,
        seed=args.seed,
        require_balanced_io=not args.disable_balanced_od_sampling,
    )
    ods = sorted(od_demand)
    if not ods:
        raise ValueError("No positive selected OD pairs.")

    pi = load_equity_shares(equity_file, od_demand)
    q_protected = {od: od_demand[od] * pi[od] for od in ods}
    q_unprotected = {od: od_demand[od] * (1.0 - pi[od]) for od in ods}
    total_demand = sum(od_demand.values())
    total_protected_demand = sum(q_protected.values())
    total_unprotected_demand = sum(q_unprotected.values())
    target_demand = coverage_share * total_demand

    K: Dict[OD, float] = {}
    B: Dict[OD, float] = {}
    C: Dict[OD, float] = {}
    warm_path_arcs: Dict[OD, List[int]] = {}
    for od in ods:
        K[od] = shortest_path_length(data, od)
        B[od] = (1.0 + args.epsilon) * K[od]
        C[od] = args.t * K[od]
        path_arcs, _ = shortest_path_arc_indices(data, od)
        warm_path_arcs[od] = path_arcs

    design_units, arc_to_design_unit, design_unit_arcs, design_unit_cost = build_undirected_design_units(data)
    feasible_arcs, pruning_stats = compute_od_feasible_arcs(
        data,
        ods,
        B,
        C,
        use_detour_pruning=not args.disable_od_arc_pruning,
        use_node_reachability_pruning=not args.disable_node_reachability_pruning,
        use_source_sink_variable_pruning=not args.disable_source_sink_variable_pruning,
        use_directed_subgraph_trimming=not args.disable_directed_subgraph_trimming,
        drop_long_uncovered_arcs=False,
    )
    feasible_arc_sets = {od: set(arcs) for od, arcs in feasible_arcs.items()}
    active_design_unit_indices = sorted({arc_to_design_unit[a] for arcs in feasible_arcs.values() for a in arcs})
    active_design_unit_set = set(active_design_unit_indices)
    active_total_design_cost = sum(design_unit_cost[q] for q in active_design_unit_indices)

    budget = float(args.budget)
    if budget < -1e-9:
        raise ValueError("--budget must be nonnegative")

    model = gp.Model("objective_I_E_refine_protected_exposure_with_coverage")
    model.Params.OutputFlag = int(args.output_flag)
    model.Params.TimeLimit = float(args.time_limit)
    model.Params.MIPGap = float(args.mip_gap)
    if args.threads is not None:
        model.Params.Threads = int(args.threads)

    delta = {
        q: model.addVar(vtype=GRB.BINARY, name=f"delta[{design_units[q][0]},{design_units[q][1]}]")
        for q in active_design_unit_indices
    }
    x = {
        (od, a): model.addVar(vtype=GRB.BINARY, name=f"x[{od[0]},{od[1]},{a}]")
        for od in ods
        for a in feasible_arcs[od]
    }
    c = {
        (od, a): model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"c[{od[0]},{od[1]},{a}]")
        for od in ods
        for a in feasible_arcs[od]
    }
    z = {
        od: model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z[{od[0]},{od[1]}]")
        for od in ods
    }

    construction_cost_expr = gp.quicksum(design_unit_cost[q] * delta[q] for q in active_design_unit_indices)
    covered_demand_expr = gp.quicksum(od_demand[od] * z[od] for od in ods)
    protected_burden_expr = gp.quicksum(
        q_protected[od] * (1.0 - z[od])
        + (q_protected[od] / K[od]) * gp.quicksum(c[(od, a)] for a in feasible_arcs[od])
        for od in ods
        if K[od] > 1e-12
    )

    model.addConstr(construction_cost_expr <= budget, name="investment_budget")
    model.addConstr(covered_demand_expr >= target_demand, name="total_coverage_requirement")
    model.ModelSense = GRB.MINIMIZE
    if args.disable_lexicographic_cost_tiebreak:
        model.setObjective(protected_burden_expr, GRB.MINIMIZE)
    else:
        model.setObjectiveN(protected_burden_expr, index=0, priority=2, name="protected_uncovered_exposure")
        model.setObjectiveN(construction_cost_expr, index=1, priority=1, name="construction_cost_tiebreak")

    origin_destination_strengthening_count = 0
    opposite_direction_x_cut_count = 0
    single_arc_uncovered_implied_cut_count = 0
    minimum_covered_length_cut_count = 0
    single_arc_detour_implied_cut_count = 0

    for od in ods:
        origin, destination = od
        feasible_set = feasible_arc_sets[od]
        for node in data.nodes:
            out_expr = gp.quicksum(x[(od, a)] for a in data.out_arcs.get(node, []) if a in feasible_set)
            in_expr = gp.quicksum(x[(od, a)] for a in data.in_arcs.get(node, []) if a in feasible_set)
            b = 1 if node == origin else (-1 if node == destination else 0)
            model.addConstr(out_expr - in_expr == b * z[od], name=f"flow_scaled[{origin},{destination},{node}]")
            model.addConstr(out_expr <= z[od], name=f"outdeg_scaled[{origin},{destination},{node}]")
            model.addConstr(in_expr <= z[od], name=f"indeg_scaled[{origin},{destination},{node}]")

        if not args.disable_origin_destination_strengthening:
            out_o = gp.quicksum(x[(od, a)] for a in data.out_arcs.get(origin, []) if a in feasible_set)
            in_o = gp.quicksum(x[(od, a)] for a in data.in_arcs.get(origin, []) if a in feasible_set)
            out_d = gp.quicksum(x[(od, a)] for a in data.out_arcs.get(destination, []) if a in feasible_set)
            in_d = gp.quicksum(x[(od, a)] for a in data.in_arcs.get(destination, []) if a in feasible_set)
            model.addConstr(in_o == 0.0, name=f"origin_no_inflow[{origin},{destination}]")
            model.addConstr(out_o == z[od], name=f"origin_one_outflow[{origin},{destination}]")
            model.addConstr(out_d == 0.0, name=f"destination_no_outflow[{origin},{destination}]")
            model.addConstr(in_d == z[od], name=f"destination_one_inflow[{origin},{destination}]")
            origin_destination_strengthening_count += 4

        if not args.disable_opposite_direction_x_cut:
            for q, arcs_in_unit in design_unit_arcs.items():
                unit_od_arcs = [a for a in arcs_in_unit if a in feasible_set]
                if len(unit_od_arcs) <= 1:
                    continue
                model.addConstr(
                    gp.quicksum(x[(od, a)] for a in unit_od_arcs) <= z[od],
                    name=f"opposite_direction_x_cut[{origin},{destination},{q}]",
                )
                opposite_direction_x_cut_count += 1

        model.addConstr(
            gp.quicksum(data.arc_len[a] * x[(od, a)] for a in feasible_arcs[od]) <= B[od] * z[od],
            name=f"detour_scaled[{origin},{destination}]",
        )
        model.addConstr(
            gp.quicksum(c[(od, a)] for a in feasible_arcs[od]) <= C[od] * z[od],
            name=f"uncovered_cap_scaled[{origin},{destination}]",
        )

        if not args.disable_minimum_covered_length_cut:
            min_covered_len = max(0.0, K[od] - C[od])
            if min_covered_len > 1e-9:
                model.addConstr(
                    gp.quicksum(data.arc_len[a] * x[(od, a)] - c[(od, a)] for a in feasible_arcs[od])
                    >= min_covered_len * z[od],
                    name=f"minimum_covered_length[{origin},{destination}]",
                )
                minimum_covered_length_cut_count += 1

    for od in ods:
        for a in feasible_arcs[od]:
            ell = data.arc_len[a]
            q = arc_to_design_unit[a]
            if q not in active_design_unit_set:
                raise RuntimeError(f"Inactive design unit {q} appeared in feasible OD arc set.")
            model.addConstr(c[(od, a)] <= ell * x[(od, a)], name=f"c_ub_x[{od[0]},{od[1]},{a}]")
            model.addConstr(c[(od, a)] <= ell * (1 - delta[q]), name=f"c_ub_d[{od[0]},{od[1]},{a}]")
            model.addConstr(c[(od, a)] >= ell * (x[(od, a)] - delta[q]), name=f"c_lb[{od[0]},{od[1]},{a}]")

            if not args.disable_single_arc_uncovered_implied_cut:
                model.addConstr(
                    ell * (x[(od, a)] - delta[q]) <= C[od] * z[od],
                    name=f"single_arc_uncovered_implied[{od[0]},{od[1]},{a}]",
                )
                single_arc_uncovered_implied_cut_count += 1
            if not args.disable_single_arc_detour_implied_cut:
                model.addConstr(
                    ell * x[(od, a)] <= B[od] * z[od],
                    name=f"single_arc_detour_implied[{od[0]},{od[1]},{a}]",
                )
                single_arc_detour_implied_cut_count += 1

    # MIP start: greedily cover high-protected-demand shortest paths within budget.
    warm_units: Set[int] = set()
    warm_covered: Set[OD] = set()
    warm_cost = 0.0
    for od in sorted(ods, key=lambda item: (-q_protected[item], -od_demand[item], item)):
        path_units = {arc_to_design_unit[a] for a in warm_path_arcs[od] if a in arc_to_design_unit}
        missing_units = path_units - warm_units
        added_cost = sum(design_unit_cost[q] for q in missing_units if q in active_design_unit_set)
        if warm_cost + added_cost <= budget + 1e-9 and all(a in feasible_arc_sets[od] for a in warm_path_arcs[od]):
            warm_units.update(q for q in missing_units if q in active_design_unit_set)
            warm_covered.add(od)
            warm_cost += added_cost

    for q in active_design_unit_indices:
        delta[q].Start = 1.0 if q in warm_units else 0.0
    for od in ods:
        z[od].Start = 1.0 if od in warm_covered else 0.0
        path_set = set(warm_path_arcs[od]) if od in warm_covered else set()
        for a in feasible_arcs[od]:
            x[(od, a)].Start = 1.0 if a in path_set else 0.0
            c[(od, a)].Start = 0.0

    start = time.time()
    model.optimize()
    runtime = time.time() - start

    def solver_attr(name: str):
        try:
            return getattr(model, name)
        except (gp.GurobiError, AttributeError):
            return None

    result: Dict[str, object] = {
        "model_name": "Objective I-E refine: budget-constrained protected exposure minimization with 80/80 coverage",
        "status": model.Status,
        "status_name": {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT"}.get(model.Status, str(model.Status)),
        "runtime": runtime,
        "solver_solution_count": model.SolCount,
        "solver_objective_value": solver_attr("ObjVal"),
        "solver_objective_bound": solver_attr("ObjBound"),
        "solver_mip_gap": solver_attr("MIPGap"),
        "network_dir": str(network_dir),
        "equity_file": str(equity_file),
        "budget": budget,
        "active_total_design_cost": active_total_design_cost,
        "budget_share_of_active_total_design_cost": budget / active_total_design_cost if active_total_design_cost > 0 else math.nan,
        "epsilon": args.epsilon,
        "t": args.t,
        "coverage_target_share": coverage_share,
        "coverage_target_demand": target_demand,
        "service_interpretation": "Budget is a feasibility cap. The 80/80 service guarantee is enforced by coverage_target_share and t: covered_demand_share must reach the target, and every covered OD has uncovered length <= t*K.",
        "od_limit": args.od_limit,
        "seed": args.seed,
        "num_ods": len(ods),
        "num_nodes": len(data.nodes),
        "num_arcs": len(data.arcs),
        "num_design_units_full": len(design_units),
        "num_design_units_active": len(active_design_unit_indices),
        "num_od_arc_variables_after_pruning": sum(len(arcs) for arcs in feasible_arcs.values()),
        "total_demand": total_demand,
        "total_protected_demand": total_protected_demand,
        "total_unprotected_demand": total_unprotected_demand,
        "protected_population_demand_share": total_protected_demand / total_demand if total_demand > 0 else math.nan,
        "od_sampling_stats": od_sampling_stats,
        "pruning_stats_summary": {
            "removed_by_source_sink": sum(v["removed_by_source_sink"] for v in pruning_stats.values()),
            "removed_by_node_reachability": sum(v["removed_by_node_reachability"] for v in pruning_stats.values()),
            "removed_by_detour": sum(v["removed_by_detour"] for v in pruning_stats.values()),
            "removed_by_directed_subgraph_trimming": sum(v["removed_by_directed_subgraph_trimming"] for v in pruning_stats.values()),
            "removed_unreachable": sum(v["removed_unreachable"] for v in pruning_stats.values()),
        },
        "cut_counts": {
            "origin_destination_strengthening": origin_destination_strengthening_count,
            "opposite_direction_x": opposite_direction_x_cut_count,
            "single_arc_uncovered_implied": single_arc_uncovered_implied_cut_count,
            "minimum_covered_length": minimum_covered_length_cut_count,
            "single_arc_detour_implied": single_arc_detour_implied_cut_count,
        },
        "warm_start": {
            "covered_od_count": len(warm_covered),
            "design_unit_count": len(warm_units),
            "cost": warm_cost,
        },
    }

    if model.SolCount > 0:
        chosen_unit_indices = [q for q in active_design_unit_indices if delta[q].X > 0.5]
        chosen_unit_set = set(chosen_unit_indices)
        chosen_links = [
            data.idx_to_arc[a]
            for a in range(len(data.arcs))
            if arc_to_design_unit.get(a) in chosen_unit_set
        ]
        z_values = {od: z[od].X for od in ods}
        model_uncovered = {
            od: sum(c[(od, a)].X for a in feasible_arcs[od])
            for od in ods
        }
        u_w = {
            od: (1.0 - z_values[od]) + model_uncovered[od] / K[od]
            for od in ods
        }
        protected_burden = sum(q_protected[od] * u_w[od] for od in ods)
        unprotected_burden = sum(q_unprotected[od] * u_w[od] for od in ods)
        covered_demand = sum(od_demand[od] * z_values[od] for od in ods)
        protected_covered_demand = sum(q_protected[od] * z_values[od] for od in ods)
        unprotected_covered_demand = sum(q_unprotected[od] * z_values[od] for od in ods)
        construction_cost = sum(design_unit_cost[q] for q in chosen_unit_indices)

        result.update(
            {
                "primary_objective_protected_burden_sum": protected_burden,
                "protected_uncovered_exposure_burden": protected_burden / total_protected_demand
                if total_protected_demand > 0
                else math.nan,
                "unprotected_uncovered_exposure_burden": unprotected_burden / total_unprotected_demand
                if total_unprotected_demand > 0
                else math.nan,
                "exposure_burden_gap_Ep_minus_Eu": (
                    protected_burden / total_protected_demand - unprotected_burden / total_unprotected_demand
                    if total_protected_demand > 0 and total_unprotected_demand > 0
                    else math.nan
                ),
                "construction_cost": construction_cost,
                "unused_budget": budget - construction_cost,
                "covered_demand": covered_demand,
                "covered_demand_share": covered_demand / total_demand if total_demand > 0 else math.nan,
                "coverage_requirement_satisfied": (
                    covered_demand / total_demand >= coverage_share - 1e-6 if total_demand > 0 else False
                ),
                "service_80_80_satisfied": (
                    covered_demand / total_demand >= 0.8 - 1e-6 and args.t <= 0.2 + 1e-9
                    if total_demand > 0
                    else False
                ),
                "protected_covered_demand": protected_covered_demand,
                "protected_coverage_ratio": protected_covered_demand / total_protected_demand
                if total_protected_demand > 0
                else math.nan,
                "unprotected_covered_demand": unprotected_covered_demand,
                "unprotected_coverage_ratio": unprotected_covered_demand / total_unprotected_demand
                if total_unprotected_demand > 0
                else math.nan,
                "covered_od_count": sum(1 for od in ods if z_values[od] > 0.5),
                "chosen_design_unit_count": len(chosen_unit_indices),
                "chosen_design_units": [unit_to_row(design_units[q], id_map) for q in chosen_unit_indices],
                "chosen_links": arc_list_to_rows(chosen_links, id_map),
            }
        )

        od_stats: Dict[str, Dict[str, object]] = {}
        od_paths: Dict[str, List[Dict[str, object]]] = {}
        for od in ods:
            covered = z_values[od] > 0.5
            if covered:
                path_arc_idx = extract_path_for_od_sparse(data, x, od, feasible_arc_sets[od])
            else:
                path_arc_idx = find_min_uncovered_reference_path(data, od, chosen_unit_set, arc_to_design_unit, B[od])
            metrics = compute_path_design_metrics(data, path_arc_idx, chosen_unit_set, arc_to_design_unit, K[od])
            path_arcs = [data.idx_to_arc[a] for a in path_arc_idx]
            key = od_text(od)
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
                "z_value": z_values[od],
                "covered": covered,
                "u_w": u_w[od],
                "path_length": metrics["path_length"],
                "shortest_length": K[od],
                "detour_cap": B[od],
                "model_uncovered_length": model_uncovered[od],
                "uncovered_cap_if_required": C[od],
                "reference_path_used": not covered,
            }
        result["od_stats"] = od_stats
        result["od_paths"] = od_paths

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Objective I-E refine experiment: minimize protected-population uncovered-exposure burden "
            "under a fixed construction budget while enforcing 80/80 service coverage."
        )
    )
    parser.add_argument("--network-dir", default=str(SCRIPT_DIR / "data" / "cropped_subgraph"))
    parser.add_argument(
        "--equity-file",
        default="",
        help=(
            f"Defaults to {DEFAULT_TARGETED_EQUITY_FILE} in --network-dir when present; "
            f"otherwise falls back to {FALLBACK_SYNTHETIC_EQUITY_FILE}."
        ),
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--prefix", default="objective_I_E_refine")
    parser.add_argument("--budget", type=float, required=True, help="Construction budget in the same units as link lengths.")
    parser.add_argument("--coverage-share", type=float, default=0.8, help="Demand share that must satisfy the OD/path service definition.")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--od-limit", type=int, default=0, help="0 means all positive OD pairs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--disable-balanced-od-sampling", action="store_true")
    parser.add_argument("--disable-od-arc-pruning", action="store_true")
    parser.add_argument("--disable-node-reachability-pruning", action="store_true")
    parser.add_argument("--disable-source-sink-variable-pruning", action="store_true")
    parser.add_argument("--disable-directed-subgraph-trimming", action="store_true")
    parser.add_argument("--disable-origin-destination-strengthening", action="store_true")
    parser.add_argument("--disable-opposite-direction-x-cut", action="store_true")
    parser.add_argument("--disable-single-arc-uncovered-implied-cut", action="store_true")
    parser.add_argument("--disable-minimum-covered-length-cut", action="store_true")
    parser.add_argument("--disable-single-arc-detour-implied-cut", action="store_true")
    parser.add_argument(
        "--disable-lexicographic-cost-tiebreak",
        action="store_true",
        help="Use only the protected burden objective; by default cost is a secondary tie-break objective.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = solve_objective_i_e_refine(args)
    id_map = load_id_map(Path(args.network_dir).expanduser().resolve() / "n2024_polygon_id_map.csv")
    output_paths = write_outputs(result, Path(args.output_dir).expanduser().resolve(), args.prefix, id_map)
    result["output_paths"] = output_paths
    Path(output_paths["json"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Objective I-E refine protected uncovered-exposure summary =====")
    print(f"Status: {result['status_name']}")
    print(f"Runtime: {result['runtime']:.2f}s")
    print(f"Budget: {result['budget']:.6f}")
    print(f"Coverage target: {result['coverage_target_share']:.2%}")
    print(f"Service cap t: {result['t']:.2f}")
    if "primary_objective_protected_burden_sum" in result:
        print(f"Protected burden sum: {result['primary_objective_protected_burden_sum']:.6f}")
        print(f"E^P: {result['protected_uncovered_exposure_burden']:.6f}")
        print(f"E^U: {result['unprotected_uncovered_exposure_burden']:.6f}")
        print(f"E^P - E^U: {result['exposure_burden_gap_Ep_minus_Eu']:.6f}")
        print(f"Construction cost: {result['construction_cost']:.6f}")
        print(f"Covered demand share: {result['covered_demand_share']:.2%}")
        print(f"Coverage constraint satisfied: {result['coverage_requirement_satisfied']}")
        print(f"Protected coverage ratio: {result['protected_coverage_ratio']:.2%}")
    print(f"Wrote JSON: {output_paths['json']}")
    print(f"Wrote OD metrics CSV: {output_paths['od_metrics_csv']}")
    print(f"Wrote chosen design units CSV: {output_paths['chosen_design_units_csv']}")


if __name__ == "__main__":
    main()
