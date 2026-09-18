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
from typing import Dict, List, Set, Tuple

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
            "service_capacity_ratio",
            "path_length",
            "shortest_length",
            "detour_cap",
            "model_uncovered_length",
            "uncovered_cap_if_required",
            "reference_path_used",
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


def solve_constraint_ii_d(args: argparse.Namespace) -> Dict[str, object]:
    network_dir = Path(args.network_dir).expanduser().resolve()
    if args.equity_file:
        equity_file = Path(args.equity_file).expanduser().resolve()
    else:
        targeted_file = network_dir / DEFAULT_TARGETED_EQUITY_FILE
        equity_file = targeted_file if targeted_file.exists() else network_dir / FALLBACK_SYNTHETIC_EQUITY_FILE
    id_map = load_id_map(network_dir / "n2024_polygon_id_map.csv")

    coverage_share = normalize_demand_share(args.coverage_share)
    eta_exposure = float(args.eta_exposure)
    if eta_exposure < -1e-12:
        raise ValueError("--eta-exposure must be nonnegative")
    if args.t < -1e-12:
        raise ValueError("--t must be nonnegative")

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
    if total_protected_demand <= 1e-9 or total_unprotected_demand <= 1e-9:
        raise ValueError("Both protected and unprotected demand totals must be positive for Constraint II-D.")
    target_demand = coverage_share * total_demand

    K: Dict[OD, float] = {}
    B: Dict[OD, float] = {}
    C: Dict[OD, float] = {}
    warm_path_arcs: Dict[OD, List[int]] = {}
    for od in ods:
        K[od] = shortest_path_length(data, od)
        if K[od] <= 1e-12:
            raise ValueError(f"OD {od_text(od)} has zero shortest-path length.")
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

    model = gp.Model("constraint_II_D_cost_min_exposure_disparity")
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
    )
    unprotected_burden_expr = gp.quicksum(
        q_unprotected[od] * (1.0 - z[od])
        + (q_unprotected[od] / K[od]) * gp.quicksum(c[(od, a)] for a in feasible_arcs[od])
        for od in ods
    )

    model.addConstr(covered_demand_expr >= target_demand, name="total_coverage_requirement")
    model.addConstr(
        protected_burden_expr
        <= (total_protected_demand / total_unprotected_demand) * unprotected_burden_expr
        + eta_exposure * total_protected_demand,
        name="uncovered_exposure_disparity_II_D",
    )
    model.ModelSense = GRB.MINIMIZE
    model.setObjective(construction_cost_expr, GRB.MINIMIZE)

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

    warm_units: Set[int] = set()
    warm_covered: Set[OD] = set()
    for od in ods:
        if all(a in feasible_arc_sets[od] for a in warm_path_arcs[od]):
            warm_covered.add(od)
            warm_units.update(
                arc_to_design_unit[a]
                for a in warm_path_arcs[od]
                if a in arc_to_design_unit and arc_to_design_unit[a] in active_design_unit_set
            )

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

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(model.Status, str(model.Status))

    result: Dict[str, object] = {
        "model_name": "Constraint II-D: cost minimization with uncovered-exposure disparity",
        "algorithm": "exact_gurobi_milp",
        "status": model.Status,
        "status_name": status_name,
        "runtime": runtime,
        "solver_solution_count": model.SolCount,
        "solver_objective_value": solver_attr("ObjVal"),
        "solver_objective_bound": solver_attr("ObjBound"),
        "solver_mip_gap": solver_attr("MIPGap"),
        "network_dir": str(network_dir),
        "equity_file": str(equity_file),
        "epsilon": args.epsilon,
        "t": args.t,
        "coverage_target_share": coverage_share,
        "coverage_target_demand": target_demand,
        "eta_exposure": eta_exposure,
        "service_interpretation": "At least coverage_target_share of total demand is covered; every covered OD has uncovered path length <= t times its shortest-path length.",
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
        "active_total_design_cost": active_total_design_cost,
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
            "strategy": "all_selected_shortest_paths",
            "covered_od_count": len(warm_covered),
            "design_unit_count": len(warm_units),
            "cost": sum(design_unit_cost[q] for q in warm_units),
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
        protected_burden_sum = sum(q_protected[od] * u_w[od] for od in ods)
        unprotected_burden_sum = sum(q_unprotected[od] * u_w[od] for od in ods)
        ep = protected_burden_sum / total_protected_demand
        eu = unprotected_burden_sum / total_unprotected_demand
        covered_demand = sum(od_demand[od] * z_values[od] for od in ods)
        protected_covered_demand = sum(q_protected[od] * z_values[od] for od in ods)
        unprotected_covered_demand = sum(q_unprotected[od] * z_values[od] for od in ods)
        construction_cost = sum(design_unit_cost[q] for q in chosen_unit_indices)
        covered_demand_share = covered_demand / total_demand if total_demand > 0 else math.nan
        exposure_gap = ep - eu

        result.update(
            {
                "objective": construction_cost,
                "construction_cost": construction_cost,
                "protected_uncovered_exposure_burden_sum": protected_burden_sum,
                "unprotected_uncovered_exposure_burden_sum": unprotected_burden_sum,
                "protected_uncovered_exposure_burden": ep,
                "unprotected_uncovered_exposure_burden": eu,
                "exposure_burden_gap_Ep_minus_Eu": exposure_gap,
                "covered_demand": covered_demand,
                "covered_demand_share": covered_demand_share,
                "coverage_requirement_satisfied": covered_demand_share >= coverage_share - 1e-6,
                "exposure_disparity_satisfied": exposure_gap <= eta_exposure + 1e-6,
                "service_80_80_satisfied": covered_demand_share >= 0.8 - 1e-6 and args.t <= 0.2 + 1e-9,
                "protected_covered_demand": protected_covered_demand,
                "protected_coverage_ratio": protected_covered_demand / total_protected_demand,
                "unprotected_covered_demand": unprotected_covered_demand,
                "unprotected_coverage_ratio": unprotected_covered_demand / total_unprotected_demand,
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
                "service_capacity_ratio": 1.0 - min(max(u_w[od], 0.0), 1.0),
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
            "Constraint II-D experiment: minimize investment cost while requiring total 80/80 service "
            "and bounding protected uncovered-exposure disparity."
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
    parser.add_argument("--prefix", default="constraint_II_D")
    parser.add_argument("--coverage-share", type=float, default=0.8, help="Demand share to cover; 0.8 means 80%.")
    parser.add_argument("--eta-exposure", type=float, default=0.05, help="Allowed E^P - E^U tolerance.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Detour tolerance: path length <= (1+epsilon) shortest path.")
    parser.add_argument("--t", type=float, default=0.2, help="Uncovered-length cap for covered ODs; 0.2 means at least 80% served.")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = solve_constraint_ii_d(args)
    id_map = load_id_map(Path(args.network_dir).expanduser().resolve() / "n2024_polygon_id_map.csv")
    output_paths = write_outputs(result, Path(args.output_dir).expanduser().resolve(), args.prefix, id_map)
    result["output_paths"] = output_paths
    Path(output_paths["json"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("===== Constraint II-D uncovered-exposure disparity summary =====")
    print(f"Status: {result['status_name']}")
    print(f"Runtime: {result['runtime']:.2f}s")
    print(f"Coverage target: {result['coverage_target_share']:.2%}")
    print(f"Service cap t: {result['t']:.2f}")
    print(f"Eta exposure: {result['eta_exposure']:.6f}")
    if "construction_cost" in result:
        print(f"Construction cost: {result['construction_cost']:.6f}")
        print(f"Covered demand share: {result['covered_demand_share']:.2%}")
        print(f"E^P: {result['protected_uncovered_exposure_burden']:.6f}")
        print(f"E^U: {result['unprotected_uncovered_exposure_burden']:.6f}")
        print(f"E^P - E^U: {result['exposure_burden_gap_Ep_minus_Eu']:.6f}")
        print(f"Coverage constraint satisfied: {result['coverage_requirement_satisfied']}")
        print(f"II-D constraint satisfied: {result['exposure_disparity_satisfied']}")
    print(f"Wrote JSON: {output_paths['json']}")
    print(f"Wrote OD metrics CSV: {output_paths['od_metrics_csv']}")
    print(f"Wrote chosen design units CSV: {output_paths['chosen_design_units_csv']}")


if __name__ == "__main__":
    main()
