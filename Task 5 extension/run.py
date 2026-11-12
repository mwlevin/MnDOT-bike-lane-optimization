#!/usr/bin/env python3
"""Portable CLI for budget-constrained bicycle-lane safety planning."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

from network import load_certificate, load_design, load_problem
from solver import solve

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def fingerprint_with_coverage(problem, share):
    identity = [problem.network_fingerprint,
                [(w.origin, w.destination, w.demand) for w in problem.trips],
                share, problem.epsilon, problem.uncovered_share]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def bundled_baseline(problem):
    """Find bundled historical evidence compatible with the current inputs."""
    manifest = ROOT / "baselines/manifest.json"
    if not manifest.exists():
        return None
    for entry in read(manifest)["networks"].values():
        if (entry["network_fingerprint"] != problem.network_fingerprint or
                entry["historical_problem_fingerprint"] != fingerprint_with_coverage(problem, .8)):
            continue
        paths = {}
        for kind, info in entry["files"].items():
            path = ROOT / "baselines" / info["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"Bundled baseline was modified: {info['path']}")
            paths[kind] = path
        return {**entry, **paths}
    return None


def save_result(directory, name, result):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    result["output_paths"] = {"json": str(path.resolve())}
    if result.get("solution_available"):
        tables = {"od_metrics": list(result.get("od_stats", {}).values()),
                  "chosen_design_units": result.get("chosen_design_units", []),
                  "new_design_units": [r for r in result.get("chosen_design_units", []) if r["new"]]}
        for suffix, rows in tables.items():
            target = directory / f"{name}_{suffix}.csv"
            fields = list(rows[0]) if rows else ["numeric_u", "numeric_v", "original_u", "original_v", "cost", "existing", "new"]
            with target.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            result["output_paths"][suffix] = str(target.resolve())
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def cached_evaluation(source, problem, units, bundled):
    """Check a fixed-layout evaluation before reusing it as a baseline."""
    source = Path(source).expanduser().resolve()
    doc = read(source)
    trusted_legacy = bundled is not None and source == bundled["evaluation"].resolve()
    expected = {problem.problem_fingerprint}
    if trusted_legacy:
        expected.add(bundled["historical_problem_fingerprint"])
    if (not doc.get("solution_available") or doc.get("objective_kind") != "j2" or
            doc.get("problem_fingerprint") not in expected or load_design(source, problem) != units):
        raise ValueError("Cached evaluation does not match the current inputs and fixed layout")
    certificate = load_certificate(source, problem, units)
    keys = {f"{w.origin}->{w.destination}" for w in problem.trips}
    if certificate is None or set(certificate["od_stats"]) != keys or set(doc["od_stats"]) != keys:
        raise ValueError("Cached evaluation has missing or invalid OD routes")
    arcs = {(a.u, a.v): a for a in problem.arcs}
    objective, covered_demand = 0.0, 0.0
    for trip in problem.trips:
        key = f"{trip.origin}->{trip.destination}"
        stat = doc["od_stats"][key]
        if not math.isclose(stat["demand"], trip.demand) or not math.isclose(stat["shortest_length"], trip.shortest):
            raise ValueError("Cached OD demand or shortest distance does not match")
        if certificate["od_stats"][key]["covered"]:
            covered_demand += trip.demand
            route = certificate["od_paths"][key]
            distance = sum(arcs[tuple(a)].length for a in route if arcs[tuple(a)].unit not in units)
            length = sum(arcs[tuple(a)].length for a in route)
            if not math.isclose(length, stat["path_length"], abs_tol=1e-5) or not math.isclose(distance, stat["model_uncovered_length"], abs_tol=1e-5):
                raise ValueError("Cached route metrics do not match the selected facilities")
        else:
            if doc["od_paths"].get(key):
                raise ValueError("Uncertified OD has a nonempty cached route")
            distance = trip.shortest
        objective += trip.demand * distance
    cost = sum(problem.costs[j] for j in units)
    if not math.isclose(objective, doc["j2"], rel_tol=1e-8, abs_tol=1e-5) or not math.isclose(cost, doc["construction_cost"], rel_tol=1e-8, abs_tol=1e-5):
        raise ValueError("Cached cost or objective does not match independent evaluation")
    if not math.isclose(covered_demand / problem.total_demand, doc["covered_demand_share"], abs_tol=1e-8):
        raise ValueError("Cached coverage does not match its route certificates")
    if trusted_legacy and (not doc.get("optimal_to_tolerance") or covered_demand < .8 * problem.total_demand - 1e-6):
        raise ValueError("Historical evaluation does not satisfy the conditions for fixed-layout reuse")
    # With fixed facilities, certifying an available route strictly reduces J
    # because t < 1. The bundled optimal fixed-layout evaluation is unchanged
    # when its historical aggregate coverage constraint is removed.
    result = copy.deepcopy(doc)
    result["source_evaluation"] = {"file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                   "problem_fingerprint": doc["problem_fingerprint"], "status_name": doc["status_name"],
                                   "runtime_seconds": doc["runtime_seconds"], "solver_mip_gap": doc.get("solver_mip_gap"),
                                   "coverage_target_share": doc.get("coverage_target_share")}
    result.update(problem_fingerprint=problem.problem_fingerprint, coverage_target_share=None,
                  aggregate_coverage_constraint=False, coverage_requirement_satisfied=None, reused_evaluation=True)
    result.pop("service_80_80_satisfied", None)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Task 5: minimize safety exposure burden under a construction budget.")
    tasks = parser.add_subparsers(dest="task", required=True)
    for task in ("full", "incremental"):
        sub = tasks.add_parser(task)
        sub.add_argument("--network", choices=["small", "large"], default="small")
        sub.add_argument("--network-dir", type=Path, help="Custom TNTP directory; overrides the bundled network.")
        sub.add_argument("--cost-file", type=Path, help="Complete physical-unit cost CSV: numeric_u,numeric_v,cost.")
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--epsilon", type=float, default=.2)
        sub.add_argument("--t", type=float, default=.2)
        sub.add_argument("--od-limit", type=int, default=0)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--solver", choices=["auto", "gurobi", "alns"], default="auto")
        sub.add_argument("--time-limit", type=float, default=18000, help="Final solver/search limit in seconds.")
        sub.add_argument("--evaluation-time-limit", type=float, default=300, help="Fixed-layout evaluation limit for a new case without a compatible cache.")
        sub.add_argument("--mip-gap", type=float, default=1e-4)
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--path-pool-size", type=int, default=12)
        sub.add_argument("--path-pool-cache", type=Path)
        sub.add_argument("--alns-iterations", type=int, default=1000000000)
        sub.add_argument("--baseline-j2-result", type=Path, help="Optional fixed-layout evaluation matching the current inputs.")
        sub.add_argument("--quiet", action="store_true")
        sub.add_argument("--write-model", action="store_true")
        if task == "full":
            sub.add_argument("--multiplier", type=float, required=True, help="Total budget as a multiple of the reference layout cost.")
            sub.add_argument("--reference-cost-result", "--minimum-cost-result", dest="reference_cost_result", type=Path,
                             help="Reference layout JSON; defaults to the bundled historical reference.")
        else:
            sub.add_argument("--existing-design", type=Path, required=True)
            sub.add_argument("--budget", type=float, required=True, help="Absolute additional budget; existing cost is excluded.")
    return parser


def run(args):
    value = args.multiplier if args.task == "full" else args.budget
    if not math.isfinite(value) or value < 0:
        raise ValueError("Budget or multiplier must be finite and nonnegative")
    if not math.isfinite(args.time_limit) or args.time_limit <= 0 or not math.isfinite(args.evaluation_time_limit) or args.evaluation_time_limit <= 0:
        raise ValueError("Time limits must be finite and positive")
    problem = load_problem(args.network_dir or ROOT / "data" / args.network, epsilon=args.epsilon,
                           uncovered_share=args.t, od_limit=args.od_limit, seed=args.seed, cost_file=args.cost_file)
    bundled = bundled_baseline(problem)
    solver_name = args.solver if args.solver != "auto" else ("alns" if args.network == "large" or len(problem.nodes) > 800 else "gurobi")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or ROOT / "results" / f"{args.task}_{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"task": args.task, "objective": "J", "network_dir": str(problem.directory),
               "problem_fingerprint": problem.problem_fingerprint, "network_fingerprint": problem.network_fingerprint,
               "aggregate_coverage_constraint": False, "epsilon": args.epsilon, "t": args.t,
               "num_ods": len(problem.trips), "seed": args.seed, "algorithm": solver_name, "stages": {}}

    def finish(status, code):
        summary["status_name"] = status
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"{status}: {output / 'run_summary.json'}", flush=True)
        return summary, code

    def stage(name, **kwargs):
        function = solve
        if solver_name == "alns":
            from alns import solve_alns
            function = solve_alns
            kwargs.update(seed=args.seed, path_pool_size=args.path_pool_size, path_pool_cache=args.path_pool_cache,
                          max_iterations=args.alns_iterations)
        result = function(problem, threads=args.threads, mip_gap=args.mip_gap, output_flag=int(not args.quiet),
                          progress_file=str(output / f"{name}_progress.json"), log_file=str(output / f"{name}.log"),
                          model_file=str(output / f"{name}.lp") if args.write_model else None, **kwargs)
        path = save_result(output, name, result)
        summary["stages"][name] = {"path": str(path), "status_name": result["status_name"], "reused": False}
        return result

    if args.task == "full":
        source = args.reference_cost_result
        if source is None:
            if args.network_dir is not None or args.cost_file is not None:
                raise ValueError("Custom full-planning inputs require --reference-cost-result")
            source = ROOT / "baselines" / args.network / "reference_cost.json"
        reference = read(source)
        units = load_design(source, problem)
        reference_cost = sum(problem.costs[j] for j in units)
        if not math.isclose(reference.get("construction_cost", reference_cost), reference_cost, rel_tol=1e-8, abs_tol=1e-5):
            raise ValueError("Reference layout cost does not match the current construction costs")
        if reference.get("network_fingerprint", problem.network_fingerprint) != problem.network_fingerprint:
            raise ValueError("Reference layout belongs to a different network or construction-cost system")
        reference_cost = reference.get("construction_cost", reference_cost)
        budget, existing = args.multiplier * reference_cost, set()
        summary.update(reference_cost=reference_cost, reference_source=str(source), multiplier=args.multiplier,
                       total_budget=budget, budget_basis="reference_layout_cost")
    else:
        source = args.existing_design
        units = load_design(source, problem)
        budget, existing = args.budget, units
        summary.update(existing_design_file=str(source), existing_cost=sum(problem.costs[j] for j in units), additional_budget=budget)

    cache = args.baseline_j2_result
    if cache is None and bundled and units == load_design(bundled["reference"], problem):
        cache = bundled["evaluation"]
    baseline_name = "baseline_j2_evaluation" if args.task == "full" else "existing_network_j2_evaluation"
    if cache is not None:
        baseline = cached_evaluation(cache, problem, units, bundled)
        path = save_result(output, baseline_name, baseline)
        summary["stages"][baseline_name] = {"path": str(path), "status_name": baseline["status_name"], "reused": True}
    else:
        baseline = stage(baseline_name, objective="j2", existing=units, fixed_design=units, budget=0,
                         warm_start=load_certificate(source, problem, units), time_limit=args.evaluation_time_limit)
    if not baseline.get("solution_available"):
        return finish("NO_FIXED_LAYOUT_EVALUATION_FOUND", 3)

    fixed_reuse = args.task == "incremental" and budget == 0 and all(cost > 0 for cost in problem.costs)
    name = "full_network_j2" if args.task == "full" else "incremental_j2"
    if fixed_reuse:
        final = copy.deepcopy(baseline)
        final.update(status_name="FIXED_LAYOUT_REUSED", algorithm="Fixed-layout evaluation", budget=0,
                     runtime_seconds=0, wall_seconds=0, source_evaluation_runtime_seconds=baseline["runtime_seconds"],
                     solver_mip_gap=None, solver_objective_bound=None, optimal_to_tolerance=False,
                     existing_cost=sum(problem.costs[j] for j in existing), added_cost=0, unused_budget=0)
        for row in final["chosen_design_units"]:
            row.update(existing=True, new=False)
        path = save_result(output, name, final)
        summary["stages"][name] = {"path": str(path), "status_name": final["status_name"], "reused": True}
    else:
        final = stage(name, objective="j2", budget=budget, existing=existing, warm_start=baseline, time_limit=args.time_limit)
    if not final.get("solution_available"):
        return finish("NO_FEASIBLE_SOLUTION_FOUND", 3)
    summary.update(final_result=final["output_paths"]["json"], baseline_j2=baseline["j2"], final_j2=final["j2"],
                   baseline_j2_evaluation_optimal=baseline["optimal_to_tolerance"],
                   j2_improvement=baseline["j2"]-final["j2"], construction_cost=final["construction_cost"],
                   added_cost=final["added_cost"], unused_budget=final["unused_budget"],
                   covered_demand_share=final["covered_demand_share"],
                   budget_satisfied=final["budget_satisfied"], solver_mip_gap=final["solver_mip_gap"],
                   optimal_to_tolerance=final["optimal_to_tolerance"])
    return finish("FIXED_LAYOUT_REUSED" if fixed_reuse else ("OPTIMAL_TO_TOLERANCE" if final["optimal_to_tolerance"]
                  else "FEASIBLE_HEURISTIC" if solver_name == "alns" else "FEASIBLE_NOT_PROVEN_OPTIMAL"), 0)


def main():
    args = build_parser().parse_args()
    try:
        return run(args)[1]
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
