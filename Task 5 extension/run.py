#!/usr/bin/env python3
"""Two tasks, not two interchangeable budget formats.

full: B_total = multiplier * imported reference construction cost, then optimize J2.
incremental: retain an existing design; absolute budget pays for NEW units only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from network import load_certificate, load_design, load_problem
from solver import solve


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data"


def save_result(directory: Path, name: str, result: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    result["output_paths"] = {"json": str(json_path.resolve())}
    tables = {
        "od_metrics": list(result.get("od_stats", {}).values()),
        "chosen_design_units": result.get("chosen_design_units", []),
        "new_design_units": [row for row in result.get("chosen_design_units", []) if row["new"]],
    }
    for suffix, rows in tables.items():
        if not result.get("solution_available"):
            continue
        path = directory / f"{name}_{suffix}.csv"
        fallback = ["numeric_u", "numeric_v", "original_u", "original_v", "cost", "existing", "new"]
        fields = list(rows[0]) if rows else fallback
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result["output_paths"][suffix] = str(path.resolve())
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 5: optimize distance-weighted safety burden J2 with 80/80 service.")
    subparsers = parser.add_subparsers(dest="task", required=True)
    for name, description in [
        ("full", "Plan the whole network; total budget = multiplier * reference construction cost."),
        ("incremental", "Retain existing lanes; an absolute budget pays for additions only."),
    ]:
        sub = subparsers.add_parser(name, help=description, description=description)
        sub.add_argument("--network", choices=["small", "large"], default="small", help="Dataset size; independent of the planning task.")
        sub.add_argument("--network-dir", type=Path, help="Custom TNTP directory; overrides --network.")
        sub.add_argument("--cost-file", type=Path, help="Optional complete physical-unit cost CSV: numeric_u,numeric_v,cost.")
        sub.add_argument("--output-dir", type=Path)
        sub.add_argument("--coverage-share", type=float, default=0.8)
        sub.add_argument("--epsilon", type=float, default=0.2)
        sub.add_argument("--t", type=float, default=0.2)
        sub.add_argument("--od-limit", type=int, default=0, help="0 = all positive ODs; positive values create a sampled experiment.")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--time-limit", type=float, default=3600, help="Seconds for the final J2 optimization.")
        sub.add_argument("--evaluation-time-limit", type=float, default=300, help="Seconds to evaluate the fixed reference network with J2.")
        sub.add_argument("--mip-gap", type=float, default=1e-4)
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--solver", choices=["auto", "gurobi", "alns"], default="auto",
                         help="auto: Gurobi for the small network, ALNS for the large network.")
        sub.add_argument("--path-pool-size", type=int, default=12)
        sub.add_argument("--path-pool-cache", type=Path)
        sub.add_argument("--alns-iterations", type=int, default=1000000000)
        sub.add_argument("--quiet", action="store_true", help="Suppress Gurobi console output.")
        sub.add_argument("--write-model", action="store_true", help="Also save each model as an LP file.")
        sub.add_argument("--baseline-j2-result", type=Path, help="Reuse a verified fixed-network J2 evaluation with identical parameters and facilities.")
        if name == "full":
            sub.add_argument("--multiplier", type=float, required=True, help="Total budget multiplier; e.g. 1.5 means 150%% of reference construction cost.")
            sub.add_argument("--minimum-cost-result", "--reference-cost-result", dest="minimum_cost_result", type=Path,
                             help="Reuse a compatible cost reference (including an explicitly labeled legacy reference) without a new cost search.")
            sub.add_argument("--baseline-time-limit", type=float, default=3600)
            sub.add_argument("--baseline-mip-gap", type=float, default=1e-6)
            sub.add_argument("--allow-unproven-baseline", action="store_true", help="If the baseline times out, explicitly allow its feasible cost as an upper-bound reference, not a certified minimum.")
        else:
            sub.add_argument("--existing-design", type=Path, required=True, help="Existing Task 4/Task 5 solution JSON or construction-unit CSV.")
            sub.add_argument("--budget", type=float, required=True, help="Absolute ADDITIONAL construction budget; existing cost is excluded.")
    return parser


def run(args: argparse.Namespace) -> tuple[dict, int]:
    if args.task == "full":
        if not math.isfinite(args.multiplier) or args.multiplier <= 0:
            raise ValueError("multiplier must be finite and positive")
    elif not math.isfinite(args.budget) or args.budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    data = args.network_dir or DEFAULT_DATA / args.network
    problem = load_problem(data, coverage_share=args.coverage_share, epsilon=args.epsilon,
                           uncovered_share=args.t, od_limit=args.od_limit, seed=args.seed,
                           cost_file=args.cost_file)
    selected_solver = args.solver
    if selected_solver == "auto":
        selected_solver = "alns" if args.network == "large" or len(problem.nodes) > 800 else "gurobi"
    if selected_solver == "alns" and args.task == "full" and not args.minimum_cost_result:
        raise ValueError("Large-network ALNS requires --reference-cost-result. Reuse the legacy construction-cost reference; no new cost minimization is run.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output_dir or ROOT / "results" / f"{args.task}_{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": args.task, "objective": "J2", "output_dir": str(output),
        "network_dir": str(problem.directory), "problem_fingerprint": problem.problem_fingerprint,
        "network_fingerprint": problem.network_fingerprint,
        "num_ods": len(problem.trips), "od_limit": args.od_limit, "seed": args.seed,
        "coverage_share": args.coverage_share, "epsilon": args.epsilon, "t": args.t,
        "od_sampling_requested": args.od_limit > 0,
        "budget_units": "construction-cost units; by default TNTP length units",
        "equity_constraints": False, "stages": {},
        "algorithm": "ALNS" if selected_solver == "alns" else "Gurobi MILP",
    }

    def stage(name, **kwargs):
        print(f"[{name}] {len(problem.trips)} ODs; building/solving model", flush=True)
        solve_function = solve
        if selected_solver == "alns":
            from alns import solve_alns
            solve_function = solve_alns
            kwargs.update(seed=args.seed, path_pool_size=args.path_pool_size,
                          path_pool_cache=args.path_pool_cache, max_iterations=args.alns_iterations)
        result = solve_function(problem, threads=args.threads, output_flag=int(not args.quiet),
                       log_file=str(output / f"{name}.log"),
                       progress_file=str(output / f"{name}_progress.json"),
                       model_file=str(output / f"{name}.lp") if args.write_model else None,
                       **kwargs)
        path = save_result(output, name, result)
        summary["stages"][name] = {"path": str(path), "status_name": result["status_name"],
                                    "solution_available": result["solution_available"],
                                    "optimal_to_tolerance": result["optimal_to_tolerance"]}
        if result["solution_available"]:
            print(f"[{name}] {result['status_name']}; cost={result['construction_cost']:.6f}; "
                  f"J2={result['j2']:.6f}; coverage={result['covered_demand_share']:.4%}", flush=True)
        else:
            print(f"[{name}] {result['status_name']}; no feasible incumbent", flush=True)
        return result

    def finish(status, code):
        summary["status_name"] = status
        (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Summary: {output / 'run_summary.json'}", flush=True)
        return summary, code

    def cached_baseline(source, units, name):
        result = json.loads(source.expanduser().read_text(encoding="utf-8"))
        if (not result.get("solution_available") or result.get("objective_kind") != "j2"
                or result.get("problem_fingerprint") != problem.problem_fingerprint
                or load_design(source, problem) != units):
            raise ValueError("baseline-j2-result does not match the current fixed network and parameters")
        path = save_result(output, name, result)
        summary["stages"][name] = {
            "path": str(path), "reused_from": str(source.expanduser().resolve()),
            "status_name": result["status_name"], "solution_available": True,
            "optimal_to_tolerance": result["optimal_to_tolerance"],
        }
        return result

    if args.task == "full":
        if args.minimum_cost_result:
            reference = json.loads(args.minimum_cost_result.expanduser().read_text(encoding="utf-8"))
            if (reference.get("objective_kind") != "cost"
                    or reference.get("problem_fingerprint") != problem.problem_fingerprint):
                raise ValueError("minimum-cost-result does not match this network, OD set, costs and service parameters")
            if reference.get("solution_available"):
                imported_units = load_design(args.minimum_cost_result, problem)
                imported_cost = sum(problem.costs[j] for j in imported_units)
                if abs(imported_cost - reference["construction_cost"]) > 1e-6 * max(1, imported_cost):
                    raise ValueError("minimum-cost-result has inconsistent construction cost")
            reference = dict(reference)
            path = save_result(output, "minimum_cost_reference", reference)
            summary["stages"]["minimum_cost_reference"] = {
                "path": str(path), "reused_from": str(args.minimum_cost_result.expanduser().resolve()),
                "status_name": reference["status_name"], "solution_available": reference["solution_available"],
                "optimal_to_tolerance": reference["optimal_to_tolerance"],
            }
        else:
            reference = stage("minimum_cost_reference", objective="cost", time_limit=args.baseline_time_limit,
                              mip_gap=args.baseline_mip_gap)
        if not reference["solution_available"]:
            return finish("NO_FEASIBLE_COST_REFERENCE", 3)
        certified = reference["optimal_to_tolerance"]
        summary.update({"multiplier": args.multiplier,
                        "reference_cost": reference["construction_cost"],
                        "minimum_cost_certified_to_tolerance": certified,
                        "reference_cost_lower_bound": reference["solver_objective_bound"],
                        "reference_cost_mip_gap": reference["solver_mip_gap"]})
        if not certified and not args.allow_unproven_baseline and reference.get("reference_basis") != "legacy_task3_cost":
            summary["message"] = "The cost reference is feasible but not certified minimum. Increase the baseline time limit or explicitly use --allow-unproven-baseline."
            return finish("BASELINE_MINIMUM_NOT_CERTIFIED", 2)
        budget = args.multiplier * reference["construction_cost"]
        summary.update({"total_budget": budget,
                        "budget_basis": reference.get("reference_basis", "minimum_cost" if certified else "unproven_feasible_cost_upper_bound"),
                        "reference_source": reference.get("reference_source")})
        baseline_units = load_design(Path(reference["output_paths"]["json"]), problem)
        if args.baseline_j2_result:
            baseline = cached_baseline(args.baseline_j2_result, baseline_units, "baseline_j2_evaluation")
        else:
            baseline = stage("baseline_j2_evaluation", objective="j2", fixed_design=baseline_units,
                             warm_start=reference, time_limit=args.evaluation_time_limit, mip_gap=args.mip_gap)
        if not baseline["solution_available"]:
            baseline = reference  # The minimum-cost incumbent is already a validated 80/80 certificate.
        final = stage("full_network_j2", objective="j2", budget=budget, warm_start=baseline,
                      time_limit=args.time_limit, mip_gap=args.mip_gap)
    else:
        existing = load_design(args.existing_design, problem)
        summary.update({"existing_design_file": str(args.existing_design.expanduser().resolve()),
                        "existing_cost": sum(problem.costs[j] for j in existing),
                        "additional_budget": args.budget})
        # Certify and evaluate the supplied design under the CURRENT OD set and service parameters.
        if args.baseline_j2_result:
            baseline = cached_baseline(args.baseline_j2_result, existing, "existing_network_j2_evaluation")
        else:
            hint = load_certificate(args.existing_design.expanduser(), problem, existing)
            baseline = stage("existing_network_j2_evaluation", objective="j2", existing=existing,
                             fixed_design=existing, budget=0, warm_start=hint,
                             time_limit=args.evaluation_time_limit, mip_gap=args.mip_gap)
        if not baseline["solution_available"]:
            summary["message"] = "The existing network was not certified to meet the requested coverage/path constraints. Check the network, OD set and parameters. A time limit without a solution is not proof of infeasibility."
            return finish("EXISTING_NETWORK_NOT_CERTIFIED", 3)
        final = stage("incremental_j2", objective="j2", budget=args.budget, existing=existing,
                      warm_start=baseline, time_limit=args.time_limit, mip_gap=args.mip_gap)

    if not final["solution_available"]:
        return finish("INFEASIBLE" if final["status_name"] == "INFEASIBLE" else "NO_FEASIBLE_J2_SOLUTION_FOUND", 3)
    summary.update({
        "final_result": final["output_paths"]["json"],
        "baseline_j2": baseline["j2"], "baseline_j2_evaluation_optimal": baseline["objective_kind"] == "j2" and baseline["optimal_to_tolerance"],
        "final_j2": final["j2"], "j2_improvement": baseline["j2"] - final["j2"],
        "construction_cost": final["construction_cost"], "added_cost": final["added_cost"],
        "unused_budget": final["unused_budget"], "covered_demand_share": final["covered_demand_share"],
        "budget_satisfied": final["budget_satisfied"], "service_80_80_satisfied": final["service_80_80_satisfied"],
        "optimal_to_tolerance": final["optimal_to_tolerance"], "solver_mip_gap": final["solver_mip_gap"],
    })
    return finish("OPTIMAL_TO_TOLERANCE" if final["optimal_to_tolerance"] else
                  ("FEASIBLE_HEURISTIC" if selected_solver == "alns" else "FEASIBLE_NOT_PROVEN_OPTIMAL"), 0)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _, code = run(args)
        return code
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
