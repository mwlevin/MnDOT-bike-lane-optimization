"""Budget-constrained Gurobi model without aggregate coverage requirements.

J2 = sum_w q_w * (K_w * (1-z_w) + sum_a c_wa).
There are no demographic weights, equity files, or disparity constraints.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
import time

from network import Problem


def solve(problem: Problem, *, objective: str = "j2", budget: float | None = None,
          existing: set[int] | None = None, fixed_design: set[int] | None = None,
          warm_start: dict | None = None, time_limit: float = 3600,
          mip_gap: float = 1e-4, threads: int = 4, output_flag: int = 1,
          log_file: str | None = None, model_file: str | None = None,
          progress_file: str | None = None) -> dict:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise RuntimeError("gurobipy is required. Use the bilk_line Python environment or install requirements.txt") from error
    if objective not in {"cost", "j2"}:
        raise ValueError("objective must be cost or j2")
    if budget is not None and (not math.isfinite(budget) or budget < 0):
        raise ValueError("Budget must be finite and nonnegative")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time-limit must be finite and positive")
    if not math.isfinite(mip_gap) or not 0 <= mip_gap < 1:
        raise ValueError("mip-gap must be in [0, 1)")
    if threads < 1:
        raise ValueError("threads must be positive")
    existing = set(existing or ())
    if not existing <= set(range(len(problem.units))):
        raise ValueError("Invalid existing construction unit")
    if fixed_design is not None:
        fixed_design = set(fixed_design)
        if not existing <= fixed_design:
            raise ValueError("fixed_design must retain all existing facilities")

    started = time.monotonic()
    def write_progress(payload):
        if progress_file:
            path = Path(progress_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
    write_progress({"stage": "building_model", "objective_kind": objective,
                    "num_ods": len(problem.trips), "wall_seconds": 0})
    model = gp.Model("task5_j2_" + objective)
    try:
        model.Params.OutputFlag = int(output_flag)
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = mip_gap
        model.Params.Threads = threads
        if log_file:
            model.Params.LogFile = log_file
        active = {problem.arcs[a].unit for w in problem.trips for a in w.arcs} | existing
        if fixed_design is not None:
            active |= fixed_design
        y = {}
        for j in sorted(active):
            lower = 1 if j in existing else 0
            upper = 1
            if fixed_design is not None:
                lower = upper = int(j in fixed_design)
            y[j] = model.addVar(lb=lower, ub=upper, vtype=GRB.BINARY, name=f"y[{j}]")
        z, x, c = {}, {}, {}
        for w, trip in enumerate(problem.trips):
            z[w] = model.addVar(vtype=GRB.BINARY, name=f"z[{w}]")
            for a in trip.arcs:
                x[w, a] = model.addVar(vtype=GRB.BINARY, name=f"x[{w},{a}]")
                c[w, a] = model.addVar(lb=0, name=f"c[{w},{a}]")

        total_cost = gp.quicksum(problem.costs[j] * y[j] for j in active)
        # Existing infrastructure is sunk cost in incremental mode.
        new_cost = gp.quicksum(problem.costs[j] * y[j] for j in active - existing)
        j2 = gp.quicksum(trip.demand * (
            trip.shortest * (1 - z[w]) + gp.quicksum(c[w, a] for a in trip.arcs)
        ) for w, trip in enumerate(problem.trips))
        if budget is not None:
            model.addConstr(new_cost <= budget, name="construction_budget")

        for w, trip in enumerate(problem.trips):
            outgoing: dict[int, list] = {}
            incoming: dict[int, list] = {}
            for a in trip.arcs:
                arc = problem.arcs[a]
                outgoing.setdefault(arc.u, []).append(x[w, a])
                incoming.setdefault(arc.v, []).append(x[w, a])
                model.addConstr(c[w, a] <= arc.length * x[w, a])
                model.addConstr(c[w, a] <= arc.length * (1 - y[arc.unit]))
                model.addConstr(c[w, a] >= arc.length * (x[w, a] - y[arc.unit]))
            nodes = set(outgoing) | set(incoming) | {trip.origin, trip.destination}
            for node in nodes:
                outflow = gp.quicksum(outgoing.get(node, []))
                inflow = gp.quicksum(incoming.get(node, []))
                balance = int(node == trip.origin) - int(node == trip.destination)
                model.addConstr(outflow - inflow == balance * z[w])
                model.addConstr(outflow <= z[w])
                model.addConstr(inflow <= z[w])
            model.addConstr(gp.quicksum(incoming.get(trip.origin, [])) == 0)
            model.addConstr(gp.quicksum(outgoing.get(trip.destination, [])) == 0)
            model.addConstr(gp.quicksum(problem.arcs[a].length * x[w, a] for a in trip.arcs)
                            <= (1 + problem.epsilon) * trip.shortest * z[w], name=f"detour[{w}]")
            model.addConstr(gp.quicksum(c[w, a] for a in trip.arcs)
                            <= problem.uncovered_share * trip.shortest * z[w], name=f"uncovered_cap[{w}]")
            # MTZ order variables exclude both attached and disconnected cycles.
            size = len(nodes)
            order = {node: model.addVar(lb=0, ub=size - 1, name=f"order[{w},{node}]") for node in nodes}
            model.addConstr(order[trip.origin] == 0)
            for a in trip.arcs:
                arc = problem.arcs[a]
                model.addConstr(order[arc.v] >= order[arc.u] + 1 - size * (1 - x[w, a]))

        model.setObjective(total_cost if objective == "cost" else j2, GRB.MINIMIZE)
        if warm_start and warm_start.get("solution_available"):
            pairs = {tuple(sorted((row["numeric_u"], row["numeric_v"])))
                     for row in warm_start["chosen_design_units"]}
            for j in active:
                y[j].Start = int(problem.units[j] in pairs)
            stats = warm_start.get("od_stats", {})
            paths = warm_start.get("od_paths", {})
            for w, trip in enumerate(problem.trips):
                key = f"{trip.origin}->{trip.destination}"
                if key not in stats:
                    continue
                z[w].Start = int(stats[key]["covered"])
                path_pairs = {tuple(pair) for pair in paths.get(key, [])}
                for a in trip.arcs:
                    arc = problem.arcs[a]
                    used = int((arc.u, arc.v) in path_pairs)
                    x[w, a].Start = used
                    c[w, a].Start = arc.length * used * int(problem.units[arc.unit] not in pairs)
        model.update()
        if model_file:
            model.write(model_file)
        last_update = [-math.inf]
        def progress_callback(callback_model, where):
            if where != GRB.Callback.MIP:
                return
            elapsed = time.monotonic() - started
            if elapsed - last_update[0] < 30:
                return
            last_update[0] = elapsed
            best = callback_model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = callback_model.cbGet(GRB.Callback.MIP_OBJBND)
            best = best if math.isfinite(best) and abs(best) < 1e90 else None
            bound = bound if math.isfinite(bound) and abs(bound) < 1e90 else None
            write_progress({"stage": "optimizing", "objective_kind": objective,
                            "wall_seconds": elapsed, "solver_seconds": callback_model.cbGet(GRB.Callback.RUNTIME),
                            "best_objective": best, "objective_bound": bound,
                            "relative_gap": None if best is None or bound is None else abs(best - bound) / max(abs(best), 1e-10),
                            "node_count": callback_model.cbGet(GRB.Callback.MIP_NODCNT),
                            "solution_count": callback_model.cbGet(GRB.Callback.MIP_SOLCNT)})
        model.optimize(progress_callback if progress_file else None)

        def attribute(name):
            try:
                value = getattr(model, name)
                return value if not isinstance(value, float) or math.isfinite(value) else None
            except (AttributeError, gp.GurobiError):
                return None

        status = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
                  GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT",
                  GRB.INTERRUPTED: "INTERRUPTED", GRB.SUBOPTIMAL: "SUBOPTIMAL"}.get(model.Status, str(model.Status))
        result = {
            "model_name": "Task 5 distance-weighted exposure J2",
            "objective_kind": objective,
            "objective_formula": "sum_w q_w * (K_w * (1-z_w) + sum_a c_wa)",
            "status_name": status, "solution_available": model.SolCount > 0,
            "optimal_to_tolerance": model.Status == GRB.OPTIMAL,
            "requested_mip_gap": mip_gap,
            "solver_objective_value": attribute("ObjVal"),
            "solver_objective_bound": attribute("ObjBound"), "solver_mip_gap": attribute("MIPGap"),
            "runtime_seconds": model.Runtime, "wall_seconds": time.monotonic() - started,
            "network_dir": str(problem.directory), "network_fingerprint": problem.network_fingerprint,
            "problem_fingerprint": problem.problem_fingerprint,
            "coverage_target_share": problem.coverage_share,
            "aggregate_coverage_constraint": False,
            "epsilon": problem.epsilon, "t": problem.uncovered_share,
            "budget": budget, "budget_counts": "new_construction_only",
            "num_nodes": len(problem.nodes), "num_arcs": len(problem.arcs),
            "num_design_units": len(problem.units), "num_ods": len(problem.trips),
            "num_variables": model.NumVars, "num_constraints": model.NumConstrs,
            "total_demand": problem.total_demand,
        }
        write_progress({"stage": "finished", "status_name": status,
                        "wall_seconds": result["wall_seconds"], "solver_seconds": model.Runtime,
                        "best_objective": result["solver_objective_value"],
                        "objective_bound": result["solver_objective_bound"],
                        "relative_gap": result["solver_mip_gap"], "solution_available": result["solution_available"]})
        if not result["solution_available"]:
            return result

        chosen = {j for j in active if y[j].X > 0.5}
        rows, paths = {}, {}
        for w, trip in enumerate(problem.trips):
            key = f"{trip.origin}->{trip.destination}"
            covered = z[w].X > 0.5
            selected = {a for a in trip.arcs if x[w, a].X > 0.5}
            path = []
            if covered:
                successor = {problem.arcs[a].u: a for a in selected}
                node, visited = trip.origin, {trip.origin}
                while node != trip.destination:
                    if node not in successor:
                        raise RuntimeError(f"Incomplete certification path for {key}")
                    a = successor[node]
                    path.append(a)
                    node = problem.arcs[a].v
                    if node in visited:
                        raise RuntimeError(f"Cyclic certification path for {key}")
                    visited.add(node)
                if set(path) != selected:
                    raise RuntimeError(f"Disconnected selected path arcs for {key}")
            elif selected:
                raise RuntimeError(f"Unserved OD {key} has nonzero path variables")
            length = sum(problem.arcs[a].length for a in path)
            uncovered = sum(problem.arcs[a].length for a in path if problem.arcs[a].unit not in chosen)
            penalty = 0.0 if covered else trip.shortest
            burden = penalty + uncovered
            rows[key] = {
                "od": key, "origin": trip.origin, "destination": trip.destination,
                "demand": trip.demand, "covered": covered, "z_value": int(covered),
                "shortest_length": trip.shortest, "path_length": length,
                "detour_cap": (1 + problem.epsilon) * trip.shortest,
                "model_uncovered_length": uncovered,
                "uncovered_cap": problem.uncovered_share * trip.shortest,
                "noncoverage_distance_penalty": penalty,
                "distance_weighted_burden": burden,
                "j2_contribution": trip.demand * burden, "u_w": burden / trip.shortest,
            }
            paths[key] = [[problem.arcs[a].u, problem.arcs[a].v] for a in path]
        construction_cost = sum(problem.costs[j] for j in chosen)
        existing_cost = sum(problem.costs[j] for j in existing)
        added_cost = sum(problem.costs[j] for j in chosen - existing)
        exposure = sum(row["j2_contribution"] for row in rows.values())
        covered_demand = sum(row["demand"] for row in rows.values() if row["covered"])
        checks = {
            "budget_satisfied": budget is None or added_cost <= budget + 1e-5 * max(1, budget),
            "existing_design_retained": existing <= chosen,
            "fixed_design_satisfied": fixed_design is None or fixed_design == chosen,
            "all_route_limits_satisfied": all(not row["covered"] or (
                row["path_length"] <= row["detour_cap"] + 1e-5
                and row["model_uncovered_length"] <= row["uncovered_cap"] + 1e-5
            ) for row in rows.values()),
            "objective_recomputed_matches_solver": abs((construction_cost if objective == "cost" else exposure)
                                                        - model.ObjVal) <= 1e-5 * max(1, abs(model.ObjVal)),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Independent solution validation failed: {checks}")
        result.update(checks)
        result["coverage_requirement_satisfied"] = None
        result.update({
            "j2": exposure,
            "mean_distance_weighted_burden": exposure / problem.total_demand,
            "j1_diagnostic": sum(row["demand"] * row["u_w"] for row in rows.values()),
            "construction_cost": construction_cost, "existing_cost": existing_cost,
            "added_cost": added_cost, "unused_budget": None if budget is None else budget - added_cost,
            "covered_demand": covered_demand, "covered_demand_share": covered_demand / problem.total_demand,
            "covered_od_count": sum(row["covered"] for row in rows.values()),
            "chosen_design_units": [{
                "numeric_u": problem.units[j][0], "numeric_v": problem.units[j][1],
                "original_u": problem.originals[problem.units[j][0]],
                "original_v": problem.originals[problem.units[j][1]],
                "cost": problem.costs[j], "existing": j in existing, "new": j not in existing,
            } for j in sorted(chosen)],
            "chosen_links": [{"numeric_u": arc.u, "numeric_v": arc.v,
                              "original_u": problem.originals[arc.u], "original_v": problem.originals[arc.v]}
                             for arc in problem.arcs if arc.unit in chosen],
            "od_stats": rows, "od_paths": paths,
        })
        return result
    finally:
        model.dispose()
