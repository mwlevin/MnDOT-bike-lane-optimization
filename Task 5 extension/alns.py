"""Budget-constrained J2 ALNS, adapted from the Task 3/4 ALNS workflow.

References in this workspace:
  MinneapolisCBD copy 2/alns_link_based_undirected_design.py
  objective_I_E_refine/run_objective_I_E_refine.py

Uses diversified detour-feasible paths, bundle/single-unit repair, adaptive
destroy/repair selection and simulated annealing. All demand is weighted by
q*K*U, not by protected shares. No Gurobi import or invocation occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import heapq
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from scipy.sparse import csr_matrix

from network import Problem


@dataclass
class State:
    chosen: frozenset[int]
    cost: float
    charged: float
    j2: float
    covered_demand: float
    uncovered: np.ndarray
    best_paths: np.ndarray
    burden: np.ndarray
    covered: np.ndarray


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def shortest_route(problem, trip, rng=None, penalties=None):
    outgoing = {}
    for a in trip.arcs:
        arc = problem.arcs[a]
        multiplier = 1 if rng is None else 1 + penalties.get(a, 0) + rng.random() * .12
        outgoing.setdefault(arc.u, []).append((arc.v, a, arc.length * multiplier))
    distances = {trip.origin: 0.0}
    predecessor, queue = {}, [(0.0, trip.origin)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == trip.destination:
            break
        for v, a, weight in outgoing.get(node, []):
            value = distance + weight
            if value < distances.get(v, math.inf):
                distances[v], predecessor[v] = value, (node, a)
                heapq.heappush(queue, (value, v))
    if trip.destination not in distances:
        return None
    path, node = [], trip.destination
    while node != trip.origin:
        previous, a = predecessor[node]
        path.append(a)
        node = previous
    return tuple(reversed(path))


def build_paths(problem, warm_start, *, count=12, seed=42, cache=None):
    if count < 1:
        raise ValueError("path-pool-size must be positive")
    warm_paths = (warm_start or {}).get("od_paths", {})
    identity = {"problem": problem.problem_fingerprint, "count": count, "seed": seed,
                "warm_paths": warm_paths}
    # JSON comparison keeps path caches deterministic and prevents mixing ODs.
    identity_text = json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def generate():
        rng, all_paths = random.Random(seed), []
        lookup = {(a.u, a.v): index for index, a in enumerate(problem.arcs)}
        for index, trip in enumerate(problem.trips):
            paths, seen = [], set()
            def add(route):
                if route is None:
                    return
                route = tuple(route)
                if not route or route in seen:
                    return
                nodes, node = {trip.origin}, trip.origin
                for a in route:
                    arc = problem.arcs[a]
                    if arc.u != node or arc.v in nodes:
                        return
                    nodes.add(arc.v)
                    node = arc.v
                if node != trip.destination or sum(problem.arcs[a].length for a in route) > (1 + problem.epsilon) * trip.shortest + 1e-7:
                    return
                seen.add(route)
                paths.append(route)
            add(shortest_route(problem, trip))
            key = f"{trip.origin}->{trip.destination}"
            try:
                rows = warm_paths.get(key, [])
                route = [lookup[(int(r["numeric_u"]), int(r["numeric_v"]))] if isinstance(r, dict)
                         else lookup[tuple(r)] for r in rows]
                add(route)
            except KeyError:
                pass
            penalties = {}
            for route in paths:
                for a in route:
                    penalties[a] = penalties.get(a, 0) + .35
            for _ in range(max(25, count * 8)):
                if len(paths) >= count:
                    break
                route = shortest_route(problem, trip, rng, penalties)
                if route:
                    for a in route:
                        penalties[a] = penalties.get(a, 0) + .35 * rng.uniform(.65, 1.35)
                    add(route)
            if not paths:
                raise ValueError(f"No candidate route for {key}")
            all_paths.append([list(route) for route in sorted(paths, key=lambda p: sum(problem.arcs[a].length for a in p))])
            if (index + 1) % 100 == 0:
                print(f"[ALNS paths] {index + 1}/{len(problem.trips)} ODs", flush=True)
        return all_paths

    if not cache:
        return generate()
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(cache.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache.exists():
            document = json.loads(cache.read_text())
            if document.get("identity") == identity_text:
                return document["paths"]
        paths = generate()
        atomic_json(cache, {"identity": identity_text, "paths": paths})
        return paths


class Evaluator:
    def __init__(self, problem, paths, locked):
        self.problem, self.locked = problem, frozenset(locked)
        self.paths, self.path_units, starts, owners, row, column, values = [], [], [], [], [], [], []
        for w, routes in enumerate(paths):
            starts.append(len(self.paths))
            for route in routes:
                p = len(self.paths)
                totals = {}
                for a in route:
                    arc = problem.arcs[a]
                    totals[arc.unit] = totals.get(arc.unit, 0) + arc.length
                for j, length in totals.items():
                    row.append(p); column.append(j); values.append(length)
                self.paths.append(tuple(route))
                self.path_units.append(totals)
                owners.append(w)
        self.starts = np.asarray(starts, dtype=int)
        self.ends = np.r_[self.starts[1:], len(self.paths)]
        self.owners = np.asarray(owners, dtype=int)
        self.lengths = np.array([sum(problem.arcs[a].length for a in path) for path in self.paths])
        self.costs = np.asarray(problem.costs)
        self.K = np.array([w.shortest for w in problem.trips])
        self.q = np.array([w.demand for w in problem.trips])
        self.caps = problem.uncovered_share * self.K
        self.matrix = csr_matrix((values, (row, column)), shape=(len(self.paths), len(problem.units)))
        self.locked_cost = sum(problem.costs[j] for j in locked)

    def __call__(self, chosen):
        chosen = frozenset(chosen)
        if not self.locked <= chosen:
            raise ValueError("ALNS attempted to remove existing facilities")
        vector = np.zeros(len(self.problem.units))
        if chosen:
            vector[list(chosen)] = 1
        uncovered = np.maximum(0, self.lengths - self.matrix @ vector)
        usable = uncovered <= self.caps[self.owners] + 1e-7
        candidate = np.where(usable, uncovered, np.inf)
        minimum = np.minimum.reduceat(candidate, self.starts)
        covered = np.isfinite(minimum)
        burden = np.where(covered, minimum, self.K)
        index = np.arange(len(self.paths))
        eligible = candidate == minimum[self.owners]
        best = np.minimum.reduceat(np.where(eligible, index, len(self.paths)), self.starts)
        # Unserved paths are diagnostics only; they never get zero burden.
        best = np.where(covered, best, self.starts)
        # Avoid platform BLAS dot products for these small reductions; use the
        # selected physical units directly for reproducible budget accounting.
        cost = sum(self.problem.costs[j] for j in chosen)
        return State(chosen, cost, cost - self.locked_cost, float(np.sum(self.q * burden)),
                     float(np.sum(self.q[covered])), uncovered, best, burden, covered)

    def key(self, state):
        return (state.j2, state.cost)


def candidates(evaluator, state, budget, *, max_paths=3):
    remaining, candidates_by_units, unit_scores = budget - state.charged, {}, {}
    for w, (start, end) in enumerate(zip(evaluator.starts, evaluator.ends)):
        if state.burden[w] <= 1e-8:
            continue
        order = sorted(range(start, end), key=lambda p: state.uncovered[p])[:max_paths]
        for p in order:
            uncovered = state.uncovered[p]
            missing = [(j, length, evaluator.costs[j]) for j, length in evaluator.path_units[p].items()
                       if j not in state.chosen]
            if not missing:
                continue
            if uncovered <= evaluator.caps[w] + 1e-7:
                for j, length, cost in missing:
                    if cost > remaining + 1e-7:
                        continue
                    benefit = evaluator.q[w] * max(0, state.burden[w] - max(0, uncovered - length))
                    unit_scores[j] = unit_scores.get(j, 0) + benefit
            else:
                selected, cost, reduction = [], 0.0, 0.0
                for j, length, value in sorted(missing, key=lambda r: r[1] / max(r[2], 1e-12), reverse=True):
                    if cost + value > remaining + 1e-7:
                        continue
                    selected.append(j); cost += value; reduction += length
                    if uncovered - reduction <= evaluator.caps[w] + 1e-7:
                        break
                if selected and uncovered - reduction <= evaluator.caps[w] + 1e-7:
                    benefit = evaluator.q[w] * max(0, state.burden[w] - max(0, uncovered - reduction))
                    units = tuple(sorted(selected))
                    candidates_by_units[units] = max(candidates_by_units.get(units, 0), benefit / max(cost, 1e-12))
    for unit, score in unit_scores.items():
        if score > 1e-8:
            candidates_by_units[(unit,)] = score / max(evaluator.costs[unit], 1e-12)
    return sorted(((score, units) for units, score in candidates_by_units.items() if score > 1e-8), reverse=True)


def repair(evaluator, state, budget, rng, deadline, *, randomized=False, max_additions=40, top_k=12):
    current = state
    for _ in range(max_additions):
        if time.monotonic() >= deadline:
            break
        ranked = candidates(evaluator, current, budget)
        if not ranked:
            break
        shortlist = ranked[:top_k]
        if randomized:
            chosen = rng.choices(shortlist, weights=[max(item[0], 1e-10) for item in shortlist], k=1)[0]
            shortlist = [chosen] + shortlist[:3]
        trial_best = None
        for _, units in shortlist:
            trial = evaluator(current.chosen | frozenset(units))
            if trial.charged > budget + 1e-6:
                continue
            if trial_best is None or evaluator.key(trial) < evaluator.key(trial_best):
                trial_best = trial
        if trial_best is None or evaluator.key(trial_best) >= evaluator.key(current):
            break
        current = trial_best
    return current


def destroy(evaluator, state, name, fraction, rng):
    mutable = sorted(state.chosen - evaluator.locked)
    if not mutable:
        return state.chosen
    count = min(len(mutable), max(1, round(len(mutable) * fraction)))
    if name == "random":
        removed = rng.sample(mutable, count)
    elif name == "costly":
        removed = sorted(mutable, key=lambda j: evaluator.costs[j], reverse=True)[:count]
    elif name == "path":
        path = evaluator.path_units[rng.choice(state.best_paths)]
        removed = list(set(path) & set(mutable))
        if not removed:
            removed = rng.sample(mutable, count)
        removed = removed[:max(count, 1)]
    else:
        sample = rng.sample(mutable, min(len(mutable), max(count, 24)))
        scores = []
        for j in sample:
            trial = evaluator(state.chosen - {j})
            loss = max(0, trial.j2 - state.j2)
            scores.append((loss / max(evaluator.costs[j], 1e-12), j))
        removed = [j for _, j in sorted(scores)[:count]]
    return state.chosen - frozenset(removed)


def result_from_state(problem, evaluator, state, *, budget, started, iterations, seed, stop_reason,
                      destroy_weights, repair_weights):
    valid = state is not None and (budget is None or state.charged <= budget + 1e-6)
    result = {"model_name": "Task 5 J2 ALNS", "algorithm": "ALNS", "objective_kind": "j2",
              "status_name": "FEASIBLE_HEURISTIC" if valid else "NO_FEASIBLE_HEURISTIC_SOLUTION",
              "solution_available": bool(valid), "optimal_to_tolerance": False,
              "solver_mip_gap": None, "solver_objective_bound": None,
              "runtime_seconds": time.monotonic() - started, "network_dir": str(problem.directory),
              "network_fingerprint": problem.network_fingerprint, "problem_fingerprint": problem.problem_fingerprint,
              "num_ods": len(problem.trips), "num_nodes": len(problem.nodes), "num_arcs": len(problem.arcs),
              "total_demand": problem.total_demand, "coverage_target_share": problem.coverage_share,
              "aggregate_coverage_constraint": False,
              "epsilon": problem.epsilon, "t": problem.uncovered_share, "budget": budget,
              "budget_counts": "new_construction_only", "seed": seed, "iterations": iterations,
              "stop_reason": stop_reason, "candidate_path_count": len(evaluator.paths),
              "candidate_route_scope": "detour-feasible diversified paths, including baseline certificate routes",
              "destroy_weights": destroy_weights, "repair_weights": repair_weights}
    if not valid:
        return result
    rows, routes = {}, {}
    for w, trip in enumerate(problem.trips):
        key = f"{trip.origin}->{trip.destination}"
        covered = bool(state.covered[w])
        path = evaluator.paths[int(state.best_paths[w])] if covered else ()
        length = sum(problem.arcs[a].length for a in path)
        uncovered = sum(problem.arcs[a].length for a in path if problem.arcs[a].unit not in state.chosen)
        if covered and (length > (1 + problem.epsilon) * trip.shortest + 1e-5 or uncovered > problem.uncovered_share * trip.shortest + 1e-5):
            raise RuntimeError("Independent ALNS route validation failed")
        burden = uncovered if covered else trip.shortest
        rows[key] = {"od": key, "origin": trip.origin, "destination": trip.destination, "demand": trip.demand,
                     "covered": covered, "z_value": int(covered), "shortest_length": trip.shortest,
                     "path_length": length, "detour_cap": (1 + problem.epsilon) * trip.shortest,
                     "model_uncovered_length": uncovered, "uncovered_cap": problem.uncovered_share * trip.shortest,
                     "noncoverage_distance_penalty": 0 if covered else trip.shortest,
                     "distance_weighted_burden": burden, "j2_contribution": trip.demand * burden,
                     "u_w": burden / trip.shortest}
        routes[key] = [[problem.arcs[a].u, problem.arcs[a].v] for a in path]
    j2 = sum(row["j2_contribution"] for row in rows.values())
    if abs(j2 - state.j2) > 1e-5 * max(1, j2):
        raise RuntimeError("Independent ALNS J2 validation failed")
    result.update({"j2": j2, "solver_objective_value": j2,
                   "construction_cost": state.cost, "existing_cost": evaluator.locked_cost,
                   "added_cost": state.charged, "unused_budget": None if budget is None else budget - state.charged,
                   "covered_demand": state.covered_demand, "covered_demand_share": state.covered_demand / problem.total_demand,
                   "covered_od_count": int(sum(state.covered)), "budget_satisfied": True,
                   "coverage_requirement_satisfied": None, "all_route_limits_satisfied": True,
                   "existing_design_retained": evaluator.locked <= state.chosen,
                   "mean_distance_weighted_burden": j2 / problem.total_demand,
                   "j1_diagnostic": sum(row["demand"] * row["u_w"] for row in rows.values()),
                   "chosen_design_units": [{"numeric_u": problem.units[j][0], "numeric_v": problem.units[j][1],
                                            "original_u": problem.originals[problem.units[j][0]], "original_v": problem.originals[problem.units[j][1]],
                                            "cost": problem.costs[j], "existing": j in evaluator.locked, "new": j not in evaluator.locked}
                                           for j in sorted(state.chosen)],
                   "chosen_links": [{"numeric_u": a.u, "numeric_v": a.v, "original_u": problem.originals[a.u],
                                     "original_v": problem.originals[a.v]} for a in problem.arcs if a.unit in state.chosen],
                   "od_stats": rows, "od_paths": routes})
    return result


def solve_alns(problem, *, objective="j2", budget=None, existing=None, fixed_design=None,
               warm_start=None, time_limit=18000, seed=42, path_pool_size=12, path_pool_cache=None,
               max_iterations=1000000000, progress_file=None, log_file=None,
               path_pool_warm_start=None, **unused):
    if objective != "j2":
        raise ValueError("Large-network ALNS only optimizes J2; import the old construction-cost reference instead")
    if not math.isfinite(time_limit) or time_limit <= 0 or (budget is not None and (not math.isfinite(budget) or budget < 0)):
        raise ValueError("Invalid ALNS time limit or budget")
    started, rng = time.monotonic(), random.Random(seed)
    deadline = started + time_limit
    if progress_file:
        atomic_json(progress_file, {"stage": "building_alns_path_pool", "algorithm": "ALNS", "num_ods": len(problem.trips)})
    paths = build_paths(problem, path_pool_warm_start if path_pool_warm_start is not None else warm_start,
                        count=path_pool_size, seed=seed, cache=path_pool_cache)
    locked = frozenset(existing or ())
    evaluator = Evaluator(problem, paths, locked)
    if fixed_design is not None:
        initial = frozenset(fixed_design)
    else:
        pairs = {tuple(sorted((row["numeric_u"], row["numeric_v"]))) for row in (warm_start or {}).get("chosen_design_units", [])}
        initial = locked | frozenset(j for j, pair in enumerate(problem.units) if pair in pairs)
    current = evaluator(initial)
    cap = float(sum(problem.costs)) if budget is None else budget
    if current.charged > cap + 1e-6:
        current = evaluator(locked)
    if fixed_design is None:
        current = repair(evaluator, current, cap, rng, deadline, max_additions=200)
    best = current if current.charged <= cap + 1e-6 else None
    destroy_weights = dict.fromkeys(["random", "costly", "low_impact", "path"], 1.0)
    repair_weights = dict.fromkeys(["greedy", "randomized"], 1.0)
    scores = {"d": {k: [0.0, 0] for k in destroy_weights}, "r": {k: [0.0, 0] for k in repair_weights}}
    temperature = max(1, .01 * current.j2)
    last_output, iteration, stop_reason = -math.inf, 0, "time_limit"

    def output(force=False):
        nonlocal last_output
        if not force and time.monotonic() - last_output < 30:
            return
        last_output = time.monotonic()
        progress = {"stage": "optimizing", "algorithm": "ALNS", "wall_seconds": time.monotonic() - started,
                    "iterations": iteration, "best_objective": None if best is None else best.j2,
                    "covered_demand_share": None if best is None else best.covered_demand / problem.total_demand,
                    "charged_cost": None if best is None else best.charged, "budget": budget,
                    "relative_gap": None, "destroy_weights": destroy_weights, "repair_weights": repair_weights}
        line = f"[ALNS] iter={iteration} best_J2={progress['best_objective']} coverage={progress['covered_demand_share']} charged={progress['charged_cost']}"
        print(line, flush=True)
        if log_file:
            with Path(log_file).open("a", encoding="utf-8") as log:
                log.write(line + "\n")
        if progress_file:
            atomic_json(progress_file, progress)
            if best is not None:
                checkpoint = result_from_state(problem, evaluator, best, budget=budget, started=started,
                                               iterations=iteration, seed=seed, stop_reason="checkpoint",
                                               destroy_weights=destroy_weights, repair_weights=repair_weights)
                atomic_json(Path(progress_file).parent / "alns_best_checkpoint.json", checkpoint)
    try:
        output(True)
        while fixed_design is None and iteration < max_iterations and time.monotonic() < deadline:
            if best is not None and best.j2 <= 1e-8:
                stop_reason = "zero_burden"
                break
            if cap <= 1e-10 and locked:
                stop_reason = "no_additional_budget"
                break
            iteration += 1
            d = rng.choices(list(destroy_weights), weights=list(destroy_weights.values()))[0]
            r = rng.choices(list(repair_weights), weights=list(repair_weights.values()))[0]
            destroyed = evaluator(destroy(evaluator, current, d, rng.uniform(.04, .22), rng))
            trial = repair(evaluator, destroyed, cap, rng, deadline, randomized=r == "randomized")
            feasible = trial.charged <= cap + 1e-6
            reward = 0
            if feasible and (best is None or trial.j2 < best.j2 - 1e-7
                             or (abs(trial.j2 - best.j2) <= 1e-7 and trial.cost < best.cost - 1e-7)):
                best, reward = trial, 8
            delta = trial.j2 - current.j2
            accepted = feasible and (delta <= 1e-8
                                     or rng.random() < math.exp(-max(0, delta) / max(temperature, 1e-12)))
            if accepted:
                reward = max(reward, 4 if delta < -1e-7 else 1)
                current = trial
            for group, key in [("d", d), ("r", r)]:
                scores[group][key][0] += reward
                scores[group][key][1] += 1
            temperature *= .9995
            if iteration % 50 == 0:
                for group, weights in [("d", destroy_weights), ("r", repair_weights)]:
                    for key in weights:
                        total, uses = scores[group][key]
                        weights[key] = max(.05, .8 * weights[key] + .2 * total / max(1, uses))
                        scores[group][key] = [0.0, 0]
            if iteration % 250 == 0 and best is not None:
                current = best
                temperature = max(temperature, .003 * best.j2)
            output()
        else:
            if fixed_design is not None:
                stop_reason = "fixed_network_evaluation"
            elif iteration >= max_iterations:
                stop_reason = "iteration_limit"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    result = result_from_state(problem, evaluator, best, budget=budget, started=started, iterations=iteration,
                               seed=seed, stop_reason=stop_reason, destroy_weights=destroy_weights, repair_weights=repair_weights)
    output(True)
    if progress_file:
        atomic_json(progress_file, {"stage": "finished", "algorithm": "ALNS", "status_name": result["status_name"],
                                   "wall_seconds": time.monotonic() - started, "iterations": iteration,
                                   "best_objective": result.get("j2"), "stop_reason": stop_reason, "relative_gap": None})
    return result
