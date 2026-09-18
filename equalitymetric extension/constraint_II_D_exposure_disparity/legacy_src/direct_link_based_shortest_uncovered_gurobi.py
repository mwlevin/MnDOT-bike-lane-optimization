from __future__ import annotations

import argparse
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB
import networkx as nx

Arc = Tuple[int, int]
OD = Tuple[int, int]


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


class TNTPNetworkData:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self.net_path = self._resolve_file(["berlin-mitte-center_net.tntp", "net.tntp"])
        self.trips_path = self._resolve_file(["berlin-mitte-center_trips.tntp", "trips.tntp"])
        self.node_path = self._resolve_file(["berlin-mitte-center_node.tntp", "node.tntp"], required=False)

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


def extract_path_for_od(data: TNTPNetworkData, x_vars: Dict[Tuple[OD, int], gp.Var], od: OD, tol: float = 1e-6) -> List[int]:
    o, d = od
    succ: Dict[int, int] = {}
    chosen_arcs: List[int] = []
    for a in range(len(data.arcs)):
        val = x_vars[(od, a)].X
        if val > 1 - tol:
            chosen_arcs.append(a)
            u, v = data.idx_to_arc[a]
            succ[u] = a

    path = []
    cur = o
    seen = {o}
    while cur != d:
        if cur not in succ:
            break
        a = succ[cur]
        path.append(a)
        _, nxt = data.idx_to_arc[a]
        if nxt in seen:
            break
        seen.add(nxt)
        cur = nxt
    return path


def build_and_solve_direct_milp(
    input_dir: str,
    epsilon: float = 0.2,
    t: float = 0.2,
    od_limit: int = 100,
    seed: int = 42,
    time_limit: float = 3600.0,
    mip_gap: float = 1e-4,
    output_flag: int = 1,
    threads: Optional[int] = None,
):
    data = TNTPNetworkData(input_dir)
    od_demand = select_ods(data.od_demand, od_limit, seed=seed)
    ods = sorted(od_demand.keys())
    num_arcs = len(data.arcs)
    num_nodes = len(data.nodes)

    K: Dict[OD, float] = {}
    B: Dict[OD, float] = {}
    for od in ods:
        K[od] = shortest_path_length(data, od)
        B[od] = (1.0 + epsilon) * K[od]

    m = gp.Model("direct_link_based_shortest_uncovered")
    m.Params.OutputFlag = int(output_flag)
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)
    if threads is not None:
        m.Params.Threads = int(threads)

    # Design variables
    delta = {a: m.addVar(vtype=GRB.BINARY, name=f"delta[{a}]") for a in range(num_arcs)}

    # OD-specific routing variables
    x = {(od, a): m.addVar(vtype=GRB.BINARY, name=f"x[{od[0]},{od[1]},{a}]") for od in ods for a in range(num_arcs)}

    # Uncovered-length contribution
    c = {(od, a): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"c[{od[0]},{od[1]},{a}]") for od in ods for a in range(num_arcs)}

    # MTZ order variables to enforce simple paths
    r = {(od, i): m.addVar(lb=0.0, ub=num_nodes - 1, vtype=GRB.CONTINUOUS, name=f"r[{od[0]},{od[1]},{i}]")
         for od in ods for i in data.nodes}

    m.setObjective(gp.quicksum(data.arc_cost[a] * delta[a] for a in range(num_arcs)), GRB.MINIMIZE)

    # Flow balance and degree constraints
    for od in ods:
        o, d = od
        for i in data.nodes:
            out_expr = gp.quicksum(x[(od, a)] for a in data.out_arcs.get(i, []))
            in_expr = gp.quicksum(x[(od, a)] for a in data.in_arcs.get(i, []))
            b = 1 if i == o else (-1 if i == d else 0)
            m.addConstr(out_expr - in_expr == b, name=f"flow[{od[0]},{od[1]},{i}]")
            m.addConstr(out_expr <= 1, name=f"outdeg[{od[0]},{od[1]},{i}]")
            m.addConstr(in_expr <= 1, name=f"indeg[{od[0]},{od[1]},{i}]")

        # Path-length bound
        m.addConstr(
            gp.quicksum(data.arc_len[a] * x[(od, a)] for a in range(num_arcs)) <= B[od],
            name=f"detour[{od[0]},{od[1]}]",
        )

        # Uncovered-length cap relative to shortest-path length
        m.addConstr(
            gp.quicksum(c[(od, a)] for a in range(num_arcs)) <= t * K[od],
            name=f"uncovered_cap[{od[0]},{od[1]}]",
        )

        # MTZ origin anchor
        m.addConstr(r[(od, o)] == 0.0, name=f"origin_rank[{od[0]},{od[1]}]")

        # MTZ acyclicity
        for a in range(num_arcs):
            i, j = data.idx_to_arc[a]
            if j == o:
                continue
            m.addConstr(
                r[(od, j)] >= r[(od, i)] + 1 - num_nodes * (1 - x[(od, a)]),
                name=f"mtz[{od[0]},{od[1]},{a}]",
            )

    # Exact linearization of uncovered length contribution c = len * x * (1-delta)
    for od in ods:
        for a in range(num_arcs):
            ell = data.arc_len[a]
            m.addConstr(c[(od, a)] <= ell * x[(od, a)], name=f"c_ub_x[{od[0]},{od[1]},{a}]")
            m.addConstr(c[(od, a)] <= ell * (1 - delta[a]), name=f"c_ub_d[{od[0]},{od[1]},{a}]")
            m.addConstr(c[(od, a)] >= ell * (x[(od, a)] - delta[a]), name=f"c_lb[{od[0]},{od[1]},{a}]")

    start = time.time()
    m.optimize()
    elapsed = time.time() - start

    result = {
        "status": m.Status,
        "status_name": {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status)),
        "runtime": elapsed,
        "objective": math.inf,
        "best_bound": getattr(m, "ObjBound", math.inf),
        "gap": math.inf,
        "num_ods": len(ods),
        "num_arcs": num_arcs,
        "num_nodes": num_nodes,
    }

    if m.SolCount > 0:
        result["objective"] = m.ObjVal
        result["best_bound"] = m.ObjBound
        result["gap"] = abs(m.ObjVal - m.ObjBound) / max(1.0, abs(m.ObjVal))
        result["chosen_links"] = [data.idx_to_arc[a] for a in range(num_arcs) if delta[a].X > 0.5]
        result["x_solution"] = {a: int(round(delta[a].X)) for a in range(num_arcs)}
        result["od_paths"] = {}
        result["od_stats"] = {}
        for od in ods:
            path_arc_idx = extract_path_for_od(data, x, od)
            path_arcs = [data.idx_to_arc[a] for a in path_arc_idx]
            path_len = sum(data.arc_len[a] for a in path_arc_idx)
            uncovered = sum(c[(od, a)].X for a in range(num_arcs))
            result["od_paths"][od] = path_arcs
            result["od_stats"][od] = {
                "path_length": path_len,
                "shortest_length": K[od],
                "detour_cap": B[od],
                "uncovered_length": uncovered,
                "uncovered_cap": t * K[od],
            }

    print("===== Direct link-based MILP summary =====")
    print(f"Status       : {result['status_name']}")
    print(f"Runtime (s)  : {result['runtime']:.2f}")
    if m.SolCount > 0:
        print(f"Incumbent UB : {result['objective']:.6f}")
        print(f"Best bound LB: {result['best_bound']:.6f}")
        print(f"Relative gap : {result['gap']:.6%}")
        print(f"Chosen links : {len(result['chosen_links'])}")
    else:
        print("No feasible incumbent found.")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Directly solve the link-based bicycle-lane MILP with Gurobi: "
            "for each OD, uncovered length must be <= t times the shortest-path length."
        )
    )
    parser.add_argument(
        "--network-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "full_network"),
    )
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--od-limit", type=int, default=0, help="0 means all OD pairs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=36000.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output-flag", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        result = build_and_solve_direct_milp(
            input_dir=args.network_dir,
            epsilon=args.epsilon,
            t=args.t,
            od_limit=args.od_limit,
            seed=args.seed,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=args.output_flag,
            threads=args.threads,
        )
        print(result)
    except gp.GurobiError as exc:
        msg = str(exc)
        if "license is for version 12.0" in msg:
            raise SystemExit(
                "Detected a Gurobi version/license mismatch. Run this script with a Python "
                "environment that has a compatible gurobipy version."
            ) from exc
        raise


if __name__ == "__main__":
    main()
