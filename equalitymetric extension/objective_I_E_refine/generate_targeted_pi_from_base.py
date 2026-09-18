#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


OD = Tuple[int, int]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SMALL_BASE = PROJECT_ROOT / "baseline_inputs" / "small_task3_baseline_80_80.json"
DEFAULT_LARGE_BASE = PROJECT_ROOT / "baseline_inputs" / "large_task3_baseline_80_80.json"


def od_key(od: OD) -> str:
    return f"{od[0]}->{od[1]}"


def parse_od_key(text: str) -> OD:
    origin, destination = text.split("->", 1)
    return int(origin), int(destination)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def read_base_covered_ods(path: Path) -> Dict[OD, bool]:
    with path.open(encoding="utf-8") as f:
        solution = json.load(f)
    od_stats = solution.get("od_stats")
    if not isinstance(od_stats, dict):
        raise ValueError(f"Base solution has no od_stats object: {path}")

    covered: Dict[OD, bool] = {}
    for key, stats in od_stats.items():
        if not isinstance(stats, dict):
            continue
        coverage_required = bool(stats.get("coverage_required"))
        z_value = float(stats.get("z_value", 0.0) or 0.0)
        covered[parse_od_key(key)] = coverage_required or z_value >= 0.5
    return covered


def generate_for_network(
    *,
    network_name: str,
    source_csv: Path,
    base_solution_json: Path,
    output_csv: Path,
    output_summary_json: Path,
    pi_low: float,
    pi_high: float,
) -> Dict[str, object]:
    covered_map = read_base_covered_ods(base_solution_json)
    rows = []
    total_demand = 0.0
    covered_demand = 0.0
    uncovered_demand = 0.0
    protected_demand = 0.0
    protected_demand_covered = 0.0
    protected_demand_uncovered = 0.0
    covered_od_count = 0
    uncovered_od_count = 0

    with source_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"origin", "destination", "demand"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source_csv} missing required columns: {sorted(missing)}")
        for row in reader:
            od = (int(row["origin"]), int(row["destination"]))
            if od not in covered_map:
                raise ValueError(f"OD {od_key(od)} appears in {source_csv}, but not in {base_solution_json}")
            demand = float(row["demand"])
            base_covered = covered_map[od]
            pi = pi_low if base_covered else pi_high
            q_protected = demand * pi
            q_unprotected = demand * (1.0 - pi)

            total_demand += demand
            protected_demand += q_protected
            if base_covered:
                covered_od_count += 1
                covered_demand += demand
                protected_demand_covered += q_protected
            else:
                uncovered_od_count += 1
                uncovered_demand += demand
                protected_demand_uncovered += q_protected

            rows.append(
                {
                    "origin": int(row["origin"]),
                    "destination": int(row["destination"]),
                    "original_origin": row.get("original_origin", ""),
                    "original_destination": row.get("original_destination", ""),
                    "demand": f"{demand:.6f}",
                    "pi_w": f"{pi:.8f}",
                    "protected_demand": f"{q_protected:.8f}",
                    "unprotected_demand": f"{q_unprotected:.8f}",
                    "base_80_80_covered": int(base_covered),
                    "base_80_80_status": "covered" if base_covered else "uncovered",
                    "generation_method": "targeted_high_pi_for_base_uncovered_low_pi_for_base_covered",
                    "pi_low_for_base_covered": f"{pi_low:.8f}",
                    "pi_high_for_base_uncovered": f"{pi_high:.8f}",
                    "base_solution_file": repo_relative(base_solution_json),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "origin",
        "destination",
        "original_origin",
        "original_destination",
        "demand",
        "pi_w",
        "protected_demand",
        "unprotected_demand",
        "base_80_80_covered",
        "base_80_80_status",
        "generation_method",
        "pi_low_for_base_covered",
        "pi_high_for_base_uncovered",
        "base_solution_file",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "network_name": network_name,
        "source_equity_file": repo_relative(source_csv),
        "base_solution_file": repo_relative(base_solution_json),
        "output_equity_file": repo_relative(output_csv),
        "pi_low_for_base_covered": pi_low,
        "pi_high_for_base_uncovered": pi_high,
        "positive_od_pairs": len(rows),
        "base_covered_od_count": covered_od_count,
        "base_uncovered_od_count": uncovered_od_count,
        "total_demand": total_demand,
        "base_covered_demand": covered_demand,
        "base_uncovered_demand": uncovered_demand,
        "base_covered_demand_share": covered_demand / total_demand if total_demand > 0 else None,
        "base_uncovered_demand_share": uncovered_demand / total_demand if total_demand > 0 else None,
        "total_protected_demand": protected_demand,
        "protected_demand_share": protected_demand / total_demand if total_demand > 0 else None,
        "protected_demand_on_base_covered_ods": protected_demand_covered,
        "protected_demand_on_base_uncovered_ods": protected_demand_uncovered,
        "protected_demand_share_on_base_uncovered_ods": (
            protected_demand_uncovered / protected_demand if protected_demand > 0 else None
        ),
        "note": (
            "This pi_w is intentionally targeted: OD pairs not covered by the original base 80/80 "
            "solution receive high pi_w, and OD pairs covered by the original solution receive low pi_w."
        ),
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate targeted OD protected shares from original base 80/80 coverage status."
    )
    parser.add_argument("--pi-low", type=float, default=0.15, help="pi_w for ODs covered by the original base 80/80 solution.")
    parser.add_argument("--pi-high", type=float, default=0.85, help="pi_w for ODs uncovered by the original base 80/80 solution.")
    parser.add_argument("--small-base-json", default=str(DEFAULT_SMALL_BASE))
    parser.add_argument("--large-base-json", default=str(DEFAULT_LARGE_BASE))
    parser.add_argument("--output-name", default="od_equity_shares_targeted_base_uncovered.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 <= args.pi_low <= 1.0 and 0.0 <= args.pi_high <= 1.0):
        raise ValueError("pi-low and pi-high must be in [0, 1].")
    if args.pi_low >= args.pi_high:
        raise ValueError("pi-low should be lower than pi-high for this targeted scenario.")

    cases = [
        (
            "cropped_subgraph",
            SCRIPT_DIR / "data" / "cropped_subgraph" / "od_equity_shares_synthetic.csv",
            Path(args.small_base_json).expanduser().resolve(),
            SCRIPT_DIR / "data" / "cropped_subgraph" / args.output_name,
            SCRIPT_DIR / "data" / "cropped_subgraph" / "targeted_base_uncovered_pi_summary.json",
        ),
        (
            "full_network",
            SCRIPT_DIR / "data" / "full_network" / "od_equity_shares_synthetic.csv",
            Path(args.large_base_json).expanduser().resolve(),
            SCRIPT_DIR / "data" / "full_network" / args.output_name,
            SCRIPT_DIR / "data" / "full_network" / "targeted_base_uncovered_pi_summary.json",
        ),
    ]
    for network_name, source_csv, base_json, output_csv, output_summary in cases:
        summary = generate_for_network(
            network_name=network_name,
            source_csv=source_csv,
            base_solution_json=base_json,
            output_csv=output_csv,
            output_summary_json=output_summary,
            pi_low=float(args.pi_low),
            pi_high=float(args.pi_high),
        )
        print(
            f"{network_name}: wrote {repo_relative(output_csv)}; "
            f"base uncovered demand share={summary['base_uncovered_demand_share']:.2%}; "
            f"protected demand on base-uncovered ODs="
            f"{summary['protected_demand_share_on_base_uncovered_ods']:.2%}"
        )


if __name__ == "__main__":
    main()
