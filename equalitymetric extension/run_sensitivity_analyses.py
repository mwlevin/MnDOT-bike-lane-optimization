#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent

OBJECTIVE_DIR = ROOT / "objective_I_E_refine"
OBJECTIVE_RUNNER = OBJECTIVE_DIR / "run_objective_I_E_refine.py"
OBJECTIVE_SUMMARY_NAME = "objective_I_E_refine_run_summary.json"

CONSTRAINT_DIR = ROOT / "constraint_II_D_exposure_disparity"
CONSTRAINT_RUNNER = CONSTRAINT_DIR / "run_constraint_II_D.py"
CONSTRAINT_SUMMARY_NAME = "constraint_II_D_run_summary.json"

COMBINED_10H_TARGETED_RUN = ROOT / "combined_model_runs" / "20260726_204835"
OBJECTIVE_DEFAULT_SUMMARY = (
    COMBINED_10H_TARGETED_RUN
    / "objective_I_E_refine"
    / "targeted_base_uncovered_p"
    / OBJECTIVE_SUMMARY_NAME
)
CONSTRAINT_DEFAULT_SUMMARY = (
    COMBINED_10H_TARGETED_RUN
    / "constraint_II_D_exposure_disparity"
    / "targeted_base_uncovered_p"
    / CONSTRAINT_SUMMARY_NAME
)

TARGETED_EQUITY_FILE = "od_equity_shares_targeted_base_uncovered.csv"
DEFAULT_OBJECTIVE_GAMMA = 1.5
DEFAULT_CONSTRAINT_ETA = 0.05

NETWORK_KEYS = {
    "small": "small_network",
    "large": "large_network",
}

CSV_FIELDS = [
    "analysis",
    "model",
    "network",
    "parameter_name",
    "parameter_value",
    "is_default_parameter",
    "source_kind",
    "summary_json",
    "detail_json",
    "status_name",
    "algorithm",
    "budget_gamma",
    "eta_exposure",
    "budget",
    "construction_cost",
    "unused_budget",
    "covered_demand_share",
    "protected_coverage_ratio",
    "unprotected_coverage_ratio",
    "protected_covered_demand",
    "unprotected_covered_demand",
    "total_protected_demand",
    "total_unprotected_demand",
    "protected_uncovered_exposure_burden",
    "unprotected_uncovered_exposure_burden",
    "exposure_burden_gap_Ep_minus_Eu",
    "coverage_requirement_satisfied",
    "exposure_disparity_satisfied",
    "service_80_80_satisfied",
    "num_ods",
    "runtime",
]


def default_python() -> str:
    env_python = Path("/opt/anaconda3/envs/bilk_line/bin/python")
    return str(env_python) if env_python.exists() else sys.executable


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.replace(";", ",").split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def unique_sorted(values: Iterable[float]) -> List[float]:
    ordered: List[float] = []
    for value in values:
        if not any(nearly_equal(value, existing) for existing in ordered):
            ordered.append(float(value))
    return sorted(ordered)


def nearly_equal(left: Any, right: Any, tol: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def value_slug(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def command_record(cmd: Sequence[str]) -> Dict[str, Any]:
    return {
        "argv": list(cmd),
        "shell_escaped": shlex.join(cmd),
    }


def run_subprocess(
    *,
    cmd: Sequence[str],
    cwd: Path,
    log_dir: Path,
    dry_run: bool,
    env: Dict[str, str],
) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    record: Dict[str, Any] = {
        "command": command_record(cmd),
        "cwd": str(cwd),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "started_at": started_at,
        "dry_run": dry_run,
    }

    if dry_run:
        stdout_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        record.update(
            {
                "returncode": None,
                "runtime_seconds": 0.0,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return record

    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        returncode = process.wait()

    record.update(
        {
            "returncode": returncode,
            "runtime_seconds": time.time() - start,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return record


def summary_matches(summary: Dict[str, Any], *, model: str, value: float, equity_file_name: str) -> bool:
    if model == "objective":
        if not nearly_equal(summary.get("budget_gamma"), value):
            return False
    elif model == "constraint":
        if not nearly_equal(summary.get("eta_exposure"), value):
            return False
    else:
        return False

    for network_key in NETWORK_KEYS.values():
        section = summary.get(network_key)
        if isinstance(section, dict) and section.get("equity_file"):
            if Path(str(section["equity_file"])).name != equity_file_name:
                return False
    return True


def build_common_runner_args(args: argparse.Namespace) -> List[str]:
    cmd = [
        "--equity-file-name",
        args.equity_file_name,
        "--small-time-limit",
        str(args.small_time_limit),
        "--large-time-limit",
        str(args.large_time_limit),
        "--small-od-limit",
        str(args.small_od_limit),
        "--large-od-limit",
        str(args.large_od_limit),
        "--large-iterations",
        str(args.large_iterations),
        "--coverage-share",
        str(args.coverage_share),
        "--epsilon",
        str(args.epsilon),
        "--t",
        str(args.t),
        "--mip-gap",
        str(args.mip_gap),
        "--threads",
        str(args.threads),
        "--large-workers",
        str(args.large_workers),
        "--output-flag",
        str(args.output_flag),
        "--report-interval",
        str(args.report_interval),
        "--path-pool-size",
        str(args.path_pool_size),
        "--path-penalty-growth",
        str(args.path_penalty_growth),
        "--path-random-noise",
        str(args.path_random_noise),
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--max-paths-per-od-scoring",
        str(args.max_paths_per_od_scoring),
        "--initial-max-additions",
        str(args.initial_max_additions),
        "--repair-max-additions",
        str(args.repair_max_additions),
        "--destroy-fraction-min",
        str(args.destroy_fraction_min),
        "--destroy-fraction-max",
        str(args.destroy_fraction_max),
        "--low-impact-sample-size",
        str(args.low_impact_sample_size),
        "--initial-temperature",
        str(args.initial_temperature),
        "--cooling-rate",
        str(args.cooling_rate),
    ]
    if args.disable_balanced_od_sampling:
        cmd.append("--disable-balanced-od-sampling")
    return cmd


def objective_command(args: argparse.Namespace, output_dir: Path, gamma: float) -> List[str]:
    return [
        args.python,
        str(OBJECTIVE_RUNNER),
        "--output-dir",
        str(output_dir),
        *build_common_runner_args(args),
        "--budget-gamma",
        str(gamma),
    ]


def constraint_command(args: argparse.Namespace, output_dir: Path, eta: float) -> List[str]:
    return [
        args.python,
        str(CONSTRAINT_RUNNER),
        "--output-dir",
        str(output_dir),
        *build_common_runner_args(args),
        "--eta-exposure",
        str(eta),
        "--prune-passes",
        str(args.prune_passes),
    ]


def detail_path_from_section(section: Dict[str, Any], summary_path: Path) -> Path | None:
    raw_path = section.get("json")
    if not raw_path:
        return None
    recorded_path = Path(str(raw_path)).expanduser()
    if recorded_path.name:
        for network_dir in ("small_exact_gurobi", "large_alns"):
            local_path = summary_path.parent / network_dir / recorded_path.name
            if local_path.exists():
                return local_path
    return recorded_path


def compute_coverage_from_od_stats(detail: Dict[str, Any], demand_key: str) -> tuple[float | None, float | None]:
    od_stats = detail.get("od_stats")
    if not isinstance(od_stats, dict):
        return None, None
    total = 0.0
    covered = 0.0
    seen = False
    for item in od_stats.values():
        if not isinstance(item, dict):
            continue
        demand = finite_float(item.get(demand_key))
        if demand is None:
            continue
        seen = True
        total += demand
        if bool(item.get("covered")):
            covered += demand
    if not seen or total <= 0.0:
        return None, None
    return covered, covered / total


def metric(detail: Dict[str, Any], section: Dict[str, Any], key: str) -> Any:
    if key in detail:
        return detail[key]
    return section.get(key, "")


def build_rows_for_summary(
    *,
    summary_path: Path,
    summary: Dict[str, Any] | None,
    analysis: str,
    model_label: str,
    parameter_name: str,
    parameter_value: float,
    default_value: float,
    source_kind: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if summary is None:
        for network in NETWORK_KEYS:
            rows.append(
                {
                    "analysis": analysis,
                    "model": model_label,
                    "network": network,
                    "parameter_name": parameter_name,
                    "parameter_value": parameter_value,
                    "is_default_parameter": nearly_equal(parameter_value, default_value),
                    "source_kind": "missing",
                    "summary_json": str(summary_path),
                    "status_name": "MISSING_SUMMARY",
                }
            )
        return rows

    for network, summary_key in NETWORK_KEYS.items():
        section = summary.get(summary_key, {})
        if not isinstance(section, dict):
            section = {}
        detail_path = detail_path_from_section(section, summary_path)
        detail = read_json(detail_path) if detail_path is not None else None
        if detail is None:
            detail = {}

        protected_covered = metric(detail, section, "protected_covered_demand")
        protected_ratio = metric(detail, section, "protected_coverage_ratio")
        if protected_covered == "" or protected_ratio == "":
            computed_covered, computed_ratio = compute_coverage_from_od_stats(detail, "protected_demand")
            protected_covered = computed_covered if protected_covered == "" else protected_covered
            protected_ratio = computed_ratio if protected_ratio == "" else protected_ratio

        unprotected_covered = metric(detail, section, "unprotected_covered_demand")
        unprotected_ratio = metric(detail, section, "unprotected_coverage_ratio")
        if unprotected_covered == "" or unprotected_ratio == "":
            computed_covered, computed_ratio = compute_coverage_from_od_stats(detail, "unprotected_demand")
            unprotected_covered = computed_covered if unprotected_covered == "" else unprotected_covered
            unprotected_ratio = computed_ratio if unprotected_ratio == "" else unprotected_ratio

        rows.append(
            {
                "analysis": analysis,
                "model": model_label,
                "network": network,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "is_default_parameter": nearly_equal(parameter_value, default_value),
                "source_kind": source_kind,
                "summary_json": str(summary_path),
                "detail_json": str(detail_path) if detail_path is not None else "",
                "status_name": metric(detail, section, "status_name"),
                "algorithm": metric(detail, section, "algorithm"),
                "budget_gamma": parameter_value if parameter_name == "budget_gamma" else "",
                "eta_exposure": parameter_value if parameter_name == "eta_exposure" else "",
                "budget": metric(detail, section, "budget"),
                "construction_cost": metric(detail, section, "construction_cost"),
                "unused_budget": metric(detail, section, "unused_budget"),
                "covered_demand_share": metric(detail, section, "covered_demand_share"),
                "protected_coverage_ratio": protected_ratio,
                "unprotected_coverage_ratio": unprotected_ratio,
                "protected_covered_demand": protected_covered,
                "unprotected_covered_demand": unprotected_covered,
                "total_protected_demand": metric(detail, section, "total_protected_demand"),
                "total_unprotected_demand": metric(detail, section, "total_unprotected_demand"),
                "protected_uncovered_exposure_burden": metric(detail, section, "protected_uncovered_exposure_burden"),
                "unprotected_uncovered_exposure_burden": metric(detail, section, "unprotected_uncovered_exposure_burden"),
                "exposure_burden_gap_Ep_minus_Eu": metric(detail, section, "exposure_burden_gap_Ep_minus_Eu"),
                "coverage_requirement_satisfied": metric(detail, section, "coverage_requirement_satisfied"),
                "exposure_disparity_satisfied": metric(detail, section, "exposure_disparity_satisfied"),
                "service_80_80_satisfied": metric(detail, section, "service_80_80_satisfied"),
                "num_ods": metric(detail, section, "num_ods"),
                "runtime": metric(detail, section, "runtime"),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_line(
    *,
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    title: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local optional package
        return f"Plot skipped because matplotlib is unavailable: {exc}"

    series: Dict[str, List[tuple[float, float]]] = {"small": [], "large": []}
    for row in rows:
        network = str(row.get("network", ""))
        x = finite_float(row.get(x_key))
        y = finite_float(row.get(y_key))
        if network in series and x is not None and y is not None:
            series[network].append((x, y))

    if all(len(points) < 2 for points in series.values()):
        return f"Plot skipped because {output_path.name} has fewer than two usable points."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=180)
    labels = {"small": "Small network", "large": "Expanded network"}
    colors = {"small": "#2A6FBB", "large": "#B64D2A"}
    for network, points in series.items():
        if not points:
            continue
        points = sorted(points)
        ax.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=labels[network],
            color=colors[network],
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return None


def make_plots(output_dir: Path, objective_rows: Sequence[Dict[str, Any]], constraint_rows: Sequence[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    objective_note = plot_line(
        rows=objective_rows,
        output_path=output_dir / "figures" / "objective_budget_sensitivity_protected_coverage.png",
        title="Objective Equity Optimization Budget Sensitivity",
        x_key="parameter_value",
        y_key="protected_coverage_ratio",
        x_label="Budget multiplier over Task 3 baseline cost",
        y_label="Protected demand coverage ratio",
    )
    if objective_note:
        notes.append(objective_note)

    constraint_note = plot_line(
        rows=constraint_rows,
        output_path=output_dir / "figures" / "constraint_gap_sensitivity_cost.png",
        title="Constraint Equity Optimization Gap Sensitivity",
        x_key="parameter_value",
        y_key="construction_cost",
        x_label="Permitted exposure gap",
        y_label="Construction cost",
    )
    if constraint_note:
        notes.append(constraint_note)
    return notes


def case_summary_path(case: Dict[str, Any]) -> Path:
    if case["source_kind"] == "default_existing":
        return Path(case["summary_path"])
    return Path(case["output_dir"]) / str(case["summary_name"])


def run_case_if_needed(
    *,
    case: Dict[str, Any],
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    manifest_path: Path,
    commands: List[str],
    env: Dict[str, str],
) -> None:
    command = case["command"]
    summary_path = case_summary_path(case)
    log_dir = Path(case["output_dir"]) / "_logs" if case.get("output_dir") else manifest_path.parent / "_logs" / case["name"]

    if case["source_kind"] == "default_existing":
        commands.append(f"# skipped default_existing {case['name']}: {summary_path}")
        print(f"[default] {case['name']} uses {summary_path}")
        case["run_record"] = {
            "returncode": 0 if summary_path.exists() else 1,
            "runtime_seconds": 0.0,
            "skipped_solver": True,
            "reason": "Default 150% budget / 5% exposure-gap case is reused from the configured existing summary.",
        }
        manifest["cases"].append(case)
        write_json(manifest_path, manifest)
        return

    commands.append(shlex.join(command))

    existing = read_json(summary_path)
    if args.reuse_existing and existing is not None and summary_matches(
        existing,
        model=str(case["model_key"]),
        value=float(case["parameter_value"]),
        equity_file_name=args.equity_file_name,
    ):
        print(f"[reuse] {case['name']} uses existing {summary_path}")
        case["run_record"] = {
            "returncode": 0,
            "runtime_seconds": 0.0,
            "skipped_solver": True,
            "reason": "Matching sensitivity result already exists in this output directory.",
        }
        manifest["cases"].append(case)
        write_json(manifest_path, manifest)
        return

    if args.collect_only:
        print(f"[collect-only] {case['name']} expects {summary_path}")
        case["run_record"] = {
            "returncode": 0 if summary_path.exists() else 1,
            "runtime_seconds": 0.0,
            "skipped_solver": True,
            "reason": "collect-only mode does not run solvers.",
        }
        manifest["cases"].append(case)
        write_json(manifest_path, manifest)
        return

    print(f"[run] {case['name']}")
    run_record = run_subprocess(cmd=command, cwd=ROOT, log_dir=log_dir, dry_run=args.dry_run, env=env)
    case["run_record"] = run_record
    manifest["cases"].append(case)
    write_json(manifest_path, manifest)

    failed = run_record.get("returncode") not in (0, None)
    if failed and args.stop_on_failure:
        raise RuntimeError(f"Sensitivity case failed: {case['name']}. See {log_dir}")


def build_cases(args: argparse.Namespace, run_root: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    gammas = unique_sorted(args.budget_gammas)
    etas = unique_sorted(args.eta_values)

    if args.model in {"all", "objective"}:
        for gamma in gammas:
            is_default = nearly_equal(gamma, DEFAULT_OBJECTIVE_GAMMA)
            source_kind = "solver_run"
            output_dir = run_root / "objective_budget_sensitivity" / f"gamma_{value_slug(gamma)}"
            summary_path = output_dir / OBJECTIVE_SUMMARY_NAME
            if is_default and not args.rerun_defaults:
                source_kind = "default_existing"
                summary_path = args.objective_default_summary
            command = objective_command(args, output_dir, gamma)
            cases.append(
                {
                    "name": f"objective_gamma_{value_slug(gamma)}",
                    "analysis": "objective_budget_sensitivity",
                    "model": "Objective Equity Optimization",
                    "model_key": "objective",
                    "parameter_name": "budget_gamma",
                    "parameter_value": gamma,
                    "default_value": DEFAULT_OBJECTIVE_GAMMA,
                    "source_kind": source_kind,
                    "runner": str(OBJECTIVE_RUNNER),
                    "summary_name": OBJECTIVE_SUMMARY_NAME,
                    "summary_path": str(summary_path),
                    "output_dir": str(output_dir),
                    "command": command,
                }
            )

    if args.model in {"all", "constraint"}:
        for eta in etas:
            is_default = nearly_equal(eta, DEFAULT_CONSTRAINT_ETA)
            source_kind = "solver_run"
            output_dir = run_root / "constraint_gap_sensitivity" / f"eta_{value_slug(eta)}"
            summary_path = output_dir / CONSTRAINT_SUMMARY_NAME
            if is_default and not args.rerun_defaults:
                source_kind = "default_existing"
                summary_path = args.constraint_default_summary
            command = constraint_command(args, output_dir, eta)
            cases.append(
                {
                    "name": f"constraint_eta_{value_slug(eta)}",
                    "analysis": "constraint_gap_sensitivity",
                    "model": "Constraint Equity Optimization",
                    "model_key": "constraint",
                    "parameter_name": "eta_exposure",
                    "parameter_value": eta,
                    "default_value": DEFAULT_CONSTRAINT_ETA,
                    "source_kind": source_kind,
                    "runner": str(CONSTRAINT_RUNNER),
                    "summary_name": CONSTRAINT_SUMMARY_NAME,
                    "summary_path": str(summary_path),
                    "output_dir": str(output_dir),
                    "command": command,
                }
            )
    return cases


def collect_rows(cases: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    objective_rows: List[Dict[str, Any]] = []
    constraint_rows: List[Dict[str, Any]] = []
    for case in cases:
        summary_path = case_summary_path(case)
        summary = read_json(summary_path)
        rows = build_rows_for_summary(
            summary_path=summary_path,
            summary=summary,
            analysis=str(case["analysis"]),
            model_label=str(case["model"]),
            parameter_name=str(case["parameter_name"]),
            parameter_value=float(case["parameter_value"]),
            default_value=float(case["default_value"]),
            source_kind=str(case["source_kind"]),
        )
        all_rows.extend(rows)
        if case["model_key"] == "objective":
            objective_rows.extend(rows)
        elif case["model_key"] == "constraint":
            constraint_rows.extend(rows)
    return all_rows, objective_rows, constraint_rows


def run(args: argparse.Namespace | None = None) -> Dict[str, Any]:
    if args is None:
        args = build_parser().parse_args()

    args.python = str(Path(args.python).expanduser())
    args.objective_default_summary = Path(args.objective_default_summary).expanduser().resolve()
    args.constraint_default_summary = Path(args.constraint_default_summary).expanduser().resolve()
    args.budget_gammas = parse_float_list(args.budget_gammas)
    args.eta_values = parse_float_list(args.eta_values)

    run_id = args.run_id or timestamp()
    run_root = Path(args.output_root).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/equalitymetric_pycache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/equalitymetric_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/equalitymetric_xdg")
    env = os.environ.copy()

    cases = build_cases(args, run_root)
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "output_root": str(run_root),
        "python": args.python,
        "equity_file_name": args.equity_file_name,
        "objective_budget_gammas": unique_sorted(args.budget_gammas),
        "constraint_eta_values": unique_sorted(args.eta_values),
        "default_objective_gamma": DEFAULT_OBJECTIVE_GAMMA,
        "default_constraint_eta": DEFAULT_CONSTRAINT_ETA,
        "rerun_defaults": args.rerun_defaults,
        "collect_only": args.collect_only,
        "dry_run": args.dry_run,
        "small_od_limit": args.small_od_limit,
        "large_od_limit": args.large_od_limit,
        "small_time_limit": args.small_time_limit,
        "large_time_limit": args.large_time_limit,
        "large_iterations": args.large_iterations,
        "large_workers": args.large_workers,
        "cases": [],
    }
    manifest_path = run_root / "sensitivity_manifest.json"
    commands_path = run_root / "commands.sh"
    write_json(manifest_path, manifest)

    commands: List[str] = []
    for case in cases:
        run_case_if_needed(
            case=case,
            args=args,
            manifest=manifest,
            manifest_path=manifest_path,
            commands=commands,
            env=env,
        )
        commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")

    all_rows, objective_rows, constraint_rows = collect_rows(cases)
    write_csv(run_root / "sensitivity_summary.csv", all_rows)
    write_csv(run_root / "objective_budget_sensitivity.csv", objective_rows)
    write_csv(run_root / "constraint_gap_sensitivity.csv", constraint_rows)

    plot_notes: List[str] = []
    if not args.no_plots:
        plot_notes = make_plots(run_root, objective_rows, constraint_rows)

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["success"] = all(
        case.get("run_record", {}).get("returncode") in (0, None) for case in manifest["cases"]
    )
    manifest["output_files"] = {
        "manifest": str(manifest_path),
        "commands": str(commands_path),
        "all_rows_csv": str(run_root / "sensitivity_summary.csv"),
        "objective_rows_csv": str(run_root / "objective_budget_sensitivity.csv"),
        "constraint_rows_csv": str(run_root / "constraint_gap_sensitivity.csv"),
        "objective_plot": str(run_root / "figures" / "objective_budget_sensitivity_protected_coverage.png"),
        "constraint_plot": str(run_root / "figures" / "constraint_gap_sensitivity_cost.png"),
    }
    manifest["plot_notes"] = plot_notes
    write_json(manifest_path, manifest)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote all sensitivity rows: {run_root / 'sensitivity_summary.csv'}")
    print(f"Wrote Objective budget table: {run_root / 'objective_budget_sensitivity.csv'}")
    print(f"Wrote Constraint gap table: {run_root / 'constraint_gap_sensitivity.csv'}")
    if not args.no_plots:
        print(f"Wrote figures under: {run_root / 'figures'}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task 4 sensitivity analyses for both network sizes. Objective Equity Optimization varies "
            "the extra budget; Constraint Equity Optimization varies the permitted exposure gap."
        )
    )
    parser.add_argument("--output-root", default=str(ROOT / "sensitivity_runs"))
    parser.add_argument("--run-id", default="", help="Optional output folder name. Defaults to a timestamp.")
    parser.add_argument("--python", default=default_python(), help="Python executable used to launch model runners.")
    parser.add_argument("--model", choices=["all", "objective", "constraint"], default="all")
    parser.add_argument(
        "--budget-gammas",
        default="1.2,1.5,1.8",
        help="Comma-separated budget multipliers for Objective Equity Optimization. Default: 1.2,1.5,1.8.",
    )
    parser.add_argument(
        "--eta-values",
        default="0,0.05,0.10",
        help="Comma-separated permitted exposure gaps for Constraint Equity Optimization. Default: 0,0.05,0.10.",
    )
    parser.add_argument(
        "--rerun-defaults",
        action="store_true",
        help="Rerun the default 1.5 Objective budget and 0.05 Constraint gap cases instead of reusing existing results.",
    )
    parser.add_argument(
        "--objective-default-summary",
        default=str(OBJECTIVE_DEFAULT_SUMMARY),
        help="Existing Objective default summary used for budget_gamma=1.5.",
    )
    parser.add_argument(
        "--constraint-default-summary",
        default=str(CONSTRAINT_DEFAULT_SUMMARY),
        help="Existing Constraint default summary used for eta_exposure=0.05.",
    )
    parser.add_argument("--equity-file-name", default=TARGETED_EQUITY_FILE)

    parser.add_argument("--coverage-share", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--small-od-limit", type=int, default=0, help="0 means all small-network ODs.")
    parser.add_argument("--large-od-limit", type=int, default=0, help="0 means all large-network ODs.")
    parser.add_argument("--small-time-limit", type=float, default=36000.0)
    parser.add_argument("--large-time-limit", type=float, default=36000.0)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--large-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--report-interval", type=float, default=300.0)
    parser.add_argument("--disable-balanced-od-sampling", action="store_true")

    parser.add_argument("--large-iterations", type=int, default=1_000_000)
    parser.add_argument("--path-pool-size", type=int, default=10)
    parser.add_argument("--path-penalty-growth", type=float, default=0.35)
    parser.add_argument("--path-random-noise", type=float, default=0.12)
    parser.add_argument("--candidate-top-k", type=int, default=25)
    parser.add_argument("--max-paths-per-od-scoring", type=int, default=4)
    parser.add_argument("--initial-max-additions", type=int, default=300)
    parser.add_argument("--repair-max-additions", type=int, default=35)
    parser.add_argument("--prune-passes", type=int, default=2)
    parser.add_argument("--destroy-fraction-min", type=float, default=0.04)
    parser.add_argument("--destroy-fraction-max", type=float, default=0.18)
    parser.add_argument("--low-impact-sample-size", type=int, default=60)
    parser.add_argument("--initial-temperature", type=float, default=25.0)
    parser.add_argument("--cooling-rate", type=float, default=0.996)

    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/JSON outputs.")
    parser.add_argument("--collect-only", action="store_true", help="Do not run solvers; aggregate existing summaries only.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands and manifest without running solvers.")
    parser.add_argument(
        "--no-reuse-existing",
        dest="reuse_existing",
        action="store_false",
        help="Rerun non-default cases even if matching summaries already exist in the selected run folder.",
    )
    parser.set_defaults(reuse_existing=True)
    parser.add_argument(
        "--no-stop-on-failure",
        dest="stop_on_failure",
        action="store_false",
        help="Continue collecting other cases when a solver subprocess fails.",
    )
    parser.set_defaults(stop_on_failure=True)
    return parser


def main() -> None:
    manifest = run()
    if not manifest.get("success", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
