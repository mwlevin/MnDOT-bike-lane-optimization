#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent

OBJECTIVE_DIR = ROOT / "objective_I_E_refine"
OBJECTIVE_RUNNER = OBJECTIVE_DIR / "run_objective_I_E_refine.py"
OBJECTIVE_GENERATOR = OBJECTIVE_DIR / "generate_targeted_pi_from_base.py"

CONSTRAINT_DIR = ROOT / "constraint_II_D_exposure_disparity"
CONSTRAINT_RUNNER = CONSTRAINT_DIR / "run_constraint_II_D.py"
CONSTRAINT_GENERATOR = CONSTRAINT_DIR / "generate_targeted_pi_from_base.py"

TARGETED_P_FILE = "od_equity_shares_targeted_base_uncovered.csv"
SYNTHETIC_P_FILE = "od_equity_shares_synthetic.csv"


def default_python() -> str:
    return sys.executable


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def nearly_equal(left: Any, right: Any, tol: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def command_record(cmd: List[str]) -> Dict[str, Any]:
    return {
        "argv": cmd,
        "shell_escaped": shlex.join(cmd),
    }


def run_subprocess(
    *,
    cmd: List[str],
    cwd: Path,
    log_dir: Path,
    dry_run: bool,
    env: Dict[str, str],
) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    record: Dict[str, Any] = {
        "command": command_record(cmd),
        "cwd": str(cwd),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
    }

    if dry_run:
        record.update(
            {
                "returncode": None,
                "runtime_seconds": 0.0,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        stdout_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return record

    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            cmd,
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


def selected_p_cases(p_mode: str) -> List[Dict[str, str]]:
    cases = [
        {
            "name": "targeted_base_uncovered_p",
            "equity_file_name": TARGETED_P_FILE,
            "description": "targeted pi_w: high for OD demand uncovered by the original baseline, low for covered OD demand",
        },
        {
            "name": "synthetic_random_p",
            "equity_file_name": SYNTHETIC_P_FILE,
            "description": "older random synthetic pi_w",
        },
    ]
    if p_mode == "targeted":
        return cases[:1]
    if p_mode == "synthetic":
        return cases[1:]
    return cases


def selected_models(model_mode: str) -> List[Dict[str, Any]]:
    models = [
        {
            "name": "objective_I_E_refine",
            "runner": OBJECTIVE_RUNNER,
            "generator": OBJECTIVE_GENERATOR,
            "summary_filename": "objective_I_E_refine_run_summary.json",
            "specific_args": lambda args: [
                "--budget-gamma",
                str(args.budget_gamma),
            ],
        },
        {
            "name": "constraint_II_D_exposure_disparity",
            "runner": CONSTRAINT_RUNNER,
            "generator": CONSTRAINT_GENERATOR,
            "summary_filename": "constraint_II_D_run_summary.json",
            "specific_args": lambda args: [
                "--eta-exposure",
                str(args.eta_exposure),
            ],
        },
    ]
    if model_mode == "objective":
        return models[:1]
    if model_mode == "constraint":
        return models[1:]
    return models


def build_runner_command(
    *,
    python_bin: str,
    runner: Path,
    output_dir: Path,
    equity_file_name: str,
    args: argparse.Namespace,
    model_specific_args: List[str],
) -> List[str]:
    time_limit_seconds = int(round(float(args.time_limit_hours) * 3600.0))
    cmd = [
        python_bin,
        str(runner),
        "--output-dir",
        str(output_dir),
        "--equity-file-name",
        equity_file_name,
        "--small-time-limit",
        str(time_limit_seconds),
        "--large-time-limit",
        str(time_limit_seconds),
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
    ]
    cmd.extend(model_specific_args)
    return cmd


def build_generator_command(python_bin: str, generator: Path) -> List[str]:
    return [python_bin, str(generator)]


def detail_json_paths(summary: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for key in ("small_network", "large_network"):
        section = summary.get(key)
        if isinstance(section, dict) and section.get("json"):
            paths.append(Path(str(section["json"])).expanduser())
    return paths


def load_detail_jsons(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    details = []
    for path in detail_json_paths(summary):
        data = read_json_if_exists(path)
        if isinstance(data, dict):
            details.append(data)
    return details


def equity_basenames(summary: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("small_network", "large_network"):
        section = summary.get(key)
        if isinstance(section, dict) and section.get("equity_file"):
            names.append(Path(str(section["equity_file"])).name)
    for detail in load_detail_jsons(summary):
        if detail.get("equity_file"):
            names.append(Path(str(detail["equity_file"])).name)
    return names


def summary_matches_p(summary: Dict[str, Any], equity_file_name: str) -> bool:
    names = equity_basenames(summary)
    return bool(names) and all(name == equity_file_name for name in names)


def summary_has_full_od_details(summary: Dict[str, Any]) -> bool:
    details = load_detail_jsons(summary)
    if len(details) < 2:
        return False
    return all(int(detail.get("od_limit", -1)) == 0 for detail in details)


def summary_has_reusable_exact_small_solution(summary: Dict[str, Any], model_name: str) -> bool:
    small = summary.get("small_network")
    large = summary.get("large_network")
    if not isinstance(small, dict) or not isinstance(large, dict):
        return False

    small_status = str(small.get("status_name", "")).upper()
    if small_status != "OPTIMAL":
        return False
    if small.get("coverage_requirement_satisfied") is not True:
        return False
    if small.get("service_80_80_satisfied") is not True:
        return False

    large_status = str(large.get("status_name", "")).upper()
    if not large_status or large_status.startswith("NO_FEASIBLE"):
        return False
    if large.get("coverage_requirement_satisfied") is not True:
        return False
    if large.get("service_80_80_satisfied") is not True:
        return False
    if model_name == "constraint_II_D_exposure_disparity" and large.get("exposure_disparity_satisfied") is not True:
        return False
    return all(path.exists() for path in detail_json_paths(summary))


def summary_matches_current_parameters(summary: Dict[str, Any], model_name: str, args: argparse.Namespace) -> bool:
    if not nearly_equal(summary.get("coverage_target_share"), args.coverage_share):
        return False
    if not nearly_equal(summary.get("service_uncovered_cap_t"), args.t):
        return False
    if model_name == "objective_I_E_refine":
        return nearly_equal(summary.get("budget_gamma"), args.budget_gamma)
    if model_name == "constraint_II_D_exposure_disparity":
        return nearly_equal(summary.get("eta_exposure"), args.eta_exposure)
    return False


def is_reusable_exact_case(
    *,
    summary_path: Path,
    model_name: str,
    equity_file_name: str,
    args: argparse.Namespace,
) -> bool:
    summary = read_json_if_exists(summary_path)
    if not isinstance(summary, dict):
        return False
    return (
        summary_has_reusable_exact_small_solution(summary, model_name)
        and summary_has_full_od_details(summary)
        and summary_matches_p(summary, equity_file_name)
        and summary_matches_current_parameters(summary, model_name, args)
    )


def iter_cache_candidates(
    *,
    model: Dict[str, Any],
    p_case: Dict[str, str],
    output_root: Path,
    run_root: Path,
    search_standalone_results: bool,
) -> List[Path]:
    summary_filename = str(model["summary_filename"])
    candidates: List[Path] = []

    if output_root.exists():
        for prior_run in output_root.iterdir():
            if not prior_run.is_dir():
                continue
            try:
                if prior_run.resolve() == run_root.resolve():
                    continue
            except FileNotFoundError:
                continue
            candidates.append(prior_run / str(model["name"]) / p_case["name"] / summary_filename)

    if search_standalone_results:
        model_dir = OBJECTIVE_DIR if model["name"] == "objective_I_E_refine" else CONSTRAINT_DIR
        candidates.extend(model_dir.glob(f"results*/{summary_filename}"))

    existing = [path for path in candidates if path.exists()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return existing


def find_reusable_exact_case(
    *,
    model: Dict[str, Any],
    p_case: Dict[str, str],
    output_root: Path,
    run_root: Path,
    args: argparse.Namespace,
) -> Path | None:
    for summary_path in iter_cache_candidates(
        model=model,
        p_case=p_case,
        output_root=output_root,
        run_root=run_root,
        search_standalone_results=args.search_standalone_results,
    ):
        if is_reusable_exact_case(
            summary_path=summary_path,
            model_name=str(model["name"]),
            equity_file_name=p_case["equity_file_name"],
            args=args,
        ):
            return summary_path.parent
    return None


def copy_reusable_case(
    *,
    source_dir: Path,
    output_dir: Path,
    case_name: str,
    cmd: List[str],
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to copy cached results into non-empty directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)

    reuse_record = {
        "case_name": case_name,
        "copied_from": str(source_dir),
        "copied_to": str(output_dir),
        "skipped_command": command_record(cmd),
        "copied_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "Previous run has a reusable OPTIMAL small-network exact solution and matching full-run outputs.",
    }
    log_dir = output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "stdout.log"
    stderr_log = log_dir / "stderr.log"
    if not stdout_log.exists():
        stdout_log.write_text(
            f"Cache hit for {case_name}; copied results from {source_dir}\n",
            encoding="utf-8",
        )
    if not stderr_log.exists():
        stderr_log.write_text("", encoding="utf-8")
    write_json(output_dir / "cache_reuse.json", reuse_record)
    return reuse_record


def run(args: argparse.Namespace | None = None) -> Dict[str, Any]:
    if args is None:
        args = build_parser().parse_args()

    python_bin = str(Path(args.python).expanduser()) if args.python else default_python()
    run_id = args.run_id or timestamp()
    run_root = Path(args.output_root).expanduser().resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/equalitymetric_pycache")

    models = selected_models(args.model)
    p_cases = selected_p_cases(args.p_mode)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "output_root": str(run_root),
        "python": python_bin,
        "time_limit_hours_per_small_and_large_solve": args.time_limit_hours,
        "large_iterations": args.large_iterations,
        "large_workers": args.large_workers,
        "threads": args.threads,
        "reuse_exact_cache": args.reuse_exact_cache,
        "search_standalone_results": args.search_standalone_results,
        "p_cases": p_cases,
        "models": [model["name"] for model in models],
        "prep": [],
        "cases": [],
    }
    manifest_path = run_root / "run_manifest.json"
    commands_path = run_root / "commands.sh"

    commands: List[str] = []

    if args.p_mode in {"both", "targeted"} and not args.skip_regenerate_targeted_p:
        for model in models:
            generator = Path(model["generator"])
            prep_log_dir = run_root / "_prep" / model["name"] / "generate_targeted_p"
            cmd = build_generator_command(python_bin, generator)
            commands.append(shlex.join(cmd))
            print(f"[prep] regenerate targeted p for {model['name']}")
            prep_record = run_subprocess(cmd=cmd, cwd=ROOT, log_dir=prep_log_dir, dry_run=args.dry_run, env=env)
            prep_record.update({"model": model["name"], "generator": str(generator)})
            manifest["prep"].append(prep_record)
            write_json(manifest_path, manifest)
            if prep_record.get("returncode") not in (0, None):
                manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
                manifest["success"] = False
                write_json(manifest_path, manifest)
                raise RuntimeError(f"Targeted p generation failed for {model['name']}. See {prep_log_dir}")

    for model in models:
        for p_case in p_cases:
            case_name = f"{model['name']}__{p_case['name']}"
            output_dir = run_root / model["name"] / p_case["name"]
            log_dir = output_dir / "_logs"
            summary_path = output_dir / model["summary_filename"]
            cmd = build_runner_command(
                python_bin=python_bin,
                runner=Path(model["runner"]),
                output_dir=output_dir,
                equity_file_name=p_case["equity_file_name"],
                args=args,
                model_specific_args=model["specific_args"](args),
            )
            commands.append(shlex.join(cmd))
            cache_dir = None
            if args.reuse_exact_cache and not args.dry_run:
                cache_dir = find_reusable_exact_case(
                    model=model,
                    p_case=p_case,
                    output_root=Path(args.output_root).expanduser().resolve(),
                    run_root=run_root,
                    args=args,
                )

            if cache_dir is not None:
                print(f"[cache] {case_name} <- {cache_dir}")
                started_at = datetime.now().isoformat(timespec="seconds")
                reuse_record = copy_reusable_case(
                    source_dir=cache_dir,
                    output_dir=output_dir,
                    case_name=case_name,
                    cmd=cmd,
                )
                case_record = {
                    "command": command_record(cmd),
                    "cwd": str(ROOT),
                    "stdout_log": str(log_dir / "stdout.log"),
                    "stderr_log": str(log_dir / "stderr.log"),
                    "started_at": started_at,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "returncode": 0,
                    "runtime_seconds": 0.0,
                    "dry_run": False,
                    "cache_hit": True,
                    "cache_reuse": reuse_record,
                    "case_name": case_name,
                    "model": model["name"],
                    "p_case": p_case["name"],
                    "p_description": p_case["description"],
                    "equity_file_name": p_case["equity_file_name"],
                    "output_dir": str(output_dir),
                    "summary_json": str(summary_path),
                    "summary": read_json_if_exists(summary_path),
                }
            else:
                print(f"[run] {case_name}")
                case_record = run_subprocess(cmd=cmd, cwd=ROOT, log_dir=log_dir, dry_run=args.dry_run, env=env)
                case_record.update(
                    {
                        "case_name": case_name,
                        "model": model["name"],
                        "p_case": p_case["name"],
                        "p_description": p_case["description"],
                        "equity_file_name": p_case["equity_file_name"],
                        "output_dir": str(output_dir),
                        "summary_json": str(summary_path),
                        "summary": None if args.dry_run else read_json_if_exists(summary_path),
                    }
                )
            manifest["cases"].append(case_record)
            write_json(manifest_path, manifest)

            failed = case_record.get("returncode") not in (0, None)
            if failed and args.stop_on_failure:
                manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
                manifest["success"] = False
                write_json(manifest_path, manifest)
                commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
                raise RuntimeError(f"Case failed: {case_name}. See {log_dir}")

    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["success"] = all(case.get("returncode") in (0, None) for case in manifest["cases"])
    write_json(manifest_path, manifest)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote commands: {commands_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Objective I-E refine and Constraint II-D for both targeted and synthetic p scenarios. "
            "Each underlying small and large solve gets a 10-hour time limit by default."
        )
    )
    parser.add_argument("--output-root", default=str(ROOT / "combined_model_runs"))
    parser.add_argument("--run-id", default="", help="Optional output folder name. Defaults to a timestamp.")
    parser.add_argument("--python", default=default_python(), help="Python executable used for the two model runners.")
    parser.add_argument("--model", choices=["all", "objective", "constraint"], default="all")
    parser.add_argument("--p-mode", choices=["both", "targeted", "synthetic"], default="both")
    parser.add_argument("--time-limit-hours", type=float, default=10.0)
    parser.add_argument(
        "--large-iterations",
        type=int,
        default=1_000_000,
        help="Large ALNS iteration cap. High default makes the 10-hour time limit the main stopping rule.",
    )
    parser.add_argument("--coverage-share", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--t", type=float, default=0.2)
    parser.add_argument("--eta-exposure", type=float, default=0.05)
    parser.add_argument("--budget-gamma", type=float, default=1.5)
    parser.add_argument("--mip-gap", type=float, default=1e-2)
    parser.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--large-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--output-flag", type=int, default=1)
    parser.add_argument("--report-interval", type=float, default=300.0)
    parser.add_argument(
        "--no-reuse-exact-cache",
        dest="reuse_exact_cache",
        action="store_false",
        help="Always rerun every case instead of copying a previous reusable exact result.",
    )
    parser.set_defaults(reuse_exact_cache=True)
    parser.add_argument(
        "--no-search-standalone-results",
        dest="search_standalone_results",
        action="store_false",
        help="Only search previous combined_model_runs, not objective/constraint results* folders.",
    )
    parser.set_defaults(search_standalone_results=True)
    parser.add_argument("--skip-regenerate-targeted-p", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write commands and manifest without running solvers.")
    return parser


def main() -> None:
    manifest = run()
    if not manifest.get("success", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
