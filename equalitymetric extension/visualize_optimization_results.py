#!/usr/bin/env python3
"""Visualize objective-equity and constraint-equity optimization results.

The script reads result JSON files produced by run_all_models_both_p.py and
draws selected design units over the TNTP network geometry. It creates
single-model maps and pairwise comparison maps for the two equity models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


Edge = Tuple[int, int]
Point = Tuple[float, float]

DEFAULT_MANIFEST = Path("combined_model_runs/20260726_204835/run_manifest.json")
DEFAULT_OUTPUT_DIR = Path("report_results/visualizations")
DEFAULT_P_CASE = "targeted_base_uncovered_p"

OBJECTIVE_MODEL = "objective_I_E_refine"
CONSTRAINT_MODEL = "constraint_II_D_exposure_disparity"

MODEL_LABELS = {
    OBJECTIVE_MODEL: "Objective Equity Optimization Model",
    CONSTRAINT_MODEL: "Constraint Equity Optimization Model",
}

MODEL_COLORS = {
    OBJECTIVE_MODEL: "#2563eb",
    CONSTRAINT_MODEL: "#dc2626",
}

P_CASE_LABELS = {
    "targeted_base_uncovered_p": "Targeted Scenario",
    "synthetic_random_p": "Random Synthetic Scenario",
}

NETWORK_LABELS = {
    "small": "Small Network",
    "large": "Expanded Network",
}


@dataclass(frozen=True)
class NetworkData:
    network_dir: Path
    nodes: Dict[int, Point]
    arcs: Set[Edge]


@dataclass(frozen=True)
class ResultCase:
    model: str
    p_case: str
    network_size: str
    result_path: Path
    result: Mapping[str, object]
    chosen_edges: Set[Edge]
    od_endpoints: Set[int]
    network_dir: Path

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PNG and HTML maps for the two equity optimization results."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"run_all_models_both_p.py manifest JSON. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--p-case",
        default=DEFAULT_P_CASE,
        help="Equity p scenario to visualize, or 'all'. Default: targeted_base_uncovered_p",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated maps. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--no-png", action="store_true", help="Skip PNG output.")
    parser.add_argument("--no-html", action="store_true", help="Skip self-contained HTML output.")
    return parser.parse_args()


def read_json(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "result"


def undirected_edge(u: object, v: object) -> Edge:
    u_i = int(u)
    v_i = int(v)
    if u_i == v_i:
        raise ValueError(f"Self-loop edge is not expected: {u_i}")
    return (u_i, v_i) if u_i < v_i else (v_i, u_i)


def edge_from_record(record: object) -> Optional[Edge]:
    try:
        if isinstance(record, Mapping):
            if "numeric_u" in record and "numeric_v" in record:
                return undirected_edge(record["numeric_u"], record["numeric_v"])
            if "u" in record and "v" in record:
                return undirected_edge(record["u"], record["v"])
        if isinstance(record, Sequence) and not isinstance(record, (str, bytes)) and len(record) >= 2:
            return undirected_edge(record[0], record[1])
    except (TypeError, ValueError):
        return None
    return None


def chosen_edge_set(result: Mapping[str, object]) -> Set[Edge]:
    records = result.get("chosen_design_units") or result.get("chosen_links") or []
    edges: Set[Edge] = set()
    if isinstance(records, Iterable):
        for record in records:
            edge = edge_from_record(record)
            if edge is not None:
                edges.add(edge)
    return edges


def od_endpoint_set(result: Mapping[str, object]) -> Set[int]:
    od_stats = result.get("od_stats") or {}
    if isinstance(od_stats, Mapping):
        rows = od_stats.values()
    elif isinstance(od_stats, Sequence) and not isinstance(od_stats, (str, bytes)):
        rows = od_stats
    else:
        rows = []

    endpoints: Set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("origin", "destination"):
            try:
                endpoints.add(int(row[key]))
            except (KeyError, TypeError, ValueError):
                continue
    return endpoints


def read_nodes(network_dir: Path) -> Dict[int, Point]:
    node_path = network_dir / "node.tntp"
    nodes: Dict[int, Point] = {}
    with node_path.open(encoding="utf-8") as f:
        for raw in f:
            tokens = raw.replace(";", " ").split()
            if len(tokens) < 3 or not tokens[0].lstrip("-").isdigit():
                continue
            node = int(tokens[0])
            nodes[node] = (float(tokens[1]), float(tokens[2]))
    if not nodes:
        raise ValueError(f"No nodes parsed from {node_path}")
    return nodes


def read_arcs(network_dir: Path) -> Set[Edge]:
    net_path = network_dir / "net.tntp"
    arcs: Set[Edge] = set()
    with net_path.open(encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("<") or stripped.startswith("~"):
                continue
            tokens = stripped.replace(";", " ").split()
            if len(tokens) < 2:
                continue
            try:
                arcs.add(undirected_edge(tokens[0], tokens[1]))
            except (TypeError, ValueError):
                continue
    if not arcs:
        raise ValueError(f"No arcs parsed from {net_path}")
    return arcs


def read_network(network_dir: Path) -> NetworkData:
    return NetworkData(network_dir=network_dir, nodes=read_nodes(network_dir), arcs=read_arcs(network_dir))


def manifest_cases(manifest_path: Path, p_case: str) -> List[ResultCase]:
    manifest = read_json(manifest_path)
    raw_cases = manifest.get("cases", [])
    if not isinstance(raw_cases, Sequence):
        raise ValueError(f"Manifest does not contain a cases list: {manifest_path}")

    cases: List[ResultCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue
        case_p = str(raw_case.get("p_case") or raw_case.get("p_scenario") or "")
        if p_case != "all" and case_p != p_case:
            continue

        model = str(raw_case.get("model") or "")
        summary = raw_case.get("summary")
        if not isinstance(summary, Mapping):
            summary_json = raw_case.get("summary_json")
            if summary_json:
                summary = read_json(Path(str(summary_json)))
        if not isinstance(summary, Mapping):
            continue

        for summary_key, network_size in (("small_network", "small"), ("large_network", "large")):
            network_summary = summary.get(summary_key)
            if not isinstance(network_summary, Mapping) or "json" not in network_summary:
                continue
            result_path = Path(str(network_summary["json"]))
            result = read_json(result_path)
            network_dir_value = result.get("network_dir")
            if not network_dir_value:
                raise ValueError(f"Result does not contain network_dir: {result_path}")
            network_dir = Path(str(network_dir_value))
            cases.append(
                ResultCase(
                    model=model,
                    p_case=case_p,
                    network_size=network_size,
                    result_path=result_path,
                    result=result,
                    chosen_edges=chosen_edge_set(result),
                    od_endpoints=od_endpoint_set(result),
                    network_dir=network_dir,
                )
            )

    if not cases:
        raise ValueError(f"No result cases found for p_case={p_case!r} in {manifest_path}")
    return cases


def fmt_float(value: object, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:,.{digits}f}"


def fmt_pct(value: object) -> str:
    try:
        numeric = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2f}%"


def exposure_gap(result: Mapping[str, object]) -> Optional[float]:
    if "exposure_burden_gap_Ep_minus_Eu" in result:
        try:
            return float(result["exposure_burden_gap_Ep_minus_Eu"])
        except (TypeError, ValueError):
            return None
    try:
        return float(result["protected_uncovered_exposure_burden"]) - float(
            result["unprotected_uncovered_exposure_burden"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def single_metrics(case: ResultCase) -> List[str]:
    result = case.result
    gap = exposure_gap(result)
    lines = [
        f"Model: {case.label}",
        f"Status: {result.get('status_name', 'n/a')}",
        f"Cost: {fmt_float(result.get('construction_cost'))}",
        f"Coverage: {fmt_pct(result.get('covered_demand_share'))}",
        f"Chosen units: {len(case.chosen_edges)}",
        (
            "Exposure: "
            f"EP={fmt_float(result.get('protected_uncovered_exposure_burden'), 3)}, "
            f"EU={fmt_float(result.get('unprotected_uncovered_exposure_burden'), 3)}, "
            f"gap={fmt_float(gap, 3)}"
        ),
    ]
    if "budget" in result:
        lines.append(f"Budget: {fmt_float(result.get('budget'))}")
    if "eta_exposure" in result:
        lines.append(f"Eta: {fmt_float(result.get('eta_exposure'), 3)}")
    return lines


def comparison_metrics(objective_case: ResultCase, constraint_case: ResultCase) -> List[str]:
    common = objective_case.chosen_edges & constraint_case.chosen_edges
    objective_only = objective_case.chosen_edges - constraint_case.chosen_edges
    constraint_only = constraint_case.chosen_edges - objective_case.chosen_edges
    obj_gap = exposure_gap(objective_case.result)
    con_gap = exposure_gap(constraint_case.result)
    return [
        "Selected design unit overlap",
        f"Common: {len(common)}",
        f"Objective-only: {len(objective_only)}",
        f"Constraint-equity only: {len(constraint_only)}",
        (
            "Objective equity: "
            f"cost={fmt_float(objective_case.result.get('construction_cost'))}, "
            f"coverage={fmt_pct(objective_case.result.get('covered_demand_share'))}, "
            f"gap={fmt_float(obj_gap, 3)}"
        ),
        (
            "Constraint equity: "
            f"cost={fmt_float(constraint_case.result.get('construction_cost'))}, "
            f"coverage={fmt_pct(constraint_case.result.get('covered_demand_share'))}, "
            f"gap={fmt_float(con_gap, 3)}"
        ),
    ]


def edge_segments(edges: Iterable[Edge], nodes: Mapping[int, Point]) -> List[Tuple[Point, Point]]:
    segments: List[Tuple[Point, Point]] = []
    for u, v in sorted(edges):
        if u in nodes and v in nodes:
            segments.append((nodes[u], nodes[v]))
    return segments


def draw_png(
    network: NetworkData,
    layers: Sequence[Tuple[str, Set[Edge], str, float]],
    od_endpoints: Set[int],
    title: str,
    metrics: Sequence[str],
    out_path: Path,
) -> None:
    mpl_config_dir = Path(tempfile.gettempdir()) / "equalitymetric_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    xdg_cache_dir = Path(tempfile.gettempdir()) / "equalitymetric_cache"
    font_cache_dir = xdg_cache_dir / "fontconfig"
    font_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
    os.environ.setdefault("FC_CACHEDIR", str(font_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    xs = [point[0] for point in network.nodes.values()]
    ys = [point[1] for point in network.nodes.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    pad_x = width * 0.04
    pad_y = height * 0.04

    fig_width = 12.0
    fig_height = min(12.0, max(7.0, fig_width * height / width * 1.15))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    ax.set_facecolor("#f8fafc")

    background = edge_segments(network.arcs, network.nodes)
    if background:
        ax.add_collection(
            LineCollection(
                background,
                colors="#cbd5e1",
                linewidths=0.45,
                alpha=0.55,
                zorder=1,
            )
        )

    for _label, edges, color, linewidth in layers:
        segments = edge_segments(edges, network.nodes)
        if not segments:
            continue
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=linewidth,
                alpha=0.94,
                zorder=3,
                capstyle="round",
                joinstyle="round",
            )
        )

    endpoint_points = [network.nodes[node] for node in sorted(od_endpoints) if node in network.nodes]
    if endpoint_points:
        ax.scatter(
            [p[0] for p in endpoint_points],
            [p[1] for p in endpoint_points],
            s=28,
            color="#f59e0b",
            edgecolors="#7c2d12",
            linewidths=0.55,
            alpha=0.9,
            zorder=4,
        )

    legend_items = [
        Line2D([0], [0], color="#cbd5e1", lw=2, label="Candidate road link"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f59e0b", markeredgecolor="#7c2d12", label="OD endpoint"),
    ]
    for label, _edges, color, linewidth in layers:
        legend_items.append(Line2D([0], [0], color=color, lw=max(2.5, linewidth), label=label))
    ax.legend(handles=legend_items, loc="lower left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8)

    ax.text(
        0.01,
        0.99,
        "\n".join(metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.4,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.93},
        zorder=5,
    )

    ax.set_title(title, fontsize=12.5, pad=12, color="#111827")
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def project_to_svg(point: Point, bounds: Tuple[float, float, float, float], width: int, height: int, pad: int) -> Tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    usable_w = max(width - 2 * pad, 1)
    usable_h = max(height - 2 * pad, 1)
    x = pad + (point[0] - min_x) / max(max_x - min_x, 1e-9) * usable_w
    y = pad + (max_y - point[1]) / max(max_y - min_y, 1e-9) * usable_h
    return x, y


def svg_line(
    segment: Tuple[Point, Point],
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
    pad: int,
    color: str,
    line_width: float,
    opacity: float,
    title: str = "",
) -> str:
    x1, y1 = project_to_svg(segment[0], bounds, width, height, pad)
    x2, y2 = project_to_svg(segment[1], bounds, width, height, pad)
    title_tag = f"<title>{escape(title)}</title>" if title else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{line_width:.2f}" stroke-opacity="{opacity:.2f}" '
        f'stroke-linecap="round">{title_tag}</line>'
    )


def write_html(
    network: NetworkData,
    layers: Sequence[Tuple[str, Set[Edge], str, float]],
    od_endpoints: Set[int],
    title: str,
    metrics: Sequence[str],
    out_path: Path,
) -> None:
    xs = [point[0] for point in network.nodes.values()]
    ys = [point[1] for point in network.nodes.values()]
    bounds = (min(xs), max(xs), min(ys), max(ys))
    map_width = 1120
    aspect = max((bounds[3] - bounds[2]) / max(bounds[1] - bounds[0], 1e-9), 0.45)
    map_height = int(min(980, max(650, map_width * aspect * 1.05)))
    pad = 36

    background_lines = [
        svg_line(segment, bounds, map_width, map_height, pad, "#cbd5e1", 0.8, 0.55)
        for segment in edge_segments(network.arcs, network.nodes)
    ]

    layer_lines: List[str] = []
    for label, edges, color, line_width in layers:
        for segment in edge_segments(edges, network.nodes):
            layer_lines.append(svg_line(segment, bounds, map_width, map_height, pad, color, line_width, 0.94, label))

    endpoint_circles: List[str] = []
    for node in sorted(od_endpoints):
        if node not in network.nodes:
            continue
        x, y = project_to_svg(network.nodes[node], bounds, map_width, map_height, pad)
        endpoint_circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="#f59e0b" '
            f'stroke="#7c2d12" stroke-width="1"><title>OD endpoint {node}</title></circle>'
        )

    legend_items = [
        '<span><i style="background:#cbd5e1"></i>Candidate road link</span>',
        '<span><i class="dot"></i>OD endpoint</span>',
    ]
    for label, _edges, color, _line_width in layers:
        legend_items.append(f'<span><i style="background:{color}"></i>{escape(label)}</span>')

    metric_html = "".join(f"<li>{escape(line)}</li>" for line in metrics)
    legend_html = "".join(legend_items)
    background_svg = "\n        ".join(background_lines)
    layer_svg = "\n        ".join(layer_lines)
    endpoint_svg = "\n        ".join(endpoint_circles)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 14px; font-size: 20px; font-weight: 650; }}
    .panel {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; }}
    .map-wrap {{ overflow: auto; background: #f8fafc; border: 1px solid #e5e7eb; }}
    svg {{ display: block; width: 100%; height: auto; }}
    .meta {{ display: grid; grid-template-columns: minmax(260px, 1fr) minmax(260px, 1.3fr); gap: 16px; margin-bottom: 14px; }}
    ul {{ margin: 0; padding-left: 18px; line-height: 1.45; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px 18px; align-content: start; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 7px; font-size: 13px; }}
    .legend i {{ width: 28px; height: 4px; display: inline-block; border-radius: 999px; }}
    .legend .dot {{ width: 9px; height: 9px; background: #f59e0b; border: 1px solid #7c2d12; }}
    @media (max-width: 760px) {{ main {{ padding: 12px; }} .meta {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>{escape(title)}</h1>
  <section class="meta">
    <div class="panel"><ul>{metric_html}</ul></div>
    <div class="panel legend">{legend_html}</div>
  </section>
  <section class="map-wrap">
    <svg viewBox="0 0 {map_width} {map_height}" role="img" aria-label="{escape(title)}">
      <rect width="{map_width}" height="{map_height}" fill="#f8fafc"></rect>
      <g>
        {background_svg}
      </g>
      <g>
        {layer_svg}
      </g>
      <g>
        {endpoint_svg}
      </g>
    </svg>
  </section>
</main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def case_title(case: ResultCase) -> str:
    p_label = P_CASE_LABELS.get(case.p_case, case.p_case)
    network_label = NETWORK_LABELS.get(case.network_size, f"{case.network_size.title()} Network")
    return f"{case.label} | {p_label} | {network_label}"


def comparison_title(p_case: str, network_size: str) -> str:
    p_label = P_CASE_LABELS.get(p_case, p_case)
    network_label = NETWORK_LABELS.get(network_size, f"{network_size.title()} Network")
    return f"Objective Equity vs Constraint Equity | {p_label} | {network_label}"


def generate_single_maps(
    cases: Sequence[ResultCase],
    network_cache: Dict[Path, NetworkData],
    output_dir: Path,
    write_png_files: bool,
    write_html_files: bool,
) -> List[Dict[str, str]]:
    generated: List[Dict[str, str]] = []
    for case in cases:
        network = network_cache.setdefault(case.network_dir, read_network(case.network_dir))
        color = MODEL_COLORS.get(case.model, "#2563eb")
        layers = [(case.label, case.chosen_edges, color, 2.6)]
        stem = slug(f"{case.p_case}_{case.network_size}_{case.model}")
        title = case_title(case)
        metrics = single_metrics(case)

        entry = {
            "kind": "single",
            "model": case.model,
            "p_case": case.p_case,
            "network_size": case.network_size,
            "source_json": str(case.result_path),
        }
        if write_png_files:
            png_path = output_dir / f"{stem}.png"
            draw_png(network, layers, case.od_endpoints, title, metrics, png_path)
            entry["png"] = str(png_path)
        if write_html_files:
            html_path = output_dir / f"{stem}.html"
            write_html(network, layers, case.od_endpoints, title, metrics, html_path)
            entry["html"] = str(html_path)
        generated.append(entry)
    return generated


def generate_comparison_maps(
    cases: Sequence[ResultCase],
    network_cache: Dict[Path, NetworkData],
    output_dir: Path,
    write_png_files: bool,
    write_html_files: bool,
) -> List[Dict[str, str]]:
    by_key: Dict[Tuple[str, str, str], ResultCase] = {}
    for case in cases:
        by_key[(case.p_case, case.network_size, case.model)] = case

    generated: List[Dict[str, str]] = []
    p_and_sizes = sorted({(case.p_case, case.network_size) for case in cases})
    for p_case, network_size in p_and_sizes:
        objective_case = by_key.get((p_case, network_size, OBJECTIVE_MODEL))
        constraint_case = by_key.get((p_case, network_size, CONSTRAINT_MODEL))
        if objective_case is None or constraint_case is None:
            continue

        common = objective_case.chosen_edges & constraint_case.chosen_edges
        objective_only = objective_case.chosen_edges - constraint_case.chosen_edges
        constraint_only = constraint_case.chosen_edges - objective_case.chosen_edges
        layers = [
            ("Common to both", common, "#16a34a", 3.2),
            ("Objective equity only", objective_only, "#2563eb", 2.6),
            ("Constraint equity only", constraint_only, "#dc2626", 2.6),
        ]
        od_endpoints = objective_case.od_endpoints | constraint_case.od_endpoints
        network = network_cache.setdefault(objective_case.network_dir, read_network(objective_case.network_dir))
        title = comparison_title(p_case, network_size)
        metrics = comparison_metrics(objective_case, constraint_case)
        stem = slug(f"{p_case}_{network_size}_objective_I_E_vs_constraint_II_D")

        entry = {
            "kind": "comparison",
            "p_case": p_case,
            "network_size": network_size,
            "objective_json": str(objective_case.result_path),
            "constraint_json": str(constraint_case.result_path),
        }
        if write_png_files:
            png_path = output_dir / f"{stem}.png"
            draw_png(network, layers, od_endpoints, title, metrics, png_path)
            entry["png"] = str(png_path)
        if write_html_files:
            html_path = output_dir / f"{stem}.html"
            write_html(network, layers, od_endpoints, title, metrics, html_path)
            entry["html"] = str(html_path)
        generated.append(entry)
    return generated


def main() -> None:
    args = parse_args()
    cases = manifest_cases(args.manifest, args.p_case)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    network_cache: Dict[Path, NetworkData] = {}
    write_png_files = not args.no_png
    write_html_files = not args.no_html
    if write_png_files and importlib.util.find_spec("matplotlib") is None:
        print("matplotlib is not available in this Python; skipping PNG output. HTML maps will still be written.")
        write_png_files = False

    generated = []
    generated.extend(generate_single_maps(cases, network_cache, output_dir, write_png_files, write_html_files))
    generated.extend(generate_comparison_maps(cases, network_cache, output_dir, write_png_files, write_html_files))

    manifest = {
        "source_manifest": str(args.manifest),
        "p_case": args.p_case,
        "output_dir": str(output_dir),
        "generated_count": len(generated),
        "generated": generated,
    }
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(generated)} visualization entries to {output_dir}")
    print(f"Wrote visualization manifest to {manifest_path}")


if __name__ == "__main__":
    main()
