# Task 5 extension

Budget-constrained bicycle-lane safety planning, extending the Task 3 route-certification framework. This folder is self-contained: copy the entire directory to another location to run it.

| Command | Infrastructure decision | Budget |
|---|---|---|
| `full` | Select the complete layout | Total budget = `multiplier * C_ref` |
| `incremental` | Retain an existing layout and select additions | `budget` is an absolute additional cost |

Both tasks use the same objective and route-certification conditions. The default solver is Gurobi MILP for the small network and ALNS for the expanded network.

## Contents

```text
Task 5 extension/
├── run.py                 Command-line interface and result export
├── network.py             TNTP input, physical construction units, path screening
├── solver.py              Gurobi MILP
├── alns.py                ALNS search
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── small/             401 nodes; 177 positive-demand OD pairs
│   └── large/             1,265 nodes; 711 positive-demand OD pairs
└── baselines/
    ├── manifest.json      Input identity and baseline file checksums
    ├── small/
    └── large/
```

Each data directory contains `net.tntp`, `trips.tntp`, and `n2024_polygon_id_map.csv`. Each baseline directory contains `reference_cost.json` and `fixed_network_j2.json`. The four Python files are the runnable code. Formal experiment results, logs, process-management scripts, and report assets are excluded.

## Installation

Use Python 3.10 or later on macOS or Linux. ALNS path caching uses `fcntl` file locking.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Gurobi requires a valid license. ALNS requires NumPy and SciPy and does not invoke Gurobi; for an ALNS-only installation, `python -m pip install numpy scipy` is sufficient.

## Objective and route certification

```text
min J = sum_w q_w [K_w (1 - z_w) + sum_a c_wa]
```

`q_w` is OD demand, `K_w` is the fixed shortest-path reference distance, `z_w` indicates whether a qualifying route is provided, and `c_wa` is the remaining unprotected distance on a selected route arc. An OD without a qualifying route contributes `q_w * K_w`. A certified OD contributes demand times its route's unprotected distance.

There is no minimum aggregate demand coverage constraint. Coverage is an output metric. With the default `--epsilon 0.2 --t 0.2`, each certified route must satisfy:

- Total path length at most `1.2 * K_w`.
- Unprotected length at most `0.2 * K_w`.

Both limits use the shortest-path reference distance, not the actual selected path length. Uncertified OD demand remains in the objective. The result field `j2` stores `J` to preserve the established output format.

Opposite travel directions share one physical construction decision. The default unit cost is the maximum directed-arc length within that physical unit, so budget units follow the network's length units.

## Historical references

The included Task 3 reference layouts and their fixed-layout objective evaluations are reused:

| Network | `C_ref` | Reference units | Fixed-layout `J` |
|---|---:|---:|---:|
| small | 5169.033897138104 | 68 | 446986.9268343237 |
| large | 19469.687353669575 | 242 | 1221979.589822862 |

`C_ref` is the construction cost of the historical reference layout. Full planning can replace that layout; incremental planning retains the input facilities. The CLI never runs a new minimum-cost optimization.

For compatible bundled inputs, the existing fixed-layout evaluation is checked and reused automatically. File hashes, network and demand identities, selected units, route conditions, costs, and the objective are checked before reuse. The files retain their historical fingerprints and solver metadata; reused outputs record the current model identity and retain the original evaluation provenance.

## Full-network planning

Small network, with a total budget of `1.5 * C_ref`:

```bash
python run.py full --network small --multiplier 1.5 \
  --time-limit 18000 --threads 4 --output-dir results/full_small_1p5
```

Expanded network, using ALNS:

```bash
python run.py full --network large --solver alns --multiplier 1.5 \
  --time-limit 18000 --path-pool-cache results/large_paths.json \
  --output-dir results/full_large_1p5
```

The supplied reference cost and fixed-layout evaluation are selected automatically. To choose an explicit reference layout, use `--reference-cost-result FILE.json`. A multiplier of `1.0` still optimizes the entire layout; it does not return the historical layout as an assumed optimum.

The report's full-planning budget multipliers are `0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0`.

## Incremental expansion

Small network with an absolute additional budget of `1000`:

```bash
python run.py incremental --network small \
  --existing-design baselines/small/reference_cost.json --budget 1000 \
  --time-limit 18000 --threads 4 --output-dir results/incremental_small_1000
```

Expanded network with an absolute additional budget of `2000`:

```bash
python run.py incremental --network large --solver alns \
  --existing-design baselines/large/reference_cost.json --budget 2000 \
  --time-limit 18000 --path-pool-cache results/large_paths.json \
  --output-dir results/incremental_large_2000
```

The budget pays only for added units. Every existing unit is retained. The report's additional-budget ratios are `0, 0.05, 0.10, 0.15, 0.20, 0.50, 0.80, 1.00` times `C_ref`; calculate the absolute amount before passing it to `--budget`.

At zero additional budget, strictly positive construction costs fix the layout. The validated fixed-layout evaluation is reused as the final result with status `FIXED_LAYOUT_REUSED`; no new construction search is started. Zero-cost candidate units are handled by the optimization model instead of this shortcut.

## Other networks and existing layouts

Use `--network-dir PATH` for a directory containing TNTP road and OD files. The numeric-to-original node mapping CSV is optional for a new dataset. An existing layout can be a result JSON with `chosen_design_units` or `chosen_links`, or a CSV with `numeric_u,numeric_v`. Original-ID columns can be included for road remapping.

For full planning on custom inputs, provide `--reference-cost-result FILE.json`. A minimal reference file is:

```json
{
  "chosen_design_units": [
    {"numeric_u": 1, "numeric_v": 2},
    {"numeric_u": 2, "numeric_v": 3}
  ]
}
```

The reference cost is computed from those units and the current cost data. If the JSON also supplies `construction_cost`, it must match the computed value.

```bash
python run.py full --network-dir /path/to/network --solver gurobi \
  --reference-cost-result /path/to/reference.json --multiplier 0.8 \
  --output-dir results/custom_full

python run.py incremental --network-dir /path/to/network --solver alns \
  --existing-design /path/to/existing.csv --budget 1000 \
  --output-dir results/custom_expansion
```

To use monetary costs, supply a complete physical-unit CSV through `--cost-file`, with columns `numeric_u,numeric_v,cost`. Path distances continue to use network lengths.

For a new case without a matching cached evaluation, the input facilities are fixed while the CLI evaluates their routes. It does not reselect baseline facilities. `--evaluation-time-limit` controls this evaluation. A compatible fixed-layout result can be supplied with `--baseline-j2-result` to reuse it. The bundled historical references are only reused when the current network, demand, costs, and route parameters match.

## Options and outputs

| Option | Default | Meaning |
|---|---:|---|
| `--time-limit` | 18000 | Final solver/search limit in seconds |
| `--evaluation-time-limit` | 300 | Fixed-layout evaluation limit for new cases |
| `--threads` | 4 | Gurobi solver threads |
| `--mip-gap` | 0.0001 | Gurobi relative gap target |
| `--seed` | 42 | Random seed |
| `--path-pool-size` | 12 | ALNS candidate-route count parameter |
| `--path-pool-cache` | none | Optional reusable ALNS path cache |
| `--quiet` | off | Suppress Gurobi console output |

Each command runs one case. Time limits cover solver/search execution; reading inputs and building the MILP also take time.

Outputs include:

```text
run_summary.json
full_network_j2.json or incremental_j2.json
<stage>_chosen_design_units.csv
<stage>_new_design_units.csv
<stage>_od_metrics.csv
<stage>_progress.json
```

Gurobi records its actual status, bound, and gap. ALNS returns validated feasible solutions with a null optimality gap. Exit code `0` indicates a validated result, `2` an input/setup error, and `3` failure to obtain the required feasible result.

For a quick interface check, use `--time-limit 5`. The bundled fixed-layout evaluations are still reused. Use the longer search limits to reproduce optimization experiments.
