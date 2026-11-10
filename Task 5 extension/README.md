# Task 5 extension

Bicycle-lane safety optimization under a fixed resource budget. This standalone package contains the code and inputs needed for two planning tasks. Copy the entire folder to run it from another location.

| Command | Planning task | Budget |
|---|---|---|
| `full` | Select the complete bicycle-lane layout | Total budget = `multiplier × C_ref` |
| `incremental` | Retain existing lanes and select additions | `budget` is an absolute additional budget |

The default method is Gurobi MILP for the small network and ALNS for the expanded network.

## Folder structure

```text
Task 5 extension/
├── run.py                         CLI, budget settings, and result export
├── network.py                     TNTP input, construction units, and path screening
├── solver.py                      Gurobi MILP
├── alns.py                        ALNS search
├── requirements.txt
├── README.md
├── data/
│   ├── small/                     401 nodes and 177 positive-demand OD pairs
│   └── large/                     1265 nodes and 711 positive-demand OD pairs
│       # Each contains net.tntp, trips.tntp, and n2024_polygon_id_map.csv
└── baselines/
    ├── small/
    └── large/
        # Each contains reference_cost.json and fixed_network_j2.json
```

The four Python files provide the runnable implementation. Network inputs, node mappings, and four baseline JSON files are included. The 26 formal experiment results, run logs, batch scheduler, and report-generation tools are omitted. `network.py`, `solver.py`, and `alns.py` retain the original implementation; `run.py` reads its default inputs from the local `data` folder.

## Installation

Use Python 3.10 or later on macOS or Linux. ALNS path caching uses `fcntl` file locking.

Open a terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Gurobi requires a valid license. To run ALNS alone, install `numpy` and `scipy`. Dependency version ranges are listed in `requirements.txt`.

## Objective and service requirements

```text
min J = sum_w q_w [K_w (1 - z_w) + sum_a c_wa]
```

Here, `q_w` is OD demand, `K_w` is the fixed shortest-path reference distance, `z_w` indicates whether a qualifying route is provided, and `c_wa` is the remaining unprotected distance on a selected route arc. Demand without a qualifying route incurs a burden of `q_w K_w`. For demand with a qualifying route, the burden is demand multiplied by the route's remaining unprotected distance.

The default service parameters are `--coverage-share 0.8 --epsilon 0.2 --t 0.2`: at least 80% of demand must receive qualifying routes, route length must not exceed `1.2 K_w`, and unprotected distance must not exceed `0.2 K_w`. Construction is charged by physical road unit, with both travel directions sharing the same construction decision.

The output field `j2` represents the objective `J` defined here and preserves the original experiment's data format.

## Reference costs and existing layouts

The included baselines reuse the historical Task 3 layouts and their fixed-layout evaluations used in the formal experiments:

| Network | `C_ref` | Baseline construction units | Baseline `J` |
|---|---:|---:|---:|
| small | 5169.033897138104 | 68 | 446986.9268343237 |
| large | 19469.687353669575 | 242 | 1221979.589822862 |

`reference_cost.json` stores the reference construction cost and layout. `fixed_network_j2.json` stores the route evaluation of that same layout under the current objective. The commands below reuse both files to proceed directly to budget optimization.

`C_ref` is the cost of a historical feasible design. The historical small-network cost solution has a MIP gap of approximately 0.9553%; the historical expanded-network layout was obtained using ALNS.

## Full-network planning

Small network with a total budget of `1.5 C_ref`:

```bash
python run.py full \
  --network small --multiplier 1.5 \
  --reference-cost-result baselines/small/reference_cost.json \
  --baseline-j2-result baselines/small/fixed_network_j2.json \
  --time-limit 18000 --threads 4 \
  --output-dir results/full_small_1p5
```

Expanded network with the same budget multiplier, using ALNS:

```bash
python run.py full \
  --network large --solver alns --multiplier 1.5 \
  --reference-cost-result baselines/large/reference_cost.json \
  --baseline-j2-result baselines/large/fixed_network_j2.json \
  --path-pool-cache results/large_paths.json \
  --time-limit 18000 \
  --output-dir results/full_large_1p5
```

A multiplier of `1.5` sets the total resource cap to 150% of the reference cost. The formal experiment uses multipliers of `1.0, 1.2, 1.5, 1.8, 2.0`.

If `--reference-cost-result` is omitted, the small-network entry point attempts a new minimum-construction-cost optimization. Expanded-network ALNS requires a reference cost file. Reuse the supplied files to retain the study's budget baseline.

## Incremental bicycle-lane expansion

Small network with an absolute additional budget of `1000`, retaining all baseline lanes:

```bash
python run.py incremental \
  --network small \
  --existing-design baselines/small/fixed_network_j2.json \
  --baseline-j2-result baselines/small/fixed_network_j2.json \
  --budget 1000 --time-limit 18000 --threads 4 \
  --output-dir results/incremental_small_1000
```

Expanded network with an absolute additional budget of `2000`:

```bash
python run.py incremental \
  --network large --solver alns \
  --existing-design baselines/large/fixed_network_j2.json \
  --baseline-j2-result baselines/large/fixed_network_j2.json \
  --budget 2000 --time-limit 18000 \
  --path-pool-cache results/large_paths.json \
  --output-dir results/incremental_large_2000
```

`--budget` pays for additions only. Existing facilities are retained, and their cost is excluded from the additional budget. The formal additional-budget levels are `0, 0.05, 0.10, 0.15, 0.20, 0.50, 0.80, 1.00` times `C_ref`. Multiply the selected level by the appropriate `C_ref` and pass the resulting absolute amount to `--budget`.

An existing layout can be supplied as a result JSON or construction-unit CSV exported by this program. When changing the layout, provide its corresponding `--baseline-j2-result`, or omit that argument to evaluate the fixed layout again.

## Common options and outputs

| Option | Meaning |
|---|---|
| `--time-limit` | Final optimization/search limit in seconds; default 3600; examples use 18000 (5 hours) |
| `--threads` | Gurobi thread count; default 4; ALNS uses one search process per case |
| `--seed` | Random seed; default 42 |
| `--path-pool-size` | ALNS candidate-route count parameter; default 12 |
| `--network-dir PATH` | Custom TNTP network directory |
| `--cost-file FILE.csv` | Complete physical-unit cost table with columns `numeric_u,numeric_v,cost` |
| `--quiet` | Suppress Gurobi console output |

The default construction cost is the maximum directed-arc length within each physical road unit. Budget units therefore follow the network's length units. To use monetary costs, supply a complete cost table through `--cost-file`. Changes to the network, OD demand, costs, or service parameters may invalidate historical baseline fingerprints; supply compatible reference results for the modified instance.

Outputs are written to the specified `--output-dir`:

```text
run_summary.json                  Budget, algorithm, baseline, and final metrics
full_network_j2.json              Full-network planning result
incremental_j2.json               Incremental planning result
<stage>_chosen_design_units.csv   All selected construction units
<stage>_new_design_units.csv      Newly built units in incremental planning
<stage>_od_metrics.csv            OD coverage, distances, and objective contributions
<stage>_progress.json             Current stage and solver/search progress
```

Each run generates the files applicable to its task. Gurobi results record the solver status, bound, and gap. ALNS results use the status `FEASIBLE_HEURISTIC` with a null gap. Exit code `0` indicates a validated feasible result, `2` indicates an input or cost-reference issue, and `3` indicates that the required feasible result was not obtained.

For an initial smoke run, replace `--time-limit 18000` in an example with `--time-limit 5`. This option limits the final optimization stage; input loading, MILP construction, and ALNS path generation also take time.
