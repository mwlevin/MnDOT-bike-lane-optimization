# Objective I-E Refine

This folder refines Objective I-E by adding an explicit 80/80 service guarantee.
The current default equity scenario is targeted to the original 80/80 baseline:
OD pairs that were not covered by the original baseline receive high protected share,
and OD pairs that were covered receive low protected share.

## Why this refinement is needed

The original Objective I-E minimizes protected-population uncovered-exposure burden under a fixed budget:

`min sum_w q_w^P U_w`

subject to the construction budget and path-certification constraints.

That objective can improve protected-group exposure, but by itself it does not logically guarantee that at least 80% of total demand is covered. The refined version adds the missing service constraint:

`sum_w q_w z_w >= rho * sum_w q_w`

with default `rho = 0.8`.

The default uncovered-length cap remains:

`sum_a c_{w,a} <= t * K_w * z_w`

with default `t = 0.2`, meaning every covered OD has at most 20% uncovered shortest-path-equivalent length. Together, `rho = 0.8` and `t = 0.2` implement the 80/80 guarantee.

## Budget interpretation

Let:

`C_80/80 = minimum construction cost needed to satisfy the 80/80 service constraints`.

In this implementation, the default reference values are taken directly from `bilk_equality_paper (3).pdf`:

- small network base-case `C_80/80 = 5169.03`
- large network base-case ALNS `C_80/80 = 19469.69`

The default multiplier is:

`gamma = 1.5`

Therefore, unless overridden, the runner uses:

- small network budget `B = 1.5 * 5169.03 = 7753.545`
- large network budget `B = 1.5 * 19469.69 = 29204.535`

More generally, if the planner sets:

`B = gamma * C_80/80`

then:

- `gamma >= 1` means the refined Objective I-E model should be feasible, assuming `C_80/80` was computed or verified on the same network, demand set, `epsilon`, and `t`;
- `gamma < 1` may be infeasible;
- budget feasibility is still verified in the output with `coverage_requirement_satisfied` and, for the exact model, the Gurobi status.

The refined model then solves:

`min protected uncovered-exposure burden`

subject to:

- budget `construction_cost <= B`;
- total coverage `covered_demand_share >= 0.8`;
- covered OD service quality `uncovered_length <= 0.2 K_w`;
- detour limit `path_length <= (1 + epsilon) K_w`.

## Files

- `solve_objective_I_E_refine.py`: exact Gurobi MILP for the small network or any selected OD subset.
- `run_objective_I_E_refine.py`: unified runner:
  - small network: exact Gurobi;
  - large network: budget-constrained ALNS heuristic with coverage repair.
- `generate_targeted_pi_from_base.py`: rebuilds the targeted `pi_w` files from the original baseline outputs.
- `data/cropped_subgraph`: small-network data, original synthetic `pi_w`, and targeted `pi_w`.
- `data/full_network`: full-network data, original synthetic `pi_w`, and targeted `pi_w`.
- `legacy_src`: copied parsing, path, pruning, and helper code.

## Current targeted `pi_w`

Default file used by both `solve_objective_I_E_refine.py` and `run_objective_I_E_refine.py`:

`od_equity_shares_targeted_base_uncovered.csv`

Construction rule:

- original baseline covered OD: `pi_w = 0.15`
- original baseline uncovered OD: `pi_w = 0.85`

The baseline references are:

- small network: `baseline_inputs/small_task3_baseline_80_80.json`
- large network: `baseline_inputs/large_task3_baseline_80_80.json`

Regenerate the targeted files:

```bash
python3 objective_I_E_refine/generate_targeted_pi_from_base.py
```

Use a different high/low contrast:

```bash
python3 objective_I_E_refine/generate_targeted_pi_from_base.py \
  --pi-low 0.10 \
  --pi-high 0.90
```

To run the older random synthetic `pi_w`, pass:

```bash
--equity-file-name od_equity_shares_synthetic.csv
```

## Run both networks

```bash
python3 objective_I_E_refine/run_objective_I_E_refine.py \
  --coverage-share 0.8 \
  --t 0.2 \
  --epsilon 0.2 \
  --budget-gamma 1.5 \
  --threads 4 \
  --large-workers 4
```

Outputs are written to `objective_I_E_refine/results/`, including:

- `objective_I_E_refine_run_summary.json`
- small exact JSON/CSV outputs
- large ALNS JSON/CSV outputs

The smoke run returned `coverage_requirement_satisfied=true` and `service_80_80_satisfied=true` for both networks.
