# Constraint II-D: Uncovered-Exposure Disparity Constraint

This folder implements Treatment II-D for the Minneapolis bike-lane design experiment.
The current default equity scenario is targeted to the original 80/80 baseline:
OD pairs that were not covered by the original baseline receive high protected share,
and OD pairs that were covered receive low protected share.

## Model meaning

The experiment minimizes investment cost while enforcing:

1. Total service coverage:

   `sum_w q_w z_w >= rho * sum_w q_w`

   Default `rho = 0.8`, so at least 80% of total demand must be covered.

2. Covered OD service quality:

   `sum_a c_{w,a} <= t * K_w * z_w`

   Default `t = 0.2`, so every covered OD has no more than 20% uncovered shortest-path-equivalent exposure. In plain terms, covered OD pairs are at least 80% served.

3. Constraint II-D:

   `E^P <= E^U + eta_E`

   where:

   `U_w = (1 - z_w) + (1 / K_w) * sum_a c_{w,a}`

   `E^P = sum_w q_w^P U_w / sum_w q_w^P`

   `E^U = sum_w q_w^U U_w / sum_w q_w^U`

The default `eta_E = 0.05` allows protected uncovered-exposure burden to exceed unprotected burden by at most 5 percentage points.

## Files

- `solve_constraint_II_D.py`: exact Gurobi MILP for the small/cropped network or any selected OD subset.
- `run_constraint_II_D.py`: unified runner with `run()`:
  - small network: exact Gurobi MILP;
  - large network: multi-start constraint-repair ALNS heuristic.
- `generate_targeted_pi_from_base.py`: rebuilds the targeted `pi_w` files from the original baseline outputs.
- `data/cropped_subgraph`: copied small-network data, original synthetic `pi_w`, and targeted `pi_w`.
- `data/full_network`: copied full-network data, original synthetic `pi_w`, and targeted `pi_w`.
- `legacy_src`: copied parsing, path, pruning, and design-unit helpers from the original project.

## Current targeted `pi_w`

Default file used by `run_constraint_II_D.py`:

`od_equity_shares_targeted_base_uncovered.csv`

Construction rule:

- original baseline covered OD: `pi_w = 0.15`
- original baseline uncovered OD: `pi_w = 0.85`

The baseline references are:

- small network: `baseline_inputs/small_task3_baseline_80_80.json`
- large network: `baseline_inputs/large_task3_baseline_80_80.json`

Regenerate the targeted files:

```bash
python3 constraint_II_D_exposure_disparity/generate_targeted_pi_from_base.py
```

Use a different high/low contrast:

```bash
python3 constraint_II_D_exposure_disparity/generate_targeted_pi_from_base.py \
  --pi-low 0.10 \
  --pi-high 0.90
```

To run the older random synthetic `pi_w`, pass:

```bash
--equity-file-name od_equity_shares_synthetic.csv
```

## Run both networks

```bash
python3 constraint_II_D_exposure_disparity/run_constraint_II_D.py \
  --coverage-share 0.8 \
  --t 0.2 \
  --eta-exposure 0.05 \
  --threads 4 \
  --large-workers 4 \
  --output-flag 1
```

Outputs are written to `constraint_II_D_exposure_disparity/results/`:

- `constraint_II_D_run_summary.json`
- `small_exact_gurobi/small_constraint_II_D_exact.json`
- `large_alns/large_constraint_II_D_alns.json`
- per-OD metrics CSV files
- chosen design-unit CSV files
