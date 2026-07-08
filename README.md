# MnDOT Bicycle-Lane Network Design Software

This folder contains the software and input data used to run the bicycle-lane
network design experiments for the Minneapolis CBD network.

## Folder Contents

```text
mndot_software_submission/
  README.md
  requirements.txt
  alns_link_based_undirected_design.py
  gurobi_link_based_undirected_design.py
  data/
    full_network/
      net.tntp
      node.tntp
      trips.tntp
      n2024_polygon_id_map.csv
      OD_geometries.csv
    cropped_subgraph/
      net.tntp
      node.tntp
      trips.tntp
      n2024_polygon_id_map.csv
      OD_geometries.csv
```

No historical run results are included in this submission folder. New results
are written to `outputs/` when the scripts are executed.

## Python Environment

Use Python 3.10 or newer.

The ALNS solver uses only the Python standard library.

Required packages for the Gurobi solver:

```bash
pip install -r requirements.txt
```

The Gurobi solver also requires a working Gurobi installation and license.

Optional command to confirm the script entry points are available:

```bash
python alns_link_based_undirected_design.py --help
python gurobi_link_based_undirected_design.py --help
```

All default paths are relative to the `mndot_software_submission/` folder. For
example, the default ALNS input folder is `data/full_network`, and the default
Gurobi input folder is `data/cropped_subgraph`.

## Run the ALNS Solver

The ALNS solver is intended for the large full network.
Run the command from the `mndot_software_submission/` folder. The default data
path is relative to this folder.

```bash
python alns_link_based_undirected_design.py
```

Default input:

```text
data/full_network/
```

Default output:

```text
outputs/alns_full_network/
```

Example with explicit parameters:

```bash
python alns_link_based_undirected_design.py \
  --network-dir data/full_network \
  --output-dir outputs/alns_full_network \
  --epsilon 0.2 \
  --t 0.2 \
  --covered-demand-share 0.8 \
  --od-limit 0 \
  --iterations 8000 \
  --time-limit 36000
```

## Run the Gurobi Solver

The Gurobi solver defaults to the cropped subgraph so that the example is
directly runnable on a smaller network.
Run the command from the `mndot_software_submission/` folder. The default data
path is relative to this folder.

```bash
python gurobi_link_based_undirected_design.py
```

Default input:

```text
data/cropped_subgraph/
```

Default output:

```text
outputs/gurobi_cropped_subgraph/
```

Example with explicit parameters:

```bash
python gurobi_link_based_undirected_design.py \
  --network-dir data/cropped_subgraph \
  --output-dir outputs/gurobi_cropped_subgraph \
  --epsilon 0.2 \
  --t 0.2 \
  --covered-demand-share 0.8 \
  --od-limit 0 \
  --time-limit 72000 \
  --mip-gap 0.01
```

The Gurobi solver can also be run on the full network:

```bash
python gurobi_link_based_undirected_design.py \
  --network-dir data/full_network \
  --output-dir outputs/gurobi_full_network \
  --epsilon 0.2 \
  --t 0.2 \
  --covered-demand-share 0.8 \
  --od-limit 20 \
  --time-limit 72000 \
  --mip-gap 0.01
```

For the full network, using a positive `--od-limit` is recommended unless a
long Gurobi run is intended.

## Quick Smoke Tests

These short runs check that the scripts can read the included relative-path
data folders and write outputs.

ALNS smoke test:

```bash
python alns_link_based_undirected_design.py \
  --od-limit 2 \
  --iterations 1 \
  --time-limit 5 \
  --output-dir outputs/alns_smoke \
  --prefix smoke \
  --report-interval 0
```

Gurobi smoke test:

```bash
python gurobi_link_based_undirected_design.py \
  --od-limit 1 \
  --time-limit 30 \
  --mip-gap 0.5 \
  --output-flag 0 \
  --output-dir outputs/gurobi_smoke \
  --prefix smoke
```

## Main Parameters

- `--epsilon`: Detour tolerance. A routed OD path must have length at most
  `(1 + epsilon)` times the shortest-path length.
- `--t`: Maximum uncovered length ratio for a covered OD. For a covered OD,
  the path length without bicycle-lane support must be at most `t` times the
  shortest-path length.
- `--covered-demand-share`: Required share of total OD demand that must satisfy
  the uncovered-length condition. Values can be written as fractions, e.g.
  `0.8`, or percentages, e.g. `80`.
- `--od-limit`: Number of positive-demand OD pairs to solve. Use `0` for all
  OD pairs.
- `--seed`: Random seed for OD sampling and ALNS search.
- `--time-limit`: Time limit in seconds.
- `--mip-gap`: Relative optimality gap target for Gurobi.

## Input Data Format

Each network folder must contain these required files.

### `net.tntp`

TNTP network file. The metadata must include:

```text
<NUMBER OF ZONES> ...
<NUMBER OF NODES> ...
<FIRST THRU NODE> ...
<NUMBER OF LINKS> ...
<END OF METADATA>
```

After the metadata, each link row must follow the TNTP link format:

```text
init_node  term_node  capacity  length  free_flow_time  b  power  speed  toll  link_type ;
```

The solvers use:

- `init_node`, `term_node`: directed arc endpoints.
- `length`: link length and construction cost.
- `link_type`: link classification. The provided data uses road links for the
  design network.

### `trips.tntp`

TNTP OD demand file. It contains the number of zones, total OD flow, and demand
blocks by origin:

```text
Origin 1
2 : 10.0; 3 : 5.0;
```

Only OD pairs with positive demand are used.

### `node.tntp`

Node coordinate file:

```text
Node  X  Y ;
1     -93.2663  44.9675 ;
```

The coordinates are used for visualization outputs.

### `n2024_polygon_id_map.csv`

Mapping from TNTP numeric IDs to original node or zone IDs:

```text
numeric_id,original_id,node_type,x,y
1,392,zone,-93.2663,44.9675
```

This file is used to report selected links and OD paths with original IDs.

### `OD_geometries.csv`

Optional zone polygon file used for map overlays:

```text
id,name,geometry
392,1312,"POLYGON ((...))"
```

The solvers can still run without this file, but the HTML maps will not show
zone polygon overlays.

## Output Files

Each run writes files using the selected `--prefix`:

- `<prefix>.json`: solution data, objective, selected links, OD paths, and OD
  statistics.
- `<prefix>.html`: interactive visualization.
- `<prefix>.svg` and `<prefix>.png`: static network visualization.
- `<prefix>_summary.html`: compact summary page generated by the Gurobi solver.
- `<prefix>_chosen_links.csv`: selected-link table generated by the ALNS solver.

## Solver Summary

Both solvers minimize the total selected bicycle-lane construction length.
Construction decisions are made on undirected physical links, so opposite
directions of the same road segment share one design decision.

The ALNS solver is a heuristic method for large instances. It builds candidate
OD paths and improves a selected link set with adaptive destroy and repair
operators.

The Gurobi solver formulates the problem as a mixed-integer optimization model
with OD path variables, uncovered-length variables, and partial demand coverage
variables.
