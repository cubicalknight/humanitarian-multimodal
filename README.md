# Sensitivity analyses

Run the following commands from the repository root. Relative sensitivity
paths are resolved from `src/`, so the examples write to the repository-level
`run_outputs/` directory.

## Cost sensitivity

All `src/main.py` commands load `src/sensitivity_config.json` by default.
Use `--config another_config.json` to override it; relative paths resolve from `src/`.

## Running cost sensitivity

To run the entire local sweep on any supported operating system, use the
cross-platform Python runner. It prepares shared inputs, runs every configured
cost combination, merges the results, validates the finished summary, and
writes the PDF heatmaps to `src/figures/`:

```bash
poetry run python scripts/run_sensitivity_local.py
```

By default this writes to `src/output/sensitivity/results_500/`. Pass
`--output-dir output/sensitivity/my_run` to choose another directory relative
to `src/`.

Prepare the network and shared scenarios once:

```bash
poetry run python src/main.py --config sensitivity_config.json --prepared-problem ../run_outputs/sensitivity/prepared_problem.pkl --prepare
```

Run one cost combination locally, replacing `0` with the desired combination
index:

```bash
poetry run python src/main.py --config sensitivity_config.json --prepared-problem ../run_outputs/sensitivity/prepared_problem.pkl --output-dir ../run_outputs/sensitivity/results --run-index 0
```

To distribute the combinations with Slurm, calculate the array size and submit
the existing batch script:

```bash
COUNT=$(poetry run python src/main.py --config sensitivity_config.json --prepared-problem ../run_outputs/sensitivity/prepared_problem.pkl --print-run-count)
export CONFIG=sensitivity_config.json
export CHUNK_SIZE=5
TASK_COUNT=$(((COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE))
MAX_CONCURRENT=1 sbatch --array=0-$((TASK_COUNT - 1))%"$MAX_CONCURRENT" --export=CONFIG,CHUNK_SIZE scripts/run_sensitivity_array.sbatch
```

After every array task completes, merge its detailed results:

```bash
poetry run python src/main.py --config sensitivity_config.json --output-dir ../run_outputs/sensitivity/results --merge
```

## Feasible-route K sensitivity

These analyses vary the candidate-path limit while solving at fixed
`base_parameters`. Both commands prepare their own inputs; no separate
`--prepare` step or prepared-problem file is needed. The sweep is currently
`K = 1, 2, 5, 10, 20`, defined by `ROUTE_PATH_LIMITS` in `src/main.py`.
Edit that constant to change the sweep; `network.recourse_path_limit` controls
ordinary preprocessing but is overridden for each K-sensitivity case.

### Feasible-path construction

`network.feasible_path_mode` in `src/sensitivity_config.json` controls route
preprocessing during `--prepare` and `--k-sensitivity`:

- `distance_only` (code default) retains up to K shortest paths ranked by
  actual leg distance.
- `seed_union` unions K raw-distance paths with K paths for every Cartesian
  pair of `parameter_values.cost_flight` and `parameter_values.cost_ground`.
  Seeded paths use `shipment.weight * leg.distance_miles * mode_unit_cost`.
  `cost_penalty_incompatibility` is not a leg cost and does not affect this
  path pool.

The supplied configuration selects `seed_union`. K is a limit **per ranking**,
not the final number of unique candidate paths: overlapping paths are
deduplicated across the distance and cost rankings.

`network.restrict_first_stage` defaults to `true`, requiring the first-stage
route to use legs from the same per-shipment candidate union used in recourse. Set
it to `false` to retain the legacy unrestricted first-stage formulation.

After changing the mode or cost grid, rerun `--prepare`: existing prepared
problem files retain their stored feasible-route sets. Result JSON records the
mode, K, and seed pairs used to produce those sets.

### Run a single preprocessing method

To compare feasible-route preprocessing independently of the cost-sensitivity
pipeline, run:

```bash
poetry run python src/main.py --config sensitivity_config.json --k-sensitivity
```

This sequentially rebuilds and solves every K using the configured
`network.feasible_path_mode` and the same random seed for comparable scenario
draws on shared routes. Objective costs stay at `base_parameters`; in
`seed_union` mode, the cost grid still determines the candidate paths.
It writes detailed `k_*.json` results and a compact
`summary.json` to `src/output/k_sensitivity/` by default. Supply `--output-dir`
to choose another result directory, with relative paths resolved from `src/`.

### Compare methods with matched path counts

To compare path-ranking methods while keeping each shipment's candidate-path
count identical, run:

```bash
poetry run python src/main.py --config sensitivity_config.json --matched-k-sensitivity
```

This sweeps every configured `ROUTE_PATH_LIMITS` value. The first case uses the
same seed union as `--k-sensitivity` with `feasible_path_mode: "seed_union"`:
K paths per configured air/ground cost pair plus K raw-distance paths, deduplicated.
Every shipment's distance-only K matches the number of unique paths in that union.
The first case retains the `cost_weighted_k_*.json` filename for compatibility,
with `feasible_path_mode: "seed_union"` in its metadata. Both cases solve at the
baseline objective costs. The result files include the per-shipment distance K.

This command always compares `seed_union` against `distance_only`, regardless
of the configured mode. Both cases use the same configured random seed.
Matching counts equalizes candidate paths, not the number of distinct legs in
their unions. Outputs default to `src/output/k_sensitivity/`:

- `cost_weighted_k_*.json`: detailed seed-union results.
- `distance_only_matched_k_*.json`: detailed distance-only results.
- `matched_k_summary.json`: objective, solver status, and runtime for each case/K.

### Plot K sensitivity

After either analysis finishes, generate the PDF figures:

```bash
poetry run python src/plot_k_sensitivity.py
```

The plotter reads `src/output/k_sensitivity/` and writes to
`src/output/figures/k_sensitivity/`. It prefers matched result files when any
are present; otherwise it reads `k_*.json`. Use separate result directories
for separate experiments to avoid mixing old and new results.

It produces `objective.pdf`, `ground_links.pdf`, `air_links.pdf`,
`first_stage_jaccard.pdf`, and `final_route_jaccard.pdf` when the corresponding
decision data are available. Link counts use final (post-recourse) routes,
with a leg active when `keep + reassign > 1e-6`. Lines show means and shaded
bands show min–max ranges across shipment/scenario routings (across shipments
for first-stage similarity). Jaccard similarity compares sets of directed,
mode-specific legs against the largest available K **within each method**;
it does not compare the two methods directly. Runs with no incumbent solution
are marked in the objective plot and omitted from route metrics.

To use custom directories or a different reference K:

```bash
poetry run python src/plot_k_sensitivity.py --input-dir run_outputs/k_sensitivity --output-dir run_outputs/figures/k_sensitivity --reference-k 10
```

Unlike `src/main.py`, the plotter resolves explicit relative paths from the
current working directory. `--reference-k` must exist in each plotted method;
`--active-threshold` changes the final-leg flow cutoff.
