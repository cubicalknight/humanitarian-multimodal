# Sensitivity analyses

Run the following commands from the repository root. Relative sensitivity
paths are resolved from `src/`, so the examples write to the repository-level
`run_outputs/` directory.

## Cost sensitivity

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

To compare feasible-route preprocessing independently of the cost-sensitivity
pipeline, run:

```bash
poetry run python src/main.py --config sensitivity_config.json --k-sensitivity
```

This sequentially rebuilds and solves the cases `K = 35, 70, 100, 150` using
only `base_parameters`. It writes detailed `k_*.json` results and a compact
`summary.json` to `src/output/k_sensitivity/` by default. Supply `--output-dir`
to choose another result directory.
