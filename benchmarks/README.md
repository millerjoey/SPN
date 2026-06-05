# Mixed Data Benchmarks

Small benchmark harness for mixed-type structure learning and prototype parameter training.

## Datasets

- `synthetic`: generated locally from a two-component mixed latent structure with numeric, count, and categorical columns plus MCAR missingness.
- `credit`: UCI Credit Approval, a small mixed categorical/real dataset with missing values.
- `adult`: UCI Adult, a larger categorical/integer dataset with missing category values.

Downloaded data is stored under `benchmarks/data/`, which is ignored by git.

## Run

```julia
julia --project=. benchmarks/mixed_data.jl
```

Useful options:

```julia
julia --project=. benchmarks/mixed_data.jl synthetic credit --max-rows=1000 --fit-iters=10
julia --project=. benchmarks/mixed_data.jl adult --max-rows=2000 --alpha=0.2 --fit-iters=5 --fit-lr=0.01 --seed=7
```

The output reports rows/columns, structure learning time, node/leaf count, initial train/test cross entropy, optional post-fit cross entropy, and synthetic oracle cross entropy when available.

## Notes

These are smoke benchmarks, not a full evaluation suite. They are intended to catch regressions in mixed-data loading, missing-value handling, structure learning, and autodiff parameter training. The synthetic benchmark is the cleanest end-to-end likelihood check because its latent structure and support are controlled, so `oracle_*_ce` is known. Real UCI datasets report `NaN` for oracle CE because their true data-generating distributions are unknown.
