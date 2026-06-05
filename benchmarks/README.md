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
julia --project=. benchmarks/mixed_data.jl adult --max-rows=2000 --alpha=0.2 --seed=7
```

The output reports rows/columns, structure learning time, node/leaf count, initial train/test mean log likelihood, and optional post-fit train/test mean log likelihood.

## Notes

These are smoke benchmarks, not a full evaluation suite. They are intended to catch regressions in mixed-data loading, missing-value handling, structure learning, and autodiff parameter training. Some real-data likelihoods can be `Inf` or `NaN` with the current distribution heuristics; the synthetic benchmark is the cleanest end-to-end likelihood check.
