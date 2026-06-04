# TODO

## Parameter Learning
- Consider structured validation error types if the training API becomes public beyond `ArgumentError`.
- Add minibatching and validation support.
- Add optimizer controls beyond a fixed Adam learning rate, including optimizer selection, early stopping, callbacks, and history metadata.
- Add regularization and priors for sum weights and leaf parameters.
- Add an AD-safe NegativeBinomial CDF/CCDF path for large finite intervals without enumeration.
- Decide semantics for finite set-valued continuous observations; current training support treats intervals as the continuous censored-data path.
- Add support for more `Distributions.jl` leaf types or a documented extension interface for custom leaves.
- Add serialization for trained parameter vectors and `ParamMap`.

## Structure Learning
- Review the distribution fitting heuristic for integer data, especially edge cases with zero variance and overdispersion.
- Add deterministic controls for all random structure-learning paths.

## API And Maintenance
- Separate public API from internal helpers and audit exports.
- Add docstrings for the stable public API.
- Add examples for interval and set-valued parameter learning.
- Add CI coverage for the README examples.
