# Fit independent species models from a validated batch specification

This batch API preserves independent-model semantics, stable species
ordering, and per-species convergence diagnostics. Set
`control$batch_solver = "torch"` to use the experimental shared-design
batched L2 solver. Presence weights can vary by species. Converged
species stop updating independently; with this solver
`regularization$lambda2` may be a scalar or a species-named vector.
Unsupported inputs return a clear error.

## Usage

``` r
maxent_fit_batch(
  x,
  species = NULL,
  background = NULL,
  presence_weights = NULL,
  ...,
  control = list()
)
```

## Arguments

- x:

  A named list of species records. Each record must contain `presence`
  and `background` tables, unless `background` is supplied separately,
  in which case each element is a presence table.

- species:

  Optional character species IDs. Required when `x` is unnamed.

- background:

  Optional shared background table.

- presence_weights:

  Optional list of positive per-row weight vectors, one per species.
  Names, when supplied, must match species IDs.

- ...:

  Arguments forwarded to
  [`maxent_fit()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_fit.md)
  except `control`.

- control:

  Solver controls forwarded to
  [`maxent_fit()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_fit.md).
  Set `batch_solver = "torch"` for the experimental shared-design
  batched solver.

## Value

An object of class `maxent_batch_model`.
