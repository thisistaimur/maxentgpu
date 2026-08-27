# Fit independent species models from a validated batch specification

This batch API preserves independent-model semantics and stable species
ordering. Set `control$batch_solver = "torch"` to use the experimental
shared-design batched L2 solver; unsupported inputs use the scalar path
unless explicitly requested, in which case a clear error is returned.

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
