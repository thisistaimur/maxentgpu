# Fit independent species models from a validated batch specification

This first batch API preserves independent-model semantics and stable
species ordering. Each species is currently fitted through the scalar
solver; the `execution` field records this explicitly until dense tensor
batching is added.

## Usage

``` r
maxent_fit_batch(x, species = NULL, background = NULL, ..., control = list())
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

- ...:

  Arguments forwarded to
  [`maxent_fit()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_fit.md)
  except `control`.

- control:

  Solver controls forwarded to
  [`maxent_fit()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_fit.md).

## Value

An object of class `maxent_batch_model`.
