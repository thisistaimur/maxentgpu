# Fit a scalar CPU maximum-entropy model

Fit a scalar CPU maximum-entropy model

## Usage

``` r
maxent_fit(
  x,
  presence,
  background = NULL,
  presence_weights = NULL,
  background_weights = NULL,
  features = c("linear", "quadratic"),
  thresholds = NULL,
  knots = NULL,
  regularization = list(lambda1 = 0, lambda2 = 1),
  control = list(max_iter = 2000L, tol = 1e-08, step = 1)
)
```

## Arguments

- x:

  Numeric presence predictors when `background` is supplied, or a
  combined table when `background` is `NULL`.

- presence:

  Logical presence indicator aligned with `x`, or a numeric presence
  predictor table when `background` is supplied.

- background:

  Optional numeric background predictor table.

- presence_weights:

  Optional positive presence weights.

- background_weights:

  Optional positive background weights.

- features:

  Feature classes: `"linear"`, `"quadratic"`, `"product"`,
  `"threshold"`, `"hinge"`, or categorical-only models.

- thresholds:

  Optional named list of numeric threshold values by predictor.

- knots:

  Optional named list of numeric hinge knots by predictor.

- regularization:

  A list with non-negative `lambda1` and `lambda2`, and optional
  feature-specific non-negative `penalty_l1` and `penalty_l2` vectors.

- control:

  A list with `max_iter`, `tol`, `step`, optional `device`, `dtype`,
  `engine` (`"analytic"` or `"torch"`), and `accelerated`.

## Value

An object of class `maxent_fit`.
