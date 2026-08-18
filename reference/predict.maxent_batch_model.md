# Predict from an independent-species batch model

Predict from an independent-species batch model

## Usage

``` r
# S3 method for class 'maxent_batch_model'
predict(
  object,
  newdata,
  type = c("raw", "link"),
  species = NULL,
  device = "cpu",
  dtype = "float64",
  batch_size = NULL,
  ...
)
```

## Arguments

- object:

  A fitted `maxent_batch_model` object.

- newdata:

  Predictor data accepted by
  [`predict.maxent_fit()`](https://thisistaimur.github.io/maxentgpu/reference/predict.maxent_fit.md).

- type:

  Prediction scale passed to the scalar models.

- species:

  Optional character IDs to select and order species.

- device:

  Execution device (`"cpu"`, `"cuda"`, or `"mps"`). Torch execution is
  used for dense shared-design prediction.

- dtype:

  Torch dtype (`"float64"` or `"float32"`).

- batch_size:

  Maximum number of species materialized per prediction chunk.

## Value

A matrix with rows corresponding to `newdata` and columns keyed by
species ID.
