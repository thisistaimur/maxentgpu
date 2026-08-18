# Predict from a fitted scalar maximum-entropy model

Predict from a fitted scalar maximum-entropy model

## Usage

``` r
# S3 method for class 'maxent_fit'
predict(object, newdata, type = c("raw", "link"), ...)
```

## Arguments

- object:

  A fitted `maxent_fit` object.

- newdata:

  Numeric predictor data.

- type:

  Prediction scale: `"link"` or `"raw"`.

- ...:

  Reserved for future device and batching controls.

## Value

A numeric prediction vector.
