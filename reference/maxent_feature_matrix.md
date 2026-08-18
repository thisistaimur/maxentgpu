# Apply a feature specification in bounded row chunks

Apply a feature specification in bounded row chunks

## Usage

``` r
maxent_feature_matrix(spec, newdata, chunk_size = NULL)
```

## Arguments

- spec:

  A `maxent_feature_spec` object.

- newdata:

  Numeric or categorical predictor data accepted by `spec`.

- chunk_size:

  Optional positive number of rows per chunk.

## Value

A numeric feature matrix.
