# Extract one fitted species model from a batch

Extraction returns the original `maxent_fit` object for the requested
species; it does not refit or copy training data beyond the model object
itself.

## Usage

``` r
maxent_batch_extract(object, species)
```

## Arguments

- object:

  A fitted `maxent_batch_model` object.

- species:

  A single species ID in `object$species`.

## Value

A fitted `maxent_fit` object.
