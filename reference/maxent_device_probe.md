# Probe Torch accelerator availability

Reports backend availability without requiring `torch` to be installed
and without failing package load on CPU-only systems. A backend is
marked available only when an actual tensor smoke operation succeeds.

## Usage

``` r
maxent_device_probe()
```

## Value

A data frame with one row per backend.
