# CUDA end-to-end fit-plus-predict performance checkpoint

This benchmark compares the same Torch engine on CPU and CUDA. It includes fitting
and prediction, uses identical controls, and requires every species to converge.

## Provenance

- Commit: `c9ff9194c0069c72cda8e4fd7c1932019958e867`
- GPU: JUWELS Booster NVIDIA A100
- Torch for R: 0.17.0
- Workload: 32 species, 5,000 background rows, 64 presences/species
- Repeats: 3
- Maximum iterations: 1,000

## Results

```text
max_abs_difference: 5.107026e-15
cuda_median_seconds: 7.818
cuda_min_seconds: 7.529
cuda_max_seconds: 8.063
cpu_median_seconds: 5.867
cpu_min_seconds: 5.710
cpu_max_seconds: 5.883
speedup_cpu_over_cuda: 0.7504477
```

All species converged. Predictions agree to near float64 roundoff. For this workload,
CPU Torch was approximately 1.33x faster than CUDA (CUDA was about 33% slower).

## Interpretation

The CUDA route is numerically correct but does not provide a performance advantage at
this workload size. The current implementation fits species independently and
synchronizes through R each iteration, so this is a correctness checkpoint rather
than evidence for GPU acceleration claims.
