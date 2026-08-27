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

## Shared-design batched solver checkpoint

The experimental `control$batch_solver = "torch"` path solves all species sharing a
linear feature design in one Torch optimization. The comparison below uses the same
JUWELS A100, Torch 0.17.0, and workload on commit `1174fa03ef42fdc0620bbe230635a4fb1227baf3`:

- Workload: 64 species, 10,000 background rows, 128 presences/species
- Repeats: 3; maximum iterations: 2,000; tolerance: `1e-6`
- Regularization: L2 only (`lambda1 = 0`, `lambda2 = 0.4`)
- Parity tolerance: `1e-4`

```text
shared_batched_cuda_median_seconds: 3.070
independent_scalar_cuda_median_seconds: 11.447
scalar_cpu_median_seconds: 9.453
max_abs_difference: 6.216432e-05
batch_speedup_over_scalar_cuda: 3.728338
shared_cuda_speedup_over_scalar_cpu: 2.892182
```

All species converged and the maximum prediction difference was within tolerance.
This supports a batching-throughput claim for the shared-design linear/L2 scenario;
it is not a claim that every scalar CUDA workload is faster than CPU.
