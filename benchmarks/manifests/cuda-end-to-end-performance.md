# CUDA end-to-end fit-plus-predict performance checkpoint

This manifest contains historical same-Torch CPU/CUDA measurements and the current
analytic-CPU versus shared-design-Torch-CUDA package-route comparison. Every accepted
measurement includes fitting and prediction, convergence checks, and parity checks.

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

## Per-species convergence-mask checkpoint

The same workload was rerun after adding per-species convergence masks on commit
`e664427abd766075c20c213d695260f275cea3ce`, using the JUWELS Booster A100 and Torch
0.17.0. Both benchmark modes used three repeats, 2,000 maximum iterations, tolerance
`1e-6`, and diagnostic interval 25. CUDA batch predictions were checked against the
scalar CPU result with tolerance `1e-4`.

```text
shared_batched_cuda_median_seconds: 2.959
shared_batched_cuda_min_seconds: 2.956
shared_batched_cuda_max_seconds: 3.082
shared_batch_cpu_median_seconds: 9.176
shared_batch_max_abs_difference: 6.256e-05

independent_scalar_cuda_median_seconds: 11.627
independent_scalar_cuda_min_seconds: 11.518
independent_scalar_cuda_max_seconds: 11.700
independent_scalar_cpu_median_seconds: 9.423
independent_scalar_max_abs_difference: 5.884182e-15
batch_speedup_over_scalar_cuda: 3.929
batch_speedup_over_scalar_cpu: 3.101
```

All species converged. The batched CUDA path remained within the parity tolerance and
was approximately 3.93x faster than independent scalar CUDA on this workload. Scalar
CUDA remained slower than scalar CPU. These measurements support the same narrowly
scoped shared-design batching claim; they do not establish performance for other
feature classes, penalties, accelerators, or raster workloads.

## CPU Torch scalar convergence follow-up

The CPU-versus-CUDA scalar Torch timing above is a historical checkpoint, not a
currently reproducible comparison. On JUWELS commit `a2f72ab`, the 64-species CPU
Torch scalar run failed its convergence gate: `species_023` reached 2,000 iterations
with `optimality_norm = 2.314621e-6` at tolerance `1e-6`. Reproducing that species
alone showed the CPU analytic solver converged in 20 iterations (`optimality_norm =
2.990195e-7`), while CPU Torch did not converge within 2,000 iterations. Changing
the starting step from 1 to 0.01 or 0.001 did not resolve the Torch result.

The end-to-end benchmark now compares the package's analytic scalar CPU route with
the shared-design Torch CUDA route. This is an end-to-end package-route comparison,
not an isolated device-only comparison. The CPU Torch scalar convergence issue
remains a separate solver investigation.

## Current analytic CPU versus shared-design CUDA checkpoint

Run on the JUWELS Booster A100 using commit `5a525c83cb953724eac199ab96e0444ec3ba7efa`,
Torch 0.17.0, and the same 64-species linear/L2 workload described above. The CPU
route used the analytic solver; CUDA used the shared-design Torch batch solver.

```text
repeats: 3
max_iter: 2000
fit_tol: 1e-6
parity_tolerance: 1e-4
max_abs_difference: 1.225353e-06
cuda_median_seconds: 3.085
cuda_min_seconds: 3.022
cuda_max_seconds: 3.111
cpu_median_seconds: 2.240
cpu_min_seconds: 2.192
cpu_max_seconds: 2.271
cpu_converged_species: 64 / 64
cpu_median_iterations: 19.5
speedup_cpu_over_cuda: 0.726094
```

All 64 CPU fits converged and CUDA predictions passed parity. On this workload, the
analytic CPU route was about 1.38x faster than the shared-design CUDA route. Thus this
checkpoint demonstrates numerical agreement and batching throughput, but not an
end-to-end GPU speedup over the package's analytic CPU route. Conclusions remain
limited to this workload and hardware.

## Diagnostic-interval and initial scaling follow-up

The stage-level benchmark was run on the A100 with 128 species, 10,000 background
rows, 128 presences/species, and the analytic CPU route. The login-node revision was
`20ce4a5f4b552ad2edb01856ed434da4d6e15a74`; Git was unavailable on the compute node,
so the script printed `commit: unavailable`. All four runs converged for all species
and passed the `1e-4` prediction-parity gate.

| CUDA diagnostic interval | CUDA fit (s) | CUDA predict (s) | CUDA end-to-end (s) | CUDA median iterations/species | CPU end-to-end (s) | Max abs. difference |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 5.778 | 0.100 | 5.884 | 48 | 4.275 | 1.512684e-06 |
| 5  | 5.759 | 0.097 | 5.848 | 50 | 4.200 | 1.416190e-06 |
| 10 | 5.684 | 0.138 | 5.866 | 50 | 4.221 | 1.229037e-06 |
| 25 | 5.726 | 0.097 | 5.831 | 50 | 4.110 | 1.229037e-06 |

Checking convergence every iteration reduced the CUDA iteration count only modestly
and did not reduce elapsed time; the runtime variation across intervals was small
relative to the CPU/CUDA gap. At all intervals the analytic CPU route was about
1.38–1.42x faster end-to-end. Initial scaling at 64, 128, and 256 species with 10,000
background rows likewise showed no break-even for the tested two-predictor design.

The current implementation uses a fixed shared-solver step (default `0.25`), whereas
the scalar analytic CPU solver uses objective backtracking. The step-size sweep below
tests whether a larger stable CUDA step can reduce the iteration gap before expanding
to more predictors and large raster-prediction workloads.

## Shared-solver step-size sweep

On the same JUWELS A100 128-species workload and revision
`0318edbd138afebe74e08c29fa3f4fede66878d8`, the CUDA shared-design step was swept with
the diagnostic interval fixed at 5. All 128 species converged in every run; all
predictions passed the `1e-4` parity gate.

| CUDA step | CUDA fit (s) | CUDA predict (s) | CUDA end-to-end (s) | CUDA median iterations/species | CPU end-to-end (s) | CPU median iterations/species | Max abs. difference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 5.737 | 0.096 | 5.824 | 50 | 4.157 | 19.5 | 1.416190e-06 |
| 0.35 | 4.758 | 0.085 | 4.843 | 35 | 4.256 | 19.5 | 1.296294e-06 |
| 0.50 | 4.055 | 0.095 | 4.152 | 25 | 4.207 | 19.5 | 1.243862e-06 |

Increasing the CUDA step improved fit time and reduced iterations without harming
convergence or parity. At step `0.5`, end-to-end medians were nearly tied: CUDA 4.152 s
and CPU 4.207 s. Their observed timing ranges overlap, so this is not evidence of a
reliable GPU speedup. Keep the package default unchanged until the larger-workload and
robustness checks are complete.
