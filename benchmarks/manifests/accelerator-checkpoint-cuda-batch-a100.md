# CUDA batch checkpoint — JUWELS Booster A100

Status: **passed for the shared-design prediction smoke case**

## Provenance

- Date: 2026-08-18
- Commit: `915f889af65c557a80978d61e90df1512e20b8b9`
- GPU: NVIDIA A100-SXM4-40GB
- Torch for R: 0.17.0
- CUDA runtime: 12.8 (`cu128`)
- CUDA device probe: available; tensor smoke operation passed

## Batch fit and prediction

Two named species (`sparrow`, `finch`) were fitted with the Torch CUDA engine. Both
converged with `stop_reason = parameter_change`. Dense batch prediction used the CUDA
device with `batch_size = 1` and was compared against independent scalar predictions.

```text
        sparrow finch
[1,] -0.3928431     0
[2,] -1.1785294     0
[3,] -1.9642157     0
[4,] -2.7499020     0
```

The scalar result was identical and the reported maximum absolute difference was:

```text
0
```

## Test suite

Command:

```bash
Rscript -e 'testthat::test_local()'
```

Result: **105 passed, 0 failed, 2 skipped**. The two skips were expected MPS
unavailable checks. Runtime: 5.7 seconds.

## Scope

This validates CUDA execution and scalar equivalence for the shared-design batch
prediction path. It is not yet a throughput claim or validation of dense batched
fitting across heterogeneous feature specifications.
