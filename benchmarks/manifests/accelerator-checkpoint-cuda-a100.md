# CUDA accelerator checkpoint — JUWELS Booster A100

Status: **passed for CUDA; MPS remains pending**

## Provenance

- Date: 2026-08-18
- Commit: `de89153fa4cfd5f11bafaa373f0747b020a8bab4`
- R: 4.4.2 (x86_64-pc-linux-gnu)
- maxentgpu: 0.1.2
- torch for R: 0.17.0
- Torch runtime: CUDA 12.8 (`cu128`)
- GPU: NVIDIA A100-SXM4-40GB
- NVIDIA driver: 595.71.05
- Driver CUDA capability: 13.2

## Device probe

```text
  backend available torch_version                        detail
1     cpu      TRUE        0.17.0 tensor smoke operation passed
2    cuda      TRUE        0.17.0 tensor smoke operation passed
3     mps     FALSE        0.17.0 torch reports MPS unavailable
```

## Test command and result

Command:

```bash
Rscript -e 'testthat::test_local()'
```

Result: **94 passed, 0 failed, 2 skipped**. Both skips were the expected MPS
unavailable checks in `tests/testthat/test-device-parity.R`. Runtime: 5.2 seconds.

## Hardware allocation

The checkpoint ran in a one-node interactive allocation with one GPU assigned.
All four A100 devices were idle in the final `nvidia-smi` report; the test process
used the allocated device only.

## Interpretation

This is a green CUDA scalar parity checkpoint for the recorded commit and software
stack. It does not substitute for the required Apple Silicon/MPS checkpoint, and it
does not establish cross-device equivalence beyond the tests listed above.
