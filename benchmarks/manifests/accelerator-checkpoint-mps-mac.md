# MPS accelerator checkpoint — Apple Silicon development Mac

Status: **pending; MPS unavailable in the installed Torch runtime**

## Provenance

- Date: 2026-08-18
- Commit: `092df22b85fe5253a1e82951e8333ebcef2c42ff`
- Host: Apple Silicon Mac (`arm64`)
- macOS kernel: Darwin 25.5.0
- R: 4.6.1 (`aarch64-apple-darwin23`)
- maxentgpu: 0.1.1
- torch for R: 0.17.0
- Torch runtime: Apple Silicon LibTorch 2.8.0 and Lantern 0.17.0 arm64

## Device probe

```text
  backend available torch_version                         detail
1     cpu      TRUE        0.17.0  tensor smoke operation passed
2    cuda     FALSE        0.17.0  torch reports CUDA unavailable
3     mps     FALSE        0.17.0  torch reports MPS unavailable
```

## Test command and result

Command:

```bash
Rscript -e 'testthat::test_local()'
```

Result: **89 passed, 0 failed, 2 skipped**. The skips were the expected CUDA
unavailable checks; MPS was also unavailable, so no MPS parity assertions ran.
Runtime: 2.2 seconds.

## Interpretation

The Apple Silicon LibTorch and Lantern archives installed successfully, but the
runtime does not expose MPS on this host. This is a pending portability checkpoint,
not a green MPS result. Gate 3 remains open until an Apple Silicon environment with
`mps` reported available can run the parity manifest.
