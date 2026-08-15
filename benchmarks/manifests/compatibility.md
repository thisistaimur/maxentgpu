# Compatibility policy

Phase 0 development targets R >= 4.3.0. The initial pinned validation stack is:

| Component | Pinned validation version | Policy |
|---|---:|---|
| R | 4.6.1 locally | CI also checks the current supported R release; minimum stays 4.3 until tested otherwise |
| torch for R | 0.17.0 | exact version in reference/accelerator manifests; package may install newer compatible versions only after parity checks |
| LibTorch | 2.8.0 bundle selected by torch 0.17.0 | record the Torch package version and the native runtime version reported by the device manifest at every accelerator checkpoint |
| terra | 1.9-34 | Suggests-only until raster Phase 5 |
| maxnet | 0.1.4 | exact reference pin |
| Java | 17 LTS reference runtime | record vendor/build; the local Java 26 installation is not the reference runtime |
| Java MaxEnt | 3.4.4 | exact reference pin; jar never committed |
| virtualspecies | 1.6.1 | Suggests-only benchmark generator in Phase 6 |

CPU is required. MPS is supported only on tested Apple Silicon combinations and is
initially float32. CUDA is supported only on tested Linux/LibTorch/CUDA combinations.
Automatic device selection may fall back; an explicitly requested unsupported backend
must fail clearly. The Phase 3 portability checkpoint archives separate MPS and DGX
Spark CUDA manifests, and neither result substitutes for the other.

Pins describe reproducible validation, not permanent dependency upper bounds. Updating
a pin requires regenerating affected fixtures, recording the old and new versions,
and reviewing numerical drift before merging.

## Known development-environment findings

- **ENV-MPS-001 (open, 2026-08-15):** Torch 0.17.0 with its LibTorch 2.8.0 arm64
  bundle executes CPU tensors on the Apple M1 Pro development host, but
  `backends_mps_is_available()` returns false on macOS 26.5.1. A direct MPS allocation
  fails in LibTorch's `empty_mps` OS-version check. Do not mark MPS validation green
  until a supported bundle or upstream fix passes a real tensor smoke operation.
