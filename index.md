# maxentgpu

[![CI /
Release](https://github.com/thisistaimur/maxentgpu/actions/workflows/ci-release.yaml/badge.svg)](https://github.com/thisistaimur/maxentgpu/actions/workflows/ci-release.yaml)
[![pkgdown](https://github.com/thisistaimur/maxentgpu/actions/workflows/pkgdown.yaml/badge.svg)](https://github.com/thisistaimur/maxentgpu/actions/workflows/pkgdown.yaml)

`maxentgpu` is an experimental R package for auditable maximum-entropy
presence-background models implemented with Torch operations. The
project is in Phase 0: its mathematical and reference contracts are
being made executable before model fitting or performance work begins.

The only current user-facing operation is a backend diagnostic:

``` r

maxentgpu::maxent_device_probe()
```

CUDA, MPS, batching, raster prediction, and claims of Java MaxEnt
compatibility are not yet released features. See `PLAN.md` and
`inst/spec/` for the staged contract.

Development documentation is published at
<https://thisistaimur.github.io/maxentgpu/>. GitHub releases are
generated from successful `main` builds; CRAN submission remains a
separate maintainer-reviewed process.
