# Contributing to maxentgpu

Preserve unrelated changes, make the smallest coherent change, and record exact
verification commands and results. Never weaken a numerical tolerance or regenerate
golden files merely to make a check pass.

Files under `sources/` are synchronized, read-only reference material and must not be
edited, renamed, moved, or deleted.

## Development commands

No task runner is required. Run commands from the repository root.

| Purpose | Command |
|---|---|
| Install development dependencies | `Rscript -e 'install.packages(c("testthat", "roxygen2", "torch", "maxnet"))'` |
| Generate documentation | `Rscript -e 'roxygen2::roxygenise()'` |
| Build the documentation site | `Rscript tools/build-pkgdown.R .` |
| Check hand fixtures | `Rscript tools/check-hand-fixtures.R .` |
| Check reference fixture integrity | `Rscript tools/check-reference-fixtures.R .` |
| Run unit tests | `Rscript -e 'testthat::test_local()'` |
| Build source package | `R CMD build .` |
| Check source package | `R CMD check --as-cran maxentgpu_*.tar.gz` |
| Generate pinned maxnet fixtures | `Rscript tools/generate-maxnet-fixtures.R .` |
| Generate pinned Java fixtures | `Rscript tools/generate-java-maxent-fixtures.R /absolute/path/to/maxent.jar .` |
| Print backend capabilities | `Rscript -e 'print(maxentgpu::maxent_device_probe())'` |

Reference fixture regeneration is an explicit maintainer action. Review inputs,
versions, commands, output diffs, and provenance together. Do not commit `maxent.jar`
or opaque benchmark binaries.

## Accelerator testing

When accelerator support is tested, archive CPU/MPS results from the Apple Silicon
development machine and CPU/CUDA results from a clean checkout on the DGX Spark using
the same pinned manifest. Record skipped or unavailable backends explicitly.

## API documentation and website

Roxygen2 and pkgdown solve different parts of the documentation pipeline:

- roxygen2 turns comments beside R functions into the `man/*.Rd` files and
  `NAMESPACE` required by R and CRAN;
- pkgdown consumes `DESCRIPTION`, `README.md`, generated `man/*.Rd` files, and
  vignettes to build the HTML website.

Run roxygen2 before pkgdown. CI rejects stale generated API documentation, while the
`pkgdown` workflow builds pull requests and deploys successful `main`/release builds
to the `gh-pages` branch. Repository administrators must configure GitHub Pages to
serve that branch once.

## Versioning and releases

Use Conventional Commit subjects so automatic Semantic Versioning is predictable:

- `fix:`, `docs:`, `chore:`, `perf:`, and other maintenance types bump the patch;
- `feat:` bumps the minor version;
- a `BREAKING CHANGE:` footer bumps the major version.

Every non-skipped push to `main` calculates a version starting at `0.1.0`. The full
R-devel/release/oldrel Linux and release macOS/Windows matrix must pass first. The
release job then updates `DESCRIPTION`, rebuilds and checks the exact source tarball,
commits the version with `[skip ci]`, tags that commit, and creates a GitHub release
with the tarball attached. The repository must allow GitHub Actions write access to
contents and allow the release job to update `main` under its branch-protection rules.

Automatic GitHub releases are not automatic CRAN submissions. Review
`CRAN-READINESS.md` and current CRAN policy before submitting a release manually.
