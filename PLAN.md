# MaxEnt-GPU (`maxentgpu`) implementation plan

## 0. Purpose and operating contract

This document is the execution plan for building **MaxEnt-GPU**, an R package named
`maxentgpu`. The initial implementation uses [`torch` for R](https://torch.mlverse.org/)
and LibTorch for tensor computation on CPU and supported accelerators: NVIDIA CUDA and
Apple Metal Performance Shaders (MPS). Backend availability and numerical capability
are detected at runtime. The initial package does **not** claim to contain custom CUDA
kernels. A native C++/CUDA extension is a later,
optional optimization and is permitted only after profiling demonstrates that
ordinary Torch operations cannot meet an agreed performance target.

The package must provide:

- mathematically specified maximum-entropy presence-background models;
- auditable feature construction and regularization;
- fitting and prediction on CPU, CUDA, and MPS where supported;
- numerically equivalent scalar and batched independent-species workflows;
- memory-bounded prediction for `terra::SpatRaster` inputs;
- reproducible correctness and performance benchmarks;
- research-software quality suitable for JOSS;
- a controlled simulation and scaling study suitable for a Methods in Ecology and
  Evolution (MEE) submission.

### Instructions for Codex (GPT-5.6 Sol)

Work one phase at a time. At the start of each phase:

1. Read this file, `AGENTS.md`, `DESCRIPTION`, and all files changed in the prior phase.
2. Inspect the working tree and preserve unrelated user changes.
3. Convert the phase tasks into a short working plan.
4. Implement the smallest coherent vertical slice.
5. Run the phase checks and record the exact commands and results in the handoff.
6. Update the checkboxes in this file only for work actually verified.
7. Stop at every **STOP/GO gate**. Do not silently weaken a tolerance or skip a
   reference comparison to pass a gate.

If an interface or mathematical convention is uncertain, write a minimal executable
fixture against the reference implementation before choosing it. Treat Java MaxEnt
and `maxnet` output as comparison targets, not as an undocumented specification.
Document intentional differences explicitly.

### Non-goals for the first stable release

- Custom `.cu` kernels, CUDA Graphs, multi-GPU execution, or distributed fitting.
- Joint, hierarchical, or interacting multispecies models. A batch is a collection
  of independent models, not a community model.
- Automatic occurrence cleaning, pseudoabsence methodology, spatial cross-validation,
  or ecological interpretation.
- Exact bitwise identity across devices or accelerator backends.
- Compatibility with every historical Java MaxEnt option.
- A claim that faster computation alone improves ecological inference.

## 1. Definition of done

Version `0.1.0` is complete only when all of the following hold:

- `R CMD check --as-cran` passes on supported CPU platforms with no errors or warnings
  attributable to the package; justified notes are documented.
- The mathematical specification is versioned in `inst/spec/` and linked from the
  developer vignette.
- Linear, quadratic, product, hinge, threshold, and categorical feature classes are
  supported, with fitted transform metadata stored in the model.
- A model fitted on CPU can predict on CUDA and vice versa after serialization.
- CPU/accelerator fits and predictions meet the declared backend/dtype tolerances.
- Batched results agree with fitting the same species one at a time.
- `SpatRaster` prediction is chunked, preserves geometry, handles `NA`, and never
  requires loading the full raster or full prediction cube into RAM/GPU memory.
- Reference fixtures cover `maxnet` and a pinned Java MaxEnt release, including
  documented mappings and known non-equivalences.
- Deterministic, compact `virtualspecies` scenarios are regenerated from configuration
  and seeds; benchmark data are not committed as giant opaque binaries.
- CI separates fast CPU checks, optional/reference checks, and CUDA checks.
- Documentation includes installation, a first model, raster projection, batching,
  output semantics, reproducibility, limitations, and troubleshooting.
- A tagged release, archived DOI, `CITATION.cff`, license, governance files, JOSS
  `paper.md`, and reproducible benchmark manifests exist.

## 2. Scientific and numerical specification

### 2.1 Canonical data model

For species \(s\), let \(P_s\) be presence samples and \(B_s\) be the background
domain. A fitted feature map \(\phi(x) \in \mathbb{R}^{F_s}\) converts predictors to
features. Background quadrature weights \(q_{sj} > 0\) are normalized to sum to one.
The unnormalized score is:

\[
  z_s(x) = \beta_s^\top \phi(x).
\]

The canonical penalized presence-background loss is:

\[
  \mathcal{L}_s(\beta_s) =
  \log\!\left(\sum_{j \in B_s}q_{sj}\exp(z_s(x_j))\right)
  - \sum_{i \in P_s}w_{si}z_s(x_i)
  + \Omega_s(\beta_s),
\]

where normalized presence weights satisfy \(\sum_i w_{si}=1\). The first term must
use a stable weighted `logsumexp`. The initial regularizer is separable:

\[
  \Omega_s(\beta_s)=
  \lambda_s\sum_k r_{sk}|\beta_{sk}|
  + \frac{\lambda_{2,s}}{2}\sum_k r^{(2)}_{sk}\beta_{sk}^{2}.
\]

The implementation must support the L1 form needed for MaxEnt-style feature-specific
regularization. L2 is included only if it can be specified and tested without
compromising the reference path. Do not represent the normalizing constant as a
freely fitted intercept: a constant feature is non-identifiable in the canonical
objective.

### 2.2 Output transforms

Every prediction must name its scale. At minimum:

- `link`: \(z(x)=\beta^\top\phi(x)\);
- `raw`: \(\exp(z(x)-\log Z)\), where
  \(\log Z=\log\sum_j q_j\exp(z(x_j))\);
- `cloglog`: computed from the model's documented entropy/normalization convention;
- `logistic`: provided only after the Java/`maxnet` convention and entropy offset are
  reproduced and tested.

The exact formulas, background-weight convention, entropy term, clamping behavior,
and extrapolation behavior belong in `inst/spec/prediction-scales.md`. Never label a
generic sigmoid as Java MaxEnt's `logistic` output.

### 2.3 Feature contract

Feature fitting and feature application are distinct operations:

```text
fit_feature_spec(training predictors, classes, options) -> immutable spec
apply_feature_spec(spec, new predictors, clamp/extrapolate policy) -> matrix/tensor
```

The immutable spec stores predictor names and order, numeric ranges, categorical
levels, knots/thresholds, scaling constants, feature names, source variables,
regularization multipliers, and a schema version. It must be possible to reconstruct
every model-matrix column from the spec alone.

Implement in this order:

1. **Linear (L):** numeric predictor.
2. **Quadratic (Q):** squared numeric predictor with the documented scaling order.
3. **Product (P):** pairwise products, deterministic column ordering, no duplicates.
4. **Threshold (T):** indicator basis at stored threshold/knots.
5. **Hinge (H):** forward and reverse hinge bases at stored knots.
6. **Categorical (C):** explicit stored levels and contrast/reference convention;
   unseen levels error by default and can optionally map to `NA`.

Feature defaults depending on sample size must live in one policy function and be
tested at every boundary. User-specified classes always override defaults.

### 2.4 Solver contract

Use a pure-Torch solver so the same implementation runs on CPU and supported
accelerators. The proposed
baseline is monotone proximal gradient/FISTA with backtracking for the smooth
log-partition loss and feature-weighted soft thresholding for L1. Requirements:

- per-species convergence state in batched fits;
- stable log-sum-exp and finite-value checks;
- objective, smooth gradient norm, parameter change, iteration count, and stop reason;
- maximum-iteration and non-convergence warnings;
- no convergence decision based only on float32 noise;
- deterministic initialization (zeros by default);
- a slow finite-difference gradient oracle for tiny tests;
- an optional unaccelerated proximal-gradient mode to diagnose FISTA behavior.

`torch` autograd may be used as the initial gradient oracle and implementation. Before
optimizing or deriving a manual gradient, test it against autograd and finite
differences. Keep optimizer state on the selected device and avoid per-iteration
device-to-host synchronization except at configurable diagnostic intervals.

### 2.5 Precision policy

- `float64` on CPU is the reference/correctness default during development; CUDA uses
  float64 where supported by the selected device.
- `float32` is an explicit performance mode with separately reported tolerances.
- MPS initially uses float32 because float64 is not generally available on that
  backend; unsupported device/dtype combinations must fail clearly or choose a
  documented fallback only when `device = "auto"`.
- Mixed precision is out of scope until the float32 path is proven stable.
- Serialize the training dtype; prediction may permit an explicit dtype conversion
  with a warning or audit entry.
- Report deterministic-algorithm settings and known nondeterministic CUDA operations.

Initial tolerances, to be calibrated from fixtures rather than loosened ad hoc:

| Comparison | float64 | float32 |
|---|---:|---:|
| feature matrices CPU vs accelerator | `rtol 1e-10`, `atol 1e-12` | `rtol 2e-5`, `atol 2e-6` |
| objective/gradient CPU vs accelerator | `rtol 1e-8`, `atol 1e-10` | `rtol 1e-4`, `atol 1e-5` |
| predictions CPU vs accelerator | `rtol 1e-7`, `atol 1e-9` | `rtol 2e-4`, `atol 2e-5` |
| scalar vs batched predictions | same as device tolerance | same as device tolerance |

Reference comparisons with `maxnet` and Java MaxEnt use fixture-specific tolerances
because optimizer, regularization, and output conventions can legitimately differ.
The expected differences must be decided before inspecting benchmark performance.

## 3. Package architecture

### 3.1 Layering

Keep dependencies directed downward:

```text
User API / S3 methods
  -> validation and model objects
  -> feature specification
  -> objective + regularization + solver
  -> scalar/batched tensor execution
  -> device, dtype, chunking, serialization

terra adapter -> prediction API (core must not depend on raster internals)
benchmarks     -> public API (never private internals unless explicitly profiling)
reference      -> isolated adapters for maxnet and Java MaxEnt
```

The core fitting API accepts tabular/matrix data. `terra` is an adapter, not the
internal data representation. No R external pointer or live device tensor is the sole
source of truth in a saved model.

### 3.2 Proposed repository structure

```text
maxent-gpu/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   │   ├── R-CMD-check.yaml
│   │   ├── test-coverage.yaml
│   │   ├── reference-tests.yaml
│   │   └── cuda-tests.yaml
│   └── dependabot.yml
├── R/
│   ├── aaa-package.R
│   ├── device.R
│   ├── validate.R
│   ├── feature-spec.R
│   ├── features-numeric.R
│   ├── features-categorical.R
│   ├── regularization.R
│   ├── objective.R
│   ├── solver-prox.R
│   ├── batch-layout.R
│   ├── fit.R
│   ├── predict.R
│   ├── predict-terra.R
│   ├── model.R
│   ├── diagnostics.R
│   ├── serialize.R
│   └── print.R
├── inst/
│   ├── extdata/
│   ├── schema/maxentgpu-model.schema.json
│   └── spec/
│       ├── objective.md
│       ├── features.md
│       ├── prediction-scales.md
│       └── reference-mapping.md
├── tests/testthat/
│   ├── helper-fixtures.R
│   ├── helper-tolerances.R
│   ├── test-features-*.R
│   ├── test-objective.R
│   ├── test-gradients.R
│   ├── test-fit.R
│   ├── test-predict.R
│   ├── test-batch.R
│   ├── test-terra.R
│   ├── test-serialization.R
│   ├── test-cpu-cuda.R
│   ├── test-reference-maxnet.R
│   └── test-reference-java.R
├── tests/fixtures/
│   ├── generated/
│   ├── maxnet/
│   └── java-maxent/
├── vignettes/
│   ├── getting-started.Rmd
│   ├── multispecies-batching.Rmd
│   ├── raster-prediction.Rmd
│   └── numerical-fidelity.Rmd
├── benchmarks/
│   ├── README.md
│   ├── config/
│   ├── R/
│   ├── scripts/
│   ├── manifests/
│   ├── results/.gitignore
│   └── reports/
├── paper/
│   ├── paper.md
│   └── paper.bib
├── man/
├── _pkgdown.yml
├── DESCRIPTION
├── NAMESPACE
├── LICENSE.md
├── README.Rmd
├── README.md
├── NEWS.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── GOVERNANCE.md
├── codemeta.json
└── PLAN.md
```

Generated `man/`, `NAMESPACE`, and `README.md` are rebuilt by documented commands.
Large benchmark outputs and Java binaries are not committed. Store checksums, source
URLs, version/license metadata, scripts, and small derived fixtures.

### 3.3 Model objects

Use documented S3 objects initially; avoid R6 unless mutable state becomes necessary.

`maxentgpu_model` contains:

- coefficients on CPU as ordinary R arrays or CPU tensors safe to serialize;
- fitted feature spec and regularization policy;
- normalization/entropy constants and training predictor ranges;
- output-scale convention;
- solver diagnostics and convergence status;
- package/spec schema version, dtype, source device, and reproducibility metadata;
- compact training summaries, never full occurrence/background data by default.

`maxentgpu_batch_model` contains shared metadata plus species IDs, a coefficient
matrix, species-specific masks/constants/diagnostics, and enough indexing metadata to
extract one `maxentgpu_model` without refitting.

## 4. Public API sketch

Names may change once during Phase 1, then require deprecation rather than silent
breakage after the Phase 4 API-freeze gate.

```r
library(maxentgpu)

fit <- maxent_fit(
  x,
  presence,
  background = NULL,
  presence_weights = NULL,
  background_weights = NULL,
  features = c("linear", "quadratic", "hinge"),
  regularization = maxent_regularization(multiplier = 1),
  device = maxent_device("auto"),
  dtype = c("float64", "float32"),
  control = maxent_control()
)

predict(fit, newdata, type = c("cloglog", "logistic", "raw", "link"),
        clamp = TRUE, device = NULL, batch_size = NULL)

batch_fit <- maxent_fit_batch(
  x,
  species,
  background = NULL,
  presence_weights = NULL,
  features = "auto",
  regularization = maxent_regularization(multiplier = 1),
  device = maxent_device("cuda"),
  dtype = "float64",
  control = maxent_control()
)

predict(batch_fit, newdata, type = "cloglog", species = NULL,
        device = NULL, batch_size = NULL)

predict_raster(
  fit_or_batch,
  predictors,                  # terra::SpatRaster
  filename = "prediction.tif",
  type = "cloglog",
  species = NULL,
  device = maxent_device("auto"),
  chunk_ncells = "auto",
  overwrite = FALSE,
  wopt = list()
)

maxent_device("auto")
maxent_accelerators()
maxent_diagnostics(fit)
maxent_feature_spec(fit)
as.data.frame(maxent_coefficients(fit))
save_maxent_model(fit, "model.rds")
read_maxent_model("model.rds", device = "cpu")
```

Input conventions must be unambiguous:

- `x` is a data frame/matrix of environmental samples or a background table;
- `presence` is a logical/0-1 vector aligned with `x`, row indices, or a separate
  validated presence table; choose one canonical internal form;
- `species` is a long table with `species_id` and row/background mapping, with helper
  constructors for list input;
- predictor names and categorical columns are mandatory for safe raster prediction;
- background sampling is never performed implicitly during `maxent_fit()`.

## 5. Phased milestones and STOP/GO gates

### Phase 0 — Repository, decisions, and executable specification

**Goal:** create a checkable package skeleton and eliminate high-risk ambiguity before
performance code exists.

Tasks:

- [x] Initialize an R package with `DESCRIPTION`, testthat edition 3, roxygen2,
  `README.Rmd`, license, and basic CI.
- [x] Record supported R, `torch`, `terra`, `maxnet`, Java, and MaxEnt versions in a
  compatibility policy; pin exact reference versions in benchmark manifests.
- [x] Add a device probe that reports CPU, CUDA, and MPS availability/capabilities
  without failing package load on CPU-only machines.
- [x] Write the four files in `inst/spec/`.
- [x] Create tiny hand-calculated fixtures: one predictor, two predictors, weighted
  background, duplicated presences, constant predictor, `NA`, extreme logits.
- [x] Build scripts that generate reference outputs from `maxnet` and the pinned Java
  MaxEnt release. Record command lines, options, hashes, and licenses.
- [ ] Decide and document the precise mappings for feature scaling, knots, default
  class selection, regularization multipliers, clamping, raw/logistic/cloglog outputs,
  duplicate records, and background weights.
- [x] Add a `make`/`just`/R-script command table to `CONTRIBUTING.md`; do not require a
  specific task runner for package users.

Acceptance criteria:

- A clean checkout installs and runs one trivial test on CPU.
- Specs contain no placeholders for the canonical objective or output scales.
- Reference fixture generation is reproducible without committing `maxent.jar`.
- Every intentional divergence from `maxnet`/Java MaxEnt has a named issue or entry in
  `reference-mapping.md`.

**STOP/GO Gate 0:** GO only if a maintainer can explain, from the specs, what exact
model is being optimized and how each output scale is computed. If reference behavior
is unresolved, add focused fixtures; do not begin CUDA work.

### Phase 1 — Features, objective, and scalar CPU fit

**Goal:** obtain a correct, inspectable single-species implementation on CPU.

Tasks:

- [x] Implement validation and canonical internal data objects.
- [x] Implement immutable feature spec fitting/application for L and Q.
- [x] Implement weighted stable objective, autograd-validated gradient, penalty, and diagnostics.
- [x] Implement proximal-gradient/FISTA solver with backtracking and convergence tests.
- [x] Add finite-difference gradient and hand-calculated objective tests.
- [x] Implement `maxent_fit()`, `predict()`, print/summary/coefficient methods.
- [x] Implement link and raw outputs; enable logistic/cloglog only when their fixture
  comparisons pass.
- [ ] Add numeric guards for all-zero weights, zero variance, overflow/underflow,
  empty presence/background sets, and rank-deficient features.
- [ ] Compare unregularized and regularized L/LQ fixtures with references.

Acceptance criteria:

- Hand-computed objective and gradients pass declared float64 tolerances.
- Fitted objective is non-increasing in monotone mode.
- Predictions are invariant to safe row reordering and equivalent weight aggregation.
- Serialization round-trip preserves predictions and diagnostics.
- Reference L and LQ predictions meet predeclared fixture tolerances or have a precise,
  tested explanation for remaining differences.

**STOP/GO Gate 1:** GO only if scalar CPU correctness passes. If coefficients differ
but predictions agree, document identifiability/scaling; if predictions differ, stop
and resolve the mathematical cause.

#### Gate 1 paper record — reference regularization mapping

The pinned `maxnet` LQ fixture uses `regmult = 1` and feature penalty factors
`0.0389711432` for linear terms and `0.0566423814` for quadratic terms. Applying
those factors directly to the package objective with `lambda1 = regmult` and no L2
penalty converges in one iteration to an all-zero coefficient vector. This is
evidence that glmnet/maxnet and the package's independently normalized
presence-background objective use different penalty scales; it is not evidence that
the model classes are incompatible.

The comparison harness records this as an unresolved `REF-REG-001` mapping. The
paper must report the objective normalization, penalty convention, conversion method,
fixture versions, and any residual coefficient/prediction differences. Until a
conversion is derived and validated on independent L/LQ fixtures, do not describe
the package as maxnet-compatible and do not use the direct factors as defaults.

### Phase 2 — Complete MaxEnt feature fidelity

**Goal:** support the required feature classes without duplicating training and
prediction logic.

Tasks:

- [ ] Implement product features with deterministic naming/order.
- [ ] Implement threshold features and stored thresholds.
- [ ] Implement forward/reverse hinge features and stored knots.
- [ ] Implement categorical levels/contrasts and unseen-level policy.
- [ ] Implement auto-feature policy with boundary tests.
- [ ] Implement feature-specific regularization and expose an auditable coefficient
  table containing class, source predictors, knot/level, scale, and penalty.
- [ ] Test feature application in chunks and whole-matrix mode for identity.
- [ ] Add property tests for permutations, constant columns, duplicated knots, extreme
  ranges, missing values, and categorical level order.
- [ ] Expand `maxnet` and Java reference fixtures one feature class at a time, then in
  supported combinations.

Acceptance criteria:

- Training and prediction call the same feature-application functions.
- Stored specs reconstruct identical design matrices after save/load.
- Each feature column has a human-readable stable identifier.
- Reference prediction comparisons pass for every advertised feature class.
- Unsupported or intentionally different reference behaviors fail clearly rather than
  being silently approximated.

**STOP/GO Gate 2:** Do not advertise “MaxEnt compatible” until the reference matrix is
green. A subset release may proceed only if package/docs name the exact supported
subset.

### Phase 3 — CUDA/MPS parity for the scalar path

**Goal:** run the same scalar implementation on CUDA and MPS, subject to each backend's
supported dtypes and operations, with controlled numerical error.

Tasks:

- [ ] Centralize device/dtype movement and forbid scattered literal `cuda` calls.
- [ ] Move feature tensors, objective, optimizer state, and prediction to the selected
  device; keep serializable model state device-neutral.
- [ ] Add CPU/CUDA and CPU/MPS parity tests for feature matrices, objective, gradient,
  fits, predictions, clamping, serialization, and errors.
- [ ] Test extreme logits and memory errors with actionable messages.
- [ ] Measure and remove accidental host/device transfers in the iteration loop.
- [ ] Add synchronization only around benchmark timing and documented transfer points.
- [ ] Add CUDA and MPS smoke workflows on controlled runners; other CI jobs skip
  unavailable accelerator tests with explicit reasons.
- [ ] Run the accelerator portability checkpoint: use the Apple Silicon/MPS development
  machine for CPU/MPS parity, then run the same pinned scalar test manifest from a
  clean checkout on the DGX Spark for CPU/CUDA parity. Archive the exact commands,
  commit SHA, package/LibTorch versions, device capability report, dtype, and test
  results. Results from one backend do not substitute for the other.

Acceptance criteria:

- CUDA parity tests pass at both declared dtypes on at least one supported NVIDIA GPU;
  MPS parity tests pass at float32 on at least one supported Apple Silicon system.
- CPU-only installation, load, examples, and checks do not require CUDA.
- Saving an accelerator-trained model and loading/predicting on CPU works.
- Solver diagnostics expose device, dtype, and convergence without retaining GPU
  memory after model removal and garbage collection.

**STOP/GO Gate 3:** GO only if numerical parity is established before any batched
specialization. If float32 is unstable for a scenario, keep float64 as the supported
mode for that scenario and file a measured issue. Gate 3 requires a green, archived
MPS checkpoint on the development device and a green, archived CUDA checkpoint on the
DGX Spark; if either device is temporarily unavailable, record the gate as pending
rather than treating a skipped backend as passing.

### Phase 4 — GPU-batched independent-species fitting and prediction

**Goal:** make species count a tensor batch dimension while preserving independent
model semantics.

Initial layout for shared background/features:

```text
background features X_b:       [N_background, F]
species coefficients beta:     [S, F]
presence row indices/mask:      padded [S, N_presence_max] + mask
feature/penalty mask:           [S, F]
scores over background:         [S_chunk, N_background]
loss/convergence/step size:      [S]
```

Do not materialize `[S, N_background, F]`. Calculate scores with matrix operations and
chunk species/background dimensions under a memory budget. Where species use different
feature specs, group them by compatible spec first; use padded masks only when measured
to be beneficial.

Tasks:

- [ ] Define long/list batch inputs, validation, stable species ordering, and extraction
  of single models.
- [ ] Implement shared-design batching for equal background and feature specs.
- [ ] Implement species-specific presence indices/weights, penalties, convergence,
  early stopping masks, and failure reporting.
- [ ] Implement prediction as `X_new %*% t(beta)` in memory-bounded cell/species chunks.
- [ ] Add a planner using available memory minus a safety reserve; permit explicit
  overrides and log the chosen layout.
- [ ] Group heterogeneous species by feature spec and regularization shape; fall back to
  scalar execution for incompatible cases with a visible diagnostic.
- [ ] Ensure one failed/non-convergent species does not corrupt successful species.
- [ ] Add scalar-vs-batch metamorphic tests on CPU/CUDA/MPS, including reordered species,
  unequal presence counts, weights, missing species, and partial extraction.
- [ ] Establish API freeze and add lifecycle/deprecation policy.

Acceptance criteria:

- For every test species, extracted batched predictions match an independent scalar fit
  within device/dtype tolerance.
- Peak tensor allocation follows the planned chunks and does not grow as
  `S * N_background * F`.
- Batch output order is stable and keyed by species ID, never implicit list order alone.
- Per-species convergence and errors are available as a tidy diagnostics table.
- On representative workloads large enough to amortize transfers, batching improves
  throughput over scalar fits on each tested accelerator; report backend-specific
  results even where CPU remains faster.

**STOP/GO Gate 4:** If batching does not improve throughput, profile layout, launch,
transfer, and convergence divergence. Continue only with a credible batched advantage
or reposition the package as a correctness-first Torch implementation. Do not invent a
speedup by excluding transfer/setup costs.

### Phase 5 — `terra` raster integration

**Goal:** project one or many fitted models over rasters without violating raster or
memory semantics.

Tasks:

- [ ] Validate predictor layer names, types, categorical levels, geometry, CRS, extent,
  resolution, and expected feature inputs.
- [ ] Implement block reads from `SpatRaster`, data-frame conversion, feature transform,
  device transfer, prediction, and block writes.
- [ ] Preserve `NA` cell masks exactly; never replace missing predictors with zero.
- [ ] Support in-memory results for small outputs and mandatory filenames for unsafe
  output sizes.
- [ ] Chunk across raster cells and species using the memory planner.
- [ ] Preserve layer names/species IDs and geometry; write scale and model metadata where
  the format permits.
- [ ] Handle interruption and partial files safely; do not leave a valid-looking partial
  product.
- [ ] Test integer/categorical layers, on-disk rasters, `NA` borders/interiors, layer
  reordering, filenames, overwrite behavior, and tiny chunk sizes.
- [ ] Compare chunked raster values with tabular predictions for sampled cells.

Acceptance criteria:

- Whole-memory and block-wise predictions agree within tolerance.
- Output geometry and `NA` mask match the input domain.
- Peak memory is bounded by configured chunk sizes for a raster larger than RAM.
- CPU and supported accelerator raster predictions agree and can be written/read by
  `terra`.
- A multispecies output can be split/resumed by explicit species subsets.

**STOP/GO Gate 5:** GO only if raster correctness is proven independently of speed. If
I/O dominates, document it and retain profiling evidence for later pipeline work.

### Phase 6 — Simulation, references, and benchmark harness

**Goal:** produce a reproducible benchmark that separates numerical fidelity,
ecological recovery, and computational scaling.

### 6.1 `virtualspecies` data generation

Use `virtualspecies` as a benchmark-generation dependency (`Suggests`, not a runtime
dependency). Store scenario configuration, RNG kind, seeds, package versions, source
code revision, raster metadata, and hashes. Generate environmental predictors with
controlled correlation/spatial autocorrelation, then virtual species with known
response functions. Include:

- linear, quadratic, interaction, threshold-like, and nonlinear response truth;
- narrow/broad niches and low/high prevalence;
- 1/2/5/10/25 influential predictors plus nuisance predictors;
- correlated and independent predictors;
- sample size, sampling bias, detection error/noise, accessible-area truncation, and
  dispersal limitation;
- shared and species-specific background domains;
- missing raster cells and categorical predictors;
- fixed small “CI” scenarios and separately generated large benchmark scenarios.

Do not treat recovery of arbitrary virtual-species generator parameters as equivalent
to recovery of MaxEnt coefficients when the generating model differs. Primary recovery
targets are the known suitability surface and response curves.

### 6.2 Reference comparison design

Run identical prepared inputs through:

1. `maxentgpu` scalar CPU;
2. `maxentgpu` scalar CUDA;
3. `maxentgpu` scalar MPS;
4. `maxentgpu` batched CPU;
5. `maxentgpu` batched CUDA;
6. `maxentgpu` batched MPS;
7. pinned `maxnet`;
8. pinned Java MaxEnt.

Compare design features where accessible, convergence, coefficient/feature tables,
link/raw/logistic/cloglog predictions, response curves, rankings, raster differences,
and failure behavior. Separate:

- **same objective/spec:** strict numerical equivalence expected;
- **mapped reference mode:** prediction equivalence expected within mapped tolerances;
- **different defaults/algorithm:** behavioral comparison, not equivalence.

### 6.3 Benchmark matrix

Use a staged design; never run the full Cartesian product blindly.

| Dimension | CI/smoke | Core paper matrix | Stress/extension |
|---|---|---|---|
| species | 1, 4 | 1, 10, 100, 1,000 | 5,000+ |
| background rows | 100, 1,000 | 10k, 100k, 1M | 10M+ |
| presences/species | 10, 50 | 25, 100, 1k, 10k | imbalanced 5–100k |
| predictors | 2, 5 | 5, 10, 25, 50 | 100 |
| feature classes | L, LQ | L, LQ, LQH, LQHPT, categorical | all/heterogeneous |
| raster cells | 1k | 100k, 1M, 10M, 100M | continental limit |
| dtype | float64 | float64, float32 | future mixed |
| device | CPU | CPU, CUDA, MPS | multiple GPU models, not multi-GPU execution |
| folds/replicates | 1 | 1, 5, 10 / 1, 10, 100 | budget-dependent |
| background | shared | shared, grouped | species-specific |

Use fractional-factorial/Latin-hypercube sampling for secondary dimensions after a
small full-factorial calibration. Pre-register the final paper matrix before the full
run. Repeat timed runs, randomize method order, include warm-up, synchronize CUDA, and
report distributions/confidence intervals rather than best runs.

### 6.4 Metrics

Correctness/ecology:

- maximum/median absolute and relative prediction error;
- Pearson/Spearman correlation and rank agreement;
- integrated squared error against true suitability;
- response-curve error and influential-variable ranking;
- spatial omission/commission and calibration metrics appropriate to presence-
  background evaluation;
- convergence/failure rate and sensitivity to sampling bias.

Performance:

- end-to-end wall time including required feature construction and transfers;
- isolated fit and prediction time as secondary decomposition;
- species fits/second and cell-species predictions/second;
- CPU RAM and peak allocated/reserved GPU memory;
- host-device bytes/transfers if measurable;
- GPU utilization/kernel timeline for selected cases;
- break-even species/background/raster sizes;
- energy only if measured with a documented calibrated method.

Record hardware model, CPU cores/threads, RAM, GPU and VRAM, driver/runtime, OS,
R/package versions, power mode, warm-up, thread variables, and commit SHA. Raw results
must be tidy, append-only, and paired with a manifest; analysis scripts regenerate all
tables/figures.

Acceptance criteria:

- One command regenerates CI simulation fixtures and one documented workflow launches
  each benchmark tier.
- Re-running a small scenario with the same environment/seed reproduces inputs and
  results within declared tolerance.
- Reference and ecological-recovery reports are distinct from performance reports.
- Benchmark failures remain rows with status/error metadata, not silently dropped.
- At least two distinct CPU systems and two NVIDIA GPU generations/configurations are
  represented before strong general scaling claims.

**STOP/GO Gate 6:** GO to paper claims only after the benchmark protocol and exclusions
are frozen. If simulation is the main MEE evidence, require at least one modest real-data
operational demonstration before submission; it validates workflow realism, not a new
biological claim.

### Phase 7 — Hardening, CI, profiling, and release engineering

**Goal:** make the package maintainable and trustworthy outside the developer machine.

Tasks:

- [ ] Run unit, integration, metamorphic/property, snapshot (sparingly), reference, and
  performance-regression tests in separate tiers.
- [ ] Add R CMD check across Linux/macOS/Windows CPU runners as supported by R `torch`;
  define documented exclusions instead of pretending unsupported combinations pass.
- [ ] Add CUDA and MPS smoke/parity CI on trusted accelerator runners, with pinned
  environments and no untrusted fork code receiving secrets or privileged runner
  access.
- [ ] Add coverage reporting, lint/static checks, dependency caching, and scheduled
  reference/extended tests.
- [ ] Set performance alert bands from stable dedicated hardware; do not fail shared CI
  on noisy microbenchmark thresholds.
- [ ] Profile representative small, break-even, and large cases using R-level timing,
  Torch profiling, CUDA timelines, memory traces, and raster I/O measurements.
- [ ] Classify bottlenecks as R orchestration, feature construction, objective/solver,
  kernel launch, tensor allocation, transfer, GPU compute, raster read, or raster write.
- [ ] Audit package size, startup behavior, GPU memory cleanup, errors, warnings,
  interrupts, temporary files, and offline documentation builds.
- [ ] Add reverse-dependency strategy once downstream packages exist.

Acceptance criteria:

- Fast CPU CI completes on every change; extended/reference/CUDA tiers have visible
  status and a documented cadence.
- No correctness test depends on benchmark timing.
- A profiling report identifies the top costs with traces and percentages for at least
  three representative workloads.
- Installation paths and failure messages have been tested by someone other than the
  primary development environment.

**STOP/GO Gate 7 (custom CUDA decision):** remain on Torch unless all are true:

1. a stable, common workload misses an explicit performance or memory target;
2. profiling attributes at least ~25% of relevant end-to-end time, or a blocking memory
   cost, to operations plausibly removable by fusion/custom layout;
3. raster I/O, transfers, R orchestration, and solver iteration count are not the main
   cause;
4. a prototype has a credible path to at least 1.5x end-to-end improvement on that
   workload, not merely a faster microkernel;
5. maintenance, build, platform, testing, and CRAN/JOSS implications are accepted.

If any condition fails, record **NO-GO** and optimize within Torch or documentation.

### Phase 8 — Documentation, JOSS, and MEE readiness

**Goal:** release a citable software artifact and a separately reproducible methods
evaluation.

### Documentation tasks

- [ ] Roxygen documentation for every export and user-visible object.
- [ ] README with scope, install, CUDA availability, minimal fit/predict example,
  benchmark caveat, citation, license, and project-status badges.
- [ ] Vignettes for getting started, multispecies batching, raster projection, and
  numerical/reference fidelity.
- [ ] A mathematical developer vignette linked to `inst/spec/`.
- [ ] Troubleshooting for LibTorch/CUDA/MPS compatibility, CPU fallback, OOM/chunk
  sizing, convergence, dtype, categorical levels, and raster layer mismatch.
- [ ] Reproducibility statement and privacy/data policy (models do not embed records by
  default).
- [ ] pkgdown site with stable-version documentation.

### JOSS readiness checklist

- [ ] Clear statement of need and comparison with Java MaxEnt, `maxnet`, and relevant
  SDM tooling without unsupported novelty claims.
- [ ] Public OSI-approved license, contribution guide, code of conduct, governance,
  issue templates, and documented support channel.
- [ ] Automated tests, continuous integration, installation docs, examples, API docs,
  and meaningful research use/case study.
- [ ] `paper/paper.md` focuses on software, architecture, validation, availability, and
  reuse; the extensive scientific benchmark stays outside the short JOSS paper.
- [ ] References, author affiliations/ORCIDs, `CITATION.cff`, codemeta, and repository
  metadata agree.
- [ ] Tagged public release archived with DOI; archive installs and reproduces the
  quick-start example.
- [ ] JOSS review checklist completed against the current guidelines at submission time.

### MEE benchmark readiness checklist

- [ ] Questions and hypotheses are frozen before the full run: fidelity, break-even,
  scaling, batching effects, and which rigorous multispecies workflows become feasible.
- [ ] Simulation design, seeds, exclusions, sample sizes, hardware, precision, and
  statistical analysis are preregistered or time-stamped.
- [ ] Controlled `virtualspecies` simulation is the primary ecological evaluation.
- [ ] A modest real-data demonstration covers operational issues (unequal records,
  missing values, categorical/environmental rasters, spatial folds, and projection)
  without turning the paper into a biological case study.
- [ ] Comparisons use equivalent inputs/options where possible and clearly label
  non-equivalent defaults.
- [ ] End-to-end runtime, accuracy, failures, memory, and break-even points are reported,
  including cases where CPU or a reference tool wins.
- [ ] All figures/tables regenerate from archived raw results and manifests.
- [ ] The central claim is ecological feasibility/rigor enabled by independent-species
  batching, not “a GPU is faster.”
- [ ] JOSS and MEE text/claims are cleanly separated to avoid duplicate publication.

**STOP/GO Gate 8:** release `0.1.0` only after Definition of Done is met. Submit to JOSS
when the software is mature and demonstrably used. Submit to MEE only when the full
benchmark supports a methodological consequence beyond a software speed demonstration.

### Optional Phase 9 — Native C++/CUDA extension (conditional)

This phase is absent from the `0.1.0` critical path. If Gate 7 says GO, create a short
architecture decision record specifying the smallest justified extension. Candidate
targets, in order of likely value:

1. fused application of hinge/threshold/product features and scoring without
   materializing the expanded feature matrix;
2. fused weighted logsumexp/objective/gradient for many species;
3. sparse/block-structured coefficient projection;
4. pinned-memory double buffering and asynchronous raster transfer;
5. only later, multi-GPU scheduling.

Requirements:

- keep the R API and serialized model schema backend-independent;
- retain the pure-Torch implementation as the correctness oracle and fallback;
- add CPU/Torch/CUDA-extension differential tests and compute-sanitizer checks;
- benchmark end-to-end with compilation/install complexity included in the decision;
- feature-detect the extension and fail back safely;
- document supported CUDA toolkit, driver, architectures, compilers, and binary policy;
- review CRAN portability and decide whether the extension belongs in the main package
  or an optional companion package.

**STOP/GO Gate 9:** ship a native extension only if it preserves numerical fidelity and
achieves the predeclared end-to-end improvement on more than one GPU generation. A
microbenchmark win alone is NO-GO.

## 6. Test strategy and required cases

### Unit tests

- validation, weights, names/order, feature specs, every feature basis;
- stable logsumexp, objective components, penalties, proximal operator;
- autograd/manual/finite-difference gradients;
- output transforms, clamping/extrapolation, unseen categories;
- device/dtype movement and serialization schema migration.

### Metamorphic/property tests

- row permutation leaves fit/predictions unchanged within tolerance;
- duplicated rows with split weights equal aggregated rows;
- scalar fit equals batch of size one;
- a batch equals independently fitted species;
- chunk size changes do not change predictions;
- CPU-trained/accelerator-predicted equals CPU-predicted and converse within the
  backend/dtype tolerance;
- save/load and model extraction preserve predictions;
- extreme constant shifts of logits do not break normalized raw output;
- irrelevant unused factor levels do not reorder fitted columns.

### Integration tests

- tabular fit through raster prediction;
- on-disk raster block read/write with interruption cleanup;
- mixed converged/non-converged species batch;
- model trained, saved, loaded in a fresh R session, and projected on another device;
- reference fixture regeneration in a pinned container/environment.

### Reference golden tests

Golden files must contain raw inputs, options, versions, commands, expected outputs,
tolerances, and provenance. Prefer small text/CSV/JSON fixtures. Regeneration must be an
explicit maintainer action; CI must detect unexplained golden-file drift.

### Performance tests

Keep microbenchmarks diagnostic. Release decisions use end-to-end cases. Track median,
dispersion, peak memory, warm/cold behavior, and correctness hash. Never use a different
dtype, feature set, tolerance, or convergence criterion to make one method look faster.

## 7. Risk register and mitigations

| Risk | Early signal | Mitigation / decision |
|---|---|---|
| Java/`maxnet` conventions differ | coefficients or scales disagree | executable mapping fixtures; compare same-objective and mapped modes separately |
| L1 solver converges differently | prediction drift near zero coefficients | KKT/prox residuals, objective checks, stricter iteration controls, tolerance sensitivity |
| Hinge/threshold expansion explodes memory | large `F`, OOM before fit | feature grouping, lazy/chunked scoring, caps/warnings; profile before fusion |
| Heterogeneous species defeat dense batching | padding/masks dominate | group compatible specs; scalar fallback; report supported fast path clearly |
| GPU slower for ordinary workloads | small break-even region | publish break-even planner and honest CPU fallback; optimize end-to-end |
| Raster I/O dominates | low GPU utilization | larger/double buffers within Torch path, compressed-output analysis; avoid premature kernels |
| Float32 changes convergence | unstable active set/output | float64 default/reference; precision-specific gates and documentation |
| CUDA CI is flaky/unavailable | untested GPU commits | controlled scheduled runner plus local release checklist and archived logs |
| Torch/CUDA installation burden | user failures | compatibility table, diagnostics, CPU mode, container/lockfile recipes |
| Benchmark becomes unmanageable | Cartesian explosion | tiered/fractional design, pre-registration, manifests, budget gates |
| Simulation is ecologically idealized | strong synthetic, weak operational story | controlled simulation primary plus modest real-data workflow demonstration |
| Package name implies universal GPU parity | reviewer/user confusion | publish a tested backend capability matrix and say “implemented with torch for R” |

## 8. Recommended execution sequence and deliverables

| Milestone | Deliverable | Depends on | Exit artifact |
|---|---|---|---|
| M0 | executable spec and skeleton | none | specs, fixture generators, CPU check |
| M1 | scalar CPU L/LQ fit/predict | M0 | correct model object and golden tests |
| M2 | full feature fidelity | M1 | green reference feature matrix |
| M3 | scalar accelerator parity | M2 | CPU/CUDA/MPS parity report |
| M4 | independent-species batching | M3 | scalar/batch equivalence and throughput report |
| M5 | `terra` projection | M4 | bounded raster integration tests |
| M6 | simulation/reference benchmark | M5 | frozen protocol and reproducible results |
| M7 | hardening/profiling | M6 | CI matrix, profiles, custom-CUDA ADR |
| M8 | release/publications | M7 | `0.1.0`, DOI, JOSS paper, MEE-ready archive |
| M9 | optional native extension | explicit Gate 7 GO | differential tests and measured end-to-end gain |

The critical path is M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8. M9 is not
allowed to delay correctness, documentation, JOSS submission, or the main MEE
benchmark unless profiling has already passed Gate 7.

## 9. First Codex work order

The first implementation task should be exactly:

> Execute Phase 0 only. Create the R package skeleton, specification files, CPU-only
> smoke test, compatibility/provenance policy, and tiny reference-fixture generators.
> Do not implement CUDA or full model fitting. At Gate 0, report unresolved mathematical
> mappings and stop for review.

This forces the highest-risk scientific decisions to become executable tests before
the codebase accumulates performance-oriented assumptions.

## 10. Authoritative resources to recheck during implementation

These are starting points, not substitutes for pinned versions and executable
fixtures. Recheck them when beginning the relevant phase and again before release:

- [`torch` for R documentation](https://torch.mlverse.org/) and
  [package source](https://github.com/mlverse/torch);
- [`terra` documentation](https://rspatial.github.io/terra/);
- [`virtualspecies` on CRAN](https://CRAN.R-project.org/package=virtualspecies);
- [`maxnet` source](https://github.com/mrmaxent/maxnet) and its pinned CRAN manual;
- [Java MaxEnt source](https://github.com/mrmaxent/Maxent) and the pinned release's
  bundled documentation;
- [Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html),
  CRAN repository policy, and current `R CMD check` behavior;
- [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
  and paper format at the time of submission;
- current MEE author guidelines and data/code policy at the time the benchmark protocol
  is frozen.
