# Reference mapping and compatibility status

Specification schema: `maxentgpu-reference-v1`

The comparison targets are `maxnet` 0.1.4 and Java MaxEnt 3.4.4. They are targets,
not an undocumented specification for `maxentgpu`. The package currently makes no
claim of reference equivalence.

## Pinned targets

| Target | Pin | Acquisition and license |
|---|---|---|
| maxnet | CRAN 0.1.4 | CRAN source tarball; MIT + file LICENSE |
| Java MaxEnt | 3.4.4 | maintainer-supplied `maxent.jar`; do not commit; verify SHA-256 before use; upstream MIT/third-party notices must accompany local acquisition |
| torch | CRAN 0.17.0 | runtime development target; MIT + file LICENSE |
| terra | CRAN 1.9-34 | later raster adapter target; GPL (>= 3) |
| virtualspecies | CRAN 1.6.1 | later benchmark generator; GPL (>= 2) |

The version policy and machine-readable pins live in `benchmarks/manifests/`.

## Settled package-native conventions

| Topic | maxentgpu convention | Reference comparison mode |
|---|---|---|
| Objective | normalized positive presence/background weights; weighted log-partition; no intercept | same-objective fixture where reference permits |
| Duplicates | retain; equivalent split/aggregated weights must agree | Java default de-duplication disabled explicitly |
| Missing training data | error | behavioral comparison |
| Numeric scaling | identity L, raw square Q, raw pairwise product P | exact maxnet feature-column comparison; verify Java |
| Auto classes | L; LQ at 10; LQH at 15; LQHP at 80; T explicit-only | mapped-default comparison |
| Clamping | predictor-range clamp before feature transform | compare both clamped and unclamped modes |
| Raw | density relative to normalized `q`, with weighted mean one | convert reference discrete raw before comparing |
| Constant features | reject | intentional divergence |

## Executable maxnet 0.1.4 findings

Inspection and execution of the pinned installed namespace established that `maxnet`:

- uses raw numeric columns for L, `x^2` for Q, and formula interactions for P;
- selects L below 10 presences, LQ from 10, LQH from 15, and LQPH from 80;
- uses 50 evenly spaced hinge locations over the combined data range and 50 interior,
  evenly spaced threshold locations;
- creates one indicator for every stored factor level;
- clamps numeric predictors to training ranges and then clamps reconstructed feature
  columns to their training ranges;
- sets `alpha = -log(sum(exp(z_background)))`, computes discrete-background entropy,
  and applies cloglog/logistic to `entropy + link`.

For uniform background weights, the package-native relative-entropy convention is
algebraically identical: its raw density is `N` times maxnet's discrete raw, while its
relative entropy is maxnet entropy minus `log(N)`, so their product `tau` is equal.
Generated LQ coefficients, predictions, formula, provenance, and hashes are stored in
`tests/fixtures/maxnet/`.

## Open Gate 0 mappings

These identifiers are blocking issues in this file until an executable fixture closes
them. They are intentionally explicit rather than placeholders in the mathematical
specification.

- **REF-FEATURE-001:** verify Java's L, Q, and P construction against the now-pinned
  `maxnet` raw-value construction.
- **REF-KNOT-001:** verify Java hinge and threshold construction and confirm how both
  targets behave for constant ranges and duplicated numeric values.
- **REF-AUTO-001:** verify whether Java 3.4.4 matches the confirmed `maxnet` boundaries and
  excludes threshold features by default under every relevant CLI mode.
- **REF-REG-001:** map reference feature-specific beta multipliers and sample-size
  adjustments onto explicit `lambda * r_k` values. The pinned maxnet LQ fixture uses
  factors `0.0389711432` (L) and `0.0566423814` (Q); direct use with package
  `lambda1 = 1` collapses the package fit to zero coefficients, confirming a scale
  mismatch that remains unresolved. maxnet's glmnet path scales the terminal
  regularization value as `mean(reg) * sum(p) / sum(p + 100 * (1 - p))`; for the
  pinned fixture this is `0.0004733343`, which is recorded in `*scales.csv`.
- **REF-SCALE-001:** confirm on Java 3.4.4 the entropy and raw-output mapping already
  established algebraically and executably for `maxnet` 0.1.4.
- **REF-WEIGHT-001:** determine which presence/background weighting operations are
  representable in each reference implementation and define fixture-specific mapped
  comparisons where exact weighting is unavailable.
- **REF-CLAMP-001:** verify whether reference clamping occurs before or within each
  feature basis and how extrapolation metadata is serialized.
- **REF-DUP-001:** verify CLI flags and behavior for duplicate removal in Java 3.4.4.
- **REF-CAT-001:** verify Java factor levels, unseen-level behavior, and naming against
  `maxnet`'s confirmed full one-hot basis.

**Closed reference acquisition:** **REF-JAVA-001** is complete for the pinned Java
3.4.4 artifact. The licensed jar is kept outside the repository; its SHA-256 and Java
17 provenance are recorded in `benchmarks/manifests/reference-versions.csv` and
`tests/fixtures/java-maxent/provenance.dcf`.

Gate 0 remains **STOP** while any open item changes the advertised reference mapping
or output-scale labels. Native objective work may proceed only after maintainer review
confirms the package-native specification is the intended model.
