# Reference comparison status

The pinned `maxnet` fixture is generated from `maxnet` 0.1.4 using the command
recorded in `tests/fixtures/maxnet/provenance.dcf`. Its coefficients and prediction
columns are retained as an external reference, not as package-native golden values.
The generator also records a `regmult = 1e-8` weakly regularized fixture because
`maxnet`/glmnet rejects an exactly zero penalty; this is a near-unregularized
diagnostic target, not an unregularized equivalence claim.

The current scalar implementation intentionally uses a different normalization
contract:

- `maxentgpu` has no freely fitted intercept or constant feature;
- presence and background weights are normalized independently;
- `raw` is density relative to the normalized background quadrature measure;
- `maxnet` reports a discrete-background convention with an intercept and its own
  entropy/output offset.

Therefore direct coefficient and raw-column equality is not an appropriate Gate 1
test. The package-native objective and predictions must first pass their hand,
finite-difference, and serialization tests. A future reference adapter must compare
equivalent normalized measures and explicitly account for the intercept/entropy
offset before claiming `maxnet` compatibility.

The command `Rscript tools/compare-reference-fixtures.R .` reports correlations plus
an affine mapping for the link scale and a multiplicative ratio for the raw scale.
These are diagnostic quantities only, not acceptance thresholds. A mapping is eligible
for a compatibility claim only after it is reproduced on independent fixtures and its
regularization and weighting assumptions are recorded here.

Java MaxEnt 3.4.4 fixtures are now available with the same provenance and checksum
discipline. Java feature-class, regularization, weighting, clamping, and output-scale
comparisons remain separate mapping decisions in `inst/spec/reference-mapping.md`.
