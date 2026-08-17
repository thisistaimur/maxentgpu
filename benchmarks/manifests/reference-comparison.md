# Reference comparison status

The pinned `maxnet` fixture is generated from `maxnet` 0.1.4 using the command
recorded in `tests/fixtures/maxnet/provenance.dcf`. Its coefficients and prediction
columns are retained as an external reference, not as package-native golden values.
The generator also records a `regmult = 1e-8` weakly regularized fixture because
`maxnet`/glmnet rejects an exactly zero penalty; this is a near-unregularized
diagnostic target, not an unregularized equivalence claim.
Each target also stores its fitted `alpha`, entropy, regularization value, and
background count in `tests/fixtures/maxnet/*scales.csv`; comparisons can therefore
remove the reference intercept explicitly instead of estimating it implicitly.
The corresponding `*penalty_factors.csv` files preserve maxnet's feature-specific
glmnet penalty factors for the regularization mapping.
The fixture set now includes an `lqph_` target covering linear, quadratic, product,
and hinge construction. Categorical and Java feature mappings remain separate.
The fixture set also includes a deterministic categorical target with stored levels
and one-hot reference predictions; its small-sample glmnet warning is retained as
fixture provenance rather than treated as a package failure.
Applying those factors directly with `lambda1 = regmult` in the package is an
executable diagnostic, not yet a matched objective: the two solvers normalize the
presence/background loss and penalty scale differently. The comparison command
reports whether that direct mapped fit collapses coefficients or converges, so the
remaining scale conversion stays visible.

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
