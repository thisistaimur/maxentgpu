# Feature specification

Specification schema: `maxentgpu-features-v1`

Feature fitting and application are separate operations. A fitted specification is
immutable and stores the schema version, predictor names and order, source types,
numeric training ranges, categorical levels, generated-column order, knots,
thresholds, scaling values, and penalty multipliers. Training and prediction must use
the same application function.

## Input contract

- Predictor names must be present, unique, and stable.
- Numeric inputs are converted to double precision before fitting transform metadata.
- Ordered and unordered factors are categorical; character columns are rejected until
  explicitly converted to factors.
- Missing and non-finite training values are errors.
- A constant numeric predictor is rejected for L/Q/P/H/T construction rather than
  silently creating a constant feature.
- Prediction columns may be reordered by name. Missing or extra required columns are
  errors; irrelevant extra columns are ignored with a diagnostic.
- Numeric prediction values are clamped to the stored training range when
  `clamp = TRUE`; `clamp = FALSE` extrapolates the fitted basis.

## Stable identifiers and column order

Predictors follow input column order. Within a predictor, feature classes follow
`L`, `Q`, `P`, `T`, `H`, `C`. Pairwise products use `(i, j)` with `i < j`, ordered
lexicographically by predictor position. Identifiers are:

```text
L:<x>
Q:<x>
P:<x1>*<x2>
T:<x>@<threshold>
H+:<x>@<knot>
H-:<x>@<knot>
C:<x>=<level>
```

Stored decimal thresholds/knots use a round-trip-safe representation, not display
rounding.

## Numeric bases

All transforms below operate after optional clamping.

- Linear: `L(x) = x` (identity scaling).
- Quadratic: `Q(x) = x^2`; squaring occurs on the raw numeric value.
- Product: `P(x, y) = x * y`, with no self-products or duplicates.
- Threshold: `T(x; t) = 1[x >= t]`.
- Forward hinge: `H+(x; k) = clip((x - k) / (max - k), 0, 1)`.
- Reverse hinge: `H-(x; k) = clip((k - x) / (k - min), 0, 1)`.

The denominator-normalized hinges range from zero to one on the training range.
Degenerate knots at zero or one are omitted.

For explicit threshold features, candidate locations are
`seq(min, max, length.out = nknots + 2)[2:(nknots + 1)]`. For hinges, start with
`seq(min, max, length.out = nknots)`: forward hinges use all but the maximum and
reverse hinges use all but the minimum. The package-native default is `nknots = 50`.
Locations are stored in the original predictor units. This matches `maxnet` 0.1.4's
location construction. A `maxnet` right-capped increasing hinge is represented as
`1 - H-`; the discarded constant is immaterial to the normalized objective, and the
coefficient sign changes. Java equivalence remains a Gate 0 question.

## Categorical basis

The fitted factor level order is stored and every level produces an indicator column,
matching `maxnet` 0.1.4. The resulting block spans a constant shift, so individual
categorical coefficients are not identifiable without the fitted penalty convention;
normalized predictions remain identifiable. Unseen levels error by default. With the
explicit `unseen = "NA"` policy, the entire affected row is marked missing. Unused
fitted factor levels are dropped before columns are created, and irrelevant unused
levels in new data cannot reorder fitted columns.

## Automatic feature policy

User-specified classes always override this policy. For `n` effective presence rows
after input validation, the package-native policy is:

| Presence rows | Classes |
|---:|---|
| 1--9 | L |
| 10--14 | L, Q |
| 15--79 | L, Q, H |
| 80+ | L, Q, H, P |

Threshold features are explicit-only. Boundary tests are required at 9/10, 14/15,
and 79/80. This mirrors the intended modern Java default shape but is not claimed to
be reference-equivalent until the reference fixtures pass.

## Regularization metadata

Each generated column stores `penalty_l1` and `penalty_l2`. Package-native defaults
are one for penalized columns. Reference-derived feature-specific multipliers must be
stored as explicit numbers with their provenance; no hidden calculation may occur
during prediction.
