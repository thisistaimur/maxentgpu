# Prediction scale specification

Specification schema: `maxentgpu-prediction-v1`

Every prediction must name its scale. Let

```text
z(x) = beta' phi(x)
logZ = log(sum(j in B, q_j * exp(z_j)))
d(x) = exp(z(x) - logZ).
```

The normalized background weights `q` sum to one, hence `sum(q * d) = 1`. In this
package `d` is a density relative to the background quadrature measure, not a discrete
probability that sums to one over background rows.

## Link

```text
link(x) = z(x).
```

No intercept or normalization offset is added.

## Raw

```text
raw(x) = d(x) = exp(z(x) - logZ).
```

The calculation must occur in log space until the final exponentiation. Overflow in
the final requested representation is an explicit error; it is not silently clipped.

## Entropy offset

Define entropy relative to the normalized background measure:

```text
entropy = -sum(j in B, q_j * d_j * log(d_j)).
```

This is the negative Kullback-Leibler divergence from the fitted density to `q`, so it
is zero for a constant score. Store the fitted scalar in the model. Its discrete Java
counterpart differs by the quadrature-measure convention; the mapping must be tested
before compatibility is claimed.

Set

```text
tau(x) = exp(entropy) * raw(x).
```

## Cloglog

```text
cloglog(x) = 1 - exp(-tau(x)).
```

Use the numerically stable form `-expm1(-tau)`. The output is in `[0, 1]` apart from
rounding. This scale is implemented only after the Java and `maxnet` fixtures confirm
the entropy/quadrature mapping.

## Logistic

```text
logistic(x) = tau(x) / (1 + tau(x))
            = plogis(entropy + log(raw(x))).
```

This is not a generic sigmoid of `z`. It is implemented and labelled `logistic` only
after the pinned Java/`maxnet` comparison passes.

## Clamping, extrapolation, and missing values

Clamping occurs in predictor space before feature application using the stored
numeric ranges. Categorical values are never numerically clamped. With clamping off,
numeric bases extrapolate according to `features.md`. If any required predictor is
missing for a prediction row, every requested scale for that row is `NA`; missing
predictors are never replaced with zero.

## Reference status

`link` and package-native `raw` are fully specified. Cloglog and logistic formulas are
specified but remain disabled until Gate 0 resolves `REF-SCALE-001` and
`REF-WEIGHT-001` in `reference-mapping.md` with executable fixtures.
