# Canonical objective specification

Specification schema: `maxentgpu-objective-v1`

## Data and normalization

For one species, `P` indexes presence rows and `B` indexes background rows. The
immutable fitted feature map is `phi(x)`. Presence weights `w_i` and background
quadrature weights `q_j` must be finite and strictly positive. They are normalized
independently in double precision so that

```text
sum(i in P, w_i) = 1
sum(j in B, q_j) = 1.
```

Zero-length sets, non-finite weights, non-positive weights, and a non-finite feature
matrix are errors. Missing predictors are rejected during fitting; prediction may
propagate a row-level missing-value mask but may not impute zero implicitly.

Duplicate presence or background rows are retained. Splitting a row's normalized
weight over identical duplicates must be equivalent to the aggregated row. No
occurrence de-duplication is performed implicitly.

## Score, partition function, and loss

There is no freely fitted intercept or constant feature. For coefficient vector
`beta`,

```text
z(x) = beta' phi(x)
logZ(beta) = log(sum(j in B, q_j * exp(z_j)))
L_smooth(beta) = logZ(beta) - sum(i in P, w_i * z_i)
```

The partition calculation must use weighted log-sum-exp:

```text
a = max(j in B, log(q_j) + z_j)
logZ = a + log(sum(j in B, exp(log(q_j) + z_j - a))).
```

This formula is required even for extreme logits. Any non-finite result after the
stable calculation is an error with the failing component named.

The initial separable penalty is

```text
Omega(beta) = lambda1 * sum(k, r1_k * abs(beta_k))
            + 0.5 * lambda2 * sum(k, r2_k * beta_k^2),
```

where `lambda1`, `lambda2`, `r1_k`, and `r2_k` are finite and non-negative. The full
objective is `L = L_smooth + Omega`. Feature-specific multipliers are stored in the
feature specification and coefficient audit table. Phase 0 defines this contract but
does not implement fitting.

## Analytic smooth gradient

Let

```text
pi_j = q_j * exp(z_j - logZ).
```

Then `sum(pi) = 1` within numerical tolerance and

```text
gradient L_smooth = sum(j in B, pi_j * phi(x_j))
                  - sum(i in P, w_i * phi(x_i))
                  + lambda2 * r2 * beta.
```

The L1 term is handled by the proximal operator, not inserted into this smooth
gradient. Tiny fixtures must compare this expression with both Torch autograd and
central finite differences before a solver is accepted.

## Identifiability and invariances

Adding a constant to every background and presence score leaves `L_smooth`
unchanged. Consequently a constant feature is non-identifiable and is rejected.
Row permutations and equivalent aggregation of duplicate weights must leave the
objective, gradient, and predictions unchanged within the declared dtype tolerance.

## Numerical reference

CPU `float64` is the Phase 1 correctness reference. The tolerances in `PLAN.md` are
normative and may only be changed with a fixture-backed decision record.
