# Hand fixtures

These compact CSV files exercise the Phase 0 numerical contract. They are source
fixtures, not reference outputs from `maxnet` or Java MaxEnt. `expected.csv` contains
values independently reproducible with `tools/check-hand-fixtures.R` using base R.

All weights shown are pre-normalization input weights. The checker normalizes presence
and background weights independently, exactly as specified in `inst/spec/objective.md`.
