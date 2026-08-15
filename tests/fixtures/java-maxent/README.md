# Java MaxEnt generated fixtures

Generated outputs belong here but are ignored until a maintainer deliberately stages
reviewed, compact golden files. Never place `maxent.jar` in this repository.

Acquire Java MaxEnt 3.4.4 under its upstream terms, place the jar outside the
repository, replace the pending SHA-256 in
`benchmarks/manifests/reference-versions.csv`, and run:

```sh
Rscript tools/generate-java-maxent-fixtures.R /absolute/path/to/maxent.jar
```
