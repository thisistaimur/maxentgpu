# Java MaxEnt generated fixtures

Generated outputs belong here and are compact golden files produced by the pinned
Java MaxEnt 3.4.4 generator. Never place `maxent.jar` in this repository.

To regenerate them, acquire Java MaxEnt 3.4.4 under its upstream terms, place the
jar outside the repository, verify its SHA-256 in
`benchmarks/manifests/reference-versions.csv`, and run:

```sh
Rscript tools/generate-java-maxent-fixtures.R /absolute/path/to/maxent.jar
```
