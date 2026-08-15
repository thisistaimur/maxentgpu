# CRAN readiness

This repository is structured as an R source package and is checked with
`R CMD check --as-cran`, but it is not yet ready for CRAN submission.

## Blocking before first submission

- Keep the designated maintainer name and directly usable email address in `Authors@R`
  current; the repository currently identifies Taimur Khan (`taimur.khan@ufz.de`).
- Finish enough of the modeling API for the package to be a non-trivial,
  publication-quality contribution. Phase 0 diagnostics alone are not a CRAN release.
- Reach the applicable STOP/GO gates in `PLAN.md`; automatic GitHub releases do not
  authorize or perform CRAN submission.
- Check the final package name against current and archived CRAN and Bioconductor
  package names immediately before submission.
- Run the built source tarball with current R-devel, R-release, and R-oldrel on Linux,
  macOS, and Windows; use win-builder and macbuilder where appropriate.
- Eliminate or explain every warning and significant note from
  `R CMD check --as-cran`.
- Review dependency availability, conditional `Suggests`, package size, test duration,
  network behavior, licenses, copyrights, URLs, examples, and vignettes against the
  current CRAN Repository Policy and Writing R Extensions.

## Release distinction

The GitHub workflow creates source-code releases from successful `main` builds using
Semantic Versioning. CRAN submission remains an explicit maintainer action because it
requires a final policy review, human confirmation, and CRAN's submission process.
