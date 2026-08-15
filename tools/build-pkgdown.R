main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")
  internal_markdown <- file.path(root, c("AGENTS.md", "PLAN.md", "CRAN-READINESS.md"))
  internal_markdown <- internal_markdown[file.exists(internal_markdown)]
  staging <- tempfile("maxentgpu-pkgdown-")
  dir.create(staging)

  restore_internal_markdown <- function() {
    restored <- TRUE
    for (source in internal_markdown) {
      staged <- file.path(staging, basename(source))
      if (file.exists(staged) && !file.rename(staged, source)) {
        warning("Could not restore ", source)
        restored <- FALSE
      }
    }
    if (restored) unlink(staging, recursive = TRUE)
  }
  on.exit(restore_internal_markdown(), add = TRUE)

  for (source in internal_markdown) {
    staged <- file.path(staging, basename(source))
    if (!file.rename(source, staged)) stop("Could not stage ", source)
  }

  pkgdown::clean_site(root, force = TRUE)
  pkgdown::build_site(root, new_process = FALSE, preview = FALSE)
}

main()
