args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L || length(args) > 2L) {
  stop("Usage: Rscript .github/scripts/set-version.R VERSION [DESCRIPTION]")
}

version <- args[[1]]
description <- if (length(args) == 2L) args[[2]] else "DESCRIPTION"
if (!grepl("^[0-9]+[.][0-9]+[.][0-9]+$", version)) {
  stop("Release version must be numeric SemVer (MAJOR.MINOR.PATCH): ", version)
}
if (!file.exists(description)) stop("DESCRIPTION file not found: ", description)

lines <- readLines(description, warn = FALSE)
version_line <- grep("^Version:[[:space:]]*", lines)
if (length(version_line) != 1L) {
  stop("Expected exactly one Version field in ", description)
}
lines[[version_line]] <- paste("Version:", version)
writeLines(lines, description, useBytes = TRUE)
message("Set ", description, " to version ", version)

news <- file.path(dirname(description), "NEWS.md")
if (file.exists(news)) {
  news_lines <- readLines(news, warn = FALSE)
  heading <- grep("^# maxentgpu[[:space:]]+", news_lines)
  if (length(heading) != 1L) stop("Expected exactly one maxentgpu heading in ", news)
  news_lines[[heading]] <- paste("# maxentgpu", version)
  writeLines(news_lines, news, useBytes = TRUE)
  message("Set ", news, " heading to version ", version)
}
