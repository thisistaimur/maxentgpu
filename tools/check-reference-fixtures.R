args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("Missing reference fixture: ", path)
}

maxnet_dir <- file.path(root, "tests", "fixtures", "maxnet")
maxnet_files <- file.path(maxnet_dir, c(
  "coefficients.csv", "predictions.csv", "formula.txt", "provenance.dcf",
  "checksums.md5"
))
invisible(lapply(maxnet_files, stop_if_missing))

checksums <- utils::read.table(
  file.path(maxnet_dir, "checksums.md5"),
  header = TRUE,
  stringsAsFactors = FALSE,
  colClasses = c("character", "character")
)
if (!identical(names(checksums), c("md5", "file"))) {
  stop("Unexpected checksum columns in maxnet fixture manifest.")
}

expected_files <- file.path(root, checksums$file)
invisible(lapply(expected_files, stop_if_missing))
actual <- unname(tools::md5sum(expected_files))
if (!identical(tolower(actual), tolower(checksums$md5))) {
  mismatch <- checksums$file[tolower(actual) != tolower(checksums$md5)]
  stop("maxnet fixture checksum mismatch: ", paste(mismatch, collapse = ", "))
}

provenance <- read.dcf(file.path(maxnet_dir, "provenance.dcf"))
required <- c("Generator", "MaxnetVersion", "RVersion", "Command")
if (!all(required %in% colnames(provenance))) {
  stop("maxnet provenance is missing: ",
       paste(setdiff(required, colnames(provenance)), collapse = ", "))
}
if (!identical(as.character(provenance[1, "MaxnetVersion"]), "0.1.4")) {
  stop("maxnet provenance is not pinned to 0.1.4.")
}

java_manifest <- file.path(root, "benchmarks", "manifests", "reference-versions.csv")
stop_if_missing(java_manifest)
manifest <- utils::read.csv(java_manifest, stringsAsFactors = FALSE)
java <- manifest[manifest$component == "java-maxent", , drop = FALSE]
if (nrow(java) != 1L || !identical(java$version[[1]], "3.4.4")) {
  stop("Java MaxEnt manifest must contain exactly one version 3.4.4 row.")
}
if (!identical(java$sha256[[1]], "REPLACE_AFTER_LICENSED_ACQUISITION")) {
  jar <- file.path(root, "tests", "fixtures", "java-maxent", "maxent.jar")
  if (!file.exists(jar)) stop("Java MaxEnt jar is missing: ", jar)
}

message("Reference fixture integrity passed: maxnet 0.1.4.")
if (identical(java$sha256[[1]], "REPLACE_AFTER_LICENSED_ACQUISITION")) {
  message("Java MaxEnt 3.4.4 remains pending a licensed jar and SHA-256.")
} else {
  message("Java MaxEnt 3.4.4 manifest is pinned.")
}
