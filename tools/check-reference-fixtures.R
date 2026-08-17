args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")

stop_if_missing <- function(path) {
  if (!file.exists(path)) stop("Missing reference fixture: ", path)
}

maxnet_dir <- file.path(root, "tests", "fixtures", "maxnet")
maxnet_files <- file.path(maxnet_dir, c(
  "coefficients.csv", "predictions.csv", "formula.txt", "provenance.dcf",
  "weakly_regularized_coefficients.csv", "weakly_regularized_predictions.csv",
  "scales.csv", "weakly_regularized_scales.csv", "checksums.md5",
  "penalty_factors.csv", "weakly_regularized_penalty_factors.csv",
  "linear_formula.txt", "linear_coefficients.csv", "linear_predictions.csv",
  "linear_scales.csv", "linear_penalty_factors.csv",
  "linear_weakly_regularized_coefficients.csv",
  "linear_weakly_regularized_predictions.csv",
  "linear_weakly_regularized_scales.csv",
  "linear_weakly_regularized_penalty_factors.csv",
  "lqph_formula.txt", "lqph_coefficients.csv", "lqph_predictions.csv",
  "lqph_scales.csv", "lqph_penalty_factors.csv"
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
  java_dir <- file.path(root, "tests", "fixtures", "java-maxent")
  java_outputs <- file.path(java_dir, c(
    "provenance.dcf", "maxentResults.csv", "tiny-species.lambdas",
    "tiny-species_backgroundPredictions.csv", "checksums.md5"
  ))
  invisible(lapply(java_outputs, stop_if_missing))
  java_checksums <- utils::read.table(
    file.path(java_dir, "checksums.md5"), header = TRUE,
    stringsAsFactors = FALSE, colClasses = c("character", "character")
  )
  java_paths <- file.path(root, java_checksums$file)
  invisible(lapply(java_paths, stop_if_missing))
  java_actual <- unname(tools::md5sum(java_paths))
  if (!identical(tolower(java_actual), tolower(java_checksums$md5))) {
    stop("Java MaxEnt fixture checksum mismatch.")
  }
  java_provenance <- read.dcf(file.path(java_dir, "provenance.dcf"))
  if (!identical(as.character(java_provenance[1, "MaxEntVersion"]), "3.4.4") ||
      !identical(as.character(java_provenance[1, "JarSHA256"]),
                 as.character(java$sha256[[1]]))) {
    stop("Java MaxEnt provenance does not match the pinned manifest.")
  }
}

message("Reference fixture integrity passed: maxnet 0.1.4.")
if (identical(java$sha256[[1]], "REPLACE_AFTER_LICENSED_ACQUISITION")) {
  message("Java MaxEnt 3.4.4 remains pending a licensed jar and SHA-256.")
} else {
  message("Java MaxEnt 3.4.4 manifest is pinned.")
}
