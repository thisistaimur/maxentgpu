args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop("Usage: Rscript tools/generate-java-maxent-fixtures.R /path/to/maxent.jar [repo-root]")
}
jar <- normalizePath(args[[1]], mustWork = TRUE)
root <- if (length(args) >= 2) normalizePath(args[[2]]) else normalizePath(".")
manifest_path <- file.path(root, "benchmarks", "manifests",
                           "reference-versions.csv")
manifest <- utils::read.csv(manifest_path, stringsAsFactors = FALSE,
                            check.names = FALSE)
row <- manifest[manifest$component == "java-maxent", ]
if (nrow(row) != 1 || !nzchar(row$sha256) ||
    grepl("^REPLACE_", row$sha256)) {
  stop("Record the licensed maxent.jar SHA-256 in ", manifest_path,
       " before generation.")
}

sha256 <- function(path) {
  command <- if (nzchar(Sys.which("sha256sum"))) "sha256sum" else "shasum"
  command_args <- if (identical(command, "shasum")) c("-a", "256", path) else path
  output <- system2(command, command_args, stdout = TRUE, stderr = TRUE)
  if (!length(output) || !identical(attr(output, "status") %||% 0L, 0L)) {
    stop("Unable to calculate SHA-256 for ", path)
  }
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}
`%||%` <- function(left, right) if (is.null(left)) right else left

actual_sha256 <- sha256(jar)
if (!identical(tolower(actual_sha256), tolower(row$sha256))) {
  stop("maxent.jar SHA-256 mismatch: expected ", row$sha256,
       ", got ", actual_sha256)
}

input_dir <- file.path(root, "tests", "fixtures", "reference", "input")
output_dir <- file.path(root, "tests", "fixtures", "java-maxent")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
java_args <- c(
  "-Xmx1g", "-jar", jar,
  paste0("samplesfile=", file.path(input_dir, "samples.csv")),
  paste0("environmentallayers=", file.path(input_dir, "background.csv")),
  paste0("outputdirectory=", output_dir),
  "autorun", "redoifexists", "removeduplicates=false", "randomseed=false",
  "threads=1", "responsecurves=false", "pictures=false", "plots=false",
  "writeplotdata=false", "writebackgroundpredictions=true",
  "outputformat=cloglog"
)
status <- system2("java", java_args, stdout = TRUE, stderr = TRUE)
writeLines(status, file.path(output_dir, "generator.log"))
exit_status <- attr(status, "status") %||% 0L
if (!identical(exit_status, 0L)) {
  stop("Java MaxEnt exited with status ", exit_status,
       "; see generator.log. REF-JAVA-001 remains open.")
}
write.dcf(data.frame(
  Generator = "tools/generate-java-maxent-fixtures.R",
  MaxEntVersion = row$version,
  JarSHA256 = actual_sha256,
  JavaVersion = paste(system2("java", "-version", stdout = TRUE,
                              stderr = TRUE), collapse = " | "),
  Command = paste(
    "java",
    paste(shQuote({
      provenance_args <- java_args
      provenance_args[[3]] <- "maxent.jar"
      provenance_args <- sub(
        "^samplesfile=.*$",
        "samplesfile=tests/fixtures/reference/input/samples.csv",
        provenance_args
      )
      provenance_args <- sub(
        "^environmentallayers=.*$",
        "environmentallayers=tests/fixtures/reference/input/background.csv",
        provenance_args
      )
      provenance_args <- sub(
        "^outputdirectory=.*$",
        "outputdirectory=tests/fixtures/java-maxent",
        provenance_args
      )
      provenance_args
    }), collapse = " ")
  ),
  stringsAsFactors = FALSE
), file.path(output_dir, "provenance.dcf"))
generated <- list.files(
  output_dir,
  pattern = "[.](csv|dcf|lambdas)$",
  full.names = TRUE
)
generated_hashes <- tools::md5sum(generated)
generated_names <- substring(normalizePath(generated), nchar(root) + 2L)
utils::write.table(
  data.frame(md5 = unname(generated_hashes), file = generated_names),
  file.path(output_dir, "checksums.md5"), row.names = FALSE, quote = FALSE
)
message("Generated Java MaxEnt fixtures in ", output_dir)
