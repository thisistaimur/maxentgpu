args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")
required_version <- "0.1.4"
if (!requireNamespace("maxnet", quietly = TRUE)) {
  stop("Install maxnet ", required_version, " before generating fixtures.")
}
actual_version <- as.character(utils::packageVersion("maxnet"))
if (!identical(actual_version, required_version)) {
  stop("Expected maxnet ", required_version, "; found ", actual_version, ".")
}

input_dir <- file.path(root, "tests", "fixtures", "reference", "input")
output_dir <- file.path(root, "tests", "fixtures", "maxnet")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
background <- utils::read.csv(file.path(input_dir, "background.csv"))
samples <- utils::read.csv(file.path(input_dir, "samples.csv"))
x <- rbind(background[c("x1", "x2")], samples[c("x1", "x2")])
p <- c(rep(FALSE, nrow(background)), rep(TRUE, nrow(samples)))
formula <- maxnet::maxnet.formula(p, x, classes = "lq")
write_fixture <- function(regmult, prefix = "") {
  fit <- maxnet::maxnet(p, x, f = formula, regmult = regmult)
  coefficient <- data.frame(feature = names(fit$betas),
                            coefficient = unname(fit$betas))
  utils::write.csv(coefficient, file.path(output_dir, paste0(prefix, "coefficients.csv")),
                   row.names = FALSE)
  prediction <- data.frame(
    row = seq_len(nrow(x)),
    link = as.numeric(stats::predict(fit, x, type = "link")),
    exponential = as.numeric(stats::predict(fit, x, type = "exponential")),
    cloglog = as.numeric(stats::predict(fit, x, type = "cloglog")),
    logistic = as.numeric(stats::predict(fit, x, type = "logistic"))
  )
  utils::write.csv(prediction, file.path(output_dir, paste0(prefix, "predictions.csv")),
                   row.names = FALSE)
  utils::write.csv(data.frame(regmult = regmult, alpha = unname(fit$alpha),
                              entropy = unname(fit$entropy),
                              n_background = nrow(background),
                              glmnet_lambda_min = mean(fit$penalty.factor) *
                                sum(p) / sum(p + (1 - p) * 100)),
                   file.path(output_dir, paste0(prefix, "scales.csv")), row.names = FALSE)
  penalty <- data.frame(feature = names(fit$penalty.factor),
                        penalty_factor = unname(fit$penalty.factor))
  utils::write.csv(penalty, file.path(output_dir, paste0(prefix, "penalty_factors.csv")),
                   row.names = FALSE)
}
write_fixture(regmult = 1)
# glmnet rejects an exactly zero penalty; this is the smallest positive value
# accepted by maxnet and is used as a near-unregularized diagnostic target.
write_fixture(regmult = 1e-8, prefix = "weakly_regularized_")
writeLines(deparse(formula), file.path(output_dir, "formula.txt"))
write.dcf(data.frame(
  Generator = "tools/generate-maxnet-fixtures.R",
  MaxnetVersion = actual_version,
  RVersion = R.version.string,
  Command = paste("Rscript tools/generate-maxnet-fixtures.R", shQuote(root)),
  stringsAsFactors = FALSE
), file.path(output_dir, "provenance.dcf"))
hash_files <- c(
  file.path(input_dir, "background.csv"),
  file.path(input_dir, "samples.csv"),
  file.path(output_dir, "coefficients.csv"),
  file.path(output_dir, "predictions.csv"),
  file.path(output_dir, "formula.txt"),
  file.path(output_dir, "provenance.dcf")
)
hash_files <- c(hash_files,
  file.path(output_dir, "weakly_regularized_coefficients.csv"),
  file.path(output_dir, "weakly_regularized_predictions.csv"),
  file.path(output_dir, "scales.csv"),
  file.path(output_dir, "weakly_regularized_scales.csv"),
  file.path(output_dir, "penalty_factors.csv"),
  file.path(output_dir, "weakly_regularized_penalty_factors.csv")
)
hashes <- tools::md5sum(hash_files)
relative_names <- substring(normalizePath(hash_files), nchar(root) + 2L)
utils::write.table(
  data.frame(md5 = unname(hashes), file = relative_names),
  file.path(output_dir, "checksums.md5"), row.names = FALSE, quote = FALSE
)
message("Generated maxnet fixtures in ", output_dir)
