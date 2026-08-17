args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")
if (!requireNamespace("maxentgpu", quietly = TRUE)) {
  stop("Install the local maxentgpu package before running this comparison.")
}

input_dir <- file.path(root, "tests", "fixtures", "reference", "input")
fixture_dir <- file.path(root, "tests", "fixtures", "maxnet")
background <- utils::read.csv(file.path(input_dir, "background.csv"))
samples <- utils::read.csv(file.path(input_dir, "samples.csv"))
presence <- samples[c("x1", "x2")]
background <- background[c("x1", "x2")]
reference <- utils::read.csv(file.path(fixture_dir, "predictions.csv"))
scales <- utils::read.csv(file.path(fixture_dir, "scales.csv"))
penalties <- utils::read.csv(file.path(fixture_dir, "penalty_factors.csv"))
if (any(!is.finite(penalties$penalty_factor)) || any(penalties$penalty_factor <= 0)) {
  stop("maxnet penalty-factor metadata is invalid.")
}

fit <- maxentgpu::maxent_fit(
  x = presence,
  presence = presence,
  background = background,
  features = c("linear", "quadratic"),
  regularization = list(lambda1 = 0, lambda2 = 1),
  control = list(max_iter = 5000L, tol = 1e-10)
)
newdata <- rbind(background, presence)
native <- data.frame(
  row = seq_len(nrow(newdata)),
  link = maxentgpu:::predict.maxent_fit(fit, newdata, type = "link"),
  raw = maxentgpu:::predict.maxent_fit(fit, newdata, type = "raw")
)
if (!identical(native$row, reference$row) || nrow(native) != nrow(reference)) {
  stop("reference and package fixture row ordering differ.")
}
if (any(!is.finite(as.matrix(native)))) stop("package-native predictions are non-finite.")

report <- data.frame(
  comparison = c("link", "raw_vs_maxnet_exponential", "link_affine_mapping", "raw_multiplicative_mapping"),
  correlation = c(stats::cor(native$link, reference$link),
                  stats::cor(native$raw, reference$exponential), NA_real_, NA_real_),
  package_scale = rep(c("canonical link", "normalized background density"), 2),
  reference_scale = rep(c("maxnet link with intercept", "maxnet discrete exponential"), 2),
  intercept = NA_real_, slope = NA_real_, rmse = NA_real_,
  ratio_median = NA_real_, log_rmse = NA_real_
)
link_mapping <- stats::lm(reference$link ~ native$link)
report[report$comparison == "link_affine_mapping", c("intercept", "slope", "rmse")] <- c(
  unname(stats::coef(link_mapping)[1]), unname(stats::coef(link_mapping)[2]),
  sqrt(mean(stats::residuals(link_mapping)^2))
)
reference_link_no_intercept <- reference$link - scales$alpha[[1]]
link_no_intercept_mapping <- stats::lm(reference_link_no_intercept ~ native$link)
report <- rbind(report, data.frame(
  comparison = "link_without_reference_intercept",
  correlation = stats::cor(native$link, reference_link_no_intercept),
  package_scale = "canonical link", reference_scale = "maxnet link minus alpha",
  intercept = unname(stats::coef(link_no_intercept_mapping)[1]),
  slope = unname(stats::coef(link_no_intercept_mapping)[2]),
  rmse = sqrt(mean(stats::residuals(link_no_intercept_mapping)^2)),
  ratio_median = NA_real_, log_rmse = NA_real_
))
raw_ratio <- reference$exponential / native$raw
report[report$comparison == "raw_multiplicative_mapping", c("ratio_median", "log_rmse")] <- c(
  stats::median(raw_ratio), sqrt(mean((log(raw_ratio) - mean(log(raw_ratio)))^2))
)
write.csv(report, file = "", row.names = FALSE)
message("Comparison is diagnostic only: scales are not declared equivalent.")
message("maxnet penalty factors: ", paste(penalties$feature, penalties$penalty_factor,
                                           sep = "=", collapse = ", "))
