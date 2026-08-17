args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")
if (!requireNamespace("maxentgpu", quietly = TRUE)) {
  stop("Install the local maxentgpu package before running this calibration.")
}

input_dir <- file.path(root, "tests", "fixtures", "reference", "input")
fixture_dir <- file.path(root, "tests", "fixtures", "maxnet")
background <- utils::read.csv(file.path(input_dir, "background.csv"))[, c("x1", "x2")]
presence <- utils::read.csv(file.path(input_dir, "samples.csv"))[, c("x1", "x2")]
reference <- utils::read.csv(file.path(fixture_dir, "predictions.csv"))
penalties <- utils::read.csv(file.path(fixture_dir, "penalty_factors.csv"))$penalty_factor
newdata <- rbind(background, presence)

lambda_grid <- seq(0.01, 0.20, by = 0.01)
results <- lapply(lambda_grid, function(lambda1) {
  fit <- maxentgpu::maxent_fit(
    x = presence, presence = presence, background = background,
    features = c("linear", "quadratic"),
    regularization = list(lambda1 = lambda1, lambda2 = 0,
                          penalty_l1 = penalties, penalty_l2 = 0),
    control = list(max_iter = 10000L, tol = 1e-9)
  )
  link <- maxentgpu:::predict.maxent_fit(fit, newdata, type = "link")
  mapping <- stats::lm(reference$link ~ link)
  data.frame(
    lambda1 = lambda1,
    converged = fit$diagnostics$converged,
    iterations = fit$diagnostics$iterations,
    nonzero_coefficients = sum(abs(fit$beta) > 0),
    correlation = stats::cor(link, reference$link),
    affine_rmse = sqrt(mean(stats::residuals(mapping)^2))
  )
})
write.csv(do.call(rbind, results), file = "", row.names = FALSE)
message("Calibration is fixture-specific and diagnostic only; it does not define a default lambda1.")
