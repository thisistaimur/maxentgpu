#!/usr/bin/env Rscript

# End-to-end benchmark: the package's analytic scalar CPU fit+predict route versus
# the shared-design Torch CUDA batch API. This is a package-route comparison, not an
# isolated device-only comparison.

root <- if (length(commandArgs(trailingOnly = TRUE))) normalizePath(commandArgs(trailingOnly = TRUE)[[1L]]) else normalizePath(".")
setwd(root)
if (!requireNamespace("torch", quietly = TRUE) || !isTRUE(torch::cuda_is_available())) stop("CUDA-enabled torch is required.", call. = FALSE)
if (!requireNamespace("maxentgpu", quietly = TRUE)) stop("Install maxentgpu first.", call. = FALSE)

species_n <- as.integer(Sys.getenv("MAXENTGPU_E2E_SPECIES", "32"))
background_n <- as.integer(Sys.getenv("MAXENTGPU_E2E_BACKGROUND", "5000"))
presence_n <- as.integer(Sys.getenv("MAXENTGPU_E2E_PRESENCE", "64"))
repeats <- as.integer(Sys.getenv("MAXENTGPU_E2E_REPEATS", "3"))
max_iter <- as.integer(Sys.getenv("MAXENTGPU_E2E_MAX_ITER", "1000"))
native <- identical(toupper(Sys.getenv("MAXENTGPU_E2E_NATIVE", "FALSE")), "TRUE")
batch_solver <- identical(toupper(Sys.getenv("MAXENTGPU_E2E_BATCH_SOLVER", "FALSE")), "TRUE")
fit_tol <- as.numeric(Sys.getenv("MAXENTGPU_E2E_TOL", "1e-8"))
parity_tol <- as.numeric(Sys.getenv("MAXENTGPU_E2E_PARITY_TOL", "1e-7"))
diagnostic_interval <- as.integer(Sys.getenv("MAXENTGPU_E2E_DIAGNOSTIC_INTERVAL", "5"))
if (any(!is.finite(c(species_n, background_n, presence_n, repeats, max_iter,
                     fit_tol, parity_tol, diagnostic_interval))) ||
    any(c(species_n, background_n, presence_n, repeats, max_iter,
          diagnostic_interval) < 1) || any(c(fit_tol, parity_tol) <= 0)) {
  stop("Benchmark controls must be positive.", call. = FALSE)
}

background <- data.frame(x1 = seq(-2, 2, length.out = background_n),
                         x2 = cos(seq(-2, 2, length.out = background_n) * 1.7))
presence <- lapply(seq_len(species_n), function(index) {
  phase <- (index - 1) / species_n
  x1 <- seq(-1.5, 1.5, length.out = presence_n) + 0.1 * sin(phase * 2 * pi)
  data.frame(x1 = x1, x2 = cos((x1 + phase) * 1.7))
})
names(presence) <- sprintf("species_%03d", seq_len(species_n))
regularization <- list(lambda1 = 0, lambda2 = 0.4)
control_cpu <- list(max_iter = max_iter, tol = fit_tol, accelerated = FALSE,
                    engine = "analytic", device = "cpu", dtype = "float64")
control_cuda <- list(max_iter = max_iter, tol = fit_tol, accelerated = FALSE,
                     engine = "torch", device = "cuda", dtype = "float64",
                     native = native, diagnostic_interval = diagnostic_interval,
                     batch_solver = if (batch_solver) "torch" else NULL)

sync <- function() torch::cuda_synchronize()
fit_cpu <- function() {
  models <- vector("list", length(presence))
  names(models) <- names(presence)
  diagnostics <- vector("list", length(presence))
  for (index in seq_along(presence)) {
    models[[index]] <- maxentgpu::maxent_fit(
      presence = presence[[index]], background = background, features = "linear",
      regularization = regularization, control = control_cpu)
    d <- maxentgpu::maxent_diagnostics(models[[index]])
    diagnostics[[index]] <- data.frame(
      species = names(presence)[[index]], iterations = d$iterations,
      converged = d$converged, stop_reason = d$stop_reason,
      parameter_change = d$parameter_change,
      optimality_norm = d$optimality_norm,
      stringsAsFactors = FALSE
    )
    if (!isTRUE(d$converged)) {
      print(diagnostics[[index]], row.names = FALSE)
      stop("CPU scalar fit did not converge for every species; refusing an incomparable timing.",
           call. = FALSE)
    }
  }
  list(models = models, diagnostics = do.call(rbind, diagnostics))
}
predict_cpu <- function(models) {
  do.call(cbind, lapply(models, predict, newdata = background, type = "link"))
}
fit_cuda <- function() {
  model <- maxentgpu::maxent_fit_batch(
    presence, background = background, features = "linear",
    regularization = regularization, control = control_cuda)
  if (!all(model$diagnostics$converged)) {
    failed <- model$diagnostics[!model$diagnostics$converged, , drop = FALSE]
    print(failed[, c("species", "iterations", "stop_reason", "parameter_change", "gradient_norm")], row.names = FALSE)
    stop("CUDA fit did not converge for every species.", call. = FALSE)
  }
  model
}
predict_cuda <- function(model) {
  predict(model, background, type = "link", device = "cuda", dtype = "float64", batch_size = 32L)
}
run_cpu <- function() {
  total_started <- proc.time()[["elapsed"]]
  fit_started <- total_started
  fitted <- fit_cpu()
  fit_seconds <- proc.time()[["elapsed"]] - fit_started
  predict_started <- proc.time()[["elapsed"]]
  prediction <- predict_cpu(fitted$models)
  predict_seconds <- proc.time()[["elapsed"]] - predict_started
  list(fit_seconds = fit_seconds, predict_seconds = predict_seconds,
       total_seconds = proc.time()[["elapsed"]] - total_started,
       prediction = prediction, diagnostics = fitted$diagnostics)
}
run_cuda <- function() {
  total_started <- proc.time()[["elapsed"]]
  fit_started <- total_started
  model <- fit_cuda()
  sync()
  fit_seconds <- proc.time()[["elapsed"]] - fit_started
  predict_started <- proc.time()[["elapsed"]]
  prediction <- predict_cuda(model)
  sync()
  predict_seconds <- proc.time()[["elapsed"]] - predict_started
  list(fit_seconds = fit_seconds, predict_seconds = predict_seconds,
       total_seconds = proc.time()[["elapsed"]] - total_started,
       prediction = prediction, diagnostics = model$diagnostics)
}
summarize_runs <- function(results) {
  summarize <- function(field) {
    elapsed <- vapply(results, `[[`, numeric(1), field)
    c(median = median(elapsed), min = min(elapsed), max = max(elapsed))
  }
  list(fit_time = summarize("fit_seconds"),
       predict_time = summarize("predict_seconds"),
       end_to_end_time = summarize("total_seconds"),
       result = results[[length(results)]])
}
measure_pair <- function() {
  runners <- list(cuda = run_cuda, cpu = run_cpu)
  results <- list(cuda = vector("list", repeats), cpu = vector("list", repeats))
  # Warm both routes, then alternate their order to reduce systematic order bias.
  invisible(run_cuda())
  invisible(run_cpu())
  for (i in seq_len(repeats)) {
    route_order <- if (i %% 2L == 1L) c("cuda", "cpu") else c("cpu", "cuda")
    for (route in route_order) results[[route]][[i]] <- runners[[route]]()
  }
  lapply(results, summarize_runs)
}
measured <- measure_pair()
cuda <- measured$cuda
cpu <- measured$cpu

cat("CUDA end-to-end fit-plus-predict benchmark\n")
git_executable <- Sys.which("git")
commit <- if (nzchar(git_executable)) {
  tryCatch({
    value <- suppressWarnings(system2(git_executable, c("rev-parse", "HEAD"),
                                      stdout = TRUE, stderr = FALSE))
    status <- attr(value, "status")
    if (is.null(status) || status == 0L) paste(value, collapse = "") else "unavailable"
  }, error = function(error) "unavailable")
} else "unavailable"
cat("commit:", commit, "\n")
cat("species:", species_n, "background:", background_n, "presence/species:", presence_n,
    "repeats:", repeats, "max_iter:", max_iter, "fit_tol:", fit_tol,
    "diagnostic_interval:", diagnostic_interval, "native:", native,
    "batch_solver:", batch_solver, "cpu_engine: analytic\n")
max_difference <- max(abs(cuda$result$prediction - cpu$result$prediction))
cat("max_abs_difference:", max_difference, "parity_tolerance:", parity_tol, "\n")
if (max_difference > parity_tol) {
  stop("fit-plus-predict parity exceeded MAXENTGPU_E2E_PARITY_TOL.", call. = FALSE)
}
for (route in c("cuda", "cpu")) {
  result <- get(route)
  cat(route, "fit_median_seconds:", result$fit_time[["median"]], "\n")
  cat(route, "fit_min_seconds:", result$fit_time[["min"]], "\n")
  cat(route, "fit_max_seconds:", result$fit_time[["max"]], "\n")
  cat(route, "predict_median_seconds:", result$predict_time[["median"]], "\n")
  cat(route, "predict_min_seconds:", result$predict_time[["min"]], "\n")
  cat(route, "predict_max_seconds:", result$predict_time[["max"]], "\n")
  cat(route, "end_to_end_median_seconds:", result$end_to_end_time[["median"]], "\n")
  cat(route, "end_to_end_min_seconds:", result$end_to_end_time[["min"]], "\n")
  cat(route, "end_to_end_max_seconds:", result$end_to_end_time[["max"]], "\n")
  cat(route, "converged_species:", sum(result$result$diagnostics$converged), "/", species_n, "\n")
  cat(route, "median_iterations:", median(result$result$diagnostics$iterations), "\n")
  cat(route, "total_iterations:", sum(result$result$diagnostics$iterations), "\n")
}
cat("speedup_cpu_over_cuda:", cpu$end_to_end_time[["median"]] /
      cuda$end_to_end_time[["median"]], "\n")
