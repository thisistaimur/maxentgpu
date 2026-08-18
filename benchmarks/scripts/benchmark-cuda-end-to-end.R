#!/usr/bin/env Rscript

# End-to-end benchmark: independent scalar CPU Torch fit+predict versus the current
# CUDA batch API. Using the same Torch engine isolates device performance.

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
                    engine = "torch", device = "cpu", dtype = "float64")
control_cuda <- list(max_iter = max_iter, tol = fit_tol, accelerated = FALSE,
                     engine = "torch", device = "cuda", dtype = "float64",
                     native = native, diagnostic_interval = diagnostic_interval)

sync <- function() torch::cuda_synchronize()
run_cpu <- function() {
  models <- lapply(presence, function(p) maxentgpu::maxent_fit(
    presence = p, background = background, features = "linear",
    regularization = regularization, control = control_cpu))
  do.call(cbind, lapply(models, predict, newdata = background, type = "link"))
}
run_cuda <- function() {
  model <- maxentgpu::maxent_fit_batch(
    presence, background = background, features = "linear",
    regularization = regularization, control = control_cuda)
  if (!all(model$diagnostics$converged)) stop("CUDA fit did not converge for every species.", call. = FALSE)
  predict(model, background, type = "link", device = "cuda", dtype = "float64", batch_size = 32L)
}

invisible(run_cuda()); sync()
invisible(run_cpu())
measure <- function(fun, synchronize = FALSE) {
  elapsed <- numeric(repeats)
  for (i in seq_len(repeats)) {
    started <- proc.time()[["elapsed"]]
    result <- fun()
    if (synchronize) sync()
    elapsed[[i]] <- proc.time()[["elapsed"]] - started
  }
  list(time = c(median = median(elapsed), min = min(elapsed), max = max(elapsed)), result = result)
}
cuda <- measure(run_cuda, synchronize = TRUE)
cpu <- measure(run_cpu)

cat("CUDA end-to-end fit-plus-predict benchmark\n")
cat("commit:", system("git rev-parse HEAD", intern = TRUE), "\n")
cat("species:", species_n, "background:", background_n, "presence/species:", presence_n,
    "repeats:", repeats, "max_iter:", max_iter, "fit_tol:", fit_tol,
    "diagnostic_interval:", diagnostic_interval, "native:", native, "\n")
max_difference <- max(abs(cuda$result - cpu$result))
if (max_difference > parity_tol) stop("fit-plus-predict parity exceeded MAXENTGPU_E2E_PARITY_TOL.", call. = FALSE)
cat("max_abs_difference:", max_difference, "parity_tolerance:", parity_tol, "\n")
cat("cuda_median_seconds:", cuda$time[["median"]], "\n")
cat("cuda_min_seconds:", cuda$time[["min"]], "\n")
cat("cuda_max_seconds:", cuda$time[["max"]], "\n")
cat("cpu_median_seconds:", cpu$time[["median"]], "\n")
cat("cpu_min_seconds:", cpu$time[["min"]], "\n")
cat("cpu_max_seconds:", cpu$time[["max"]], "\n")
cat("speedup_cpu_over_cuda:", cpu$time[["median"]] / cuda$time[["median"]], "\n")
