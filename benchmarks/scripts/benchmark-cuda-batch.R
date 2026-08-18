#!/usr/bin/env Rscript

# Benchmark shared-design CUDA prediction against the current independent scalar
# prediction path. Fitting is performed once before timing and is not included in
# either prediction measurement.

root <- if (length(commandArgs(trailingOnly = TRUE))) {
  normalizePath(commandArgs(trailingOnly = TRUE)[[1L]])
} else normalizePath(".")
setwd(root)

if (!requireNamespace("torch", quietly = TRUE) || !isTRUE(torch::cuda_is_available())) {
  stop("CUDA-enabled torch is required for this benchmark.", call. = FALSE)
}
if (!requireNamespace("maxentgpu", quietly = TRUE)) {
  stop("Install maxentgpu into the active R library before running this benchmark.", call. = FALSE)
}

species_n <- as.integer(Sys.getenv("MAXENTGPU_BENCH_SPECIES", "64"))
background_n <- as.integer(Sys.getenv("MAXENTGPU_BENCH_BACKGROUND", "2000"))
presence_n <- as.integer(Sys.getenv("MAXENTGPU_BENCH_PRESENCE", "32"))
repeats <- as.integer(Sys.getenv("MAXENTGPU_BENCH_REPEATS", "5"))
chunk_size <- as.integer(Sys.getenv("MAXENTGPU_BENCH_CHUNK", "16"))
max_iter <- as.integer(Sys.getenv("MAXENTGPU_BENCH_MAX_ITER", "2000"))
if (any(!is.finite(c(species_n, background_n, presence_n, repeats, chunk_size, max_iter))) ||
    any(c(species_n, background_n, presence_n, repeats, chunk_size, max_iter) < 1L)) {
  stop("Benchmark sizes must be positive integers.", call. = FALSE)
}

background <- data.frame(
  x1 = seq(-2, 2, length.out = background_n),
  x2 = cos(seq(-2, 2, length.out = background_n) * 1.7)
)
presence <- lapply(seq_len(species_n), function(index) {
  phase <- (index - 1) / species_n
  x1 <- seq(-1.5, 1.5, length.out = presence_n) + 0.1 * sin(phase * 2 * pi)
  data.frame(x1 = x1, x2 = cos((x1 + phase) * 1.7))
})
names(presence) <- sprintf("species_%03d", seq_len(species_n))

batch <- maxentgpu::maxent_fit_batch(
  presence, background = background, features = "linear",
  regularization = list(lambda1 = 0, lambda2 = 0.4),
  control = list(max_iter = max_iter, tol = 1e-6, accelerated = FALSE,
                 engine = "torch", device = "cuda", dtype = "float64")
)
if (!all(batch$diagnostics$converged)) {
  stop("Benchmark requires all species to converge; increase MAXENTGPU_BENCH_MAX_ITER.", call. = FALSE)
}

sync <- function() if (isTRUE(torch::cuda_is_available())) torch::cuda_synchronize()
measure <- function(fun) {
  elapsed <- numeric(repeats)
  for (iteration in seq_len(repeats)) {
    started <- proc.time()[["elapsed"]]
    invisible(fun())
    sync()
    elapsed[[iteration]] <- proc.time()[["elapsed"]] - started
  }
  c(median = median(elapsed), min = min(elapsed), max = max(elapsed))
}

# Warm up both paths before timing.
invisible(predict(batch, background, type = "link", device = "cuda",
                  dtype = "float64", batch_size = chunk_size))
invisible(do.call(cbind, lapply(batch$models, predict, newdata = background, type = "link")))
sync()

dense_time <- measure(function() {
  predict(batch, background, type = "link", device = "cuda",
          dtype = "float64", batch_size = chunk_size)
})
scalar_time <- measure(function() {
  do.call(cbind, lapply(batch$models, predict, newdata = background, type = "link"))
})
dense <- predict(batch, background, type = "link", device = "cuda",
                  dtype = "float64", batch_size = chunk_size)
scalar <- do.call(cbind, lapply(batch$models, predict, newdata = background, type = "link"))

cat("CUDA batch prediction benchmark\n")
cat("commit:", system("git rev-parse HEAD", intern = TRUE), "\n")
cat("species:", species_n, "background:", background_n,
    "presence/species:", presence_n, "repeats:", repeats,
    "chunk:", chunk_size, "max_iter:", max_iter, "\n")
cat("converged_species:", sum(batch$diagnostics$converged), "/", species_n, "\n")
cat("max_abs_difference:", max(abs(dense - scalar)), "\n")
cat("dense_cuda_median_seconds:", dense_time[["median"]], "\n")
cat("dense_cuda_min_seconds:", dense_time[["min"]], "\n")
cat("dense_cuda_max_seconds:", dense_time[["max"]], "\n")
cat("scalar_cpu_median_seconds:", scalar_time[["median"]], "\n")
cat("scalar_cpu_min_seconds:", scalar_time[["min"]], "\n")
cat("scalar_cpu_max_seconds:", scalar_time[["max"]], "\n")
cat("speedup_scalar_cpu_over_dense_cuda:", scalar_time[["median"]] / dense_time[["median"]], "\n")
