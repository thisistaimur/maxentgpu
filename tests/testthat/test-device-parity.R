backend_available <- function(backend) {
  probe <- maxent_device_probe()
  isTRUE(probe$available[match(backend, probe$backend)])
}

test_that("Torch objective is numerically portable to available accelerators", {
  skip_if_not_installed("torch")
  presence_phi <- matrix(c(0, 1, 1, 2), nrow = 2, byrow = TRUE)
  background_phi <- matrix(c(2, 3, 3, 4, 4, 5), nrow = 3, byrow = TRUE)
  beta <- c(0.15, -0.2)
  args <- list(beta = beta, presence_phi = presence_phi, background_phi = background_phi,
               w = c(0.25, 0.75), q = c(0.2, 0.3, 0.5), lambda1 = 0,
               lambda2 = 0.4, r1 = c(1, 1), r2 = c(1, 1))
  analytic <- do.call(maxentgpu:::objective_components, args)

  for (backend in c("cuda", "mps")) {
    skip_if(!backend_available(backend), paste(toupper(backend), "is unavailable"))
    dtype <- if (backend == "mps") "float32" else "float64"
    result <- do.call(maxentgpu:::torch_objective_components,
                      c(args, list(device = backend, dtype = dtype)))
    tolerance <- if (backend == "mps") 2e-4 else 1e-7
    expect_equal(result$value, analytic$value, tolerance = tolerance,
                 info = paste("objective on", backend))
    expect_equal(result$gradient, analytic$gradient, tolerance = tolerance,
                 info = paste("gradient on", backend))
  }
})

test_that("scalar Torch fits agree across CPU and available accelerators", {
  skip_if_not_installed("torch")
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  cpu <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                    regularization = list(lambda1 = 0, lambda2 = 0.4),
                    control = list(max_iter = 500L, tol = 1e-8, engine = "torch"))
  cpu_pred <- predict(cpu, x, type = "raw")

  for (backend in c("cuda", "mps")) {
    skip_if(!backend_available(backend), paste(toupper(backend), "is unavailable"))
    dtype <- if (backend == "mps") "float32" else "float64"
    accelerated <- maxent_fit(
      x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
      regularization = list(lambda1 = 0, lambda2 = 0.4),
      control = list(max_iter = 500L, tol = 1e-8, engine = "torch",
                     device = backend, dtype = dtype)
    )
    tolerance <- if (backend == "mps") 3e-3 else 1e-6
    expect_equal(predict(accelerated, x, type = "raw"), cpu_pred,
                 tolerance = tolerance, info = paste("predictions on", backend))
    expect_identical(maxent_diagnostics(accelerated)$device, backend)
    expect_identical(maxent_diagnostics(accelerated)$engine, "torch")
  }
})
