test_that("L/Q feature specifications are immutable and deterministic", {
  x <- data.frame(x1 = c(0, 1, 2), x2 = c(1, 2, 4))
  spec <- maxent_feature_spec(maxent_fit(x, c(TRUE, TRUE, FALSE), features = "linear"))
  expect_identical(spec$columns, c("L:x1", "L:x2"))
  expect_equal(unname(apply_feature_spec(spec, x)), unname(as.matrix(x)))
})

test_that("CPU fit has stable objective and link/raw predictions", {
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  fit <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                    control = list(max_iter = 500L, tol = 1e-9, step = 1))
  diagnostics <- maxent_diagnostics(fit)
  expect_true(all(diff(diagnostics$objective) <= 1e-10))
  expect_true(all(is.finite(predict(fit, x, type = "link"))))
  expect_true(all(predict(fit, x, type = "raw") > 0))
  expect_equal(sum(fit$background_weights * predict(fit, fit$background, type = "raw")), 1, tolerance = 1e-8)
})

test_that("weighted duplicate rows preserve the normalized objective", {
  presence <- data.frame(x1 = c(0, 0, 1))
  background <- data.frame(x1 = c(2, 3, 4))
  expanded <- maxent_fit(presence, presence, background = background,
                         presence_weights = c(1, 1, 2),
                         features = "linear")
  aggregated <- maxent_fit(data.frame(x1 = c(0, 1)), data.frame(x1 = c(0, 1)),
                           background = background, presence_weights = c(2, 2),
                           features = "linear")
  expect_equal(expanded$diagnostics$final_objective,
               aggregated$diagnostics$final_objective, tolerance = 1e-7)
})

test_that("analytic smooth gradient agrees with central differences", {
  presence_phi <- matrix(c(0, 1, 1, 2), nrow = 2, byrow = TRUE)
  background_phi <- matrix(c(2, 3, 3, 4, 4, 5), nrow = 3, byrow = TRUE)
  w <- c(0.25, 0.75)
  q <- c(0.2, 0.3, 0.5)
  beta <- c(0.15, -0.2)
  lambda2 <- 0.4
  objective <- function(value) {
    maxentgpu:::objective_components(beta = value,
      presence_phi = presence_phi, background_phi = background_phi,
      w = w, q = q, lambda1 = 0, lambda2 = lambda2,
      r1 = c(1, 1), r2 = c(1, 1))$value
  }
  analytic <- maxentgpu:::objective_components(beta, presence_phi, background_phi,
    w, q, 0, lambda2, c(1, 1), c(1, 1))$gradient
  epsilon <- 1e-6
  numeric <- vapply(seq_along(beta), function(index) {
    plus <- beta
    minus <- beta
    plus[index] <- plus[index] + epsilon
    minus[index] <- minus[index] - epsilon
    (objective(plus) - objective(minus)) / (2 * epsilon)
  }, numeric(1))
  expect_equal(analytic, numeric, tolerance = 1e-7)
})

test_that("Torch autograd agrees with the stable analytic gradient", {
  skip_if_not_installed("torch")
  presence_phi <- matrix(c(0, 1, 1, 2), nrow = 2, byrow = TRUE)
  background_phi <- matrix(c(2, 3, 3, 4, 4, 5), nrow = 3, byrow = TRUE)
  w <- c(0.25, 0.75)
  q <- c(0.2, 0.3, 0.5)
  beta <- c(0.15, -0.2)
  lambda2 <- 0.4
  analytic <- maxentgpu:::objective_components(beta, presence_phi, background_phi,
    w, q, 0, lambda2, c(1, 1), c(1, 1))$gradient
  beta_t <- torch::torch_tensor(beta, dtype = torch::torch_float64(), requires_grad = TRUE)
  presence_t <- torch::torch_tensor(presence_phi, dtype = torch::torch_float64())
  background_t <- torch::torch_tensor(background_phi, dtype = torch::torch_float64())
  w_t <- torch::torch_tensor(w, dtype = torch::torch_float64())
  q_t <- torch::torch_tensor(q, dtype = torch::torch_float64())
  z_presence <- presence_t$matmul(beta_t)
  z_background <- background_t$matmul(beta_t)
  logz <- torch::torch_logsumexp(torch::torch_log(q_t) + z_background, dim = 1)
  loss <- logz - (w_t * z_presence)$sum() + 0.5 * lambda2 * (beta_t^2)$sum()
  loss$backward()
  expect_equal(as.numeric(beta_t$grad), analytic, tolerance = 1e-7)
})
