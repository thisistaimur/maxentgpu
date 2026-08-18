test_that("L/Q feature specifications are immutable and deterministic", {
  x <- data.frame(x1 = c(0, 1, 2), x2 = c(1, 2, 4))
  spec <- maxent_feature_spec(maxent_fit(x, c(TRUE, TRUE, FALSE), features = "linear"))
  expect_identical(spec$columns, c("L:x1", "L:x2"))
  expect_equal(unname(apply_feature_spec(spec, x)), unname(as.matrix(x)))
})

test_that("product features have deterministic pair ordering", {
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 4, 1), x3 = c(2, 1, 3, 4))
  fit <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "product")
  spec <- maxent_feature_spec(fit)
  expect_identical(spec$columns, c("P:x1*x2", "P:x1*x3", "P:x2*x3"))
  expect_equal(unname(apply_feature_spec(spec, x)[1, ]), c(0, 0, 2))
})

test_that("threshold features store knots and apply deterministic indicators", {
  x <- data.frame(x1 = c(0, 1, 2, 3, 4), x2 = c(1, 2, 4, 1, 3))
  fit <- maxent_fit(x, c(TRUE, TRUE, TRUE, FALSE, FALSE), features = "threshold",
                    thresholds = list(x1 = c(1.5, 2.5), x2 = 2.5))
  spec <- maxent_feature_spec(fit)
  expect_identical(spec$columns, c("T:x1<=1.5", "T:x1<=2.5", "T:x2<=2.5"))
  expect_equal(unname(apply_feature_spec(spec, x)[1, ]), c(1, 1, 1))
  expect_equal(spec$thresholds$x1, c(1.5, 2.5))
})

test_that("hinge features store knots and apply forward and reverse bases", {
  x <- data.frame(x1 = c(0, 1, 2, 3, 4), x2 = c(1, 2, 4, 1, 3))
  spec <- maxentgpu:::new_feature_spec(x, classes = "hinge",
                                       knots = list(x1 = 2, x2 = 2.5))
  expect_identical(spec$columns, c("H+:x1>2", "H-:x1<2", "H+:x2>2.5", "H-:x2<2.5"))
  expect_equal(unname(apply_feature_spec(spec, x)[1, ]), c(0, 2, 0, 1.5))
  expect_equal(spec$knots$x1, 2)
})

test_that("categorical features store levels and reject unseen values", {
  x <- data.frame(habitat = c("forest", "grass", "forest", "wetland", "grass", "wetland", "forest", "grass", "wetland"),
                  region = factor(c("north", "south", "east", "south", "north", "east", "east", "north", "south")))
  fit <- maxent_fit(x, c(TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE), features = "categorical")
  spec <- maxent_feature_spec(fit)
  expect_identical(spec$columns, c("C:habitat=forest", "C:habitat=grass", "C:habitat=wetland",
                                   "C:region=east", "C:region=north", "C:region=south"))
  expect_equal(rowSums(apply_feature_spec(spec, x)), rep(2, nrow(x)))
  expect_error(predict(fit, data.frame(habitat = "desert", region = "north"), type = "link"),
               "unseen levels")
})

test_that("auto feature policy has documented sample-size boundaries", {
  expect_identical(maxent_auto_features(9), "linear")
  expect_identical(maxent_auto_features(10), c("linear", "quadratic"))
  expect_identical(maxent_auto_features(14), c("linear", "quadratic"))
  expect_identical(maxent_auto_features(15), c("linear", "quadratic", "hinge"))
  expect_identical(maxent_auto_features(79), c("linear", "quadratic", "hinge"))
  expect_identical(maxent_auto_features(80), c("linear", "quadratic", "hinge", "product"))
  expect_error(maxent_auto_features(0), "positive")
})

test_that("chunked feature application equals whole-matrix application", {
  x <- data.frame(x1 = c(0, 1, 2, 3, 4), x2 = c(1, 2, 4, 1, 3))
  spec <- maxentgpu:::new_feature_spec(x, classes = c("linear", "quadratic", "product"))
  whole <- maxent_feature_matrix(spec, x)
  chunked <- maxent_feature_matrix(spec, x, chunk_size = 2)
  expect_equal(chunked, whole)
  permutation <- c(5, 2, 4, 1, 3)
  expect_equal(maxent_feature_matrix(spec, x[permutation, ]), whole[permutation, ])
  expect_error(maxent_feature_matrix(spec, x, chunk_size = 0), "positive")
})

test_that("feature properties guard duplicates, missing values, extremes, and level order", {
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 4, 1))
  threshold_spec <- maxentgpu:::new_feature_spec(x, classes = "threshold",
                                                  thresholds = list(x1 = c(1.5, 1.5), x2 = 2))
  expect_identical(threshold_spec$columns, c("T:x1<=1.5", "T:x2<=2"))
  hinge_spec <- maxentgpu:::new_feature_spec(x, classes = "hinge",
                                             knots = list(x1 = c(1.5, 1.5), x2 = 2))
  expect_identical(hinge_spec$columns, c("H+:x1>1.5", "H-:x1<1.5", "H+:x2>2", "H-:x2<2"))
  expect_error(maxentgpu:::apply_feature_spec(hinge_spec,
                                              data.frame(x1 = NA_real_, x2 = 2)), "non-finite")
  extreme <- maxentgpu:::new_feature_spec(data.frame(x = c(-1e200, 1e200)), classes = "quadratic")
  expect_error(maxentgpu:::apply_feature_spec(extreme, data.frame(x = c(-1e200, 1e200))), "non-finite")
  categorical <- maxentgpu:::new_feature_spec(
    data.frame(group = factor(c("b", "a"), levels = c("b", "a"))), classes = "categorical")
  expect_identical(categorical$columns, c("C:group=a", "C:group=b"))
})

test_that("CPU fit has stable objective and link/raw predictions", {
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  fit <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                    control = list(max_iter = 500L, tol = 1e-9, step = 1))
  diagnostics <- maxent_diagnostics(fit)
  expect_true(all(diff(diagnostics$objective) <= 1e-10))
  expect_true(diagnostics$stop_reason %in% c("parameter_change", "max_iter"))
  expect_true(is.finite(diagnostics$parameter_change))
  expect_true(is.finite(diagnostics$smooth_gradient_norm))
  expect_true(all(is.finite(predict(fit, x, type = "link"))))
  expect_true(all(predict(fit, x, type = "raw") > 0))
  expect_equal(sum(fit$background_weights * predict(fit, fit$background, type = "raw")), 1, tolerance = 1e-8)
})

test_that("monotone FISTA agrees with non-accelerated proximal updates", {
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  accelerated <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                            control = list(max_iter = 1000L, tol = 1e-9,
                                           step = 1, accelerated = TRUE))
  plain <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                      control = list(max_iter = 1000L, tol = 1e-9,
                                     step = 1, accelerated = FALSE))
  expect_true(all(diff(maxent_diagnostics(accelerated)$objective) <= 1e-10))
  expect_equal(predict(accelerated, x, type = "raw"),
               predict(plain, x, type = "raw"), tolerance = 1e-6)
})

test_that("model serialization preserves predictions and diagnostics", {
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  fit <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear")
  path <- tempfile(fileext = ".rds")
  expect_identical(save_maxent_model(fit, path), path)
  restored <- read_maxent_model(path)
  expect_equal(predict(restored, x, type = "raw"), predict(fit, x, type = "raw"))
  expect_equal(maxent_diagnostics(restored), maxent_diagnostics(fit))
  expect_equal(summary(restored)$entropy, summary(fit)$entropy)
})

test_that("invalid training inputs fail explicitly", {
  expect_error(maxent_fit(data.frame(x = c(1, 2)), c(TRUE, FALSE),
                          presence_weights = c(1, 0)), "strictly positive")
  expect_error(maxent_fit(data.frame(x = c(1, 1)), c(TRUE, FALSE)), "constant")
  expect_error(maxent_fit(data.frame(x = c(1, 2)), c(TRUE, FALSE),
                          features = "bogus"), "linear.*quadratic.*product.*threshold.*hinge")
  fit <- maxent_fit(data.frame(x = c(1, 2, 3)), c(TRUE, FALSE, FALSE), features = "linear")
  expect_error(predict(fit, data.frame(y = 1), type = "raw"), "missing predictors")
  expect_error(predict(fit, data.frame(x = 1), type = "cloglog"), "should be one of")
  expect_error(maxent_fit(data.frame(x1 = c(1, 2, 3), x2 = c(1, 2, 3)),
                          c(TRUE, FALSE, FALSE), features = "linear"),
               "rank-deficient")
  expect_error(maxent_fit(data.frame(x = c(1, 2, 3)), c(TRUE, FALSE, FALSE),
                          control = list(device = "cuda")), "requires control engine")
  expect_error(maxent_fit(data.frame(x = c(1, 2, 3)), c(TRUE, FALSE, FALSE),
                          control = list(dtype = "float32")), "requires dtype")
})

test_that("numeric guards reject invalid partition inputs", {
  expect_error(maxentgpu:::stable_logz(c(1, Inf), c(0.5, 0.5)), "finite")
  expect_error(maxentgpu:::stable_logz(c(1, 2), c(0, 1)), "strictly positive")
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

test_that("feature-specific regularization multipliers are audited", {
  x <- data.frame(x1 = c(0, 1, 2, 3, 4), x2 = c(1, 2, 4, 1, 3))
  fit <- maxent_fit(x, c(TRUE, TRUE, TRUE, FALSE, FALSE), features = c("linear", "quadratic"),
                    regularization = list(lambda1 = 0.1, lambda2 = 0.2,
                                          penalty_l1 = c(1, 2, 3, 4),
                                          penalty_l2 = c(4, 3, 2, 1)))
  coefficients <- maxent_coefficients(fit)
  expect_equal(coefficients$penalty_l1, c(1, 2, 3, 4))
  expect_equal(coefficients$penalty_l2, c(4, 3, 2, 1))
  expect_true(all(c("class", "source", "knot", "level") %in% names(coefficients)))
  expect_error(maxent_fit(x, c(TRUE, TRUE, TRUE, FALSE, FALSE),
                          regularization = list(penalty_l1 = c(1, 2))),
               "every feature")
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

test_that("package Torch objective oracle agrees with analytic objective", {
  skip_if_not_installed("torch")
  presence_phi <- matrix(c(0, 1, 1, 2), nrow = 2, byrow = TRUE)
  background_phi <- matrix(c(2, 3, 3, 4, 4, 5), nrow = 3, byrow = TRUE)
  beta <- c(0.15, -0.2)
  args <- list(beta = beta, presence_phi = presence_phi, background_phi = background_phi,
               w = c(0.25, 0.75), q = c(0.2, 0.3, 0.5), lambda1 = 0,
               lambda2 = 0.4, r1 = c(1, 1), r2 = c(1, 1))
  analytic <- do.call(maxentgpu:::objective_components, args)
  oracle <- do.call(maxentgpu:::torch_objective_components, args)
  expect_equal(oracle$value, analytic$value, tolerance = 1e-7)
  expect_equal(oracle$gradient, analytic$gradient, tolerance = 1e-7)
})

test_that("Torch solver engine matches the analytic CPU engine", {
  skip_if_not_installed("torch")
  x <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  analytic <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                          control = list(max_iter = 500L, tol = 1e-8, engine = "analytic"))
  torch_fit <- maxent_fit(x, c(TRUE, TRUE, FALSE, FALSE), features = "linear",
                          control = list(max_iter = 500L, tol = 1e-8, engine = "torch"))
  torch_raw <- predict(torch_fit, x, type = "raw")
  analytic_raw <- predict(analytic, x, type = "raw")
  expect_lt(max(abs((torch_raw - analytic_raw) / analytic_raw)), 1e-4)
  expect_identical(maxent_diagnostics(torch_fit)$engine, "torch")
})

test_that("optional solver profiling reports objective evaluation counts", {
  fit <- maxent_fit(data.frame(x = c(0, 1, 2, 3)), c(TRUE, TRUE, FALSE, FALSE),
                    features = "linear",
                    control = list(max_iter = 20L, tol = 1e-6, profile = TRUE))
  profile <- maxent_diagnostics(fit)$profile
  expect_true(profile$enabled)
  expect_gte(profile$objective_evaluations, 1L)
  expect_equal(profile$host_synchronizations, 0L)
  expect_gte(profile$objective_seconds, 0)
})
