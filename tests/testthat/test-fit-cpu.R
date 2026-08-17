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
