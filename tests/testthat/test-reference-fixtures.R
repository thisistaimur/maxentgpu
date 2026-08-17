test_that("pinned reference fixtures are readable and finite", {
  maxnet <- read.csv(test_path("..", "fixtures", "maxnet", "predictions.csv"))
  java <- read.csv(test_path("..", "fixtures", "java-maxent", "maxentResults.csv"),
                   check.names = FALSE)
  expect_true(all(is.finite(maxnet$link)))
  expect_true(all(is.finite(maxnet$exponential)))
  expect_true(nrow(java) >= 1L)
  expect_true(all(is.finite(java$`Regularized training gain`)))
})

test_that("package-native LQ fit aligns with reference fixture rows", {
  input_dir <- test_path("..", "fixtures", "reference", "input")
  background <- read.csv(file.path(input_dir, "background.csv"))[, c("x1", "x2")]
  presence <- read.csv(file.path(input_dir, "samples.csv"))[, c("x1", "x2")]
  fit <- maxent_fit(x = presence, presence = presence, background = background,
                    features = c("linear", "quadratic"),
                    regularization = list(lambda1 = 0, lambda2 = 1),
                    control = list(max_iter = 5000L, tol = 1e-10))
  predictions <- predict(fit, rbind(background, presence), type = "raw")
  expect_length(predictions, 16L)
  expect_true(all(is.finite(predictions)))
  expect_equal(sum(fit$background_weights * predict(fit, background, type = "raw")),
               1, tolerance = 1e-8)
  expect_identical(maxent_feature_spec(fit)$columns,
                   c("L:x1", "L:x2", "Q:x1", "Q:x2"))
})
