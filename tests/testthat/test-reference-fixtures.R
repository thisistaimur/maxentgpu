test_that("pinned reference fixtures are readable and finite", {
  maxnet <- read.csv(test_path("..", "fixtures", "maxnet", "predictions.csv"))
  maxnet_weakly_regularized <- read.csv(test_path("..", "fixtures", "maxnet", "weakly_regularized_predictions.csv"))
  maxnet_linear <- read.csv(test_path("..", "fixtures", "maxnet", "linear_predictions.csv"))
  maxnet_lqph <- read.csv(test_path("..", "fixtures", "maxnet", "lqph_predictions.csv"))
  maxnet_categorical <- read.csv(test_path("..", "fixtures", "maxnet", "categorical_predictions.csv"))
  scales <- read.csv(test_path("..", "fixtures", "maxnet", "scales.csv"))
  penalties <- read.csv(test_path("..", "fixtures", "maxnet", "penalty_factors.csv"))
  java <- read.csv(test_path("..", "fixtures", "java-maxent", "maxentResults.csv"),
                   check.names = FALSE)
  expect_true(all(is.finite(maxnet$link)))
  expect_true(all(is.finite(maxnet$exponential)))
  expect_equal(nrow(maxnet_weakly_regularized), nrow(maxnet))
  expect_true(all(is.finite(maxnet_weakly_regularized$link)))
  expect_equal(nrow(maxnet_linear), nrow(maxnet))
  expect_true(all(is.finite(maxnet_linear$link)))
  expect_equal(nrow(maxnet_lqph), nrow(maxnet))
  expect_true(all(is.finite(maxnet_lqph$link)))
  expect_equal(nrow(maxnet_categorical), 9L)
  expect_true(all(is.finite(maxnet_categorical$link)))
  expect_equal(scales$n_background, 8L)
  expect_true(is.finite(scales$alpha) && is.finite(scales$entropy))
  expect_lt(abs(scales$glmnet_lambda_min - 0.0004733343), 1e-8)
  expect_equal(penalties$feature, c("x1", "x2", "I(x1^2)", "I(x2^2)"))
  expect_true(all(is.finite(penalties$penalty_factor) & penalties$penalty_factor > 0))
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
