test_that("pinned reference fixtures are readable and finite", {
  maxnet <- read.csv(test_path("..", "fixtures", "maxnet", "predictions.csv"))
  java <- read.csv(test_path("..", "fixtures", "java-maxent", "maxentResults.csv"),
                   check.names = FALSE)
  expect_true(all(is.finite(maxnet$link)))
  expect_true(all(is.finite(maxnet$exponential)))
  expect_true(nrow(java) >= 1L)
  expect_true(all(is.finite(java$`Regularized training gain`)))
})
