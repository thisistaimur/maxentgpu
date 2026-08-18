test_that("batch fitting validates species and preserves stable extraction order", {
  background <- data.frame(x1 = c(0, 1, 2), x2 = c(1, 2, 3))
  presence <- list(
    sparrow = data.frame(x1 = c(0, 1), x2 = c(1, 2)),
    finch = data.frame(x1 = c(1, 2), x2 = c(2, 3))
  )
  batch <- maxent_fit_batch(presence, background = background, features = "linear",
                            regularization = list(lambda1 = 0, lambda2 = 0.4),
                            control = list(max_iter = 100L, tol = 1e-7))
  expect_s3_class(batch, "maxent_batch_model")
  expect_identical(batch$species, c("sparrow", "finch"))
  expect_identical(batch$execution, "independent_scalar")
  expect_identical(colnames(predict(batch, background, type = "link")), batch$species)
  expect_equal(unname(predict(batch, background, species = "finch", type = "raw")),
               unname(matrix(predict(batch$models$finch, background, type = "raw"), ncol = 1L)),
               tolerance = 1e-12)
})

test_that("batch records can carry species-specific backgrounds and weights", {
  records <- list(
    a = list(presence = data.frame(x = c(0, 1)), background = data.frame(x = c(2, 3)),
             presence_weights = c(1, 2)),
    b = list(presence = data.frame(x = c(1, 2)), background = data.frame(x = c(3, 4)))
  )
  batch <- maxent_fit_batch(records, features = "linear",
                            regularization = list(lambda1 = 0, lambda2 = 0.2),
                            control = list(max_iter = 80L, tol = 1e-7))
  expect_equal(nrow(batch$diagnostics), 2L)
  expect_true(all(batch$diagnostics$species == c("a", "b")))
  expect_equal(dim(predict(batch, data.frame(x = c(0, 1, 2)), type = "link")), c(3L, 2L))
})

test_that("batch rejects unstable or mismatched species IDs", {
  expect_error(maxent_fit_batch(list(data.frame(x = 1)), background = data.frame(x = 2)), "species")
  expect_error(maxent_fit_batch(list(a = data.frame(x = 1), b = data.frame(x = 2)),
                                species = c("b", "a"), background = data.frame(x = 3)), "match")
})
