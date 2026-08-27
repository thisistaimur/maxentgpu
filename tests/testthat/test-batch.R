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
  extracted <- maxent_batch_extract(batch, "finch")
  expect_s3_class(extracted, "maxent_fit")
  expect_identical(extracted$beta, batch$models$finch$beta)
  expect_identical(colnames(predict(batch, background, type = "link")), batch$species)
  expect_equal(unname(predict(batch, background, species = "finch", type = "raw")),
               unname(matrix(predict(batch$models$finch, background, type = "raw"), ncol = 1L)),
               tolerance = 1e-12)
})

test_that("batch extraction validates species IDs", {
  batch <- structure(list(species = "a", models = list(a = structure(list(), class = "maxent_fit"))),
                     class = "maxent_batch_model")
  expect_error(maxent_batch_extract(batch, "missing"), "existing species")
  expect_error(maxent_batch_extract(batch, c("a", "a")), "single")
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

test_that("shared-design batch prediction uses Torch matrix multiplication", {
  skip_if_torch_unavailable()
  background <- data.frame(x1 = c(0, 1, 2), x2 = c(1, 2, 3))
  presence <- list(a = data.frame(x1 = c(0, 1), x2 = c(1, 2)),
                  b = data.frame(x1 = c(1, 2), x2 = c(2, 3)))
  batch <- maxent_fit_batch(presence, background = background, features = "linear",
                            regularization = list(lambda1 = 0, lambda2 = 0.4),
                            control = list(max_iter = 100L, tol = 1e-7))
  newdata <- data.frame(x1 = c(0.5, 1.5), x2 = c(1.5, 2.5))
  dense <- predict(batch, newdata, type = "link", device = "cpu", batch_size = 1L)
  scalar <- do.call(cbind, lapply(batch$models, predict, newdata = newdata, type = "link"))
  expect_equal(dense, scalar, tolerance = 1e-12)
})

test_that("experimental shared-design batched solver preserves species order", {
  skip_if_torch_unavailable()
  background <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  presence <- list(a = data.frame(x1 = c(0, 1), x2 = c(1, 2)),
                  b = data.frame(x1 = c(1, 2), x2 = c(2, 3)))
  batch <- maxent_fit_batch(
    presence, background = background, features = "linear",
    regularization = list(lambda1 = 0, lambda2 = 0.4),
    control = list(batch_solver = "torch", engine = "torch", device = "cpu",
                   max_iter = 500L, tol = 1e-5, diagnostic_interval = 25L)
  )
  expect_identical(batch$execution, "shared_design_torch")
  expect_identical(batch$species, c("a", "b"))
  expect_true(all(batch$diagnostics$device == "cpu"))
  expect_equal(dim(predict(batch, background, type = "link")), c(4L, 2L))
})

test_that("weighted shared-design solver matches scalar weighted fits", {
  skip_if_torch_unavailable()
  background <- data.frame(x1 = c(0, 1, 2, 3), x2 = c(1, 2, 3, 4))
  presence <- list(a = data.frame(x1 = c(0, 1), x2 = c(1, 2)),
                  b = data.frame(x1 = c(1, 2), x2 = c(2, 3)))
  weights <- list(a = c(1, 3), b = c(2, 1))
  regularization <- list(lambda1 = 0, lambda2 = 0.4)
  batch <- maxent_fit_batch(
    presence, background = background, presence_weights = weights,
    features = "linear", regularization = regularization,
    control = list(batch_solver = "torch", engine = "torch", device = "cpu",
                   max_iter = 500L, tol = 1e-5, diagnostic_interval = 25L)
  )
  scalar <- lapply(seq_along(presence), function(index) maxent_fit(
    x = presence[[index]], presence = presence[[index]], background = background,
    presence_weights = weights[[index]], features = "linear",
    regularization = regularization,
    control = list(engine = "torch", device = "cpu", max_iter = 500L, tol = 1e-5)
  ))
  expect_equal(predict(batch, background, type = "link", device = "cpu"),
               do.call(cbind, lapply(scalar, predict, newdata = background, type = "link")),
               tolerance = 2e-5)
})

test_that("batch rejects unstable or mismatched species IDs", {
  expect_error(maxent_fit_batch(list(data.frame(x = 1)), background = data.frame(x = 2)), "species")
  expect_error(maxent_fit_batch(list(a = data.frame(x = 1), b = data.frame(x = 2)),
                                species = c("b", "a"), background = data.frame(x = 3)), "match")
})
