test_that("device probe always reports the three declared backends", {
  probe <- maxent_device_probe()

  expect_s3_class(probe, "data.frame")
  expect_identical(probe$backend, c("cpu", "cuda", "mps"))
  expect_type(probe$available, "logical")
  expect_true(all(nzchar(probe$detail)))
})

test_that("available accelerator helper agrees with the probe", {
  probe <- maxent_device_probe()
  expect_identical(maxent_accelerators(), probe$backend[probe$available])
})
