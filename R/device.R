#' Probe Torch accelerator availability
#'
#' Reports backend availability without requiring `torch` to be installed and
#' without failing package load on CPU-only systems. A backend is marked
#' available only when an actual tensor smoke operation succeeds.
#'
#' @return A data frame with one row per backend.
#' @export
maxent_device_probe <- function() {
  torch_version <- if (requireNamespace("torch", quietly = TRUE)) {
    as.character(utils::packageVersion("torch"))
  } else {
    NA_character_
  }

  rows <- list(
    probe_backend("cpu", torch_version),
    probe_backend("cuda", torch_version),
    probe_backend("mps", torch_version)
  )
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' List available MaxEnt-GPU compute backends
#'
#' @return A character vector. CPU is returned only when Torch can execute the
#'   CPU smoke operation.
#' @export
maxent_accelerators <- function() {
  probe <- maxent_device_probe()
  probe$backend[probe$available]
}

probe_backend <- function(backend, torch_version) {
  base <- data.frame(
    backend = backend,
    available = FALSE,
    torch_version = torch_version,
    detail = "torch package is not installed",
    stringsAsFactors = FALSE
  )
  if (is.na(torch_version)) {
    return(base)
  }

  reported_available <- tryCatch(
    switch(
      backend,
      cpu = TRUE,
      cuda = isTRUE(torch::cuda_is_available()),
      mps = isTRUE(torch::backends_mps_is_available())
    ),
    error = function(error) FALSE
  )
  if (!reported_available) {
    base$detail <- paste("torch reports", toupper(backend), "unavailable")
    return(base)
  }

  result <- tryCatch({
    device <- torch::torch_device(backend)
    value <- torch::torch_tensor(1, device = device)$to(device = "cpu")$item()
    if (!identical(as.numeric(value), 1)) {
      stop("tensor smoke operation returned an unexpected value")
    }
    list(available = TRUE, detail = "tensor smoke operation passed")
  }, error = function(error) {
    first_line <- strsplit(conditionMessage(error), "\n", fixed = TRUE)[[1]][[1]]
    list(available = FALSE, detail = first_line)
  })

  base$available <- result$available
  base$detail <- result$detail
  base
}
