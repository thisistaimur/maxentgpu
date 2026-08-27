#' Fit independent species models from a validated batch specification
#'
#' This batch API preserves independent-model semantics and stable species ordering.
#' Set `control$batch_solver = "torch"` to use the experimental shared-design
#' batched L2 solver; unsupported inputs use the scalar path unless explicitly
#' requested, in which case a clear error is returned.
#'
#' @param x A named list of species records. Each record must contain `presence`
#'   and `background` tables, unless `background` is supplied separately, in which
#'   case each element is a presence table.
#' @param species Optional character species IDs. Required when `x` is unnamed.
#' @param background Optional shared background table.
#' @param ... Arguments forwarded to [maxent_fit()] except `control`.
#' @param control Solver controls forwarded to [maxent_fit()]. Set
#'   `batch_solver = "torch"` for the experimental shared-design batched solver.
#' @return An object of class `maxent_batch_model`.
#' @export
#' @importFrom stats predict
#' @importFrom utils head tail
#' @noRd

torch_batch_solve_l2 <- function(presence_phi, background_phi, control, lambda2,
                                 device = "cpu", dtype = "float64") {
  if (!requireNamespace("torch", quietly = TRUE)) stop("torch is required for batched fitting.", call. = FALSE)
  torch_dtype <- if (identical(dtype, "float32")) torch::torch_float32() else torch::torch_float64()
  species_n <- length(presence_phi)
  feature_n <- ncol(background_phi)
  background_t <- torch::torch_tensor(background_phi, dtype = torch_dtype, device = device)
  presence_t <- lapply(presence_phi, torch::torch_tensor, dtype = torch_dtype, device = device)
  weights_t <- lapply(presence_phi, function(phi) {
    torch::torch_tensor(rep(1 / nrow(phi), nrow(phi)), dtype = torch_dtype, device = device)
  })
  beta <- torch::torch_zeros(c(species_n, feature_n), dtype = torch_dtype, device = device)
  max_iter <- as.integer(control$max_iter %||% 2000L)
  tol <- as.numeric(control$tol %||% 1e-8)
  step <- as.numeric(control$step %||% 1)
  q_t <- torch::torch_tensor(rep(1 / nrow(background_phi), nrow(background_phi)), dtype = torch_dtype, device = device)
  converged <- rep(FALSE, species_n)
  parameter_change <- rep(Inf, species_n)
  gradient_norm <- rep(Inf, species_n)
  for (iteration in seq_len(max_iter)) {
    beta <- beta$detach()$requires_grad_(TRUE)
    scores_b <- background_t$matmul(beta$t())
    logz <- torch::torch_logsumexp(torch::torch_log(q_t) + scores_b, dim = 1)
    presence_loss <- lapply(seq_len(species_n), function(index) {
      (weights_t[[index]] * presence_t[[index]]$matmul(beta[index, ]))$sum()
    })
    value <- logz$sum() - Reduce(`+`, presence_loss) +
      0.5 * lambda2 * (beta^2)$sum()
    value$backward()
    candidate <- torch::with_no_grad({ beta - step * beta$grad })
    if (iteration %% as.integer(control$diagnostic_interval %||% 25L) == 0L || iteration == max_iter) {
      delta <- as.matrix(torch::as_array((candidate - beta)$to(device = "cpu")))
      grad <- as.matrix(torch::as_array(beta$grad$to(device = "cpu")))
      parameter_change <- apply(abs(delta), 1L, max)
      gradient_norm <- apply(abs(grad), 1L, max)
      converged <- parameter_change <= tol
      if (all(converged)) break
    }
    beta <- candidate
  }
  list(beta = as.matrix(torch::as_array(beta$to(device = "cpu"))), iterations = iteration,
       converged = converged, parameter_change = parameter_change,
       gradient_norm = gradient_norm)
}

#' Fit independent species models from a validated batch specification
#'
#' This batch API preserves independent-model semantics and stable species ordering.
#' Set `control$batch_solver = "torch"` to use the experimental shared-design
#' batched L2 solver; unsupported inputs use the scalar path unless explicitly
#' requested, in which case a clear error is returned.
#'
#' @param x A named list of species records. Each record must contain `presence`
#'   and `background` tables, unless `background` is supplied separately, in which
#'   case each element is a presence table.
#' @param species Optional character species IDs. Required when `x` is unnamed.
#' @param background Optional shared background table.
#' @param ... Arguments forwarded to [maxent_fit()] except `control`.
#' @param control Solver controls forwarded to [maxent_fit()]. Set
#'   `batch_solver = "torch"` for the experimental shared-design batched solver.
#' @return An object of class `maxent_batch_model`.
#' @export
maxent_fit_batch <- function(x, species = NULL, background = NULL, ..., control = list()) {
  if (!is.list(x) || !length(x)) stop("x must be a non-empty list of species inputs.", call. = FALSE)
  ids <- names(x)
  if (is.null(ids) || any(!nzchar(ids))) {
    if (is.null(species) || length(species) != length(x) || any(!nzchar(species)) || anyDuplicated(species)) {
      stop("species must provide unique non-empty IDs when x is unnamed.", call. = FALSE)
    }
    ids <- as.character(species)
  } else {
    if (anyDuplicated(ids)) stop("x must have unique species names.", call. = FALSE)
    if (!is.null(species)) {
      if (length(species) != length(x) || !identical(as.character(species), ids)) {
        stop("species must match the names of x when both are supplied.", call. = FALSE)
      }
    }
  }
  if (!is.null(background) && (!is.data.frame(background) || !nrow(background))) {
    stop("background must be a non-empty data frame when supplied.", call. = FALSE)
  }
  dots <- list(...)
  if ("control" %in% names(dots)) stop("supply control through the control argument, not ...", call. = FALSE)
  use_batched_solver <- identical(control$batch_solver, "torch")
  if (use_batched_solver && is.null(background)) {
    stop("control batch_solver = 'torch' requires a shared background.", call. = FALSE)
  }
  fits <- lapply(seq_along(x), function(index) {
    record <- x[[index]]
    if (is.null(background)) {
      if (!is.list(record) || is.null(record$presence) || is.null(record$background)) {
        stop("each x element must contain presence and background when no shared background is supplied.", call. = FALSE)
      }
      args <- c(list(presence = record$presence, background = record$background), dots,
                list(control = control))
      if (!is.null(record$presence_weights)) args$presence_weights <- record$presence_weights
      if (!is.null(record$background_weights)) args$background_weights <- record$background_weights
    } else {
      if (!is.data.frame(record) || !nrow(record)) stop("each species presence input must be a non-empty data frame.", call. = FALSE)
      args <- c(list(presence = record, background = background), dots,
                list(control = control))
    }
    do.call(maxent_fit, args)
  })
  if (use_batched_solver) {
    if (!requireNamespace("torch", quietly = TRUE)) stop("batch_solver = 'torch' requires the torch package.", call. = FALSE)
    if (!identical(tolower(as.character(dots$features %||% "linear")), "linear") ||
        !identical(as.numeric((dots$regularization %||% list(lambda1 = 0))$lambda1 %||% 0), 0)) {
      stop("batch_solver = 'torch' currently supports linear features and L2-only regularization.", call. = FALSE)
    }
    specs <- lapply(fits, `[[`, "feature_spec")
    if (!all(vapply(specs[-1L], identical, logical(1), specs[[1L]]))) {
      stop("batch_solver = 'torch' requires identical feature specifications.", call. = FALSE)
    }
    solver <- torch_batch_solve_l2(
      lapply(fits, function(fit) apply_feature_spec(specs[[1L]], fit$presence)),
      apply_feature_spec(specs[[1L]], background),
      control = control, lambda2 = as.numeric((dots$regularization %||% list(lambda2 = 1))$lambda2 %||% 1),
      device = maxent_diagnostics(fits[[1L]])$device,
      dtype = maxent_diagnostics(fits[[1L]])$dtype
    )
    fits <- lapply(seq_along(fits), function(index) {
      fit <- fits[[index]]
      beta <- solver$beta[index, ]
      final <- objective_components(beta, apply_feature_spec(specs[[1L]], fit$presence),
                                   apply_feature_spec(specs[[1L]], background),
                                   fit$presence_weights, fit$background_weights, 0,
                                   as.numeric((dots$regularization %||% list(lambda2 = 1))$lambda2 %||% 1),
                                   fit$feature_spec$penalty_l1, fit$feature_spec$penalty_l2)
      fit$beta <- beta
      fit$logz <- final$logz
      fit$diagnostics$iterations <- solver$iterations
      fit$diagnostics$converged <- solver$converged[[index]]
      fit$diagnostics$stop_reason <- if (fit$diagnostics$converged) "parameter_change" else "max_iter"
      fit$diagnostics$parameter_change <- solver$parameter_change[[index]]
      fit$diagnostics$smooth_gradient_norm <- solver$gradient_norm[[index]]
      fit
    })
  }
  diagnostics <- do.call(rbind, lapply(seq_along(fits), function(index) {
    d <- maxent_diagnostics(fits[[index]])
    data.frame(species = ids[[index]], iterations = d$iterations,
               converged = d$converged, stop_reason = d$stop_reason,
               parameter_change = d$parameter_change,
               gradient_norm = d$smooth_gradient_norm,
               device = d$device, dtype = d$dtype, engine = d$engine,
               stringsAsFactors = FALSE)
  }))
  names(fits) <- ids
  structure(list(species = ids, models = fits, diagnostics = diagnostics,
                 execution = if (use_batched_solver) "shared_design_torch" else "independent_scalar"),
            class = "maxent_batch_model")
}

#' Predict from an independent-species batch model
#'
#' @param object A fitted `maxent_batch_model` object.
#' @param newdata Predictor data accepted by [predict.maxent_fit()].
#' @param type Prediction scale passed to the scalar models.
#' @param species Optional character IDs to select and order species.
#' @param device Execution device (`"cpu"`, `"cuda"`, or `"mps"`). Torch
#'   execution is used for dense shared-design prediction.
#' @param dtype Torch dtype (`"float64"` or `"float32"`).
#' @param batch_size Maximum number of species materialized per prediction chunk.
#' @param ... Additional arguments passed to [predict.maxent_fit()].
#' @return A matrix with rows corresponding to `newdata` and columns keyed by species ID.
#' @export
predict.maxent_batch_model <- function(object, newdata, type = c("raw", "link"), species = NULL,
                                       device = "cpu", dtype = "float64", batch_size = NULL, ...) {
  if (!inherits(object, "maxent_batch_model")) stop("object is not a maxent_batch_model.", call. = FALSE)
  type <- match.arg(type)
  selected <- if (is.null(species)) object$species else as.character(species)
  if (!length(selected) || anyDuplicated(selected) || any(!selected %in% object$species)) {
    stop("species must contain unique IDs present in the batch model.", call. = FALSE)
  }
  if (length(device) != 1L || !device %in% c("cpu", "cuda", "mps")) {
    stop("device must be one of 'cpu', 'cuda', or 'mps'.", call. = FALSE)
  }
  if (length(dtype) != 1L || !dtype %in% c("float32", "float64")) {
    stop("dtype must be 'float32' or 'float64'.", call. = FALSE)
  }
  if (is.null(batch_size)) batch_size <- length(selected)
  batch_size <- as.integer(batch_size)
  if (length(batch_size) != 1L || is.na(batch_size) || batch_size < 1L) {
    stop("batch_size must be a positive integer.", call. = FALSE)
  }
  specs <- lapply(object$models[selected], `[[`, "feature_spec")
  shared_design <- length(specs) == 1L || all(vapply(specs[-1L], identical, logical(1), specs[[1L]]))
  if (!shared_design || device == "cpu" && !requireNamespace("torch", quietly = TRUE)) {
    out <- vapply(selected, function(id) predict(object$models[[id]], newdata, type = type, ...),
                  numeric(nrow(newdata)))
    if (is.null(dim(out))) out <- matrix(out, ncol = 1L, dimnames = list(NULL, selected))
    else colnames(out) <- selected
    return(out)
  }
  if (!requireNamespace("torch", quietly = TRUE)) stop("Torch is required for dense batch prediction.", call. = FALSE)
  if (device != "cpu" && !isTRUE(maxent_device_probe()$available[match(device, maxent_device_probe()$backend)])) {
    stop("requested prediction device is unavailable.", call. = FALSE)
  }
  if (device == "cpu" && dtype != "float64") stop("CPU prediction currently requires dtype = 'float64'.", call. = FALSE)
  if (device == "mps" && dtype == "float64") stop("MPS prediction requires dtype = 'float32'.", call. = FALSE)
  torch_dtype <- if (dtype == "float32") torch::torch_float32() else torch::torch_float64()
  phi <- apply_feature_spec(specs[[1L]], newdata)
  chunks <- split(seq_along(selected), ceiling(seq_along(selected) / batch_size))
  result <- lapply(chunks, function(indices) {
    ids <- selected[indices]
    beta <- do.call(cbind, lapply(ids, function(id) object$models[[id]]$beta))
    phi_t <- torch::torch_tensor(phi, dtype = torch_dtype, device = device)
    beta_t <- torch::torch_tensor(beta, dtype = torch_dtype, device = device)
    scores <- phi_t$matmul(beta_t)
    if (type == "raw") {
      logz <- torch::torch_tensor(vapply(ids, function(id) object$models[[id]]$logz, numeric(1)),
                                  dtype = torch_dtype, device = device)
      scores <- torch::torch_exp(scores - logz$reshape(c(1L, length(ids))))
    }
    as.matrix(torch::as_array(scores))
  })
  out <- do.call(cbind, result)
  colnames(out) <- selected
  out
}
