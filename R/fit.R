`%||%` <- function(left, right) if (is.null(left)) right else left

normalize_weights <- function(weights, n, name) {
  if (is.null(weights)) weights <- rep(1, n)
  if (length(weights) != n || any(!is.finite(weights)) || any(weights <= 0)) {
    stop(name, " must contain one finite, strictly positive value per row.", call. = FALSE)
  }
  as.numeric(weights) / sum(weights)
}

normalize_penalty <- function(value, n, name, default = 1) {
  if (is.null(value)) value <- rep(default, n)
  if (length(value) == 1L) value <- rep(value, n)
  if (length(value) != n || any(!is.finite(value)) || any(value < 0)) {
    stop(name, " must contain non-negative finite multipliers for every feature.", call. = FALSE)
  }
  as.numeric(value)
}

normalize_execution <- function(control) {
  device <- tolower(as.character(control$device %||% "cpu"))
  dtype <- tolower(as.character(control$dtype %||% "float64"))
  if (length(device) != 1L || !device %in% c("auto", "cpu", "cuda", "mps")) {
    stop("control device must be one of 'auto', 'cpu', 'cuda', or 'mps'.", call. = FALSE)
  }
  if (length(dtype) != 1L || !dtype %in% c("float32", "float64")) {
    stop("control dtype must be 'float32' or 'float64'.", call. = FALSE)
  }
  engine <- tolower(as.character(control$engine %||% "analytic"))
  if (length(engine) != 1L || !engine %in% c("analytic", "torch")) {
    stop("control engine must be 'analytic' or 'torch'.", call. = FALSE)
  }
  if (engine == "torch" && !requireNamespace("torch", quietly = TRUE)) {
    stop("control engine = 'torch' requires the torch package.", call. = FALSE)
  }
  if (device == "auto") {
    if (requireNamespace("torch", quietly = TRUE) && isTRUE(torch::cuda_is_available())) device <- "cuda"
    else if (requireNamespace("torch", quietly = TRUE) && isTRUE(torch::backends_mps_is_available())) device <- "mps"
    else device <- "cpu"
  }
  if (device != "cpu" && engine != "torch") stop("accelerator fitting requires control engine = 'torch'.", call. = FALSE)
  if (device == "cuda" && (!requireNamespace("torch", quietly = TRUE) || !isTRUE(torch::cuda_is_available()))) {
    stop("CUDA is unavailable for the installed torch runtime.", call. = FALSE)
  }
  if (device == "mps" && (!requireNamespace("torch", quietly = TRUE) || !isTRUE(torch::backends_mps_is_available()))) {
    stop("MPS is unavailable for the installed torch runtime.", call. = FALSE)
  }
  if (device == "cpu" && dtype != "float64") stop("CPU fitting currently requires dtype = 'float64'.", call. = FALSE)
  if (device == "mps" && dtype == "float64") stop("MPS fitting requires dtype = 'float32'.", call. = FALSE)
  list(device = device, dtype = dtype, engine = engine)
}

stable_logz <- function(z, weights) {
  if (!length(z) || length(weights) != length(z) || any(!is.finite(z)) ||
      any(!is.finite(weights)) || any(weights <= 0)) {
    stop("log-partition inputs must be finite and have strictly positive weights.", call. = FALSE)
  }
  terms <- log(weights) + z
  pivot <- max(terms)
  value <- pivot + log(sum(exp(terms - pivot)))
  if (!is.finite(value)) stop("weighted log-partition is non-finite.", call. = FALSE)
  value
}

objective_components <- function(beta, presence_phi, background_phi, w, q, lambda1, lambda2, r1, r2) {
  presence_z <- drop(presence_phi %*% beta)
  background_z <- drop(background_phi %*% beta)
  logz <- stable_logz(background_z, q)
  smooth <- logz - sum(w * presence_z)
  penalty <- lambda1 * sum(r1 * abs(beta)) + 0.5 * lambda2 * sum(r2 * beta^2)
  pi <- q * exp(background_z - logz)
  gradient <- drop(crossprod(background_phi, pi) - crossprod(presence_phi, w)) + lambda2 * r2 * beta
  list(value = smooth + penalty, smooth = smooth, logz = logz,
       gradient = gradient, pi = pi)
}

torch_objective_components <- function(beta, presence_phi, background_phi, w, q, lambda1, lambda2, r1, r2,
                                       device = "cpu", dtype = "float64") {
  if (!requireNamespace("torch", quietly = TRUE)) stop("torch is required for the autograd oracle.", call. = FALSE)
  torch_dtype <- if (identical(dtype, "float32")) torch::torch_float32() else torch::torch_float64()
  beta_t <- torch::torch_tensor(beta, dtype = torch_dtype, device = device, requires_grad = TRUE)
  presence_t <- torch::torch_tensor(presence_phi, dtype = torch_dtype, device = device)
  background_t <- torch::torch_tensor(background_phi, dtype = torch_dtype, device = device)
  w_t <- torch::torch_tensor(w, dtype = torch_dtype, device = device)
  q_t <- torch::torch_tensor(q, dtype = torch_dtype, device = device)
  z_presence <- presence_t$matmul(beta_t)
  z_background <- background_t$matmul(beta_t)
  logz <- torch::torch_logsumexp(torch::torch_log(q_t) + z_background, dim = 1)
  smooth <- logz - (w_t * z_presence)$sum()
  penalty <- lambda1 * (torch::torch_tensor(r1, dtype = torch_dtype, device = device) * torch::torch_abs(beta_t))$sum() +
    0.5 * lambda2 * (torch::torch_tensor(r2, dtype = torch_dtype, device = device) * beta_t^2)$sum()
  value <- smooth + penalty
  value$backward()
  list(value = as.numeric(value$item()), gradient = as.numeric(beta_t$grad))
}

torch_objective_factory <- function(presence_phi, background_phi, w, q, lambda1, lambda2,
                                    r1, r2, device = "cpu", dtype = "float64") {
  if (!requireNamespace("torch", quietly = TRUE)) stop("torch is required for the autograd oracle.", call. = FALSE)
  torch_dtype <- if (identical(dtype, "float32")) torch::torch_float32() else torch::torch_float64()
  presence_t <- torch::torch_tensor(presence_phi, dtype = torch_dtype, device = device)
  background_t <- torch::torch_tensor(background_phi, dtype = torch_dtype, device = device)
  w_t <- torch::torch_tensor(w, dtype = torch_dtype, device = device)
  q_t <- torch::torch_tensor(q, dtype = torch_dtype, device = device)
  r1_t <- torch::torch_tensor(r1, dtype = torch_dtype, device = device)
  r2_t <- torch::torch_tensor(r2, dtype = torch_dtype, device = device)
  function(beta) {
    beta_t <- torch::torch_tensor(beta, dtype = torch_dtype, device = device, requires_grad = TRUE)
    z_presence <- presence_t$matmul(beta_t)
    z_background <- background_t$matmul(beta_t)
    logz <- torch::torch_logsumexp(torch::torch_log(q_t) + z_background, dim = 1)
    smooth <- logz - (w_t * z_presence)$sum()
    penalty <- lambda1 * (r1_t * torch::torch_abs(beta_t))$sum() +
      0.5 * lambda2 * (r2_t * beta_t^2)$sum()
    value <- smooth + penalty
    value$backward()
    list(value = as.numeric(value$item()), gradient = as.numeric(beta_t$grad))
  }
}

torch_native_solve <- function(presence_phi, background_phi, w, q, lambda2, r2,
                               device, dtype, max_iter, step, tol, diagnostic_interval) {
  torch_dtype <- if (identical(dtype, "float32")) torch::torch_float32() else torch::torch_float64()
  presence_t <- torch::torch_tensor(presence_phi, dtype = torch_dtype, device = device)
  background_t <- torch::torch_tensor(background_phi, dtype = torch_dtype, device = device)
  w_t <- torch::torch_tensor(w, dtype = torch_dtype, device = device)
  q_t <- torch::torch_tensor(q, dtype = torch_dtype, device = device)
  r2_t <- torch::torch_tensor(r2, dtype = torch_dtype, device = device)
  beta <- torch::torch_tensor(rep(0, ncol(presence_phi)), dtype = torch_dtype, device = device)
  values <- rep(NA_real_, max_iter)
  previous <- beta
  converged <- FALSE
  parameter_change <- Inf
  for (iteration in seq_len(max_iter)) {
    beta <- beta$detach()$requires_grad_(TRUE)
    z_presence <- presence_t$matmul(beta)
    z_background <- background_t$matmul(beta)
    logz <- torch::torch_logsumexp(torch::torch_log(q_t) + z_background, dim = 1)
    value <- logz - (w_t * z_presence)$sum() + 0.5 * lambda2 * (r2_t * beta^2)$sum()
    value$backward()
    candidate <- torch::with_no_grad({ beta - step * beta$grad })
    if (iteration %% diagnostic_interval == 0L || iteration == max_iter) {
      values[[iteration]] <- as.numeric(value$item())
      parameter_change <- max(abs(as.numeric((candidate - beta)$to(device = "cpu"))))
      if (parameter_change <= tol) {
        beta <- candidate
        converged <- TRUE
        break
      }
    }
    previous <- beta
    beta <- candidate
  }
  list(beta = as.numeric(beta$to(device = "cpu")), values = values[seq_len(iteration)],
       iterations = iteration, converged = converged,
       parameter_change = parameter_change)
}

#' Fit a scalar CPU maximum-entropy model
#'
#' @param x Numeric presence predictors when `background` is supplied, or a
#'   combined table when `background` is `NULL`.
#' @param presence Logical presence indicator aligned with `x`, or a numeric
#'   presence predictor table when `background` is supplied.
#' @param background Optional numeric background predictor table.
#' @param presence_weights Optional positive presence weights.
#' @param background_weights Optional positive background weights.
#' @param features Feature classes: `"linear"`, `"quadratic"`, `"product"`,
#'   `"threshold"`, `"hinge"`, or categorical-only models.
#' @param thresholds Optional named list of numeric threshold values by predictor.
#' @param knots Optional named list of numeric hinge knots by predictor.
#' @param regularization A list with non-negative `lambda1` and `lambda2`, and
#'   optional feature-specific non-negative `penalty_l1` and `penalty_l2` vectors.
#' @param control A list with `max_iter`, `tol`, `step`, optional `device`,
#'   `dtype`, `engine` (`"analytic"` or `"torch"`), `accelerated`, and
#'   `profile` (to collect objective timing and host-synchronization counts).
#'   Set `native = TRUE` for the experimental fixed-step Torch-native mode;
#'   it currently requires `engine = "torch"`, `lambda1 = 0`, and
#'   `accelerated = FALSE`.
#' @return An object of class `maxent_fit`.
#' @export
maxent_fit <- function(x, presence, background = NULL,
                       presence_weights = NULL, background_weights = NULL,
                       features = c("linear", "quadratic"),
                       thresholds = NULL,
                       knots = NULL,
                       regularization = list(lambda1 = 0, lambda2 = 1),
                       control = list(max_iter = 2000L, tol = 1e-8, step = 1)) {
  if (length(features) == 1L && identical(tolower(features), "auto")) {
    presence_count <- if (is.null(background)) sum(as.logical(presence)) else nrow(presence)
    features <- select_auto_features(presence_count)
  }
  classes <- normalize_feature_classes(features)
  categorical <- identical(classes, "categorical")
  validate_training <- function(value, name) {
    if (categorical) {
      if (!is.data.frame(value) || !nrow(value) || any(!vapply(value, function(column) is.factor(column) || is.character(column), logical(1)))) {
        stop(name, " must be a non-empty data frame of factor or character columns for categorical features.", call. = FALSE)
      }
      value
    } else validate_numeric_table(value, name)
  }
  if (is.null(background)) {
    x <- validate_training(x, "x")
    if (length(presence) != nrow(x) || !is.logical(presence)) {
      stop("presence must be a logical vector aligned with x when background is NULL.", call. = FALSE)
    }
    presence_x <- x[presence, , drop = FALSE]
    background_x <- x[!presence, , drop = FALSE]
  } else {
    presence_x <- validate_training(presence, "presence")
    background_x <- validate_training(background, "background")
    if (!identical(colnames(presence_x), colnames(background_x))) {
      stop("presence and background must have identical predictor names and order.", call. = FALSE)
    }
  }
  if (!nrow(presence_x) || !nrow(background_x)) stop("presence and background must both be non-empty.", call. = FALSE)
  combined <- if (categorical) rbind(presence_x, background_x) else rbind(presence_x, background_x)
  spec <- new_feature_spec(combined, classes,
                           thresholds = thresholds, knots = knots)
  presence_phi <- apply_feature_spec(spec, presence_x)
  background_phi <- apply_feature_spec(spec, background_x)
  design_rank <- qr(rbind(presence_phi, background_phi), tol = 1e-10)$rank
  if (!categorical && design_rank < ncol(presence_phi)) {
    stop("feature design matrix is rank-deficient; remove duplicated or collinear features.", call. = FALSE)
  }
  w <- normalize_weights(presence_weights, nrow(presence_phi), "presence_weights")
  q <- normalize_weights(background_weights, nrow(background_phi), "background_weights")
  lambda1 <- as.numeric(regularization$lambda1 %||% 0)
  lambda2 <- as.numeric(regularization$lambda2 %||% 1)
  if (length(lambda1) != 1L || length(lambda2) != 1L || !is.finite(lambda1) || !is.finite(lambda2) || lambda1 < 0 || lambda2 < 0) {
    stop("regularization lambda1 and lambda2 must be finite and non-negative scalars.", call. = FALSE)
  }
  spec$penalty_l1 <- normalize_penalty(regularization$penalty_l1, length(spec$columns), "regularization penalty_l1")
  spec$penalty_l2 <- normalize_penalty(regularization$penalty_l2, length(spec$columns), "regularization penalty_l2")
  max_iter <- as.integer(control$max_iter %||% 2000L)
  tol <- as.numeric(control$tol %||% 1e-8)
  step <- as.numeric(control$step %||% 1)
  execution <- normalize_execution(control)
  accelerated <- isTRUE(control$accelerated %||% TRUE)
  profile <- isTRUE(control$profile %||% FALSE)
  objective_impl <- if (execution$engine == "torch") {
    torch_eval <- torch_objective_factory(
      presence_phi, background_phi, w, q, lambda1, lambda2,
      spec$penalty_l1, spec$penalty_l2,
      device = execution$device, dtype = execution$dtype
    )
    function(beta_value, ...) {
      result <- torch_eval(beta_value)
      list(value = result$value, gradient = result$gradient,
           smooth = NA_real_, logz = NA_real_, pi = NULL)
    }
  } else objective_components
  objective_evaluations <- 0L
  objective_seconds <- 0
  objective_eval <- function(...) {
    objective_evaluations <<- objective_evaluations + 1L
    started <- if (profile) proc.time()[["elapsed"]] else 0
    result <- objective_impl(...)
    if (profile) objective_seconds <<- objective_seconds + proc.time()[["elapsed"]] - started
    result
  }
  if (max_iter < 1L || !is.finite(tol) || tol <= 0 || !is.finite(step) || step <= 0) stop("invalid solver control values.", call. = FALSE)
  native <- isTRUE(control$native %||% FALSE)
  if (native && execution$engine != "torch") stop("control native requires engine = 'torch'.", call. = FALSE)
  if (native && lambda1 != 0) stop("control native currently requires lambda1 = 0.", call. = FALSE)
  if (native && accelerated) stop("control native currently requires accelerated = FALSE.", call. = FALSE)
  diagnostic_interval <- as.integer(control$diagnostic_interval %||% 25L)
  if (native && (length(diagnostic_interval) != 1L || is.na(diagnostic_interval) || diagnostic_interval < 1L)) {
    stop("diagnostic_interval must be a positive integer.", call. = FALSE)
  }
  native_result <- if (native) torch_native_solve(
    presence_phi, background_phi, w, q, lambda2, spec$penalty_l2,
    execution$device, execution$dtype, max_iter, step, tol, diagnostic_interval
  ) else NULL
  beta <- if (is.null(native_result)) numeric(ncol(presence_phi)) else native_result$beta
  y <- beta
  momentum <- 1
  values <- numeric(max_iter)
  converged <- FALSE
  stop_reason <- "max_iter"
  parameter_change <- Inf
  smooth_gradient_norm <- Inf
  if (is.null(native_result)) for (iteration in seq_len(max_iter)) {
    baseline <- objective_eval(beta, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    values[iteration] <- baseline$value
    current <- objective_eval(y, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    candidate <- y - step * current$gradient
    candidate <- sign(candidate) * pmax(abs(candidate) - step * lambda1 * spec$penalty_l1, 0)
    trial <- objective_eval(candidate, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    if (trial$value > baseline$value && accelerated) {
      y <- beta
      momentum <- 1
      current <- baseline
      candidate <- beta - step * current$gradient
      candidate <- sign(candidate) * pmax(abs(candidate) - step * lambda1 * spec$penalty_l1, 0)
      trial <- objective_eval(candidate, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    }
    while (trial$value > baseline$value && step > 1e-12) {
      step <- step / 2
      candidate <- y - step * current$gradient
      candidate <- sign(candidate) * pmax(abs(candidate) - step * lambda1 * spec$penalty_l1, 0)
      trial <- objective_eval(candidate, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    }
    if (max(abs(candidate - beta)) <= tol) {
      parameter_change <- max(abs(candidate - beta))
      smooth_gradient_norm <- max(abs(trial$gradient - lambda2 * spec$penalty_l2 * candidate))
      beta <- candidate
      converged <- TRUE
      stop_reason <- "parameter_change"
      break
    }
    parameter_change <- max(abs(candidate - beta))
    smooth_gradient_norm <- max(abs(trial$gradient - lambda2 * spec$penalty_l2 * candidate))
    previous <- beta
    beta <- candidate
    if (accelerated) {
      next_momentum <- (1 + sqrt(1 + 4 * momentum^2)) / 2
      y <- beta + ((momentum - 1) / next_momentum) * (beta - previous)
      momentum <- next_momentum
    } else {
      y <- beta
    }
  }
  if (!is.null(native_result)) {
    iteration <- native_result$iterations
    values <- native_result$values
    converged <- native_result$converged
    stop_reason <- if (converged) "parameter_change" else "max_iter"
    parameter_change <- native_result$parameter_change
    objective_evaluations <- ceiling(iteration / diagnostic_interval)
  }
  final <- objective_components(beta, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
  if (!converged || !is.null(native_result)) {
    parameter_change <- if (is.finite(parameter_change)) parameter_change else NA_real_
    smooth_gradient_norm <- max(abs(final$gradient - lambda2 * spec$penalty_l2 * beta))
  }
  values <- values[seq_len(iteration)]
  structure(list(beta = beta, feature_spec = spec, presence = presence_x,
                 background = background_x, presence_weights = w,
                 background_weights = q, logz = final$logz,
                 entropy = -sum(q * exp(drop(background_phi %*% beta) - final$logz) *
                                  (drop(background_phi %*% beta) - final$logz)),
                 diagnostics = list(iterations = iteration, converged = converged,
                                    objective = values, final_objective = final$value,
                                    parameter_change = parameter_change,
                                    smooth_gradient_norm = smooth_gradient_norm,
                                    stop_reason = stop_reason,
                                    device = execution$device, dtype = execution$dtype,
                                    engine = execution$engine,
                                    profile = list(enabled = profile || native,
                                                   objective_evaluations = objective_evaluations,
                                                   objective_seconds = if (native) NA_real_ else objective_seconds,
                                                   host_synchronizations = if (execution$engine == "torch") objective_evaluations else 0L))),
            class = "maxent_fit")
}

#' Predict from a fitted scalar maximum-entropy model
#'
#' @param object A fitted `maxent_fit` object.
#' @param newdata Numeric predictor data.
#' @param type Prediction scale: `"link"` or `"raw"`.
#' @param ... Reserved for future device and batching controls.
#' @return A numeric prediction vector.
#' @export
predict.maxent_fit <- function(object, newdata, type = c("raw", "link"), ...) {
  type <- match.arg(type)
  phi <- apply_feature_spec(object$feature_spec, newdata)
  z <- drop(phi %*% object$beta)
  if (type == "link") return(z)
  exp(z - object$logz)
}

#' Return fit diagnostics
#' @param object A fitted `maxent_fit` object.
#' @return A diagnostics list.
#' @export
maxent_diagnostics <- function(object) {
  if (!inherits(object, "maxent_fit")) stop("object is not a maxent_fit model.", call. = FALSE)
  object$diagnostics
}

#' Return fitted coefficients as an auditable data frame
#' @param object A fitted `maxent_fit` object.
#' @return A data frame of feature names and coefficients.
#' @export
maxent_coefficients <- function(object) {
  if (!inherits(object, "maxent_fit")) stop("object is not a maxent_fit model.", call. = FALSE)
  feature <- object$feature_spec$columns
  feature_class <- sub("^([^:]+):.*$", "\\1", feature)
  source <- sub("^[^:]+:", "", feature)
  knot <- ifelse(grepl("[<>]=?", source), sub("^.*[<>]=?", "", source), NA_character_)
  level <- ifelse(grepl("=", source), sub("^.*=", "", source), NA_character_)
  source_predictors <- sub("[<>]=?.*$", "", sub("=.*$", "", source))
  data.frame(feature = feature, class = feature_class, source = source_predictors,
             knot = knot, level = level, coefficient = unname(object$beta),
             penalty_l1 = object$feature_spec$penalty_l1,
             penalty_l2 = object$feature_spec$penalty_l2,
             row.names = NULL)
}

#' @export
print.maxent_fit <- function(x, ...) {
  cat("maxent_fit\n")
  cat("  features:", length(x$beta), "  iterations:", x$diagnostics$iterations,
      "  converged:", x$diagnostics$converged, "\n")
  invisible(x)
}

#' Summarize a fitted maximum-entropy model
#'
#' @param object A fitted `maxent_fit` object.
#' @param ... Unused.
#' @return A compact model summary, invisibly.
#' @export
summary.maxent_fit <- function(object, ...) {
  if (!inherits(object, "maxent_fit")) stop("object is not a maxent_fit model.", call. = FALSE)
  out <- list(
    predictors = object$feature_spec$predictors,
    features = object$feature_spec$columns,
    coefficients = maxent_coefficients(object),
    diagnostics = object$diagnostics,
    entropy = object$entropy
  )
  class(out) <- "summary.maxent_fit"
  out
}

#' Save a fitted model
#'
#' @param object A fitted `maxent_fit` object.
#' @param file Destination `.rds` path.
#' @return `file`, invisibly.
#' @export
save_maxent_model <- function(object, file) {
  if (!inherits(object, "maxent_fit")) stop("object is not a maxent_fit model.", call. = FALSE)
  if (length(file) != 1L || !nzchar(file)) stop("file must be a non-empty path.", call. = FALSE)
  saveRDS(object, file = file)
  invisible(file)
}

#' Load a fitted model
#'
#' @param file Path created by `save_maxent_model()`.
#' @return A fitted `maxent_fit` object.
#' @export
read_maxent_model <- function(file) {
  object <- readRDS(file)
  if (!inherits(object, "maxent_fit")) stop("file does not contain a maxent_fit model.", call. = FALSE)
  object
}
