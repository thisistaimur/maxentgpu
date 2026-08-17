`%||%` <- function(left, right) if (is.null(left)) right else left

normalize_weights <- function(weights, n, name) {
  if (is.null(weights)) weights <- rep(1, n)
  if (length(weights) != n || any(!is.finite(weights)) || any(weights <= 0)) {
    stop(name, " must contain one finite, strictly positive value per row.", call. = FALSE)
  }
  as.numeric(weights) / sum(weights)
}

stable_logz <- function(z, weights) {
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

#' Fit a scalar CPU maximum-entropy model
#'
#' @param x Numeric presence predictors when `background` is supplied, or a
#'   combined table when `background` is `NULL`.
#' @param presence Logical presence indicator aligned with `x`, or a numeric
#'   presence predictor table when `background` is supplied.
#' @param background Optional numeric background predictor table.
#' @param presence_weights Optional positive presence weights.
#' @param background_weights Optional positive background weights.
#' @param features Feature classes: `"linear"`, `"quadratic"`, or both.
#' @param regularization A list with non-negative `lambda1` and `lambda2`.
#' @param control A list with `max_iter`, `tol`, and `step`.
#' @return An object of class `maxent_fit`.
#' @export
maxent_fit <- function(x, presence, background = NULL,
                       presence_weights = NULL, background_weights = NULL,
                       features = c("linear", "quadratic"),
                       regularization = list(lambda1 = 0, lambda2 = 1),
                       control = list(max_iter = 2000L, tol = 1e-8, step = 1)) {
  if (is.null(background)) {
    x <- validate_numeric_table(x, "x")
    if (length(presence) != nrow(x) || !is.logical(presence)) {
      stop("presence must be a logical vector aligned with x when background is NULL.", call. = FALSE)
    }
    presence_x <- x[presence, , drop = FALSE]
    background_x <- x[!presence, , drop = FALSE]
  } else {
    presence_x <- validate_numeric_table(presence, "presence")
    background_x <- validate_numeric_table(background, "background")
    if (!identical(colnames(presence_x), colnames(background_x))) {
      stop("presence and background must have identical predictor names and order.", call. = FALSE)
    }
  }
  if (!nrow(presence_x) || !nrow(background_x)) stop("presence and background must both be non-empty.", call. = FALSE)
  classes <- normalize_feature_classes(features)
  spec <- new_feature_spec(rbind(presence_x, background_x), classes)
  presence_phi <- apply_feature_spec(spec, presence_x)
  background_phi <- apply_feature_spec(spec, background_x)
  w <- normalize_weights(presence_weights, nrow(presence_phi), "presence_weights")
  q <- normalize_weights(background_weights, nrow(background_phi), "background_weights")
  lambda1 <- as.numeric(regularization$lambda1 %||% 0)
  lambda2 <- as.numeric(regularization$lambda2 %||% 1)
  if (length(lambda1) != 1L || length(lambda2) != 1L || !is.finite(lambda1) || !is.finite(lambda2) || lambda1 < 0 || lambda2 < 0) {
    stop("regularization lambda1 and lambda2 must be finite and non-negative scalars.", call. = FALSE)
  }
  max_iter <- as.integer(control$max_iter %||% 2000L)
  tol <- as.numeric(control$tol %||% 1e-8)
  step <- as.numeric(control$step %||% 1)
  accelerated <- isTRUE(control$accelerated %||% TRUE)
  if (max_iter < 1L || !is.finite(tol) || tol <= 0 || !is.finite(step) || step <= 0) stop("invalid solver control values.", call. = FALSE)
  beta <- numeric(ncol(presence_phi))
  y <- beta
  momentum <- 1
  values <- numeric(max_iter)
  converged <- FALSE
  for (iteration in seq_len(max_iter)) {
    baseline <- objective_components(beta, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    values[iteration] <- baseline$value
    current <- objective_components(y, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    candidate <- y - step * current$gradient
    candidate <- sign(candidate) * pmax(abs(candidate) - step * lambda1 * spec$penalty_l1, 0)
    trial <- objective_components(candidate, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    if (trial$value > baseline$value && accelerated) {
      y <- beta
      momentum <- 1
      current <- baseline
      candidate <- beta - step * current$gradient
      candidate <- sign(candidate) * pmax(abs(candidate) - step * lambda1 * spec$penalty_l1, 0)
      trial <- objective_components(candidate, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    }
    while (trial$value > baseline$value && step > 1e-12) {
      step <- step / 2
      candidate <- y - step * current$gradient
      candidate <- sign(candidate) * pmax(abs(candidate) - step * lambda1 * spec$penalty_l1, 0)
      trial <- objective_components(candidate, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
    }
    if (max(abs(candidate - beta)) <= tol) {
      beta <- candidate
      converged <- TRUE
      break
    }
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
  final <- objective_components(beta, presence_phi, background_phi, w, q, lambda1, lambda2, spec$penalty_l1, spec$penalty_l2)
  values <- values[seq_len(iteration)]
  structure(list(beta = beta, feature_spec = spec, presence = presence_x,
                 background = background_x, presence_weights = w,
                 background_weights = q, logz = final$logz,
                 entropy = -sum(q * exp(drop(background_phi %*% beta) - final$logz) *
                                  (drop(background_phi %*% beta) - final$logz)),
                 diagnostics = list(iterations = iteration, converged = converged,
                                    objective = values, final_objective = final$value,
                                    device = "cpu", dtype = "float64")),
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
  data.frame(feature = object$feature_spec$columns, coefficient = unname(object$beta),
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
