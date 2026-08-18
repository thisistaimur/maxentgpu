#' Fit independent species models from a validated batch specification
#'
#' This first batch API preserves independent-model semantics and stable species
#' ordering. Each species is currently fitted through the scalar solver; the
#' `execution` field records this explicitly until dense tensor batching is added.
#'
#' @param x A named list of species records. Each record must contain `presence`
#'   and `background` tables, unless `background` is supplied separately, in which
#'   case each element is a presence table.
#' @param species Optional character species IDs. Required when `x` is unnamed.
#' @param background Optional shared background table.
#' @param ... Arguments forwarded to [maxent_fit()] except `control`.
#' @param control Solver controls forwarded to [maxent_fit()].
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
  diagnostics <- do.call(rbind, lapply(seq_along(fits), function(index) {
    d <- maxent_diagnostics(fits[[index]])
    data.frame(species = ids[[index]], iterations = d$iterations,
               converged = d$converged, stop_reason = d$stop_reason,
               device = d$device, dtype = d$dtype, engine = d$engine,
               stringsAsFactors = FALSE)
  }))
  names(fits) <- ids
  structure(list(species = ids, models = fits, diagnostics = diagnostics,
                 execution = "independent_scalar"),
            class = "maxent_batch_model")
}

#' Predict from an independent-species batch model
#'
#' @param object A fitted `maxent_batch_model` object.
#' @param newdata Predictor data accepted by [predict.maxent_fit()].
#' @param type Prediction scale passed to the scalar models.
#' @param species Optional character IDs to select and order species.
#' @param ... Reserved for future chunking and device controls.
#' @return A matrix with rows corresponding to `newdata` and columns keyed by species ID.
#' @export
predict.maxent_batch_model <- function(object, newdata, type = c("raw", "link"), species = NULL, ...) {
  if (!inherits(object, "maxent_batch_model")) stop("object is not a maxent_batch_model.", call. = FALSE)
  type <- match.arg(type)
  selected <- if (is.null(species)) object$species else as.character(species)
  if (!length(selected) || anyDuplicated(selected) || any(!selected %in% object$species)) {
    stop("species must contain unique IDs present in the batch model.", call. = FALSE)
  }
  out <- vapply(selected, function(id) predict(object$models[[id]], newdata, type = type, ...),
                numeric(nrow(newdata)))
  if (is.null(dim(out))) out <- matrix(out, ncol = 1L, dimnames = list(NULL, selected))
  else colnames(out) <- selected
  out
}
