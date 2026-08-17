normalize_feature_classes <- function(classes) {
  classes <- tolower(classes)
  classes[classes %in% c("l", "linear")] <- "linear"
  classes[classes %in% c("q", "quadratic")] <- "quadratic"
  if (!length(classes) || any(!classes %in% c("linear", "quadratic"))) {
    stop("features must contain only 'linear' and/or 'quadratic'.", call. = FALSE)
  }
  unique(classes)
}

validate_numeric_table <- function(x, name) {
  if (is.data.frame(x)) {
    bad <- !vapply(x, is.numeric, logical(1))
    if (any(bad)) stop(name, " must contain numeric columns only.", call. = FALSE)
    x <- as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || !ncol(x) || !nrow(x)) {
    stop(name, " must be a non-empty numeric data frame or matrix.", call. = FALSE)
  }
  storage.mode(x) <- "double"
  if (any(!is.finite(x))) stop(name, " contains non-finite values.", call. = FALSE)
  if (is.null(colnames(x))) colnames(x) <- paste0("x", seq_len(ncol(x)))
  if (any(!nzchar(colnames(x))) || anyDuplicated(colnames(x))) {
    stop(name, " must have unique, non-empty column names.", call. = FALSE)
  }
  x
}

new_feature_spec <- function(x, classes = c("linear", "quadratic"), clamp = TRUE) {
  x <- validate_numeric_table(x, "presence/background predictors")
  classes <- normalize_feature_classes(classes)
  ranges <- rbind(min = apply(x, 2L, min), max = apply(x, 2L, max))
  if (any(ranges["min", ] == ranges["max", ])) {
    stop("constant predictors are not supported.", call. = FALSE)
  }
  columns <- unlist(lapply(classes, function(class) {
    if (class == "linear") paste0("L:", colnames(x)) else paste0("Q:", colnames(x))
  }), use.names = FALSE)
  structure(list(
    schema = "maxentgpu-features-v1",
    predictors = colnames(x),
    classes = classes,
    ranges = ranges,
    columns = columns,
    clamp = isTRUE(clamp),
    penalty_l1 = rep(1, length(columns)),
    penalty_l2 = rep(1, length(columns))
  ), class = "maxent_feature_spec")
}

apply_feature_spec <- function(spec, newdata, clamp = spec$clamp) {
  if (!inherits(spec, "maxent_feature_spec")) stop("invalid feature specification.", call. = FALSE)
  newdata <- validate_numeric_table(newdata, "newdata")
  missing <- setdiff(spec$predictors, colnames(newdata))
  if (length(missing)) stop("newdata is missing predictors: ", paste(missing, collapse = ", "), call. = FALSE)
  x <- newdata[, spec$predictors, drop = FALSE]
  if (isTRUE(clamp)) {
    x <- pmax(x, matrix(spec$ranges["min", ], nrow(x), ncol(x), byrow = TRUE))
    x <- pmin(x, matrix(spec$ranges["max", ], nrow(x), ncol(x), byrow = TRUE))
  }
  out <- list()
  for (class in spec$classes) {
    if (class == "linear") {
      out[[length(out) + 1L]] <- x
    } else {
      out[[length(out) + 1L]] <- x^2
    }
  }
  result <- do.call(cbind, out)
  colnames(result) <- spec$columns
  result
}

#' Return a fitted model's immutable feature specification
#'
#' @param object A fitted `maxent_fit` object.
#' @return A `maxent_feature_spec` object.
#' @export
maxent_feature_spec <- function(object) {
  if (!inherits(object, "maxent_fit")) stop("object is not a maxent_fit model.", call. = FALSE)
  object$feature_spec
}
