normalize_feature_classes <- function(classes) {
  classes <- tolower(classes)
  classes[classes %in% c("l", "linear")] <- "linear"
  classes[classes %in% c("q", "quadratic")] <- "quadratic"
  classes[classes %in% c("p", "product", "pairwise")] <- "product"
  classes[classes %in% c("t", "threshold", "thresholds")] <- "threshold"
  if (!length(classes) || any(!classes %in% c("linear", "quadratic", "product", "threshold"))) {
    stop("features must contain only 'linear', 'quadratic', 'product', and/or 'threshold'.", call. = FALSE)
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

new_feature_spec <- function(x, classes = c("linear", "quadratic"), clamp = TRUE,
                             thresholds = NULL) {
  x <- validate_numeric_table(x, "presence/background predictors")
  classes <- normalize_feature_classes(classes)
  ranges <- rbind(min = apply(x, 2L, min), max = apply(x, 2L, max))
  if (any(ranges["min", ] == ranges["max", ])) {
    stop("constant predictors are not supported.", call. = FALSE)
  }
  threshold_values <- lapply(seq_len(ncol(x)), function(index) {
    value <- if (is.null(thresholds)) NULL else thresholds[[colnames(x)[index]]]
    if (is.null(value)) {
      unique_values <- sort(unique(x[, index]))
      value <- if (length(unique_values) > 1L) (head(unique_values, -1L) + tail(unique_values, -1L)) / 2 else numeric()
    }
    value <- sort(unique(as.numeric(value)))
    if (any(!is.finite(value)) || any(value <= ranges["min", index]) || any(value >= ranges["max", index])) {
      stop("thresholds must be finite and strictly inside predictor ranges.", call. = FALSE)
    }
    value
  })
  names(threshold_values) <- colnames(x)
  if ("threshold" %in% classes && !any(lengths(threshold_values))) {
    stop("threshold features require at least one threshold.", call. = FALSE)
  }
  columns <- unlist(lapply(classes, function(class) {
    if (class == "linear") paste0("L:", colnames(x))
    else if (class == "quadratic") paste0("Q:", colnames(x))
    else if (class == "product" && ncol(x) < 2L) stop("product features require at least two predictors.", call. = FALSE)
    else if (class == "product") {
      pairs <- utils::combn(colnames(x), 2L)
      paste0("P:", pairs[1L, ], "*", pairs[2L, ])
    } else {
      unlist(Map(function(name, values) paste0("T:", name, "<=", format(values, trim = TRUE, scientific = FALSE)),
                 names(threshold_values), threshold_values), use.names = FALSE)
    }
  }), use.names = FALSE)
  structure(list(
    schema = "maxentgpu-features-v1",
    predictors = colnames(x),
    classes = classes,
    ranges = ranges,
    thresholds = threshold_values,
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
    } else if (class == "quadratic") {
      out[[length(out) + 1L]] <- x^2
    } else {
      if (class == "product") {
        pairs <- utils::combn(seq_len(ncol(x)), 2L)
        out[[length(out) + 1L]] <- vapply(seq_len(ncol(pairs)), function(pair) {
          x[, pairs[1L, pair]] * x[, pairs[2L, pair]]
        }, numeric(nrow(x)))
      } else {
        out[[length(out) + 1L]] <- do.call(cbind, Map(function(name, values) {
          outer(x[, name], values, FUN = "<=") * 1
        }, names(spec$thresholds), spec$thresholds))
      }
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
