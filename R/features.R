normalize_feature_classes <- function(classes) {
  classes <- tolower(classes)
  classes[classes %in% c("l", "linear")] <- "linear"
  classes[classes %in% c("q", "quadratic")] <- "quadratic"
  classes[classes %in% c("p", "product", "pairwise")] <- "product"
  classes[classes %in% c("t", "threshold", "thresholds")] <- "threshold"
  classes[classes %in% c("h", "hinge", "hinges")] <- "hinge"
  classes[classes %in% c("c", "categorical", "factor")] <- "categorical"
  if (!length(classes) || any(!classes %in% c("linear", "quadratic", "product", "threshold", "hinge", "categorical"))) {
    stop("features must contain only 'linear', 'quadratic', 'product', 'threshold', 'hinge', and/or 'categorical'.", call. = FALSE)
  }
  unique(classes)
}

select_auto_features <- function(n_presence) {
  if (length(n_presence) != 1L || !is.finite(n_presence) || n_presence < 1) {
    stop("n_presence must be a positive finite scalar.", call. = FALSE)
  }
  n_presence <- as.integer(n_presence)
  if (n_presence < 10L) return("linear")
  if (n_presence < 15L) return(c("linear", "quadratic"))
  if (n_presence < 80L) return(c("linear", "quadratic", "hinge"))
  c("linear", "quadratic", "hinge", "product")
}

#' Select default feature classes by presence sample size
#'
#' @param n_presence Number of presence records.
#' @return Character vector of feature classes.
#' @export
maxent_auto_features <- function(n_presence) select_auto_features(n_presence)

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
                             thresholds = NULL, knots = NULL) {
  if (identical(classes, "categorical")) {
    if (!is.data.frame(x) || !nrow(x) || !ncol(x) ||
        any(!vapply(x, function(column) is.factor(column) || is.character(column), logical(1)))) {
      stop("categorical features require a non-empty data frame of factor or character columns.", call. = FALSE)
    }
    levels <- lapply(x, function(column) sort(unique(as.character(column))))
    names(levels) <- names(x)
    columns <- unlist(Map(function(name, values) paste0("C:", name, "=", values), names(levels), levels), use.names = FALSE)
    return(structure(list(schema = "maxentgpu-features-v1", predictors = names(x),
                          classes = classes, levels = levels, columns = columns,
                          clamp = FALSE, penalty_l1 = rep(1, length(columns)),
                          penalty_l2 = rep(1, length(columns))), class = "maxent_feature_spec"))
  }
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
  knot_values <- lapply(seq_len(ncol(x)), function(index) {
    value <- if (is.null(knots)) NULL else knots[[colnames(x)[index]]]
    if (is.null(value)) value <- threshold_values[[index]]
    value <- sort(unique(as.numeric(value)))
    if (any(!is.finite(value)) || any(value <= ranges["min", index]) || any(value >= ranges["max", index])) {
      stop("knots must be finite and strictly inside predictor ranges.", call. = FALSE)
    }
    value
  })
  names(knot_values) <- colnames(x)
  if ("hinge" %in% classes && !any(lengths(knot_values))) {
    stop("hinge features require at least one knot.", call. = FALSE)
  }
  columns <- unlist(lapply(classes, function(class) {
    if (class == "linear") paste0("L:", colnames(x))
    else if (class == "quadratic") paste0("Q:", colnames(x))
    else if (class == "product" && ncol(x) < 2L) stop("product features require at least two predictors.", call. = FALSE)
    else if (class == "product") {
      pairs <- utils::combn(colnames(x), 2L)
      paste0("P:", pairs[1L, ], "*", pairs[2L, ])
    } else if (class == "threshold") {
      unlist(Map(function(name, values) paste0("T:", name, "<=", format(values, trim = TRUE, scientific = FALSE)),
                 names(threshold_values), threshold_values), use.names = FALSE)
    } else {
      unlist(Map(function(name, values) c(
        paste0("H+:", name, ">", format(values, trim = TRUE, scientific = FALSE)),
        paste0("H-:", name, "<", format(values, trim = TRUE, scientific = FALSE))),
        names(knot_values), knot_values), use.names = FALSE)
    }
  }), use.names = FALSE)
  structure(list(
    schema = "maxentgpu-features-v1",
    predictors = colnames(x),
    classes = classes,
    ranges = ranges,
    thresholds = threshold_values,
    knots = knot_values,
    columns = columns,
    clamp = isTRUE(clamp),
    penalty_l1 = rep(1, length(columns)),
    penalty_l2 = rep(1, length(columns))
  ), class = "maxent_feature_spec")
}

apply_feature_spec <- function(spec, newdata, clamp = spec$clamp) {
  if (!inherits(spec, "maxent_feature_spec")) stop("invalid feature specification.", call. = FALSE)
  if (identical(spec$classes, "categorical")) {
    if (!is.data.frame(newdata)) stop("newdata must be a data frame for categorical features.", call. = FALSE)
    missing <- setdiff(spec$predictors, names(newdata))
    if (length(missing)) stop("newdata is missing predictors: ", paste(missing, collapse = ", "), call. = FALSE)
    result <- do.call(cbind, Map(function(name, values) {
      observed <- as.character(newdata[[name]])
      if (any(!observed %in% values)) stop("newdata contains unseen levels for ", name, ".", call. = FALSE)
      vapply(values, function(value) as.numeric(observed == value), numeric(nrow(newdata)))
    }, spec$predictors, spec$levels))
    colnames(result) <- spec$columns
    return(result)
  }
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
      } else if (class == "threshold") {
        out[[length(out) + 1L]] <- do.call(cbind, Map(function(name, values) {
          outer(x[, name], values, FUN = "<=") * 1
        }, names(spec$thresholds), spec$thresholds))
      } else {
        out[[length(out) + 1L]] <- do.call(cbind, Map(function(name, values) {
          cbind(outer(x[, name], values, FUN = function(value, knot) pmax(value - knot, 0)),
                outer(x[, name], values, FUN = function(value, knot) pmax(knot - value, 0)))
        }, names(spec$knots), spec$knots))
      }
    }
  }
  result <- do.call(cbind, out)
  if (any(!is.finite(result))) stop("feature transform produced non-finite values.", call. = FALSE)
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

#' Apply a feature specification in bounded row chunks
#'
#' @param spec A `maxent_feature_spec` object.
#' @param newdata Numeric or categorical predictor data accepted by `spec`.
#' @param chunk_size Optional positive number of rows per chunk.
#' @return A numeric feature matrix.
#' @export
maxent_feature_matrix <- function(spec, newdata, chunk_size = NULL) {
  if (!inherits(spec, "maxent_feature_spec")) stop("invalid feature specification.", call. = FALSE)
  if (is.null(chunk_size)) {
    result <- apply_feature_spec(spec, newdata)
    rownames(result) <- NULL
    return(result)
  }
  chunk_size <- as.integer(chunk_size)
  if (length(chunk_size) != 1L || is.na(chunk_size) || chunk_size < 1L) {
    stop("chunk_size must be a positive integer.", call. = FALSE)
  }
  rows <- seq_len(nrow(newdata))
  pieces <- lapply(split(rows, ceiling(rows / chunk_size)), function(index) {
    apply_feature_spec(spec, newdata[index, , drop = FALSE])
  })
  result <- do.call(rbind, pieces)
  rownames(result) <- NULL
  result
}
