args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) normalizePath(args[[1]]) else normalizePath(".")
fixture_dir <- file.path(root, "tests", "fixtures", "hand")
expected <- utils::read.csv(file.path(fixture_dir, "expected.csv"),
                            stringsAsFactors = FALSE)

stable_logsumexp <- function(score, weight) {
  weight <- weight / sum(weight)
  terms <- log(weight) + score
  anchor <- max(terms)
  anchor + log(sum(exp(terms - anchor)))
}

check_close <- function(actual, expected, label, tolerance = 1e-12) {
  if (!isTRUE(all.equal(actual, expected, tolerance = tolerance,
                        check.attributes = FALSE))) {
    stop(label, " mismatch: actual=", format(actual, digits = 17),
         ", expected=", format(expected, digits = 17))
  }
}

check_numeric_case <- function(name, beta) {
  data <- utils::read.csv(file.path(fixture_dir, paste0(name, ".csv")),
                          stringsAsFactors = FALSE)
  predictors <- grep("^x", names(data), value = TRUE)
  if (anyNA(data[predictors])) stop("missing predictor")
  if (any(vapply(data[predictors], function(x) diff(range(x)) == 0,
                 logical(1)))) stop("constant predictor")
  background <- data$role == "background"
  presence <- data$role == "presence"
  x <- as.matrix(data[predictors])
  score <- drop(x %*% beta)
  q <- data$weight[background]
  q <- q / sum(q)
  w <- data$weight[presence]
  w <- w / sum(w)
  log_z <- stable_logsumexp(score[background], q)
  presence_score <- sum(w * score[presence])
  pi <- q * exp(score[background] - log_z)
  gradient <- colSums(x[background, , drop = FALSE] * pi) -
    colSums(x[presence, , drop = FALSE] * w)
  list(log_z = log_z, presence_score = presence_score,
       objective = log_z - presence_score, gradient = gradient)
}

cases <- list(
  `one-predictor` = c(1),
  `two-predictors` = c(1, -0.5),
  `weighted-background` = c(1),
  `duplicated-presences` = c(1)
)

for (name in names(cases)) {
  observed <- check_numeric_case(name, cases[[name]])
  wanted <- expected[expected$case == name, ]
  check_close(observed$log_z, wanted$log_z, paste(name, "logZ"))
  check_close(observed$presence_score, wanted$presence_score,
              paste(name, "presence score"))
  check_close(observed$objective, wanted$smooth_objective,
              paste(name, "objective"))
  check_close(observed$gradient[[1]], wanted$gradient1,
              paste(name, "gradient1"))
  if (length(observed$gradient) == 2) {
    check_close(observed$gradient[[2]], wanted$gradient2,
                paste(name, "gradient2"))
  }
}

extreme <- utils::read.csv(file.path(fixture_dir, "extreme-logits.csv"))
background <- extreme$role == "background"
presence <- extreme$role == "presence"
log_z <- stable_logsumexp(extreme$score[background],
                          extreme$weight[background])
objective <- log_z - weighted.mean(extreme$score[presence],
                                   extreme$weight[presence])
wanted <- expected[expected$case == "extreme-logits", ]
check_close(log_z, wanted$log_z, "extreme logZ", tolerance = 1e-10)
check_close(objective, wanted$smooth_objective, "extreme objective",
            tolerance = 1e-10)

for (name in c("constant-predictor", "missing-value")) {
  failed <- tryCatch({
    check_numeric_case(name, c(1))
    FALSE
  }, error = function(error) TRUE)
  if (!failed) stop(name, " was expected to fail validation")
}

message("Hand fixture checks passed (7 cases).")
