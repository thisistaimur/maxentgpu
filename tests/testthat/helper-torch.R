skip_if_torch_unavailable <- function() {
  skip_if_not_installed("torch")
  available <- tryCatch({
    torch::torch_tensor(0, dtype = torch::torch_float64())
    TRUE
  }, error = function(error) FALSE)
  skip_if(!available, "Torch runtime is not installed or failed to load")
}
