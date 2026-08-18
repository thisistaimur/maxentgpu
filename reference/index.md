# Package index

## Backend diagnostics

- [`maxent_device_probe()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_device_probe.md)
  : Probe Torch accelerator availability
- [`maxent_accelerators()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_accelerators.md)
  : List available MaxEnt-GPU compute backends

## Feature construction

- [`maxent_feature_spec()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_feature_spec.md)
  : Return a fitted model's immutable feature specification
- [`maxent_feature_matrix()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_feature_matrix.md)
  : Apply a feature specification in bounded row chunks
- [`maxent_auto_features()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_auto_features.md)
  : Select default feature classes by presence sample size

## Scalar fitting and models

- [`maxent_fit()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_fit.md)
  : Fit a scalar CPU maximum-entropy model
- [`predict(`*`<maxent_fit>`*`)`](https://thisistaimur.github.io/maxentgpu/reference/predict.maxent_fit.md)
  : Predict from a fitted scalar maximum-entropy model
- [`maxent_diagnostics()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_diagnostics.md)
  : Return fit diagnostics
- [`maxent_coefficients()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_coefficients.md)
  : Return fitted coefficients as an auditable data frame
- [`summary(`*`<maxent_fit>`*`)`](https://thisistaimur.github.io/maxentgpu/reference/summary.maxent_fit.md)
  : Summarize a fitted maximum-entropy model
- [`save_maxent_model()`](https://thisistaimur.github.io/maxentgpu/reference/save_maxent_model.md)
  : Save a fitted model
- [`read_maxent_model()`](https://thisistaimur.github.io/maxentgpu/reference/read_maxent_model.md)
  : Load a fitted model

## Independent-species batches

- [`maxent_fit_batch()`](https://thisistaimur.github.io/maxentgpu/reference/maxent_fit_batch.md)
  : Fit independent species models from a validated batch specification
- [`predict(`*`<maxent_batch_model>`*`)`](https://thisistaimur.github.io/maxentgpu/reference/predict.maxent_batch_model.md)
  : Predict from an independent-species batch model
