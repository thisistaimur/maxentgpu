# CUDA batch prediction performance checkpoint

These measurements compare shared-design dense CUDA prediction with the current
independent scalar CPU prediction path. Fitting is performed before timing and is
excluded from both measurements. All species converged and all comparisons had zero
maximum absolute difference.

Environment: JUWELS Booster A100, Torch 0.17.0, commit
`b48c9c61e8866e9814dc08ec20e46927c006e3f5`.

| Species | Background | Presence/species | Repeats | Dense CUDA median (s) | Scalar CPU median (s) | CUDA/CPU ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2,000 | 32 | 5 | 0.023 | 0.021 | 1.095 |
| 128 | 10,000 | 64 | 5 | 0.090 | 0.087 | 1.034 |
| 256 | 20,000 | 64 | 3 | 0.329 | 0.262 | 1.256 |

The ratios show that CUDA was slower in all three prediction-only cases: approximately
9.5%, 3.4%, and 25.6% slower, respectively. These are diagnostic measurements, not
end-to-end performance claims. GPU launch and host/device transfer overhead remain
material, and the current scalar baseline predicts on CPU.

Conclusion: retain the dense CUDA path for correctness and future profiling, but do
not advertise a batch prediction speedup. A credible speedup requires profiling and
an end-to-end fit-plus-predict workload or a substantially larger feature/problem size.
