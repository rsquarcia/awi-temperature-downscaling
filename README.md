# Temperature downscaling across climates

Code accompanying **Generalization and Interpretability of U-Net and EOF–CCA in Temperature Downscaling Across Climates**.

We compare residual U-Net and EOF–CCA against bilinear interpolation for daily 2-m temperature downscaling. Models trained on present-day simulations are applied unchanged to present-day, mid-Holocene and SSP5-8.5 data.

## Contents

- [U-Net](code/unet/) and [EOF–CCA](code/cca/): model and training code.
- [Evaluation](code/evaluation/) and [diagnostics](code/diagnostics/): inference, metrics and error analysis.
- [Configurations](configs/) and [results](results/): paper settings and numerical summaries.

## Use

```bash
pip install -r requirements.txt
python code/evaluation/run_inference.py --help
python -m unittest discover -s tests -v
```

Developed on Linux. Full experiments require external prepared data, fitted model artifacts and original metadata; these are not included. Tests are small synthetic checks, not reproductions of the experiments.

Set `DATA_ROOT`, `MODEL_ROOT` and `RESULTS_ROOT` as required by each script. Replace `${...}` paths in JSON configurations explicitly; environment variables are not expanded there automatically. For CCA, relocate file/output paths only: preserve pinned metadata and identity fields, including embedded historical paths.

## Paper settings

- **U-Net:** base width **96**, depth **6**, five input channels, GroupNorm and pole-aware latitude padding. See [run configuration](configs/unet/unet_run_config.json).
- **EOF–CCA:** Kx = 10592, Ky = 512, r = 512, ridge = 0.

Paper skill is `1 − RMSE_method / RMSE_reference`. Some legacy reducers also output MSE-ratio skill; use the `rmse-skill` command in [the final reducer](code/evaluation/ssp585/reduce_and_report.py) for the paper statistic.

## Availability and citation

U-Net weights and archival DOI: **[TO DO]**. License: [pending](LICENSE_PENDING.md). Citation: [CITATION.cff](CITATION.cff).

