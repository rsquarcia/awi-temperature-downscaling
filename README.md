# Temperature downscaling across climates

Code and compact results for **Generalization and Interpretability of U-Net and EOF–CCA in Temperature Downscaling Across Climates**, by Riccardo Squarcia, Nina Öhlckers, Alexander Thorneloe, and Gerrit Lohmann.

The study compares a residual U-Net, EOF–CCA, and bilinear interpolation for daily 2-m temperature downscaling. Both learned models are trained on present-day AWI-CM3 output and evaluated, without refitting, on present-day, mid-Holocene, and SSP5-8.5 data.

## Where to start

| Location | Contents |
|---|---|
| `code/unet/` | Network, data loading, normalization, and training |
| `code/cca/` | EOF bases, score projection, CCA search, and selected-model evaluation |
| `code/evaluation/` | One shared inference runner and climate-specific numerical reduction |
| `code/diagnostics/` | Error decomposition, uniform-shift response, predictor-dimension sensitivity, and map diagnostics |
| `configs/` | Selected settings, normalization constants, and grid definitions |
| `results/` | Compact numerical tables and summaries |
| `tests/test_checks.py` | Small offline checks |

For the **U-Net**, start with `code/unet/training/global_unet.py`; the training runner is `global_unet_second_stage.py` in the same folder. The selected network has five input channels, base width 96, depth 6, GroupNorm, periodic longitude, and pole-aware latitude padding. Its settings and frozen normalization are in `configs/unet/`.

The chronological training split contains 10,593 days. The selected U-Net optimizes a fixed 10,336-day pool and selects its checkpoint using 548 every-other validation days; subsequent candidate selection uses all 1,095 validation days. The saved run configuration records these executed settings.

For **EOF–CCA**, follow the numbered scripts in `code/cca/`. The selected configuration is Kx = 10592, Ky = 512, r = 512, and ridge = 0. The `03b` script retains the original physical-unit reporting correction.

For **inference**, use `code/evaluation/run_inference.py`. Its `unet`, `project-scores`, and `cca-shard` commands take `--split test`, `--split mh`, or `--split ssp585`. Run `python code/evaluation/run_inference.py --help` for the command list; add a command followed by `--help` for its arguments. The climate-specific input paths, fixed settings, and SSP5-8.5 window checks are retained.

For the **error analysis**, start with `code/diagnostics/decomposition_exact.py` and the `uniform_shift_*.py` scripts.

Both U-Net inference entry points execute the shipped model and dataset modules under `code/unet/` and check their imported paths. The original training-source hash is checked against checkpoint metadata; it is not the identity of the public source. Neither entry point requires the private executed snapshot or its manifest.

Each Python file begins with an overview. The saved configurations specify the paper's selected settings; some original script defaults also support earlier experiments.

## Results and definitions

The paper uses **RMSE skill = 1 − RMSE_method / RMSE_reference**. Positive values indicate improvement. The additive error decomposition is in squared-error units (K²).

`results/headline_metrics.csv` contains global estimates and final bootstrap intervals. `regional_metrics.csv` covers 12 overlapping regions and five periods per climate. `kx_sensitivity.csv` distinguishes ΔB from ΔM. The remaining files contain the grid search, spectra, decomposition, uniform-shift response, map correlations, and XIOS sensitivity.

The original climate-specific reduction routines also produce historical **MSE-ratio skill** columns. Those are not the paper's final RMSE-ratio skill. The final calculation is `three_climate_rmse_skill_statistics` in `code/evaluation/ssp585/reduce_and_report.py`, available through its `rmse-skill` command:

```bash
python code/evaluation/ssp585/reduce_and_report.py rmse-skill \
  --pd-root /path/to/retained_pd \
  --mh-root /path/to/retained_mh \
  --retention-root /path/to/retained_ssp585 \
  --output-root /path/to/new_statistics
```

This reads retained daily partials, their metadata, and the existing SSP5-8.5 reference CSV. It exports estimates and intervals without plotting or rerunning inference. The original reference-consistency check remains. The bootstrap uses 10,000 paired circular moving-block replicates, 60-day blocks, seed 20260715, and climate draw order PD, MH, SSP5-8.5. RMSE ratios are computed within each replicate. The separate bilinear baselines used by the two numerical pipelines are preserved.

## Execution and scope

Install dependencies with `pip install -r requirements.txt`. The research workflows were developed on Linux and require external simulation data, preprocessed stores, and fitted model artifacts. They do not run end to end from this repository alone. Original input-identity and checkpoint checks remain.

Scripts refer to external roots through environment variables such as `DATA_ROOT`, `RESULTS_ROOT`, and `MODEL_ROOT`; training also uses its original repository/snapshot settings. JSON paths containing `${...}` are templates and must be substituted before loading: setting an environment variable alone does not expand JSON strings.

From the repository root, an inference invocation is:

```bash
export DATA_ROOT=/path/to/prepared_data
export MODEL_ROOT=/path/to/model_artifacts
export RESULTS_ROOT=/path/to/external_results
export NORM_STATS_PATH="$PWD/configs/unet/norm_stats.json"
python code/evaluation/run_inference.py unet \
  --split test --input-store "$DATA_ROOT/zarr/awi_downscaling_test.zarr" \
  --start-date 2012-01-01 --end-date 2014-12-31 \
  --output-root /path/to/new_pd_predictions --device cuda --n-shards 20
```

`MODEL_ROOT` supplies `checkpoint/best_val_mse.pt`. Both inference entry points default to the bundled, byte-compatible `configs/unet/norm_stats.json`; `NORM_STATS_PATH` explicitly selects that same file above. Its checksum and numerical equality with the checkpoint remain checked, without copying anything into the data directories. The public training module can be invoked with `PYTHONPATH="$PWD/code/unet" python -m training.global_unet_second_stage --help`.

Prepared Zarr stores must contain normalized dynamic inputs, target temperature, ordered dates, and the raw land/lake static inputs with the completion and channel metadata expected by the dataset. The canonical MH and SSP store locations are under `RESULTS_ROOT` as listed in the inference runner. SSP uses the complete 2096–2098 window (1,096 days); its population/window safeguards remain mandatory.

CCA inference requires the frozen X (maxK10592) and Y (maxK1024, reused July 3) EOF bases and `selected_cca_model.npz`; retained per-climate X scores may replace fresh projection. Stages 02–04 also need the original train/validation score arrays with unmodified `.done.json` records, prepared target-normalized baseline/residual shards, and the hash-pinned source manifest. Stage 01 expects the training baseline mean and Stage A/B, Stage00 run/preflight/input-hash and grid-recommendation metadata named in `eof_basis_config.json`; this release starts from those prepared external products. Stage 04 additionally requires the original hash-pinned selected metadata, grid summary, metric constants and statics. The descriptive metadata copy under `configs/` does not replace the original hash-pinned reference file.

Relocate only actual CCA file locators (such as `path`, `npz` and `done_json`) and output locations. Leave `expected_ident`, `expected_basis_ident` and all hash pins unchanged, including the historical `basis_npz` strings. Preserve original metadata bytes. Paths embedded in the pinned shard manifest must remain accessible, for example through a read-only mount or symlink; rewriting that manifest would invalidate its identity.

`python code/diagnostics/decomposition_bootstrap.py /path/to/new_decomposition` reads the retained mechanism arrays under `RESULTS_ROOT/ssp585_pd_mh_full_mechanism_20260802T095350Z`. Its independent full-precision references are bundled in `results/decomposition_reference_values.json`, preserving coefficient-space and pixel-space quantities, including their distinct M values. Private audit and triage reports are not required; the original consistency checks and tolerances remain.

This is a curated research-code release. Numerical functions were retained without rewriting their calculations; duplicated helpers are shared in `code/downscaling_numerics.py` and `code/cca/cca_utils.py`. The optimizer builder was moved unchanged into the training script. The three inference runners were consolidated, preserving their climate-specific settings and checks. Standalone CCA development smoke modes, the training self-test command, and training-history plot rendering were removed; production checks, training/validation operations, checkpoints and numerical outputs remain. Old figure-rendering, exploratory review-report, and review-package assembly code was omitted from the evaluation scripts. This repository is not a complete manuscript-figure rendering package. Scientific settings and numerical result values are preserved.

The small tests use NumPy and, for network checks, PyTorch:

```bash
python -m unittest discover -s tests -v
```

They check synthetic arrays and imports, not the full paper experiments. Large simulation output, Zarr stores, predictions, EOF bases, and fitted CCA arrays are not included. A link to the trained U-Net weights will be added when their Zenodo deposit is available.

## Review status and citation

The licence and public repository URL remain to be confirmed. See `LICENSE_PENDING.md` and `CITATION.cff`. Publication details and the archival DOI will be added when available. The intended GitHub and Zenodo source releases contain the same files.
