# csfr — compressed sensing fragment reconstruction

reproducibility package for the paper *constraint-driven forensic image
reconstruction using auditable compressed sensing* (benjamin tei partey, yaw
osei adjei, knust).

csfr reconstructs corrupted image fragments as a regularised inverse problem
under four forensic constraints. it preserves observed pixels exactly, carries a
conditional recovery bound, then emits a per-pixel provenance record. this repo
holds the solver, the evaluation harness, the released results, the manuscript.

## verification gate

one command verifies that the paper's numbers are reproducible without
recomputing experiments:

```bash
bash scripts/verify.sh
```

this runs the code tests, a downstream smoke test (into a temporary directory),
regenerates all tables from the released result CSVs, and checks that the
manuscript builds with no undefined references. **wall-clock time: ~50 seconds**
on a modern machine (Python 3.10+, torch CPU).

a passing gate confirms:
- all 58 code tests pass; 4 tests skip deliberately (see note below)
- the downstream harness works end-to-end
- every table in the paper regenerates identically from its source CSV
- the LaTeX build succeeds with no undefined references or citations

**test skips:** four tests in `tests/test_inpainter.py` that load LaMa model
weights (several gb) are skipped by default because downloading them during
routine verification would burden every clone. to run them once on a GPU box
after caching the weights:
```bash
CSFR_RUN_INPAINTER_TESTS=1 python -m pytest tests/test_inpainter.py -q
```
see the docstring in `tests/test_inpainter.py` for details.

## what is released and what requires recomputation

**released (committed to repo):**
- solver code (`src/`), test suite, evaluation harness
- 5 input patch corpora: DFRWS 2006 (300 patches), GovDocs1 (300), Nick Mikus
  carved images FAT and Ext2 (20 and 26 colour patches)
- 3 result directories containing pre-computed metrics:
  - `results/csfr_sweep_2d_v2/`: primary sweep (20 cells × 3 seeds × 11 methods)
  - `results/csfr_sweep_govdocs1/`: external validation (20 cells × 3 seeds × 6 methods)
  - `results/downstream/`: digit CNN utility (20 cells × 7 methods × 1000 patches)
  - supporting results: `paired_stats/`, `bench/`, `budget_sensitivity/`,
    `sensitivity/`, `provenance_example/`
- frozen DigitCNN classifier (`results/downstream/classifier.pt`, 2.4 mb,
  committed deliberately so every downstream number is bit-identical)
- generated table rows (`results/tables_tex/`)
- manuscript source and PDF

**not released (regenerate on demand):**
- solver checkpoints (`*.pt`), reconstructed arrays (`x_hat`, `y`, `mask` as
  `.npy`), and sensitivity sweeps (`*.npz`); these are in `.gitignore` and
  regenerate from the corpora + config
- SVHN/MNIST raw dataset (automatically fetched by `src/downstream.py`)
- LaMa inpainting weights (fetched by `simple-lama-inpainting` on first use)

## device split and experimental setup

**all numbers in the paper except the LaMa baseline were computed on CPU:**
- Windows 11, Python 3.10–3.14, torch CPU build
- solver is deterministic across runs on the same hardware
- CSFR tables, figures, and ablation studies use only CPU results

**LaMa baseline (external-prior method) was computed separately on GPU:**
- Kaggle notebook with T4 GPU, Python 3.12, torch 2.10.0+cu128
- see `kaggle/README.md` for the reproducible runbook
- LaMa results are reported separately and **not compared directly to CPU-only
  methods in any table row**; each table specifies its methods

**why the split?**
CSFR is CPU-efficient and all ablations fit on a laptop. LaMa requires a GPU
to run in reasonable time. Rather than hobble the primary results, we
acknowledge the split in the paper and keep the baselines honest by device.

## reproduction path with timings

to verify the released results (gate pass: ~50 seconds, no recompute):
```bash
bash scripts/verify.sh
```

to regenerate the full pipeline from scratch (several hours of computation):

```bash
# primary sweep (DFRWS 2006, 20 cells × 3 seeds, CPU)
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml        # ~3 hours

# external validation (GovDocs1, CPU)
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_govdocs1.yaml  # ~40 min

# downstream utility experiment (SVHN, CPU)
python scripts/run_downstream.py --config configs/downstream.yaml --n-test 1000   # ~15 min
python scripts/run_downstream_stats.py                                            # <1 min

# paired inference stats (cluster-robust CIs)
python scripts/run_paired_stats.py                                          # ~10 min

# microbenchmarks (runtime, peak memory)
python scripts/run_bench.py                                                 # ~20 min

# sensitivity sweeps
python scripts/run_budget_sensitivity.py                                    # ~30 min
python scripts/run_frontier_sensitivity.py                                  # ~20 min

# figures (all CPU)
python scripts/render_figures.py && python scripts/render_frontier.py       # figures f1-f9 (seconds)
python scripts/render_qualitative.py && python scripts/render_fullres_colour.py  # f10 f11 (seconds)

# LaMa baseline (GPU required; see kaggle/README.md)
# [separate Kaggle notebook; results in kaggle/]
```

seeds are `[0, 1, 2]` (config key `seeds:`). the solver is deterministic on
fixed hardware; each result directory records its environment in
`run_metadata.json`.

## where every number comes from

**primary results table (PSNR, SSIM):**
- source: `results/csfr_sweep_2d_v2/*/seed*/metrics.json`
- script: `scripts/update_paper_tables.py` splices `results/tables_tex/psnr_ssim_rows.tex`
  into the manuscript

**ablation table (CSFR variants):**
- source: `results/csfr_sweep_2d_v2/*/seed*/metrics.json`
- script: `scripts/update_paper_tables.py` splices `results/tables_tex/ablation_rows.tex`

**forensic metrics (CVR, HRP):**
- source: `results/csfr_sweep_2d_v2/*/seed*/metrics.json`
- script: `scripts/update_paper_tables.py` splices `results/tables_tex/cvr_rows.tex`
  and `hrp_rows.tex`

**external validation (GovDocs1):**
- source: `results/csfr_sweep_govdocs1/*/seed*/metrics.json`
- script: `scripts/render_stats_tables.py` splices `results/tables_tex/external_rows.tex`

**paired inference (cluster-robust CIs):**
- source: `results/paired_stats/summary.csv`
- script: `scripts/render_stats_tables.py` splices `results/tables_tex/paired_rows.tex`

**microbenchmarks (runtime, memory):**
- source: `results/bench/summary.csv`
- script: `scripts/render_stats_tables.py` splices `results/tables_tex/bench_rows.tex`

**budget sensitivity sweep:**
- source: `results/budget_sensitivity/*.csv`
- script: `scripts/render_stats_tables.py` splices `results/tables_tex/budget_rows.tex`

**downstream utility (downstream accuracy, ECE, McNemar):**
- source: `results/downstream/*.csv` (clean, metrics, stats)
- script: `scripts/render_stats_tables.py` splices `results/tables_tex/downstream_rows.tex`
  and `downstream_abstain_rows.tex`

**figures F1–F9 (method comparison, sensitivity, frontier):**
- source: `results/csfr_sweep_2d_v2/*/seed*/metrics.json`, sensitivity CSVs
- script: `scripts/render_figures.py` generates SVG overlays; `scripts/render_frontier.py`
  generates F6 (method frontier)

**figure F10 (qualitative DFRWS patches):**
- source: `results/csfr_sweep_2d_v2/*/seed0/x_hat.npy`, `y.npy`, `mask.npy`
- script: `scripts/render_qualitative.py` renders 2×3 panel

**figure F11 (full-resolution colour cases):**
- source: `data/full_mikus_*/` (carved full images), `results/csfr_sweep_2d_v2/CF4_L5/seed0/`
- script: `scripts/render_fullres_colour.py` reconstructs and renders

## downstream utility experiment

the paper argues that hallucinatory reconstruction without per-pixel provenance
cannot be trusted near decision boundaries. the downstream utility harness tests
this claim empirically by corrupting SVHN digits, reconstructing them with each
method, and measuring classification accuracy with a frozen CNN trained on clean
data.

the harness selects a class-balanced 1000-image evaluation subset from the SVHN
test set (seeded for reproducibility), draws 20 corruption masks per image (one
mask draw, applied to all methods for fair comparison), runs each reconstruction
method, and scores predictions against labels. output is per-patch CSVs and
aggregated metrics.

**classical baselines** (zero-fill, bilinear, Teila, Navier–Stokes, dictionary
learning, CSFR) are all implemented in `src/` and run on CPU as part of the main
pipeline.

**LaMa baseline (generative inpainting):** the external-prior method requires
installing `simple-lama-inpainting`, which provides big-lama weights. this is
optional; if not installed, the harness reports classical baselines only.

⚠️ **note:** installing `simple-lama-inpainting` downgrades numpy and pillow to
versions incompatible with the rest of the repo. if you install it, reinstall
the repo requirements afterward:
```bash
pip install simple-lama-inpainting
pip install -r requirements.txt   # restore pinned versions
```
see `kaggle/README.md` for a GPU-friendly alternative using a Kaggle notebook
with torch 2.10.0+cu128 and T4 GPU.

## frozen classifier

the downstream evaluation uses a small frozen DigitCNN trained on clean SVHN,
saved at `results/downstream/classifier.pt` (2.4 mb). it is committed as an
exception to the no-checkpoints policy because bit-exact reproducibility matters:
every downstream accuracy and ECE number in the paper refers to this exact
classifier. retraining on a different machine (particularly a GPU) would
silently change the accuracy ceiling and invalidate all comparisons.

clean accuracy:
- full SVHN test set (10,000 images): 87.83%
- class-balanced evaluation subset (1,000 images, seed=0): 86.90%

## layout

```
src/        solver, metrics, provenance
scripts/    experiment harness, statistics, figures, tables
configs/    sweep configs (primary + external)
data/       patch corpora, cluster labels, manifests
results/    released metrics, statistics, generated table rows
paper/      manuscript tex, bib, pdf, figures
tests/      unit + smoke tests
```

## files needed to reproduce

each line gives the file with a brief conventional-commit label.

**source**
- `src/__init__.py` — chore: package marker
- `src/csfr_reconstruct_2d.py` — feat: csfr dct-2d solver
- `src/metrics.py` — feat: forensic metrics psnr ssim cvr hrp ts
- `src/provenance.py` — feat: per-pixel provenance via implicit differentiation
- `src/downstream.py` — feat: frozen DigitCNN classifier, SVHN/MNIST loader, training helpers
- `src/generative_inpainter.py` — feat: LaMa and Stable Diffusion inpainting backend

**experiment scripts**
- `scripts/run_csfr_sweep.py` — feat: corruption sweep with constraint ablation
- `scripts/run_paired_stats.py` — feat: paired cluster-robust inference
- `scripts/run_bench.py` — feat: runtime and peak-memory benchmark
- `scripts/run_budget_sensitivity.py` — feat: iteration-budget sensitivity
- `scripts/run_frontier_sensitivity.py` — feat: frontier-weight sensitivity
- `scripts/extract_patches.py` — feat: carve and dice patch corpora
- `scripts/run_downstream.py` — feat: downstream utility: corrupt SVHN → reconstruct → classify
- `scripts/run_downstream_stats.py` — feat: accuracy, ECE, McNemar and bootstrap CIs on downstream

**figures and tables**
- `scripts/render_figures.py` — feat: render figures f1 f2 f6 f8 f9
- `scripts/render_frontier.py` — feat: render method-frontier figure
- `scripts/render_qualitative.py` — feat: render qualitative panel f10
- `scripts/render_fullres_colour.py` — feat: render full-res colour cases f11
- `scripts/render_stats_tables.py` — feat: splice paired bench external budget tables
- `scripts/update_paper_tables.py` — feat: splice psnr cvr hrp ablation tables
- `scripts/verify.sh` — test: build-and-test gate

**configs**
- `configs/csfr_sweep_2d.yaml` — chore: primary sweep config
- `configs/csfr_sweep_govdocs1.yaml` — chore: external-validation config
- `configs/csfr_sweep_2d_t2400.yaml` — chore: matched-compute sweep (T=2400, not yet executed)
- `configs/downstream.yaml` — chore: downstream utility sweep config

**data**
- `data/patches.npy` — data: dfrws 2006 patch corpus
- `data/patches_cluster.npy` — data: dfrws source-image labels
- `data/patches_manifest.json` — docs: dfrws corpus manifest
- `data/patches_govdocs1.npy` — data: govdocs1 external corpus
- `data/patches_govdocs1_cluster.npy` — data: govdocs1 source-image labels
- `data/patches_govdocs1_manifest.json` — docs: govdocs1 manifest
- `data/patches_mikus_11-carve-fat.npy` — data: mikus fat colour corpus
- `data/patches_mikus_12-carve-ext2.npy` — data: mikus ext2 colour corpus
- `data/full_mikus_11-carve-fat/` — data: carved full images (fat)
- `data/full_mikus_12-carve-ext2/` — data: carved full images (ext2)
- `data/README.md` — docs: data provenance with rebuild steps

**results**
- `results/csfr_sweep_2d_v2/` — data: primary sweep metrics + summary
- `results/csfr_sweep_govdocs1/` — data: external sweep metrics
- `results/paired_stats/` — data: per-patch metrics + paired tests
- `results/bench/` — data: runtime and memory
- `results/budget_sensitivity/` — data: budget sweep
- `results/sensitivity/` — data: frontier sensitivity
- `results/provenance_example/` — data: example provenance manifest
- `results/downstream/` — data: per-cell classifications, metrics, statistics
- `results/downstream/classifier.pt` — data: frozen DigitCNN (committed, 2.4 mb)
- `results/tables_tex/` — data: generated latex table rows

**tests**
- `tests/test_csfr_smoke.py` — test: solver smoke test
- `tests/test_metrics.py` — test: metrics unit tests
- `tests/test_provenance.py` — test: provenance unit tests
- `tests/test_downstream.py` — test: digit data load, classifier forward, determinism
- `tests/test_downstream_stats.py` — test: McNemar, bootstrap CIs, abstention metrics
- `tests/test_inpainter.py` — test: LaMa + Stable Diffusion composition (opt-in, gpu-heavy)

**manuscript**
- `paper/paper2_reconstruction.tex` — docs: manuscript source
- `paper/paper2_reconstruction.bib` — docs: bibliography
- `paper/paper2_reconstruction.pdf` — docs: compiled manuscript
- `paper/figures/` — docs: manuscript figures

**meta**
- `requirements.txt` — build: pinned dependencies
- `README.md` — docs: reproducibility guide
- `LICENSE` — chore: mit licence

## licence

mit. see [LICENSE](LICENSE).
