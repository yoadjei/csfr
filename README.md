# csfr: compressed sensing fragment reconstruction

reproducibility package for the paper *constraint-driven forensic image
reconstruction using auditable compressed sensing*

csfr reconstructs corrupted image fragments as a regularised inverse problem
under four forensic constraints. it preserves observed pixels exactly, carries a
conditional recovery bound, then emits a per-pixel provenance record. this repo
holds the solver, the evaluation harness, the released results, the manuscript.

## setup

```bash
python -m venv .venv && source .venv/Scripts/activate   # windows git-bash
pip install -r requirements.txt                          # python >= 3.10
```

## reproduce

one command runs the tests, regenerates the tables from the released csvs, then
builds the pdf:

```bash
bash scripts/verify.sh
```

to rerun the experiments from scratch (cpu, a few hours for the primary sweep):

```bash
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml        # primary sweep
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_govdocs1.yaml  # external corpus
python scripts/run_paired_stats.py                                          # paired inference
python scripts/run_bench.py                                                 # runtime + memory
python scripts/run_budget_sensitivity.py                                    # budget sweep
python scripts/run_frontier_sensitivity.py                                  # frontier weights
python scripts/render_figures.py && python scripts/render_frontier.py       # figures f1-f9
python scripts/render_qualitative.py && python scripts/render_fullres_colour.py  # f10 f11
```

seeds are `[0, 1, 2]` (config key `seeds:`). the solver is deterministic on
fixed hardware; each result directory records its environment in
`run_metadata.json`.

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

**experiment scripts**
- `scripts/run_csfr_sweep.py` — feat: corruption sweep with constraint ablation
- `scripts/run_paired_stats.py` — feat: paired cluster-robust inference
- `scripts/run_bench.py` — feat: runtime and peak-memory benchmark
- `scripts/run_budget_sensitivity.py` — feat: iteration-budget sensitivity
- `scripts/run_frontier_sensitivity.py` — feat: frontier-weight sensitivity
- `scripts/extract_patches.py` — feat: carve and dice patch corpora

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
- `results/tables_tex/` — data: generated latex table rows

**tests**
- `tests/test_csfr_smoke.py` — test: solver smoke test
- `tests/test_metrics.py` — test: metrics unit tests
- `tests/test_provenance.py` — test: provenance unit tests

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
