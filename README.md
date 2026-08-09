# csfr: compressed sensing fragment reconstruction

reproducibility package for the paper *constraint-driven forensic image
reconstruction using auditable compressed sensing*

csfr reconstructs corrupted image fragments as a regularised inverse problem
under four forensic constraints. it preserves observed pixels exactly, carries a
conditional recovery bound, and emits a per-pixel provenance record saying which
observed pixels determined each imputed one. the claim is auditability, not
perceptual quality.

## verify the released work

```bash
bash scripts/verify.sh
```

about 35 seconds. runs the test suite, a downstream smoke test into a temporary
directory, regenerates every table from the released csvs, and builds the
manuscript. exit 0 only if all of it passes with zero undefined references and
zero latex errors.

this checks the released numbers without recomputing them. full recomputation
takes hours of cpu plus a gpu.

## what is released

| path | what |
|---|---|
| `src/` | solver, provenance, downstream classifier, external-prior inpainter |
| `scripts/` | sweep, downstream, statistics, figure and table generators |
| `configs/` | experiment configurations |
| `data/` | patch corpora with cluster labels and manifests, see `data/README.md` |
| `results/` | per-cell metrics, statistics, released table bodies |
| `paper/` | manuscript, supplement, figures |
| `tests/` | 67 tests, including checks that every number in the paper matches its csv |

not released, by policy: solver checkpoints, per-seed `x_hat`, `y` and `mask`
arrays, raw corpora, model weights. all regenerate from the committed patch sets
and configs. one exception is `results/downstream/classifier.pt` (2.4 mb), the
frozen classifier, committed so anyone re-running gets bit-identical downstream
numbers.

## where each number comes from

every table in the paper is generated, never typed.

| table | generator | source |
|---|---|---|
| psnr/ssim, cvr, hrp, ablation | `update_paper_tables.py` | `results/csfr_sweep_2d_v2/summary.csv` |
| paired inference | `render_stats_tables.py` | `results/paired_stats/` |
| runtime, external, budget | `render_stats_tables.py` | `results/bench/`, `csfr_sweep_govdocs1/`, `budget_sensitivity/` |
| downstream, abstention | `render_stats_tables.py` | `results/downstream/` |
| f9 frontier | `render_frontier.py` | `results/csfr_sweep_2d_v2/summary.csv` |
| f12 provenance | `render_provenance_walkthrough.py` | `results/provenance_example/` |

prose numbers are hand-written. `tests/test_manuscript_numbers.py` recomputes
each one from the csvs and fails if a claim is missing as well as if it is wrong.

## cpu and gpu

everything except the `lama` method was computed on cpu (windows 11, python
3.14, torch cpu build). the `lama` results were computed on an nvidia t4
(python 3.12, torch 2.10.0+cu128). these are separate runs reported separately.
no table row in the paper mixes devices.

## reproducing from scratch

```bash
pip install -r requirements.txt
```

cpu, roughly 4 to 6 hours:

```bash
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_govdocs1.yaml
python scripts/run_paired_stats.py
python scripts/run_bench.py
python scripts/run_budget_sensitivity.py
python scripts/run_frontier_sensitivity.py
```

gpu, about an hour, see `kaggle/README.md`:

```bash
python scripts/run_downstream.py --config configs/downstream.yaml --n-test 1000
python scripts/run_downstream_stats.py
```

figures and tables:

```bash
python scripts/render_figures.py
python scripts/render_frontier.py
python scripts/render_provenance_walkthrough.py
python scripts/update_paper_tables.py
python scripts/render_stats_tables.py
```

there is no one-command full reproduction. the gate verifies, recomputation is a
documented multi-step process.

## external-prior baseline

the paper argues against generative restoration in evidential use, so it runs
one: lama, via `simple-lama-inpainting`, weights fetched on first use and not
committed.

installing it downgrades numpy and pillow. restore them before importing
anything, in a separate cell or shell:

```bash
pip install simple-lama-inpainting
pip install --upgrade "numpy>=2" "pillow>=10.1"
```

numpy's c extension cannot be reloaded in a live process, so install first,
import second.

## tests

```bash
python -m pytest tests/ -q
```

67 passed, 4 skipped. the skips are the mnist tests when the cache is absent and
the heavy inpainter tests when model weights are absent. run the heavy ones with:

```bash
CSFR_RUN_INPAINTER_TESTS=1 python -m pytest tests/test_inpainter.py -q
```

## licence

see `LICENSE`.
