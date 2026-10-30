# csfr: compressed sensing fragment reconstruction

reproducibility package for the paper *constraint-driven forensic image
reconstruction using auditable compressed sensing*

csfr reconstructs corrupted image fragments in the decoded pixel domain by
minimising a penalised objective under four constraint families. it preserves
observed pixels exactly in the released output, runs deterministically without
a learned prior, and emits an approximate local sensitivity record for sampled
imputed pixels. an idealised estimator carries a conditional recovery bound; the
fixed-budget implementation does not. the claim is auditability of the
evaluated implementation, not perceptual quality and not admissibility.

## verify the released work

```bash
bash scripts/verify.sh
```

about three minutes, most of it the test suite. runs the tests, a downstream
smoke test into a temporary directory, regenerates every table from the
released csvs, builds the manuscript, the supplement and the ieee access
version of the same source, and refreshes the submission-named pdf copies so a
stale one cannot be handed out. exit 0 only if all of it passes with zero
undefined references and zero latex errors.

## terminology

the paper was revised after review to name things by what the code does. the
code keeps its earlier identifiers so released manifests and scripts resolve.

| paper | code | meaning |
|---|---|---|
| audit record coverage (arc), metric m5 | `provenance.measured_audit_record_coverage` (alias of `measured_traceability`); `metrics.audit_record_coverage`; json key `arc` beside `ts` | fraction of declared queries for which the implementation produced a finite, non-empty sensitivity record. says the artefact exists, not that it is exact or causal |
| approximate local sensitivity record | `src/provenance.py`, `provenance_row`, `write_manifest`, `results/provenance_example/` | signed jacobian row of the solver's stationarity condition, normalised to unit absolute sum. not contribution shares |
| exploratory auditability composite | `render_frontier.py`, `run_frontier_sensitivity.py` (`ARC` dict) | equal-weight combination of cvr, normalised hrp and arc. describes the evaluated software, not forensic validity |
| c3, global intensity bounds | `clamp_(LO, HI)` in `csfr_reconstruct_2d.py` | clipping to [0, 255] every step. neighbourhood bounds are not implemented |

## ieee access build

`scripts/build_ieee.py` derives `paper/paper2_ieeeaccess.tex` from the
elsarticle source: class, front matter, alphabetical keywords, ieee headings,
`IEEEtran` bibliography style and biographies from `paper/authors_ieee.json`.
it uses the journal's `ieeeaccess.cls` when the template is unzipped into
`paper/ieeeaccess/` (not committed; download it from the ieee access author
page) and falls back to `IEEEtran` otherwise. `--build` compiles it.
`scripts/build_ieee_docx.py` turns the generated tex into the word manuscript
the journal accepts for submission: pandoc with the ieee citation style,
the official word template (`Access-Template-2024.docx`, also in
`paper/ieeeaccess/`) as the reference document, and a python-docx pass for
ieee heading numbering, equation numbers, numbered captions and table
borders. it compares every number in every table with the tex and, with
`--pdf`, prints the file through word for a visual check.
`scripts/render_graphical_abstract.py` draws the 660 x 295 graphical abstract
the journal asks for, live from the released solver on one cf1 patch, into
the gitignored `submission/` folder.

this checks the released numbers without recomputing them. full recomputation
takes hours of cpu plus a gpu.

## what is released

| path | what |
|---|---|
| `src/` | solver, sensitivity audit, downstream classifier, external-prior inpainter |
| `scripts/` | sweep, downstream, statistics, figure and table generators |
| `configs/` | experiment configurations |
| `data/` | patch corpora with cluster labels and manifests, see `data/README.md` |
| `results/` | per-cell metrics, statistics, released table bodies |
| `paper/` | manuscript, supplement, figures |
| `tests/` | 92 tests, including checks that every number in the paper matches its csv, that no caption describes a table other than the one under it, and that wording withdrawn after review does not return |

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
| f9 exploratory composite | `render_frontier.py` (the only generator of that file) | `results/csfr_sweep_2d_v2/summary.csv` |
| f12 sensitivity queries | `render_provenance_walkthrough.py` | `results/csfr_sweep_2d_v2/CF3_L5/seed0/`, `results/provenance_example/` |

prose numbers are hand-written. `tests/test_manuscript_numbers.py` recomputes
each one from the csvs and fails if a claim is missing as well as if it is wrong.

## cpu and gpu

two environments, and the artefacts record which produced what.

| results | environment | recorded in |
|---|---|---|
| primary sweep, downstream | linux, python 3.12.13, torch 2.10.0+cu128, nvidia t4 | `csfr_sweep_2d_v2/run_metadata.json`, `downstream/environment.json` |
| external validation, runtime bench | windows 11, python 3.14.6, torch 2.13.0+cpu | `csfr_sweep_govdocs1/run_metadata.json`, `bench/environment.json` |

`requirements.txt` pins the cpu environment. the gpu runs used torch
2.10.0+cu128, scipy 1.16.3, scikit-learn 1.6.1, scikit-image 0.25.2.

within the primary sweep every method reconstructed the same patches in one
job. csfr and lama are the torch methods and used the gpu; the opencv and
scikit-learn baselines ran on cpu because those libraries offer no gpu path.
no method got a separate run. runtime is measured on its own uniform cpu path,
so the device split does not reach the cost comparison.

`budget_sensitivity/` and the non-lama downstream runs predate environment
capture and carry no metadata.

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

88 passed, 4 skipped. the skips are the mnist tests when the cache is absent and
the heavy inpainter tests when model weights are absent. run the heavy ones with:

```bash
CSFR_RUN_INPAINTER_TESTS=1 python -m pytest tests/test_inpainter.py -q
```

## licence

see `LICENSE`.
