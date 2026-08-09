Needs the GPU:

| Job | Why | Cost on one T4 |
|---|---|---|
| **A. Generative inpainter (external-prior baseline)** | the paper argues against deep-learning restoration without ever running one | LaMa ~30-55 min full grid; diffusion ~14 h full grid, ~4 h on the severe-cell subset |
| **B. Matched-compute sweep at T=2400** | shows the CF2/CF3/CF4 deficit is mostly a stopping-rule artefact | 20-60 min |
| **C. Re-run downstream stats and tables** | only if A adds a method to the grid | under a minute, CPU |

Job B is currently **paused** at cell 1 of 15 and is resumable.

Job A's feasibility is partly answered already, by accident. A local run pulled
the weights and loaded the pipeline successfully: 3.26 gb over about 44 minutes
on a domestic connection, 15 files, pipeline components loaded without error. So
the model resolves, downloads and initialises. What is still unmeasured is
per-patch throughput and output quality, which is what section 3.1 is for.

### 0.1 The scheduling problem, read this first

A full-grid diffusion run is **20 cells x 1000 patches = 20,000 images**. On a
T4 at 20 steps and 512x512 in float16 that is roughly 2-3 seconds each, so about
14 hours. **Kaggle kills the session at 12 hours.** It does not fit.

Three ways out, in order of preference:

1. **Use LaMa instead of Stable Diffusion.** Roughly 50-100 ms per image on a T4,
   so the full grid is under an hour. This is also the plan's original first
   choice. See 1.6.
2. **Use both T4s.** The `T4 x2` accelerator gives two devices. Run the grid split
   across them (section 3.3), which halves wall-clock to about 7 hours. Still one
   session, still tight.
3. **Run the severe-cell subset only** (CF2/CF3/CF4 at L4 and L5, 6 cells, 6000
   images, about 4 hours on one T4). The paper then reports the external-prior
   comparison on severe corruption alone, which is defensible as long as it is
   stated plainly rather than implied.

Do not attempt the full diffusion grid in one session on a single GPU. It will be
killed at 12 hours with the grid incomplete, and it will cost most of the weekly
quota for nothing.

### 0.2 Kaggle quotas and limits that matter here

- **30 GPU-hours per week**, free tier. A full diffusion grid is roughly half of
  that in one go.
- **12 hours maximum** per GPU session (9 hours on some accelerators; check the
  sidebar when the session starts).
- Interactive sessions **stop when idle**, around 20-40 minutes with no activity.
  For anything long, use `Save Version -> Save & Run All (Commit)`, which runs
  the notebook top to bottom detached and does not care whether your browser is
  open.
- **Internet is off by default.** Turn it on in the notebook sidebar under
  `Settings -> Internet`. It requires a phone-verified account. Without it, pip
  installs and the Hugging Face download both fail.
- `/kaggle/working` is **20 gb** and is what gets saved as the notebook's output.
  `/kaggle/input` is read-only. `/tmp` is scratch and is discarded.

### 0.3 Which accelerator

Pick **GPU T4 x2**.

- **T4** has tensor cores, so float16 inference is genuinely fast, which is what
  the diffusion path needs. Two of them also allow the split in 3.3.
- **P100** has no tensor cores. Its float16 is roughly 2x its float32 rather than
  the far larger factor on a T4, so diffusion is meaningfully slower. It has more
  memory bandwidth, which does not help here.
- Job B (the CSFR solver) runs comfortably on either.

The code already picks float16 on CUDA and float32 elsewhere via
`resolve_dtype`, so no edit is needed for either card.

---

## 1. Code fixes: done, and what is still open

The four defects that would have wasted GPU time are **fixed and verified**
locally. Recorded here because they change what the run means, not just whether
it works.

| Fix | What it was | Now |
|---|---|---|
| Device | `device="cpu"` hardcoded in both `baseline_lama` hooks, so the GPU would sit idle | threaded from `pick_device`; `float16` on CUDA, `float32` elsewhere |
| Silent failure | `except Exception: return None`, which is the harness's "skip this method" signal, so a failed download or OOM looked like success | only `ImportError` returns `None`; every real failure raises |
| Observed pixels | the 32 -> 512 -> 32 round trip overwrote observed pixels, which every other baseline preserves and CSFR enforces as hard C1 | composited: `x_hat = y*M + pred*(1-M)`, observed data bit-exact |
| Pipeline reload | `from_pretrained` inside the call, reloading several gb per cell and seed | cached at module level by `(model_id, device, dtype)` |

Compositing is the one that matters scientifically. With observed pixels held
fixed, the only difference between methods is what they invent in the holes,
which is exactly the paper's argument. Without it, the inpainter's PSNR would
have been depressed by resampling loss that has nothing to do with hallucination.

Verified: `resolve_dtype` returns `float16` for cuda and `float32` for cpu, the
composite keeps observed pixels bit-exact, and both hooks now take a `device`
argument.

### 1.5 Model id: checked, no action needed

Contrary to what I first assumed, the primary id resolves and the obvious
fallback does not. Checked against the hub:

```
OK    runwayml/stable-diffusion-inpainting                 8a4288a76071
OK    stable-diffusion-v1-5/stable-diffusion-inpainting    8a4288a76071   (same revision)
OK    benjamin-paine/stable-diffusion-v1-5-inpainting      705090e31033
FAIL  stabilityai/stable-diffusion-2-inpainting            RepositoryNotFoundError
OK    kandinsky-community/kandinsky-2-2-decoder-inpaint    db790ad5cbca
```

`DEFAULT_MODEL_ID` stays `runwayml/stable-diffusion-inpainting`; `FALLBACK_MODEL_ID`
is now the community mirror at the identical revision. `get_model_info` records
the resolved revision sha into the environment capture, so the paper can name the
external prior exactly.

### 1.6 Still open, and now more important than before: prefer LaMa

On Kaggle this stops being a nicety. LaMa (`simple-lama-inpainting`) is
purpose-built for inpainting rather than repurposed text-to-image, needs no
prompt, and runs in tens of milliseconds per image instead of seconds. That is
the difference between a full grid that fits in one session and one that does
not.

The current code has only the diffusers path. If LaMa installs cleanly, add it as
a second backend and prefer it; keep diffusers as the fallback. Whichever runs,
the paper needs exactly one external-prior method, clearly labelled.

Note the diffusers path also passes the prompt `"a detailed image"` against SVHN
house numbers. That domain mismatch is worth a sentence in the limitations
either way.

---

## 2. Session setup

Notebook settings: **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence:
Files only** if offered.

### 2.1 Get the repo onto Kaggle

There is no Drive mount. Two options.

**Option A, a private Kaggle Dataset (recommended).** From the local machine,
make a clean archive of the working tree and upload it as a private dataset via
the Kaggle UI (`Datasets -> New Dataset -> Upload`) or the CLI:

Exclude the heavy artefacts by name rather than by extension, so the one `.pt`
that must travel, the frozen classifier, is kept:

```bash
cd b:/paper2_recon
tar --exclude='.git' --exclude='__pycache__' \
    --exclude='data/svhn' --exclude='data/mnist' \
    --exclude='ckpt_*.pt' \
    --exclude='x_hat.npy' --exclude='y.npy' --exclude='mask.npy' \
    --exclude='*.npz' \
    -czf /tmp/paper2_recon_src.tar.gz .
```

The solver checkpoints are all named `ckpt_csfr*.pt`, and the frozen classifier
is `results/downstream/classifier.pt`, so this drops the gigabytes and keeps the
2.4 mb that matters. Do not use a blanket `--exclude='*.pt'`: it removes the
classifier, and you cannot append it afterwards because `tar -r` does not work on
a compressed archive.

Check the result before uploading:

```bash
ls -lh /tmp/paper2_recon_src.tar.gz                      # expect tens of mb
tar -tzf /tmp/paper2_recon_src.tar.gz | grep classifier  # must be present
```

In the notebook, the dataset mounts read-only at a nested path:
`/kaggle/input/datasets/<username>/<dataset-slug>/`. Copy it into writable space:

```bash
mkdir -p /kaggle/working/paper2_recon
tar -xzf /kaggle/input/datasets/<username>/<dataset-slug>/paper2_recon_src.tar.gz \
    -C /kaggle/working/paper2_recon
cd /kaggle/working/paper2_recon
ls scripts/ configs/
```

If you uploaded a `.tar.gz` file, Kaggle automatically extracts it in the input
directory, so the archive path is correct as written. Check the exact slug in
the Datasets sidebar if the path is wrong.

**Option B, clone from a remote.** If the branch is pushed somewhere reachable:

```bash
cd /kaggle/working
git clone <your-remote> paper2_recon
cd paper2_recon && git checkout feat/downstream-and-final
```

Whichever route, what must come across: `src/`, `scripts/`, `configs/`,
`data/patches.npy`, `results/downstream/classifier.pt`, and
`results/csfr_sweep_2d_v2/summary.csv` for the consistency check. Everything else
regenerates.

Work in `/kaggle/working/paper2_recon`. Never write into `/kaggle/input`, which
is read-only.

### 2.2 Keep the model cache out of the output directory

`/kaggle/working` is capped at 20 gb and everything in it is saved as the
notebook's output. The Stable Diffusion weights are 3.3 gb and must not go there.
Send the Hugging Face cache to `/tmp`, which is scratch:

```python
import os
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

Set this **before** importing `diffusers` or `huggingface_hub`. The cost is that
the weights are re-downloaded in a new session, which on Kaggle's connection is a
few minutes rather than the 44 it took locally.

### 2.3 Dependencies

**Do not run `pip install -r requirements.txt`.** It pins `torch==2.13.0`, the
CPU build used locally, and installing it will replace Kaggle's CUDA torch and
leave you on the CPU with no warning.

**Measured on the Kaggle GPU image, 2026-08: torch, torchvision, cv2, skimage,
sklearn, yaml, psutil, scipy, diffusers, transformers and accelerate are all
pre-installed.** No packages need installing for the CSFR solver or the
diffusion baseline. Only if you add the LaMa backend:

```bash
pip install simple-lama-inpainting
```

**Important: LaMa downgrades critical dependencies.** After installing
`simple-lama-inpainting`, numpy is downgraded to 1.26 (from 2.x) and pillow to
9.5 (from 10.x). You must restore both in a fresh cell before any imports:

```python
pip install --upgrade "numpy>=2" "pillow>=10.1"
```

This must be in a separate cell and run before importing anything, because
numpy's C extension cannot be reloaded in a live process. Run this only once
per session.

The torch version on the GPU (2.10.0+cu128) differs from the local CPU build
(2.13.0). That is expected and is exactly why GPU results are reported as their
own runs rather than spliced into the CPU tables. Record both versions in the
run metadata.

Two things seen locally that are expected rather than errors: the weights
repository has no `.safetensors` for the VAE or UNet, so diffusers falls back to
pickle deserialisation and says so; and `transformers` 5.x prints a
`CLIPImageProcessor requires torchvision` style notice on some paths.

### 2.4 Data

`data/patches.npy` (the 300 DFRWS 64x64 patches) travels with the repo. SVHN is
not committed, by policy, and is re-fetched:

```python
import sys; sys.path.insert(0, "src")
from downstream import load_digit_dataset
imgs, labs, corpus = load_digit_dataset(prefer="svhn", split="test")
print(corpus, imgs.shape, labs.shape)
```

That pulls `test_32x32.mat` (64 mb) into `data/svhn/`. Add `split="train"` only
if you intend to retrain the classifier, which you should not: the frozen
`classifier.pt` travels precisely so the classifier is identical to the one
behind the released results. Retraining on a different device would silently
change the ceiling and invalidate every downstream number already computed.

Sanity check that the classifier reproduces its ceiling:

```python
from downstream import load_classifier, predict
m = load_classifier("results/downstream/classifier.pt")
p, c = predict(m, imgs if imgs.ndim == 4 else imgs.unsqueeze(1))
print((p.numpy() == labs.numpy()).mean())   # expect 0.8783
```

If that is not 0.8783, stop and work out why before running anything else.

---

## 3. Job A: the external-prior baseline

### 3.1 Prove it runs before running a grid

This is the step that was skipped locally. Do not launch a grid until a handful
of patches come back with a sane PSNR, and do not launch a long one until you
have timed a short one.

First, the opt-in model tests. They are skipped by default on purpose: the
weights are several gb and `verify.sh` runs the suite, so a download in the
default path would make the project's verification gate pull gigabytes on any
clean machine. On the GPU box, once, run them explicitly:

```bash
CSFR_RUN_INPAINTER_TESTS=1 python -m pytest tests/test_inpainter.py -q
```

That covers the interface contract, that observed pixels come back bit-exact,
and that a fixed seed reproduces. Then the trial:

```bash
python scripts/trial_lama.py     # inspect and adapt; point output at /tmp
```

What to confirm and write down: the resolved model id and revision, **per-patch
wall-clock on this card**, the measured PSNR against ground truth on a few
patches, that output is in [0, 255] with the right shape, that observed pixels
are preserved bit-exact, and that two runs with the same seed give identical
output.

Measured throughput on a T4: approximately 35 ms per patch, or about 20 seconds
per 1000-digit cell. This scales to roughly 6-7 minutes per corruption family
(all 5 levels) and about 25-30 minutes for the full grid (CF1-CF4). Multiply
your trial time by 20,000 for the full count. If the answer exceeds about 10
hours, take one of the routes in 0.1.

If PSNR is far below the classical baselines, look at the resize path before
concluding anything about the method.

### 3.2 Add it to the downstream grid

The harness runs one method at a time, so the existing six methods do not need
recomputing:

```bash
python scripts/run_downstream.py \
  --config configs/downstream.yaml \
  --n-test 1000 \
  --only-method lama
```

`results/downstream/<CELL>/lama.csv` is deliberately never treated as cached, so
this always recomputes.

**If diffusion is the only option, run the severe-cell subset**, where the
hallucination argument actually bites:

```bash
for LVL in L4 L5; do
  for FAM in CF2 CF3 CF4; do
    python scripts/run_downstream.py --config configs/downstream.yaml \
      --n-test 1000 --only-method lama --only-family $FAM --only-level $LVL
  done
done
```

Six cells, about 4 hours on one T4. Each cell writes its own CSV as it finishes,
so a session that dies partway still leaves usable completed cells.

### 3.3 Using both T4s

`T4 x2` gives two devices, and the loop above splits cleanly by family. Pin each
half to a card and run them concurrently:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash -c 'for LVL in L4 L5; do
  python scripts/run_downstream.py --config configs/downstream.yaml \
    --n-test 1000 --only-method lama --only-family CF2 --only-level $LVL; done' \
  > /tmp/lama_gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash -c 'for LVL in L4 L5; do
  for FAM in CF3 CF4; do
  python scripts/run_downstream.py --config configs/downstream.yaml \
    --n-test 1000 --only-method lama --only-family $FAM --only-level $LVL; done; done' \
  > /tmp/lama_gpu1.log 2>&1 &
```

Each process sees exactly one GPU, so `pick_device` resolves to `cuda:0` inside
both and they do not contend. They write to different cells, so there is no file
conflict. Watch both logs.

This is worth doing only for the diffusion path. With LaMa the whole grid fits
comfortably on one card and the added complexity buys nothing.

### 3.4 Add it to the sweep and the frontier

```bash
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml
python scripts/render_frontier.py
```

The sweep skips cells whose metrics already exist, so this fills in `lama` only.
The frontier figure places it at traceability score 0, which is by construction
and not a measurement; that wording matters and is already in the code comments.

### 3.5 Then regenerate the analysis

```bash
python scripts/run_downstream_stats.py
python scripts/render_stats_tables.py
```

`run_downstream_stats.py` asserts the grid is complete before emitting anything.
If you ran a subset in 3.2, that assertion will fire. Extend it to accept a
declared subset rather than disabling it: an assertion that is switched off when
inconvenient is worse than no assertion.

---

## 4. Job B: matched-compute sweep at T=2400

Already configured in `configs/csfr_sweep_2d_t2400.yaml`: CF2/CF3/CF4, all five
levels, 300 patches, 3 seeds, `variants: [csfr]`, `baselines: []`,
`max_iter: 2400`. The primary audited result stays at T=600; this is the
secondary result showing how much of the deficit is under-convergence.

`baselines: []` matters and is new. `variants` only selects the CSFR ablation
arms; the baseline list used to be unconditional, so this config ran all five
classical baselines **and the diffusion inpainter** in every cell. A local run
hit exactly that: it spent 44 minutes downloading 3.26 gb of weights and then
started diffusing 300 patches for a number this run has no use for, since no
baseline depends on the solver budget. The harness now takes a `baselines:` key,
defaulting to all of them when the key is absent.

Because no baseline runs here, the free consistency check against the primary
sweep is no longer automatic. If you want it, run one cell with
`baselines: [zero_fill, bilinear]` and confirm those match
`results/csfr_sweep_2d_v2/summary.csv`. If they do not, something about the
corpus or the masks differs and the whole run is suspect.

### 4.1 Start and resume: the same command

```bash
python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d_t2400.yaml
```

There is no separate resume flag. The harness writes `metrics.json` per cell and
skips any cell that already has one, so re-issuing the identical command picks up
where it stopped. `--force` is the opposite: it recomputes everything. Do not
pass `--force` when resuming.

`device: auto` resolves to CUDA on Kaggle with no edit. The run was interrupted
locally partway through CF2_L1, which will simply recompute.

Run it detached so an idle disconnect does not kill it:

```bash
nohup python scripts/run_csfr_sweep.py \
  --config configs/csfr_sweep_2d_t2400.yaml > /tmp/t2400.log 2>&1 &
echo $! > /tmp/t2400.pid
```

Progress:

```bash
grep -E "^\[sweep2d\]" /tmp/t2400.log | tail -5     # cell x of 15
ls results/csfr_sweep_2d_t2400/                      # completed cells
```

If you are running Job A on the other card, give this one the free device with
`CUDA_VISIBLE_DEVICES=1`.

### 4.2 Halt

```bash
kill $(cat /tmp/t2400.pid)
```

Or from a notebook cell, `!pkill -f csfr_sweep_2d_t2400`. Halting costs only the
cell in flight; everything already written stays valid, because a cell's
`metrics.json` is written after that cell completes. Resume with the plain
command in 4.1.

To confirm it actually stopped, `!pgrep -af run_csfr_sweep` should print nothing.
Worth checking: locally a run once appeared to have exited while the process was
still alive and writing, and a second run was launched over the top of it.

Expect 20-60 minutes on a T4 against hours on the local CPU.

---

## 5. Getting results off Kaggle and back into this repo

### 5.1 What comes back

The reproducibility policy is lean-set only, and `.gitignore` already enforces
most of it. Never bring back `*.pt` solver checkpoints, `x_hat`/`y`/`mask`
`.npy`, `*.npz`, or model weights. These regenerate from `patches.npy` plus the
config. The Hugging Face cache lives in `/tmp/hf` per 2.2, outside the output
directory, which is where it should stay.

Bring back:

- `results/downstream/<CELL>/lama.csv` (one per cell run)
- `results/downstream/{metrics.csv,stats.csv,summary.csv,environment.json}`
- `results/csfr_sweep_2d_t2400/**/metrics.json` plus its summary CSV
- `results/tables_tex/*.tex`
- `paper/figures/paper2_F9_method_frontier.png` if the frontier was redrawn
- the run logs, so the wall-clock and any warnings are recoverable

### 5.2 Package it into the notebook output

Anything left in `/kaggle/working` at the end of the session becomes the
notebook's downloadable output. Build one small archive rather than leaving the
whole tree, so the 20 gb cap is never in play and nothing heavy escapes:

```bash
cd /kaggle/working/paper2_recon
cp /tmp/t2400.log /tmp/lama_gpu*.log . 2>/dev/null || true
tar --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
    -czf /kaggle/working/results_gpu.tar.gz \
    results/downstream results/csfr_sweep_2d_t2400 results/tables_tex \
    paper/figures/paper2_F9_method_frontier.png *.log
ls -lh /kaggle/working/results_gpu.tar.gz
```

The exclusions are belt and braces: `.gitignore` stops these being committed, but
this stops them being carried at all. Expect a few megabytes. If the archive is
large, something heavy slipped in and should be found before downloading.

Then either download `results_gpu.tar.gz` from the notebook's **Output** tab, or
pull it with the CLI from the local machine:

```bash
kaggle kernels output <your-username>/<notebook-slug> -p /tmp/kaggle_out
ls -lh /tmp/kaggle_out
```

If the notebook was run with `Save & Run All (Commit)`, the output is attached to
that committed version and is downloadable without keeping a session alive.

### 5.3 Putting it back into this repo

From the repo root on the local machine:

```bash
tar -tzf /tmp/kaggle_out/results_gpu.tar.gz | head -30   # look before unpacking
tar -xzf /tmp/kaggle_out/results_gpu.tar.gz              # paths are repo-relative
git status --short
```

The tar paths are already repo-relative, so it unpacks into place. Inspect the
listing first: unpacking over `results/downstream/` overwrites the per-cell CSVs,
and those are the released per-patch record.

Then:

```bash
python scripts/run_downstream_stats.py     # regenerate with lama included
python scripts/render_stats_tables.py
python -m pytest tests/ -q                 # expect 54 passed, 2 skipped
bash scripts/verify.sh                     # expect PASS
```

Check before committing: `git status` should show only CSVs, `.tex` table
bodies, `environment.json`, and possibly the frontier PNG. If it shows a `.pt`,
`.npy` or `.npz`, stop and work out how it got past the exclusions.

One caution about mixing devices. The T=2400 sweep and the inpainter are new runs
reported separately, so computing them on CUDA is fine. Do not regenerate any
existing T=600 primary number on the GPU and splice it into the current tables:
floating-point behaviour differs between devices, and a table mixing CPU and GPU
numbers would be indefensible.

---

## 6. Verify before calling it done

Back on the local machine:

```bash
python -m pytest tests/ -q          # expect 54 passed, 2 skipped
bash scripts/verify.sh              # expect PASS, 0 undefined refs, 0 latex errors
```

`verify.sh` will report `marker DOWNSTREAM absent; wrote file only` for the two
downstream table blocks. That is expected until Phase 5 inserts the section into
the manuscript.

The two skips are the MNIST tests, which skip when the cache is absent and the
network is unreachable. On a machine with `data/mnist/` populated they run.

Then check the numbers actually changed in the direction claimed, and that no
released figure still refers to a method that was not run.

---

## 7. Practical Kaggle notes

- **Sessions end at 12 hours, and idle interactive sessions end far sooner.** Both
  harnesses are resumable at cell granularity, so prefer many short invocations
  over one long one, and check `results/` between them. For anything over an hour
  use `Save & Run All (Commit)` rather than an interactive session.
- **Save output before the session ends.** Unlike a Drive mount, nothing persists
  automatically. A session that dies with results only in `/tmp` loses them. Write
  results under `/kaggle/working` and archive early rather than at the very end.
- **Watch the weekly quota.** 30 GPU-hours goes quickly if a long diffusion run is
  restarted a few times. Time a small run first (3.1) and extrapolate.
- The GPU changes floating-point behaviour. CSFR results computed on CUDA will
  not be bit-identical to the CPU numbers already in the paper. That is fine for
  the T=2400 secondary result, which is its own run reported separately, but do
  not regenerate any T=600 primary number on the GPU and splice it into the
  existing tables. Mixed-device numbers in one table would be indefensible.
- Record `torch.__version__`, the CUDA version and the GPU name in the run
  metadata. The paper states the hardware for the runtime benchmark, and the
  external-prior method needs its provenance recorded to be auditable at all.
  Note the card actually used: a T4 and a P100 are different machines and the
  runtime table must say which.
- The frozen classifier must stay on CPU-trained weights as committed. Load it,
  do not retrain it.
