# running the gpu work on kaggle

only the external-prior baseline needs a gpu. everything else runs on cpu.

## session setup

accelerator **gpu t4 x2**, internet **on** (needs a phone-verified account),
persistence **files only**.

nothing needs installing for the diffusion path. measured on the kaggle gpu
image, august 2026:

```
python 3.12.13   torch 2.10.0+cu128   cuda 12.8   2x tesla t4
torchvision 0.25.0   cv2 4.13.0   skimage 0.25.2   sklearn 1.6.1
yaml 6.0.3   psutil 5.9.5   scipy 1.16.3
diffusers 0.37.1   transformers 5.0.0   accelerate 1.13.0
```

do not run `pip install -r requirements.txt`. it pins the cpu torch build and
will replace kaggle's cuda torch with no warning.

## lama, and the numpy trap

lama is the preferred backend, roughly two orders of magnitude faster than
diffusion. installing it downgrades numpy to 1.26 and pillow to 9.5, which
breaks cv2. restore them, and do it in a separate cell before anything imports
numpy:

```python
!pip install -q simple-lama-inpainting
!pip install -q --upgrade "numpy>=2" "pillow>=10.1"
```

then, in a second cell, import. numpy's c extension cannot be reloaded in a live
process, so installing after importing gives `cannot load module more than once
per process` and needs a session restart.

## getting the repo in

there is no drive mount. upload the working tree as a private kaggle dataset,
excluding the heavy artefacts by name so the frozen classifier survives:

```bash
tar --exclude='.git' --exclude='__pycache__' \
    --exclude='data/svhn' --exclude='data/mnist' \
    --exclude='ckpt_*.pt' \
    --exclude='x_hat.npy' --exclude='y.npy' --exclude='mask.npy' \
    --exclude='*.npz' \
    -czf /tmp/paper2_recon_src.tar.gz .
```

about 22 mb. do not use a blanket `--exclude='*.pt'`: it drops
`results/downstream/classifier.pt`, and you cannot append to a compressed
archive afterwards.

kaggle auto-extracts the tarball and mounts it at a nested path, typically
`/kaggle/input/datasets/<user>/<slug>/`. copy it into writable space:

```python
!mkdir -p /kaggle/working/paper2_recon
!cp -r /kaggle/input/datasets/<user>/<slug>/. /kaggle/working/paper2_recon/
%cd /kaggle/working/paper2_recon
```

keep model caches out of `/kaggle/working`, which is capped at 20 gb and is
saved as the notebook output:

```python
import os
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["TORCH_HOME"] = "/tmp/torch"
```

## check before spending gpu time

```python
from downstream import load_classifier, predict, load_digit_dataset
imgs, labs, _ = load_digit_dataset(prefer="svhn", split="test")
m = load_classifier("results/downstream/classifier.pt")
p, c = predict(m, imgs if imgs.ndim == 4 else imgs.unsqueeze(1))
print((p.numpy() == labs.numpy()).mean())    # must be 0.8783
```

if it is not 0.8783, the classifier that travelled is not the one behind the
released numbers. stop.

then a smoke test on 8 patches. lama at native resolution scores about 36.6 db
on cf1 at 30% loss, against telea 34.6 and bilinear 32.8. if it scores in the
teens, `LAMA_WORK_SIZE` is not `None` and the resize round trip is costing 8 to
22 db.

## the run

```python
!python scripts/run_downstream.py --config configs/downstream.yaml \
    --n-test 1000 --only-method lama --only-family CF1
```

one family at a time, so each writes its five cell csvs before the next starts.
about 20 seconds per cell at 35 ms per patch, roughly 7 minutes for all four
families.

for the sweep, the same command with the primary config fills in `lama` only,
since every other method is already cached:

```python
!python scripts/run_csfr_sweep.py --config configs/csfr_sweep_2d.yaml
```

do not split across both gpus. `run_pipeline` writes `summary.csv` and
`clean.csv` at the top level on every invocation, so two concurrent processes
race on the same two files.

## bringing results back

```bash
tar --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
    -czf /kaggle/working/results_gpu.tar.gz \
    results/downstream results/csfr_sweep_2d_v2 results/tables_tex
```

a few megabytes. if it is large, something heavy slipped in. download from the
notebook output panel, then unpack at the repo root, since the paths are
repo-relative. inspect the listing first: unpacking over `results/downstream/`
overwrites the released per-patch record.

then locally:

```bash
python scripts/run_downstream_stats.py
python scripts/render_stats_tables.py
bash scripts/verify.sh
```

## limits worth knowing

- 30 gpu-hours per week, 12 hours per session
- interactive sessions stop when idle. use save and run all for anything long
- gpu results are not bit-identical to cpu results. the lama runs are reported
  as their own runs and no table row in the paper mixes devices. do not
  regenerate an existing cpu number on the gpu and splice it into a cpu table

## round 2: the downstream stage (e12, e13)

`kaggle/downstream_v2_kernel.py` runs the workstation's downstream stage,
`bash scripts/run_round2_campaign.sh downstream`, in a kaggle gpu session with
`DOWNSTREAM_DEVICE=cuda OCR_GPU=1`: the resnet training, lama and easyocr use the
gpu, every other step runs as configured (classifiers and reconstructions on the
cpu). it builds its own environments, so the warning above about
`requirements.txt` does not apply: the main one is `requirements.txt` exactly, in
an isolated venv that cannot see the image's torch or diffusers. it stops if any
environment cannot run a cuda convolution, if the round-1 cnn does not score
0.8783 on svhn test, if the evaluation or calibration subset hashes differ from
the workstation's (`EXPECTED`), or if a govdocs1 jpeg differs from its manifest.

one-time setup: a phone-verified account (gpu and internet), a token from
kaggle > settings > api > create new token saved as `~/.kaggle/kaggle.json`, and
the cli in its own venv (`python -m venv .venv-kaggle`, then
`.venv-kaggle/Scripts/pip install kaggle six`: kaggle 2.2.5 imports six without
declaring it).

```bash
K=.venv-kaggle/Scripts/kaggle; U=<kaggle user>; D=<scratch dir>
python kaggle/downstream_v2_kernel.py stage $D/ds --user $U     # refuses uncommitted changes
$K datasets create -p $D/ds                                    # private; later: datasets version -p $D/ds -m <msg>
python kaggle/downstream_v2_kernel.py kernel-dir $D/k1 --user $U --session 1
$K kernels push -p $D/k1 --accelerator NvidiaTeslaT4
$K kernels status $U/csfr-downstream-v2-s1
$K kernels output $U/csfr-downstream-v2-s1 -p $D/out1 \
    --file-pattern '^(STATUS|kaggle_env\.json|stage\.log|results\.tar\.gz)$'
```

the file pattern leaves `state/` (up to ~14 gb of arrays) on kaggle, where the
next session reads it.

a session stops the stage at its time budget (12 h less 45 min for saving) and,
until the stage is complete, saves `state/` with the arrays; session n+1
(`kernel-dir ... --session 2`) takes session n's output as an input and
continues, since every step resumes. `STATUS` says complete, incomplete or
failed with the reason; `kaggle_env.json` records the gpu, the cpu count and
`pip freeze` of each environment.

bringing it back: `results.tar.gz` holds `results/downstream_v2`, `results/ocr`
and `data/ocr` without arrays or the training checkpoint. move the
workstation's own `results/downstream_v2` aside first (it holds the partial cpu
training), inspect the listing, unpack at the repo root, then
`python kaggle/downstream_v2_kernel.py check` must print `complete`. the
workstation's campaign must then run without its downstream stage
(`-Stages "forensic idle"`), or it would redo lama on the cpu over the gpu
outputs.
