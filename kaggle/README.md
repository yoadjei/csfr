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
