# data

## committed patch corpora

| file | content | source images | largest cluster |
|---|---|---|---|
| `patches.npy` | 300 grayscale 64x64 patches, uint8 | 6 (dfrws 2006) | 41.3% |
| `patches_govdocs1.npy` | 300 grayscale 64x64 patches, uint8 | 176 (govdocs1) | 1.7% |
| `patches_mikus_11-carve-fat.npy` | 20 colour 64x64x3 patches | 3 (dftt #11, fat32) | 80.0% |
| `patches_mikus_12-carve-ext2.npy` | 26 colour 64x64x3 patches | 3 (dftt #12, ext2) | 61.5% |

each corpus ships a `*_cluster.npy` (per-patch source-image label) and a
`*_manifest.json` recording the extraction protocol, per-image patch counts,
patch coordinates, and the sha-256 of every carved byte range.

the dfrws corpus is concentrated: two images supply 77% of its patches, so
cluster-level inference is not possible on it. govdocs1 carries the
generalisation claim for that reason. rebuild either with
`scripts/extract_patches.py`.

## downstream corpus

svhn, street-view house numbers, 32x32 rgb converted to grayscale to match the
paper's evaluation. not committed. it is fetched automatically on first use:

```python
from downstream import load_digit_dataset
imgs, labels, corpus = load_digit_dataset(prefer="svhn", split="test")
```

`download_svhn()` fetches both train and test regardless of the split asked for,
about 246 mb. mnist is a documented fallback if svhn is unreachable, and the
paper states the weaker proxy if that path is used.

the experiment uses a class-balanced subset of 1000 digits, seed 0, drawn by
`select_class_balanced` in `scripts/run_downstream.py`. its clean accuracy is
0.8690, which is the ceiling accuracy recovery is computed against. the full
test set gives 0.8783.

## frozen classifier

`results/downstream/classifier.pt` (2.4 mb) is committed deliberately, as the
one exception to the no-checkpoints policy. retraining it, on a different device
or a different torch build, would move the ceiling and silently invalidate every
downstream number already released. load it, do not retrain it.

## external corpora, not committed

- govdocs1 zipfiles
- dfrws 2006 challenge images
- nick mikus dftt carving images (#11 fat32, #12 ext2)

paths are recorded in each manifest. the committed `.npy` corpora are the
released artefacts; the raw sources are third-party downloads.
