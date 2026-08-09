# Data

## Committed patch corpora

| File | Content | Source images | Largest cluster |
|---|---|---|---|
| `patches.npy` | 300 grayscale 64x64 patches, uint8 | 6 (DFRWS 2006) | 41.3% |
| `patches_govdocs1.npy` | 300 grayscale 64x64 patches, uint8 | 176 (GovDocs1) | 1.7% |
| `patches_mikus_11-carve-fat.npy` | 20 colour 64x64x3 patches | 3 (DFTT #11, FAT32) | 80.0% |
| `patches_mikus_12-carve-ext2.npy` | 26 colour 64x64x3 patches | 3 (DFTT #12, Ext2) | 61.5% |

Each corpus ships a `*_cluster.npy` (per-patch source-image label) and a
`*_manifest.json` recording the extraction protocol, per-image patch counts,
patch coordinates, and the SHA-256 of every carved byte range.

**Patches from one source image are not independent.** Use the cluster labels
for any inference over these corpora; `scripts/run_paired_stats.py` does this
automatically when a `*_cluster.npy` is present.

## Rebuilding the corpora

`scripts/extract_patches.py` implements the sampling protocol (signature carve
-> decode -> min-side filter -> non-overlapping dice -> variance filter ->
per-image cap -> seeded sample). Examples:

```bash
# external validation corpus
python scripts/extract_patches.py --source-type zip     --input <GovDocs1>/zipfiles --name govdocs1 --n-target 300

# naturally corrupted, full-resolution, colour case studies
python scripts/extract_patches.py --source-type diskimage     --input <DFTT>/11-carve-fat/11-carve-fat.dd --name mikus_11-carve-fat     --colour --min-side 64 --keep-full-images data/full_mikus_11-carve-fat
```

`patches.npy` (DFRWS 2006) was produced by an earlier revision of this protocol
and is shipped as-is so that the published numbers refer to the exact array
that generated them; `data/patches_manifest.json` documents its composition and
records the array's SHA-256.

## External data (download separately)

| Source | URL | Used for |
|--------|-----|----------|
| DFRWS 2006 challenge image | https://www.dfrws.org/2006-challenge/ | primary sweep corpus |
| GovDocs1 | https://digitalcorpora.org/corpora/files/ | external validation corpus |
| DFTT carving images #11, #12 (Nick Mikus) | http://dftt.sourceforge.net/ | full-resolution colour cases |
| FFT-75 | https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset | protocol extension (specified, not executed) |
| SVHN (`train_32x32.mat`, `test_32x32.mat`) | http://ufldl.stanford.edu/housenumbers/ | downstream digit CNN (preferred) |
| MNIST IDX gz | https://ossci-datasets.s3.amazonaws.com/mnist/ | downstream digit CNN (fallback) |

## Downstream digit corpus

The CSFR downstream utility experiment needs labelled digits for classifying
reconstructed image patches. Prefer **SVHN**: 32×32 RGB street-view house
numbers from Stanford, automatically converted to grayscale luminance by
`src/downstream.py`. The raw SVHN distribution encodes digit zero as label `10`;
the loader maps this to `0` for consistency with other datasets.

SVHN is **automatically fetched on first use** when running `scripts/run_downstream.py` or
the test suite. The downloader calls `download_svhn()` to fetch `train_32x32.mat`
and `test_32x32.mat` (64 mb each) from http://ufldl.stanford.edu/housenumbers/
into `data/svhn/`. The `data/svhn/` directory is gitignored; re-fetching on a
fresh clone is the intended path.

You can manually fetch SVHN without running experiments:
```bash
python -c "from src.downstream import download_svhn; download_svhn()"
```

**note:** `download_svhn()` fetches both train and test regardless of the split
requested. See `src/downstream.py` for the implementation.

**evaluation subset:** The downstream utility harness uses a class-balanced
1000-image evaluation subset selected from the SVHN test set, using seed=0 for
reproducibility. This subset is used consistently across all downstream metrics
in the paper; 100 images are drawn per digit class (0–9) from the test set,
preserving class balance.

**frozen classifier weights:** The downstream evaluation uses a small DigitCNN
trained on clean SVHN, frozen and committed at `results/downstream/classifier.pt`
(2.4 mb). Retraining on a different machine (especially a GPU) would silently
change the accuracy ceiling and invalidate all downstream comparisons. The
classifier is therefore an **exception to the no-checkpoints policy**: it is
deliberately committed so that every downstream accuracy and ECE number in the
paper refers to the same bit-identical classifier.

Clean top-1 accuracy (reported once in the README, not recalculated):
- Full SVHN test set (10,000 images): 87.83%
- Class-balanced evaluation subset (1,000 images, seed=0): 86.90%

The evaluation subset is used consistently for all downstream metrics in the
paper.

**MNIST fallback.** If the SVHN host is unreachable, `load_digit_dataset(prefer="svhn")`
falls back to MNIST (grayscale, 28×28, resized to 32×32 via nearest-neighbour).
MNIST is a weaker proxy for street-view digits: cleaner glyphs, centred, no
outdoor lighting variations. Always check the returned corpus name when
interpreting downstream accuracy and ECE numbers.
