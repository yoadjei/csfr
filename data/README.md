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
