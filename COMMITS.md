# commits

the nine commits on `feat/csfr-release`, in order. messages reproduced
verbatim so they can be reapplied without rewriting.

## 1. `87898d3`

```
feat(downstream): svhn corpus, frozen classifier and the utility harness

adds the downstream utility experiment's data and execution layer. reconstruction
quality alone does not establish that a reconstruction is useful, so the paper
needs an end-to-end task.

src/downstream.py loads svhn (mnist as a documented fallback), converts to
grayscale to match the paper's evaluation, and trains a small cnn on clean
digits which is then frozen. the checkpoint is committed deliberately, as the
one exception to the no-checkpoints policy, so the classifier behind every
released number is bit-identical for anyone re-running.

scripts/run_downstream.py draws masks with the paper's own make_mask over a
seeded class-balanced subset, reconstructs with each method, and classifies.
one mask draw per image over 1000 independent digits keeps the analysis unit
the image, avoiding the nested sampling that would arise from reusing images
across seeds.

per-patch rows record prediction, confidence, correctness, the bounded-unknown
flag and its continuous statistic, and two mask geometry columns. clean
predictions are written before the cell loop so an interrupted run cannot leave
them describing a different subset.
```

## 2. `50db33c`

```
feat(baseline): run the generative inpainter the paper argues against

the paper contends that sampling from a learned prior forfeits the error
characterisation and per-pixel derivation evidential use requires. it previously
made that argument without running such a method, which is the weakest position
to argue from.

src/generative_inpainter.py wraps two external-prior backends behind one
interface, lama preferred and stable diffusion as fallback, and both harnesses
gain it as method "lama". the backend that actually ran is recorded in the
environment capture: two different models with different failure modes must not
be indistinguishable in the released record.

observed pixels are composited back bit-exact. without that the resize round
trip would also resample observed data, and the resulting psnr loss would have
nothing to do with what the prior invented, which is the only thing being
compared.

inference runs at the patch's native resolution. an earlier version upscaled to
256, which measured 8 to 22 db worse: 14.50 db against 36.57 at cf1 0.30, where
telea scores 34.55. that would have crippled the opponent and flattered csfr,
and nothing in the code or tests would have flagged it, so a regression test now
asserts reconstruction quality clears a threshold a resize round trip fails.

run_csfr_sweep.py gains a baselines: config key, since no baseline depends on
the solver budget and re-running the inpainter inside a budget sweep costs hours
for a number that run has no use for.
```

## 3. `f105ad4`

```
feat(downstream): metrics and inference for the utility experiment

computes per-cell accuracy recovery, ece, the confident-error pair and the
abstention statistics, then tests them: mcnemar with holm correction on
accuracy, and seeded bootstrap intervals on confident-error differences over
the signal-covering subset.

the confident-error rate is reported two ways because the joint rate
P(wrong and confident) alone rewards a method that is never confident: zero-fill
scores 0.000 on cf1 purely because its accuracy is 0.115. the conditional rate
P(wrong | confident) controls for that, and both are meaningless without the
count of confident predictions beside them.

the columns name their own metric, since this file carries three families of
statistic and citing one as evidence for another is a defect. the mcnemar sign
convention is documented with a worked example verified against the released
data rather than restated from memory, because it had been documented backwards.

conditional intervals exclude resamples in which a method has no confident
predictions, and the count of valid resamples is reported: at cf1 l3 against
zero-fill that rule fires on 27 per cent of resamples, and an interval computed
from a heavily filtered set is not the interval it appears to be.

optional methods are discovered from the results rather than hardcoded, so a
baseline present in every cell cannot be silently dropped from the tables.
```

## 4. `17f00fb`

```
data(results): release the downstream grid and the sweep with the external prior

results/downstream/: 20 cells by 7 methods by 1000 patches, one seed, plus the
clean-image predictions the accuracy-recovery baseline is computed against, the
metrics and statistics tables, and the environment capture.

results/csfr_sweep_2d_v2/: the external prior added across all 20 cells and 3
seeds. it achieves the best psnr on 13 of them, beating csfr by up to 9.08 db on
structured corruption, so the paper now compares against the strongest method
that was actually run rather than the strongest classical one.

the lama results were computed on an nvidia t4 while every other method is
cpu-computed. these are separate runs reported separately and no table row mixes
devices, which the paper states rather than leaving to be discovered.

heavy artefacts stay out by policy: solver checkpoints, per-seed x_hat, y and
mask arrays, and the raw corpora, all of which regenerate from the committed
patch sets and configs.
```

## 5. `cf8a41c`

```
docs(figures): measured provenance walkthrough, corrected frontier

paper2_F12_provenance.png makes the auditability claim inspectable. two queries
in one band-corrupted patch: one pixel from the observed boundary the attribution
concentrates, top 5 contributors carrying 15.9 per cent; 23 pixels deep it
disperses across the whole observed band, top 5 carrying 2.8 per cent with no
observed 4-neighbours. provenance concentrates where evidence is adjacent and
spreads where it is not, and the record says which regime a pixel is in.
regenerated arrays were checked against the released manifest's mask and
observed-pixel hashes.

the frontier figure carried three errors. it labelled the method "diffusion"
when lama is fourier-convolutional. its shaded band was called the TS=0 region
occupied by generative methods and declared them inadmissible: six of the seven
methods have TS=0 and five plot above that band, the only point inside it was
zero-fill, and asserting inadmissibility contradicts this paper's own position
that admissibility is for a court. the band now marks the 2/3 ceiling, which is
the real structural claim, since a method with TS=0 cannot exceed two thirds of
the composite however good its reconstruction.

both figures used green with orange beside the red used for the external prior,
the deuteranopia failure triplet, and those were the three clustered markers.
both now use the okabe-ito palette with marker shape carrying the distinction
independently, so they survive greyscale printing.

the F6 generator is deleted rather than kept: it drew alpha_1=0.42, alpha_2=0.31
and alpha_3=0.27 as attribution weights, hardcoded rather than computed,
illustrating the one thing this paper claims to measure.
```

## 6. `702bd51`

```
refactor(paper): condense to 26 pages, fold defensive tables into a supplement

74 review pages to a 26-page journal final plus a 3-page supplement, with no
result, table or figure lost that carried a finding.

claims corrected against the artefacts. the research question asked whether csfr
reconstructs at a quality competitive with classical inpainting; the results
refute that, so it now asks what a downstream analysis gains and gives up. the
downstream experiment, the paper's strongest evidence, was missing from the
contributions list. related work claimed generative methods were "cited rather
than run" while citing the one that was run, and the same stale excuse appeared
in the limitations. the psnr discussion called lama a diffusion model.

numbers corrected: the external prior's largest margin is 9.08 db, reported as
8.4; the cf1 l1 standard deviations were wrong by up to 0.5; the cf2 margin was
given as a flat 6 to 7 db when it runs 1.81 at l1 to 7.01 at l5; and the
downstream ceiling quoted the full test set while accuracy recovery is computed
against the evaluation subset.

a gap is now disclosed rather than left implicit: the external prior appears in
no paired test, because its per-patch reconstructions were not persisted, so its
margins are point estimates and the paper says so.

limitations went from twelve items to five. the quality deficit and the solver
cost moved to results, where they are findings rather than concessions.

ablation, runtime, budget sensitivity and abstention moved to the supplement.
external validation stays in the body because it carries the generalisation
claim. both AUTO generators now splice into either document and treat a marker
found in neither as fatal, so a relocated table cannot silently freeze while its
csv moves on.
```

## 7. `479706b`

```
test(paper): lock every hand-written manuscript number to its source csv

tables are AUTO-generated and cannot drift. prose numbers are typed by hand and
had drifted in four places, each of which survived a numerical audit that read
captions and checked they were internally coherent rather than opening the csv.

nine tests recompute each claim from the released artefacts. a test fails when
the claim goes missing from the manuscript as well as when it goes wrong, so
rewording a sentence forces its number to be re-verified rather than quietly
dropping out of the check.

the deficit ranges against the classical baselines are computed without the
external prior, matching the paired inference that covers those baselines only.
getting that wrong would make a correct claim look broken.
```

## 8. `386d212`

```
docs(repro): reviewer-facing reproducibility, one-command verification gate

scripts/verify.sh now runs the test suite, a downstream smoke test into a
temporary directory, regeneration of both the primary and the aggregate table
blocks, and the manuscript build, in about 35 seconds. the smoke test writes to
a temp path deliberately: the harness's default output directory is the released
per-patch record, and a gate that overwrites the evidence it is checking would
be worse than no gate. table regeneration was added because those tables could
previously drift from their csvs without anything noticing, which is how the
manuscript came to assert one ranking while its own table showed another.

readme maps every table and figure to the script and csv that produced it,
states the cpu and gpu split with the rule that no table row mixes devices,
gives honest timings, and documents that installing the inpainter downgrades
numpy and pillow along with how to restore them.

kaggle/README.md is a gpu runbook written against a real session: quotas,
session limits, the internet toggle, keeping model weights out of the 20 gb
output directory, and the measured throughput of 35 ms per patch.

configs/csfr_sweep_2d_t2400.yaml ships so the matched-compute run is
reproducible. its results do not: that sweep was started, paused and never
completed, and one orphan cell in a released tree would imply an experiment the
paper does not report.
```

## 9. `da7e0b7`

```
docs: trim readmes, add commit list

readmes cut from 1015 lines to 334, lowercase, no long dashes. the readme now
answers what a reviewer needs first: one command to verify, what is released,
which script and csv produced each table, and the cpu gpu split.

COMMITS.md lists the eight commits verbatim.
```
