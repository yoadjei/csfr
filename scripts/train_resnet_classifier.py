"""Train the second downstream classifier (round 2, E12).

    python scripts/train_resnet_classifier.py        # about an hour on 8 CPU cores

DigitResNet on the SVHN training split (90 percent; 10 percent monitors
validation accuracy), 15 epochs, Adam 1e-3 with cosine decay, batch 128, weight
decay 5e-4, seed 0, CPU. Saves results/downstream_v2/classifier_resnet.pt with
the clean test accuracy in its metadata. Stops without saving if the clean test
accuracy is below 0.95. The training state is saved after every epoch
(--checkpoint), and a rerun resumes after the last saved epoch with the result
an uninterrupted run gives. Temperatures are fitted by the downstream harness, for
both classifiers, on SVHN test images disjoint from the evaluation subset.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from downstream import (clean_top1_accuracy, load_digit_dataset, save_classifier,  # noqa: E402
                        set_seed, sha256_file, train_classifier)

MIN_CLEAN_ACC = 0.95


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/downstream_v2/classifier_resnet.pt")
    ap.add_argument("--checkpoint", default="results/downstream_v2/classifier_resnet_training.pt")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    train_x, train_y, corpus = load_digit_dataset(prefer="svhn", split="train")
    test_x, test_y, _ = load_digit_dataset(prefer=corpus, split="test", download=False)
    if corpus != "svhn":
        raise SystemExit(f"SVHN unavailable (got {corpus}); the downstream experiment is defined on SVHN")
    t0 = time.time()
    model, info = train_classifier(train_x, train_y, epochs=args.epochs, batch_size=128, lr=1e-3,
                                   seed=args.seed, arch="DigitResNet", weight_decay=5e-4, cosine=True,
                                   checkpoint=ROOT / args.checkpoint)
    acc = clean_top1_accuracy(model, test_x, test_y)
    print(f"[resnet] clean SVHN test top-1 {acc:.4f} after {time.time() - t0:.0f}s", flush=True)
    if acc < MIN_CLEAN_ACC:
        print(f"[resnet] below {MIN_CLEAN_ACC}: not saved; report before continuing", flush=True)
        return 2
    out = ROOT / args.out
    meta = {"corpus": corpus, "clean_test_top1": acc, "train_seconds": time.time() - t0,
            "torch": torch.__version__, "threads": args.threads, **info}
    save_classifier(model, out, meta=meta)
    print(f"[resnet] saved {out} sha256 {sha256_file(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
