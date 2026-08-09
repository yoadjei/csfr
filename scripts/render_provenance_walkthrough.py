"""Provenance walkthrough figure (Phase 4): contrast attribution in two regimes.

Band corruption (CF3_L5_patch5) compared at two depths: a pixel one row from
the observed boundary (row 44) shows concentrated support in immediate
neighbours. The same patch, 22 rows deeper inside (row 22), shows distributed
support across all observed pixels. The contrast shows how provenance regime
shifts with depth: evidence concentrates when observed data is adjacent,
disperses when the pixel is deep inside a missing region.

Output: paper/figures/paper2_F12_provenance.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from provenance import provenance_row


def compute_stats(weights, M, y, pixel_r, pixel_c):
    """Compute attribution statistics for a query pixel."""
    weights_obs = np.where(M, weights, 0)
    weight_flat = weights_obs.ravel()
    order = np.argsort(np.abs(weight_flat))[::-1]
    top_5_indices = order[:5]
    top_5_pixels = [(int(idx // 64), int(idx % 64)) for idx in top_5_indices]
    top_5_weights = [weight_flat[idx] for idx in top_5_indices]

    top_5_abs_sum = sum(abs(w) for w in top_5_weights)
    total_abs_weight = np.abs(weights_obs).sum()
    top_5_share = top_5_abs_sum / total_abs_weight if total_abs_weight > 0 else 0

    obs_rows, obs_cols = np.where(M == 1)
    if len(obs_rows) > 0:
        dists = np.sqrt((obs_rows - pixel_r) ** 2 + (obs_cols - pixel_c) ** 2)
        nearest_dist = float(dists.min())
        nearest_obs_idx = int(dists.argmin())
        nearest_obs = (int(obs_rows[nearest_obs_idx]), int(obs_cols[nearest_obs_idx]))
    else:
        nearest_dist = np.inf
        nearest_obs = None

    four_neighbors = [(pixel_r-1, pixel_c), (pixel_r+1, pixel_c),
                      (pixel_r, pixel_c-1), (pixel_r, pixel_c+1)]
    neighbor_weights = []
    for nr, nc in four_neighbors:
        if 0 <= nr < 64 and 0 <= nc < 64 and M[nr, nc] == 1:
            neighbor_weights.append(abs(weights[nr, nc]))
    four_neighbor_share = sum(neighbor_weights) / total_abs_weight if total_abs_weight > 0 else 0

    max_weight = max(abs(w) for w in top_5_weights) if top_5_weights else 0

    return {
        "top_5_pixels": top_5_pixels,
        "top_5_weights": top_5_weights,
        "top_5_share": top_5_share,
        "four_neighbor_share": four_neighbor_share,
        "nearest_obs": nearest_obs,
        "nearest_dist": nearest_dist,
        "max_weight": max_weight,
        "total_weight": total_abs_weight,
    }


def main() -> int:
    # Load data
    data_dir = ROOT / "results" / "csfr_sweep_2d_v2" / "CF3_L5" / "seed0"
    M_all = np.load(data_dir / "mask.npy").astype(np.float32)
    y_all = np.load(data_dir / "y.npy").astype(np.float32)
    x_hat_all = np.load(data_dir / "x_hat.npy").astype(np.float32)

    patch_idx = 5
    M = M_all[patch_idx]
    y = y_all[patch_idx]
    x_hat = x_hat_all[patch_idx]

    # Get solver config from manifest
    manifest_path = ROOT / "results" / "provenance_example" / "CF1_L3_patch0.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = manifest["solver_configuration"]

    # Two queries: near boundary (row 44) and deep interior (row 22)
    queries = [
        {"name": "Near boundary", "pixel": (44, 32), "row_desc": "row 44 (1 pixel from row 45)"},
        {"name": "Deep interior", "pixel": (22, 32), "row_desc": "row 22 (22 pixels from row 45)"},
    ]

    results = []
    for q in queries:
        pixel_r, pixel_c = q["pixel"]
        print(f"\n{q['name']}: ({pixel_r}, {pixel_c})")

        x_hat_t = torch.from_numpy(x_hat[None, ...]).to(torch.float32)
        y_t = torch.from_numpy(y[None, ...]).to(torch.float32)
        M_t = torch.from_numpy(M[None, ...]).to(torch.float32)

        weights, cg_info = provenance_row(
            x_hat_t[0], y_t[0], M_t[0], (pixel_r, pixel_c),
            lam_l1=cfg["lam_l1"], lam_tv=cfg["lam_tv"], lam_fc=cfg["lam_fc"],
        )

        stats = compute_stats(weights, M, y, pixel_r, pixel_c)
        stats["cg_iters"] = cg_info["cg_iters"]
        stats["residual"] = cg_info["residual"]
        stats["weights"] = weights
        stats["value"] = float(x_hat[pixel_r, pixel_c])

        print(f"  Top 5 share: {stats['top_5_share']:.1%}")
        print(f"  4-neighbor share: {stats['four_neighbor_share']:.1%}")
        print(f"  Nearest observed: {stats['nearest_obs']} at {stats['nearest_dist']:.1f} px")

        results.append((q, stats))

    # Build 2-row, 3-column figure
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.3, height_ratios=[1, 1])

    for row_idx, (query, stats) in enumerate(results):
        pixel_r, pixel_c = query["pixel"]
        weights = stats["weights"]

        # Column 1: Full patch
        ax = fig.add_subplot(gs[row_idx, 0])
        patch_display = y.copy()
        patch_display[M == 0] = y.min() * 0.65
        ax.imshow(patch_display, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

        # Mark query pixel
        rect = mpatches.Rectangle((pixel_c - 0.5, pixel_r - 0.5), 1, 1,
                                   linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

        # Mark nearest observed
        obs_r, obs_c = stats["nearest_obs"]
        ax.plot(obs_c, obs_r, "g*", markersize=12, markeredgewidth=0.5)

        # Draw line and distance label
        ax.plot([pixel_c, obs_c], [pixel_r, obs_r], "g--", linewidth=1, alpha=0.5)
        mid_r, mid_c = (pixel_r + obs_r) / 2, (pixel_c + obs_c) / 2
        ax.text(mid_c + 2, mid_r, f"{stats['nearest_dist']:.1f}px", fontsize=7, color="green")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({chr(65 + row_idx*3)}) {query['name']}", fontsize=9, loc="left")
        ax.text(0.5, -0.05, query['row_desc'], transform=ax.transAxes,
                fontsize=7, ha="center", style="italic")

        # Column 2: Full-patch attribution
        ax = fig.add_subplot(gs[row_idx, 1])
        w_abs_max = np.abs(weights).max()
        im = ax.imshow(weights, cmap="RdBu_r", interpolation="nearest",
                       vmin=-w_abs_max, vmax=w_abs_max)
        ax.plot(pixel_c, pixel_r, "r+", markersize=12, markeredgewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({chr(66 + row_idx*3)}) Attribution (limit: +/- {w_abs_max:.3g})",
                     fontsize=9, loc="left")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("weight", fontsize=7)

        # Column 3: Compact stats
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.axis("off")

        text_lines = [
            f"Top 5 share: {stats['top_5_share']:.1%}",
            f"4-nbr share: {stats['four_neighbor_share']:.1%}",
            f"Dist to obs: {stats['nearest_dist']:.1f} px",
            f"Max weight: {stats['max_weight']:+.6f}",
            "",
            f"CG: {stats['cg_iters']} iter",
            f"  residual: {stats['residual']:.2e}",
            "",
            "Top 5:",
        ]

        for rank, ((tr, tc), w) in enumerate(zip(stats["top_5_pixels"], stats["top_5_weights"])):
            text_lines.append(f"  {rank+1}. ({tr},{tc:2d}) {w:+.5f}")

        y_pos = 0.95
        for line in text_lines:
            ax.text(0.05, y_pos, line, fontsize=7, family="monospace",
                    verticalalignment="top", transform=ax.transAxes)
            y_pos -= 0.045

        ax.set_title(f"({chr(67 + row_idx*3)}) Statistics", fontsize=9, loc="left")

    # Footer with constraints
    fig.text(0.5, 0.01,
             "Constraints: C1 hard (consistency), C2 soft (λ_TV=0.1), "
             "C3 hard (bounds), C4 soft (λ_FC=0.005). "
             "Weights describe stationarity condition at 600 iterations.",
             ha="center", fontsize=7, style="italic")

    fig.suptitle("Provenance: attribution depth dependence (CF3_L5_patch5, band corruption)",
                 fontsize=11, y=0.98)

    out = ROOT / "paper" / "figures" / "paper2_F12_provenance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Report statistics for manuscript
    print("\n" + "="*60)
    print("MANUSCRIPT SUMMARY")
    print("="*60)
    for query, stats in results:
        print(f"\n{query['name']} (pixel {query['pixel']}):")
        print(f"  Top 5 weight share: {stats['top_5_share']:.1%}")
        print(f"  4-neighbor weight share: {stats['four_neighbor_share']:.1%}")
        print(f"  Distance to nearest observed: {stats['nearest_dist']:.1f} pixels")

    return 0


if __name__ == "__main__":
    sys.exit(main())
