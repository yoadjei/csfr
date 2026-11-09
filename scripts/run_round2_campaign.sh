#!/usr/bin/env bash
# Round-2 experiment campaign, in dependency order. Every step is resumable, so
# a stage can be re-run after an interruption. Logs go to results/<dir>/run_*.log.
#
#   bash scripts/run_round2_campaign.sh sweeps      # E1-E4: DFRWS, GovDocs1 (+ ablation), Kodak
#   bash scripts/run_round2_campaign.sh lama        # LaMa outputs for the sweeps, then scored
#   bash scripts/run_round2_campaign.sh analysis    # stats, fidelity (E6, E7), oracle, BU, solver comparison
#   bash scripts/run_round2_campaign.sh forensic    # mask robustness (E10), JPEG corruption (E9) + LaMa
#   bash scripts/run_round2_campaign.sh downstream  # ResNet, SVHN for both classifiers (E12) + LaMa, OCR (E13)
#   bash scripts/run_round2_campaign.sh idle        # bench v2 (E15) and full resolution (E11): idle machine only
set -euo pipefail
cd "$(dirname "$0")/.."
LAMA=".venv-lama/Scripts/python"; [ -x "$LAMA" ] || LAMA=".venv-lama/bin/python"
OCR=".venv-ocr/Scripts/python"; [ -x "$OCR" ] || OCR=".venv-ocr/bin/python"
SWEEPS="dfrws govdocs1 kodak"

stage_sweeps() {
  for s in $SWEEPS; do
    mkdir -p "results/sweep_v2_$s"
    python scripts/run_sweep_v2.py --config "configs/sweep_v2_$s.yaml" > "results/sweep_v2_$s/run.log" 2>&1
  done
}

stage_lama() {
  for s in $SWEEPS; do
    "$LAMA" scripts/run_lama_cells.py --root "results/sweep_v2_$s" > "results/sweep_v2_$s/run_lama.log" 2>&1
    python scripts/run_sweep_v2.py --config "configs/sweep_v2_$s.yaml" --only-method lama > "results/sweep_v2_$s/run_score_lama.log" 2>&1
  done
}

stage_analysis() {
  for s in $SWEEPS; do
    python scripts/run_stats_v2.py --sweep-dir "results/sweep_v2_$s" > "results/sweep_v2_$s/run_stats.log" 2>&1
  done
  mkdir -p results/sensitivity_fidelity results/oracle_diagnostic results/bu_evaluation results/solver_comparison
  python scripts/run_sensitivity_fidelity.py > results/sensitivity_fidelity/run.log 2>&1
  python scripts/run_sensitivity_fidelity.py --tol-study > results/sensitivity_fidelity/run_tol.log 2>&1
  python scripts/run_sensitivity_fidelity.py --records-compare > results/sensitivity_fidelity/run_records.log 2>&1
  python scripts/run_oracle_diagnostic.py > results/oracle_diagnostic/run.log 2>&1
  python scripts/run_bu_evaluation.py > results/bu_evaluation/run.log 2>&1
  python scripts/run_solver_comparison.py > results/solver_comparison/run.log 2>&1
}

stage_forensic() {
  mkdir -p results/mask_robustness results/jpeg_corruption
  python scripts/run_mask_robustness.py > results/mask_robustness/run.log 2>&1
  "$LAMA" scripts/run_lama_cells.py --root results/mask_robustness/cases > results/mask_robustness/run_lama.log 2>&1
  python scripts/run_mask_robustness.py > results/mask_robustness/run_score_lama.log 2>&1
  python -I scripts/run_jpeg_corruption.py --config configs/jpeg_corruption.yaml > results/jpeg_corruption/run.log 2>&1
  "$LAMA" scripts/run_lama_cells.py --root results/jpeg_corruption/cases > results/jpeg_corruption/run_lama.log 2>&1
  python -I scripts/run_jpeg_corruption.py --config configs/jpeg_corruption.yaml > results/jpeg_corruption/run_score_lama.log 2>&1
}

stage_downstream() {
  mkdir -p results/downstream_v2 results/ocr
  [ -f results/downstream_v2/classifier_resnet.pt ] || \
    python scripts/train_resnet_classifier.py > results/downstream_v2/run_train_resnet.log 2>&1
  python scripts/run_downstream.py --config configs/downstream_v2_cnn.yaml > results/downstream_v2/run_cnn.log 2>&1
  "$LAMA" scripts/run_lama_cells.py --root results/downstream_v2/cases > results/downstream_v2/run_lama.log 2>&1
  python scripts/run_downstream.py --config configs/downstream_v2_cnn.yaml > results/downstream_v2/run_cnn_lama.log 2>&1
  python scripts/run_downstream.py --config configs/downstream_v2_resnet.yaml > results/downstream_v2/run_resnet.log 2>&1
  [ -f data/ocr/crops.npy ] || "$OCR" -I scripts/run_ocr_task.py crops > results/ocr/run_crops.log 2>&1
  python scripts/run_ocr_task.py reconstruct > results/ocr/run_reconstruct.log 2>&1
  "$LAMA" scripts/run_lama_cells.py --root results/ocr/cases > results/ocr/run_lama.log 2>&1
  "$OCR" scripts/run_ocr_task.py recognise > results/ocr/run_recognise.log 2>&1
  python scripts/run_ocr_task.py score > results/ocr/run_score.log 2>&1
}

stage_idle() {
  mkdir -p results/bench_v2 results/fullres
  python scripts/run_bench_v2.py > results/bench_v2/run.log 2>&1
  python -I scripts/run_fullres.py > results/fullres/run.log 2>&1
  "$LAMA" scripts/run_lama_cells.py --root results/fullres/cases > results/fullres/run_lama.log 2>&1
  python -I scripts/run_fullres.py > results/fullres/run_score_lama.log 2>&1
}

for stage in "$@"; do
  echo "[campaign] $(date '+%F %T') stage $stage"
  "stage_$stage"
  echo "[campaign] $(date '+%F %T') stage $stage done"
done
