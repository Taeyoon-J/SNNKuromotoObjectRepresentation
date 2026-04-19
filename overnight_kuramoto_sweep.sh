#!/usr/bin/env bash
set -u

HDF5=${HDF5:-/Data0/tkim1/datasets/object_centric_data/clevr_10-full.hdf5}
GPU=${GPU:-1}
ROOT=${ROOT:-outputs/overnight_kuramoto_sweep}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "${ROOT}" "${LOG_DIR}"

run_one () {
  local name=$1
  local readout=$2
  local attr=$3
  local schedule=$4
  local attr_end=$5
  local decay_steps=$6
  local alpha=$7
  local floor=$8
  local dt=$9
  local output_dir="${ROOT}/${name}"

  echo "===== ${name} ====="
  echo "readout=${readout} attr=${attr} schedule=${schedule} end=${attr_end} alpha=${alpha} floor=${floor} dt=${dt}"

  CUDA_VISIBLE_DEVICES=${GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_clevr_sweep.py \
    --hdf5_path "${HDF5}" \
    --output_dir "${output_dir}" \
    --max_runs 1 \
    --max_eval_batches 5 \
    --epochs 6 \
    --train_samples 20 \
    --image_size 64 \
    --patch_size 64 \
    --patch_stride 64 \
    --object_patches_only \
    --min_object_pixels 120 \
    --batch_size 1 \
    --dt "${dt}" \
    --coupling 1.0 \
    --attraction_strength "${attr}" \
    --attraction_strength_schedule "${schedule}" \
    --attraction_strength_end "${attr_end}" \
    --attraction_strength_decay_steps "${decay_steps}" \
    --recurrent_scale 1.0 \
    --coupling_chunk_size 16 \
    --osc_dim 4 \
    --t_limits 30 \
    --classifier_start_steps 20 \
    --readout_update_intervals "${readout}" \
    --spike_update_offsets 0 \
    --loss_functions 1234 \
    --feedback_magnitudes 0.1 \
    --background_suppression_weight 2.0 \
    --object_coverage_weight 4.0 \
    --score_quantile 0.45 \
    --min_pixels 64 \
    --fixed_alpha_during_training \
    --fixed_alpha_value "${alpha}" \
    --preserve_gamma_value_amplitude \
    --gamma_value_floor "${floor}" \
    --gamma_encoder_skip_scale 0.10 \
    --visual_steps 5,10,15,20,25,30

  local run_dir="${output_dir}/0000_readout${readout}_spike_same_t_classifier20_steps30_loss1234_feedback0p1"
  if [[ -f "${run_dir}/history_maps.npz" ]]; then
    python diagnose_mean_iou.py \
      --run_dir "${run_dir}" \
      --image_height 64 \
      --image_width 64 \
      --classifier_start_step 20 \
      --score_quantiles 0.35,0.40,0.45,0.50,0.55,0.60 \
      --min_pixels 32,64,96,128,192

    python diagnose_dynamics.py \
      --run_dir "${run_dir}" \
      --image_height 64 \
      --image_width 64 \
      --classifier_start_step 20 \
      --score_quantiles 0.35,0.40,0.45,0.50,0.55,0.60 \
      --min_pixels 32,64,96,128,192
  else
    echo "Missing history_maps.npz for ${name}; skipping dynamics diagnostics"
  fi
}

# Baselines that were informative before.
run_one baseline_attr3_alpha0_floor0_readout5 5 3 constant 3 30 0.0 0.00 0.15
run_one baseline_attr7_alpha05_floor005_readout5 5 7 constant 7 30 0.5 0.05 0.20

# Does early strong gamma pull followed by weaker pull improve object binding?
run_one attr7to1_alpha05_floor005_readout5 5 7 linear_decay 1 30 0.5 0.05 0.20
run_one attr7to2_alpha05_floor005_readout5 5 7 linear_decay 2 30 0.5 0.05 0.20
run_one attr5to1_alpha05_floor005_readout5 5 5 linear_decay 1 30 0.5 0.05 0.20
run_one attr5to2_alpha05_floor005_readout5 5 5 linear_decay 2 30 0.5 0.05 0.20

# Is alpha helping or randomizing background?
run_one attr7to1_alpha0_floor005_readout5 5 7 linear_decay 1 30 0.0 0.05 0.20
run_one attr7to1_alpha1_floor005_readout5 5 7 linear_decay 1 30 1.0 0.05 0.20

# Does background anchoring help?
run_one attr7to1_alpha05_floor0_readout5 5 7 linear_decay 1 30 0.5 0.00 0.20
run_one attr7to1_alpha05_floor002_readout5 5 7 linear_decay 1 30 0.5 0.02 0.20

# Does readout10 hurt because updates are too sparse?
run_one attr7to1_alpha05_floor005_readout10 10 7 linear_decay 1 30 0.5 0.05 0.20

python - <<'PY'
from pathlib import Path
import csv
import json

root = Path("outputs/overnight_kuramoto_sweep")
rows = []
for result_path in sorted(root.glob("*/sweep_results.csv")):
    experiment = result_path.parent.name
    with result_path.open() as f:
        result_rows = list(csv.DictReader(f))
    row = {"experiment": experiment}
    if result_rows:
        row.update(result_rows[0])
    run_dirs = sorted(result_path.parent.glob("0000_*"))
    if run_dirs:
        dyn_path = run_dirs[0] / "dynamics_diagnostics.json"
        if dyn_path.exists():
            with dyn_path.open() as f:
                dyn = json.load(f)
            row.update({f"dyn_{k}": v for k, v in dyn.items() if k != "run_dir"})
    rows.append(row)

def as_float(row, key):
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0

rows.sort(
    key=lambda row: (
        as_float(row, "score_mean_one_to_one_iou"),
        as_float(row, "score_50_coverage"),
        as_float(row, "dyn_spike_object_background_gap_late_mean"),
        as_float(row, "dyn_theta_phase_object_coherence_final") - as_float(row, "dyn_theta_phase_background_coherence_final"),
    ),
    reverse=True,
)

out = root / "summary_sorted.csv"
fieldnames = []
for row in rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print("Wrote", out)
print("Top runs:")
for row in rows[:10]:
    print(
        row.get("experiment"),
        "meanIoU=", row.get("score_mean_one_to_one_iou"),
        "cov50=", row.get("score_50_coverage"),
        "gapLate=", row.get("dyn_spike_object_background_gap_late_mean"),
        "thetaObjCoh=", row.get("dyn_theta_phase_object_coherence_final"),
        "thetaBgCoh=", row.get("dyn_theta_phase_background_coherence_final"),
        "bestGrid=", row.get("dyn_best_grid_mean_iou"),
    )
PY
