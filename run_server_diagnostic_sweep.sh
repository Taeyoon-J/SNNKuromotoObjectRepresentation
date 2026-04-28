#!/usr/bin/env bash
set -euo pipefail

# Run Kuramoto-Gamma diagnostic sweeps on the server.
#
# Default behavior:
# - Computes spikes/classifier masks and Aij/alpha.
# - Does NOT let Aij/alpha affect Kuramoto unless EXTRA_ARGS includes
#   --feedback_affects_kuramoto.
# - Runs sequentially by default to avoid overloading one GPU.
#
# Example:
#   cd /Data0/tkim1/SNNKuromotoObjectRepresentation/rezero
#   source ../.venv311/bin/activate
#   bash run_server_diagnostic_sweep.sh
#
# Optional overrides:
#   MAX_PARALLEL=2 bash run_server_diagnostic_sweep.sh
#   INITIALIZERS="patch_conv encoder" INTERVALS="8 16" bash run_server_diagnostic_sweep.sh

HDF5_PATH="${HDF5_PATH:-/Data0/tkim1/datasets/object_centric_data/clevr_10-full.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/kuramoto_gamma_diagnostic_sweep}"
LOG_DIR="${LOG_DIR:-logs/kuramoto_gamma_diagnostic_sweep}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
STEPS="${STEPS:-30}"
VISUAL_STEPS="${VISUAL_STEPS:-1,8,16,24,30}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
DEVICE="${DEVICE:-cuda}"

INITIALIZERS="${INITIALIZERS:-encoder patch_conv channel_compress}"
INTERVALS="${INTERVALS:-8 16 24}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

CLASSIFIER_SIMILARITY_THRESHOLD="${CLASSIFIER_SIMILARITY_THRESHOLD:-0.60}"
GAMMA_PATCH_SIZE="${GAMMA_PATCH_SIZE:-2}"
GAMMA_UPDATE_SCALE="${GAMMA_UPDATE_SCALE:-1.0}"
GLOBAL_COUPLING_STRENGTH="${GLOBAL_COUPLING_STRENGTH:-1.0}"
FEEDBACK_MODE="${FEEDBACK_MODE:-spike_feedback}"

PRETRAIN_GAMMA_EPOCHS="${PRETRAIN_GAMMA_EPOCHS:-5}"
PRETRAIN_GAMMA_SAMPLES="${PRETRAIN_GAMMA_SAMPLES:-50}"
PRETRAIN_GAMMA_BATCH_SIZE="${PRETRAIN_GAMMA_BATCH_SIZE:-4}"

# Extra args are appended to every run. Example:
#   EXTRA_ARGS="--no-pretrain_gamma_encoder" bash run_server_diagnostic_sweep.sh
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${MAX_PARALLEL}" ]; do
    sleep 5
  done
}

run_one() {
  local initializer="$1"
  local interval="$2"
  local schedule_name="$3"
  local output_dir="$4"
  local log_file="$5"
  shift 5

  echo "launch ${schedule_name} initializer=${initializer} interval=${interval}"
  nohup env PYTHONPATH=. python -m re_zero.diagnose_kuramoto_gamma \
    --hdf5_path "${HDF5_PATH}" \
    --output_dir "${output_dir}" \
    --image_size "${IMAGE_SIZE}" \
    --steps "${STEPS}" \
    --gamma_update_interval "${interval}" \
    --gamma_initialization "${initializer}" \
    --gamma_patch_size "${GAMMA_PATCH_SIZE}" \
    --classifier_similarity_threshold "${CLASSIFIER_SIMILARITY_THRESHOLD}" \
    --global_coupling_strength "${GLOBAL_COUPLING_STRENGTH}" \
    --feedback_mode "${FEEDBACK_MODE}" \
    --fixed_affinity_value 1.0 \
    --fixed_alpha_value 0.0 \
    --gamma_update_scale "${GAMMA_UPDATE_SCALE}" \
    --pretrain_gamma_epochs "${PRETRAIN_GAMMA_EPOCHS}" \
    --pretrain_gamma_samples "${PRETRAIN_GAMMA_SAMPLES}" \
    --pretrain_gamma_batch_size "${PRETRAIN_GAMMA_BATCH_SIZE}" \
    --sample_idx "${SAMPLE_IDX}" \
    --device "${DEVICE}" \
    --visual_steps "${VISUAL_STEPS}" \
    "$@" \
    ${EXTRA_ARGS} \
    > "${log_file}" 2>&1 &
}

for initializer in ${INITIALIZERS}; do
  for interval in ${INTERVALS}; do
    wait_for_slot
    run_one \
      "${initializer}" \
      "${interval}" \
      "attract_1p8_to_1p0" \
      "${OUTPUT_ROOT}/${initializer}_interval${interval}_attract_1p8_to_1p0" \
      "${LOG_DIR}/${initializer}_interval${interval}_attract_1p8_to_1p0.log" \
      --gamma_attraction_schedule step_decay \
      --gamma_attraction_values 1.8,1.5,1.2,1.0 \
      --gamma_attraction_boundaries 8,16,24

    wait_for_slot
    run_one \
      "${initializer}" \
      "${interval}" \
      "attract_2p0_to_1p5" \
      "${OUTPUT_ROOT}/${initializer}_interval${interval}_attract_2p0_to_1p5" \
      "${LOG_DIR}/${initializer}_interval${interval}_attract_2p0_to_1p5.log" \
      --gamma_attraction_schedule step_decay \
      --gamma_attraction_values 2.0,1.8,1.6,1.5 \
      --gamma_attraction_boundaries 8,16,24

    wait_for_slot
    run_one \
      "${initializer}" \
      "${interval}" \
      "attract_const_1p5" \
      "${OUTPUT_ROOT}/${initializer}_interval${interval}_attract_const_1p5" \
      "${LOG_DIR}/${initializer}_interval${interval}_attract_const_1p5.log" \
      --gamma_attraction_schedule constant \
      --gamma_attraction_strength 1.5

    wait_for_slot
    run_one \
      "${initializer}" \
      "${interval}" \
      "attract_const_1p8" \
      "${OUTPUT_ROOT}/${initializer}_interval${interval}_attract_const_1p8" \
      "${LOG_DIR}/${initializer}_interval${interval}_attract_const_1p8.log" \
      --gamma_attraction_schedule constant \
      --gamma_attraction_strength 1.8
  done
done

wait
echo "All diagnostic sweep jobs completed."
echo "Outputs: ${OUTPUT_ROOT}"
echo "Logs: ${LOG_DIR}"
