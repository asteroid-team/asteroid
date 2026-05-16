#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-convtasnet-three-src-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-ablation_feb17_three_src}"
WANDB_TAGS="${WANDB_TAGS:-ablation,feb17,convtasnet,3src,rt,1000ms,50ms}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

GPU_SLOTS="${GPU_SLOTS:-0;1;2;3}"
INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TRAIN_FAIL_POLICY="${TRAIN_FAIL_POLICY:-continue}" # continue|abort

# Common training
EPOCHS_STAGE1="${EPOCHS_STAGE1:-40}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAMPLE_RATE="${SAMPLE_RATE:-8000}"
SEGMENT="${SEGMENT:-1.0}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20000}"
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-2000}"

SRC_TRAIN="${SRC_TRAIN:-/home/mkeller/data/librimix/LibriSpeech/train-clean-360}"
SRC_DEV="${SRC_DEV:-/home/mkeller/data/librimix/LibriSpeech/dev-clean}"
SRC_TEST="${SRC_TEST:-/home/mkeller/data/librimix/LibriSpeech/test-clean}"
FULL_TEST_DIR="${FULL_TEST_DIR:-/home/mkeller/data/librimix/Libri3Mix/wav8k/min/metadata}"

WINDOW_MS="${WINDOW_MS:-1000}"
HOP_MS="${HOP_MS:-50}"
ACTIVE_RMS_DB="${ACTIVE_RMS_DB:--40.0}"

PROMOTE_TOPK="${PROMOTE_TOPK:-3}"
MANIFEST="${MANIFEST:-local/ablation_manifest_feb17.json}"

RUN_ROOT="exp/ablation_feb17_three_src_${DATE_TAG}"
LOG_ROOT="${RUN_ROOT}/logs"
REPORT_ROOT="${RUN_ROOT}/reports"
mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"
MASTER_LOG="${LOG_ROOT}/master.log"

log() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

run_cmd() {
  local name="$1"
  shift
  log ""
  log "===== ${name} ====="
  log "CMD: $*"
  "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

ensure_wandb() {
  if [[ "${USE_WANDB}" != "true" ]]; then
    return 0
  fi
  if ! command -v wandb >/dev/null 2>&1; then
    log "W&B enabled but 'wandb' CLI not found. Install with: pip install wandb"
    exit 1
  fi
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import wandb
wandb.Api().viewer.username
PY
  then
    log "W&B enabled but authentication check failed. Run: wandb login"
    exit 1
  fi
}

read_manifest_ids() {
  "${PYTHON_BIN}" - <<'PY' "${MANIFEST}"
import json, sys
p=sys.argv[1]
d=json.load(open(p))
for c in d.get("candidates", []):
    print(c["id"])
PY
}

load_cfg_args() {
  local run_id="$1"
  "${PYTHON_BIN}" - <<'PY' "${MANIFEST}" "${run_id}"
import json, sys
mf, rid = sys.argv[1], sys.argv[2]
d = json.load(open(mf))
cand = None
for c in d.get("candidates", []):
    if c["id"] == rid:
        cand = c
        break
if cand is None:
    raise SystemExit(f"run id not in manifest: {rid}")
mask = cand.get("masknet", {})
fb = cand.get("filterbank", {})
args = []
for k in ["n_blocks", "n_repeats", "bn_chan", "skip_chan", "hid_chan"]:
    if k in mask:
        args += [f"--{k}", str(mask[k])]
for k in ["n_filters"]:
    if k in fb:
        args += [f"--{k}", str(fb[k])]
print("\n".join(args))
PY
}

wait_for_jobs() {
  local running="$1"
  while (( running > 0 )); do
    if wait -n; then
      running=$((running - 1))
    else
      running=$((running - 1))
      if [[ "${TRAIN_FAIL_POLICY}" == "abort" ]]; then
        log "At least one background job failed. Aborting (TRAIN_FAIL_POLICY=abort)."
        exit 1
      fi
      log "A background job failed. Continuing (TRAIN_FAIL_POLICY=continue)."
    fi
  done
}

launch_train() {
  local run_id="$1"
  local stage="$2"
  local epochs="$3"
  local slot_devices="$4"

  local exp_dir="${RUN_ROOT}/${run_id}"
  mkdir -p "${exp_dir}"

  mapfile -t cfg_args < <(load_cfg_args "${run_id}")

  local -a wandb_args=(
    --use_wandb "${USE_WANDB}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_GROUP}_${DATE_TAG}_${stage}"
    --wandb_run_name "${run_id}_${stage}"
    --wandb_tags "${WANDB_TAGS},${run_id},${stage}"
    --wandb_job_type train
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    wandb_args+=(--wandb_entity "${WANDB_ENTITY}")
  fi

  local -a cmd=(
    env CUDA_VISIBLE_DEVICES="${slot_devices}" "${PYTHON_BIN}" train.py
    --exp_dir "${exp_dir}"
    --dataset_type online_mix
    --source_dir "${SRC_TRAIN}"
    --valid_source_dir "${SRC_DEV}"
    --task sep_clean
    --sample_rate "${SAMPLE_RATE}"
    --segment "${SEGMENT}"
    --n_src 3
    --min_speakers 1
    --max_speakers 3
    --speaker_count_weights 0.20,0.30,0.50
    --num_examples "${NUM_EXAMPLES}"
    --val_num_examples "${VAL_NUM_EXAMPLES}"
    --epochs "${epochs}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    "${cfg_args[@]}"
    "${wandb_args[@]}"
  )

  log ""
  log "===== train_${run_id}_${stage} ====="
  log "CUDA_VISIBLE_DEVICES=${slot_devices}"
  log "CMD: ${cmd[*]}"

  "${cmd[@]}" >"${LOG_ROOT}/train_${run_id}_${stage}.log" 2>&1 &
}

run_validations_for_model() {
  local run_id="$1"
  local exp_dir="${RUN_ROOT}/${run_id}"

  if [[ ! -f "${exp_dir}/best_model.pth" ]]; then
    log "[skip] missing best_model.pth for ${run_id}"
    return
  fi

  run_cmd "fullctx_${run_id}" env CUDA_VISIBLE_DEVICES="${INFER_CUDA_DEVICES}" \
    "${PYTHON_BIN}" eval.py \
    --exp_dir "${exp_dir}" \
    --test_dir "${FULL_TEST_DIR}" \
    --task sep_clean \
    --use_gpu 1 \
    --compute_wer 0 \
    --n_save_ex 10 \
    --out_dir eval_fullclip_feb17

  run_cmd "streamviz_${run_id}" env CUDA_VISIBLE_DEVICES="${INFER_CUDA_DEVICES}" \
    "${PYTHON_BIN}" validate.py \
    --exp_dir "${exp_dir}" \
    --source_dir "${SRC_TEST}" \
    --speaker_range 1,2,3 \
    --stream_window_ms "${WINDOW_MS}" \
    --stream_hop_ms "${HOP_MS}" \
    --active_rms_db "${ACTIVE_RMS_DB}" \
    --out_dir "${exp_dir}/validation_rt1s_hop50ms"

  run_cmd "compare_${run_id}" env CUDA_VISIBLE_DEVICES="${INFER_CUDA_DEVICES}" \
    "${PYTHON_BIN}" local/compare_models_nsrc3.py \
    --model "${run_id}::${exp_dir}/best_model.pth::3" \
    --source_dir "${SRC_TEST}" \
    --sample_rate "${SAMPLE_RATE}" \
    --num_examples_per_n 300 \
    --window_ms "${WINDOW_MS}" \
    --hop_ms "${HOP_MS}" \
    --out "${exp_dir}/reports/compare_models_nsrc3.json" \
    --summary_md "${exp_dir}/reports/compare_models_nsrc3.md"
}

select_promotions() {
  local stage1_ids=("$@")
  local tmp_json="${REPORT_ROOT}/stage1_collect.json"
  local tmp_md="${REPORT_ROOT}/stage1_collect.md"
  run_cmd "collect_stage1" "${PYTHON_BIN}" local/collect_ablation_metrics_feb17.py \
    --run_root "${RUN_ROOT}" \
    --eval_tag eval_fullclip_feb17 \
    --compare_json_name compare_models_nsrc3.json \
    --out_json "${tmp_json}" \
    --out_md "${tmp_md}"

  "${PYTHON_BIN}" - <<'PY' "${tmp_json}" "${PROMOTE_TOPK}" "${REPORT_ROOT}/promoted_ids.txt"
import json, sys
payload=json.load(open(sys.argv[1]))
topk=int(sys.argv[2])
outf=sys.argv[3]
models=payload.get("models", {})
if not models:
    open(outf, "w").close()
    raise SystemExit(0)
best_full=max((v.get("full_si_sdr_db") for v in models.values() if v.get("full_si_sdr_db") is not None), default=None)
rows=[]
for rid, m in models.items():
    fs=m.get("full_si_sdr_db")
    lat=m.get("stream_avg_latency_ms")
    size=m.get("model_size_mb")
    if fs is None or lat is None or size is None:
        continue
    drop=(best_full-fs) if best_full is not None else 0.0
    if drop <= 1.0:
        rows.append((lat, size, -fs, rid))
rows.sort()
p=[r[-1] for r in rows[:topk]]
with open(outf, "w") as f:
    for x in p:
        f.write(x+"\n")
print("promoted:", p)
PY
}

main() {
  ensure_wandb

  if [[ ! -f "${MANIFEST}" ]]; then
    log "Manifest not found: ${MANIFEST}"
    exit 1
  fi

  if [[ ! -d "${FULL_TEST_DIR}" ]]; then
    log "Missing full-context test dir: ${FULL_TEST_DIR}"
    exit 1
  fi

  IFS=';' read -r -a SLOT_ARR <<< "${GPU_SLOTS}"
  if (( ${#SLOT_ARR[@]} == 0 )); then
    log "GPU_SLOTS resolved to empty"
    exit 1
  fi

  mapfile -t run_ids < <(read_manifest_ids)
  if (( ${#run_ids[@]} == 0 )); then
    log "No candidates in manifest"
    exit 1
  fi

  log "Starting ablation run root: ${RUN_ROOT}"
  log "Candidates: ${run_ids[*]}"

  # Stage 1: launch all candidates in waves with MAX_PARALLEL.
  running=0
  idx=0
  for rid in "${run_ids[@]}"; do
    slot="${SLOT_ARR[$((idx % ${#SLOT_ARR[@]}))]}"
    launch_train "${rid}" "s1" "${EPOCHS_STAGE1}" "${slot}"
    running=$((running + 1))
    idx=$((idx + 1))
    if (( running >= MAX_PARALLEL )); then
      wait_for_jobs "${running}"
      running=0
    fi
  done
  if (( running > 0 )); then
    wait_for_jobs "${running}"
  fi

  # Stage 1 validations
  for rid in "${run_ids[@]}"; do
    run_validations_for_model "${rid}"
  done

  # Promotion selection
  select_promotions "${run_ids[@]}"
  mapfile -t promoted < "${REPORT_ROOT}/promoted_ids.txt"
  if (( ${#promoted[@]} == 0 )); then
    log "No promoted candidates after stage 1 filter. Stopping."
    exit 0
  fi
  log "Promoted candidates: ${promoted[*]}"

  # Stage 2: continue promoted to 200 epochs (fresh run with same exp_dir and larger epochs).
  running=0
  idx=0
  for rid in "${promoted[@]}"; do
    slot="${SLOT_ARR[$((idx % ${#SLOT_ARR[@]}))]}"
    launch_train "${rid}" "s2" "${EPOCHS_STAGE2}" "${slot}"
    running=$((running + 1))
    idx=$((idx + 1))
    if (( running >= MAX_PARALLEL )); then
      wait_for_jobs "${running}"
      running=0
    fi
  done
  if (( running > 0 )); then
    wait_for_jobs "${running}"
  fi

  # Stage 2 validations
  for rid in "${promoted[@]}"; do
    run_validations_for_model "${rid}"
  done

  run_cmd "collect_final" "${PYTHON_BIN}" local/collect_ablation_metrics_feb17.py \
    --run_root "${RUN_ROOT}" \
    --eval_tag eval_fullclip_feb17 \
    --compare_json_name compare_models_nsrc3.json \
    --out_json "${REPORT_ROOT}/ablation_feb17_summary.json" \
    --out_md "${REPORT_ROOT}/ablation_feb17_summary.md"

  log "Ablation complete."
  log "Run root: ${RUN_ROOT}"
  log "Summary JSON: ${REPORT_ROOT}/ablation_feb17_summary.json"
  log "Summary MD: ${REPORT_ROOT}/ablation_feb17_summary.md"
}

main "$@"
