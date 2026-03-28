#!/usr/bin/env bash
set -euo pipefail

MODEL="r50"
VARIANT="hrca"
MODE="train"
GPUS="2,3"
IMS_PER_BATCH=""
BASE_LR=""
LOG_DIR="logs"
RUN_NAME=""
OUTPUT_DIR=""
OUTPUT_ROOT="output"
BACKGROUND="0"
RESUME="0"
WEIGHTS=""
EXTRA_OPTS=()

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

usage() {
  cat <<'EOF'
Usage:
  scripts/train_boundary.sh [options] [-- <extra detectron2 opts>]

Options:
  --model <r50|r101>           Backbone preset. Default: r50
  --variant <befbm|hrca>       Experiment preset. Default: hrca
  --mode <train|eval>          Run training or eval-only. Default: train
  --gpus <ids>                 CUDA_VISIBLE_DEVICES, e.g. "2" or "2,3"
  --ims-per-batch <int>        Override SOLVER.IMS_PER_BATCH
  --base-lr <float>            Override SOLVER.BASE_LR
  --weights <path>             Required for --mode eval
  --log-dir <dir>              Log directory. Default: logs
  --name <run_name>            Custom log prefix
  --output-dir <dir>           Override OUTPUT_DIR. Default: output/<run_name>_<timestamp>
  --resume                     Resume from last checkpoint
  --background                 Run with nohup in background
  -h, --help                   Show help

Examples:
  scripts/train_boundary.sh
  scripts/train_boundary.sh --variant befbm --gpus 2
  scripts/train_boundary.sh --model r101 --variant hrca --background
  scripts/train_boundary.sh --mode eval --weights output/model_final.pth
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --ims-per-batch)
      IMS_PER_BATCH="$2"
      shift 2
      ;;
    --base-lr)
      BASE_LR="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --name)
      RUN_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    --background)
      BACKGROUND="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OPTS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$MODEL" != "r50" && "$MODEL" != "r101" ]]; then
  echo "Unsupported --model: $MODEL" >&2
  exit 2
fi

if [[ "$VARIANT" != "befbm" && "$VARIANT" != "hrca" ]]; then
  echo "Unsupported --variant: $VARIANT" >&2
  exit 2
fi

if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
  echo "Unsupported --mode: $MODE" >&2
  exit 2
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in current environment." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "mask2former" ]]; then
  echo "Warning: current conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected 'mask2former'." >&2
fi

case "$MODEL:$VARIANT" in
  r50:befbm)
    CONFIG="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml"
    DEFAULT_IMS_PER_BATCH="4"
    DEFAULT_BASE_LR="0.00005"
    ;;
  r50:hrca)
    CONFIG="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary_hrca.yaml"
    DEFAULT_IMS_PER_BATCH="4"
    DEFAULT_BASE_LR="0.00005"
    ;;
  r101:befbm)
    CONFIG="configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary.yaml"
    DEFAULT_IMS_PER_BATCH="4"
    DEFAULT_BASE_LR="0.00005"
    ;;
  r101:hrca)
    CONFIG="configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary_hrca.yaml"
    DEFAULT_IMS_PER_BATCH="4"
    DEFAULT_BASE_LR="0.00005"
    ;;
esac

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if [[ -z "$IMS_PER_BATCH" ]]; then
  IMS_PER_BATCH="$DEFAULT_IMS_PER_BATCH"
fi

if [[ -z "$BASE_LR" ]]; then
  BASE_LR="$DEFAULT_BASE_LR"
fi

if [[ "$MODE" == "eval" && -z "$WEIGHTS" ]]; then
  echo "--weights is required when --mode eval" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"

mkdir -p "$LOG_DIR"
timestamp="$(date +%F_%H-%M-%S)"
gpu_tag="${GPUS//,/_}"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="${MODEL}_${VARIANT}_gpu${gpu_tag}_${MODE}"
fi
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${timestamp}.log"

find_latest_output_dir() {
  local latest_dir=""
  latest_dir="$(
    find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${RUN_NAME}_*" | sort | tail -n 1
  )"
  echo "$latest_dir"
}

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$RESUME" == "1" ]]; then
    OUTPUT_DIR="$(find_latest_output_dir)"
    if [[ -z "$OUTPUT_DIR" && -f "${OUTPUT_ROOT}/last_checkpoint" ]]; then
      OUTPUT_DIR="$OUTPUT_ROOT"
      echo "Warning: using legacy flat output directory '${OUTPUT_DIR}' for resume." >&2
    fi
    if [[ -z "$OUTPUT_DIR" ]]; then
      echo "No previous output directory found for resume. Use --output-dir to specify one." >&2
      exit 2
    fi
  else
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${timestamp}"
  fi
fi

mkdir -p "$OUTPUT_DIR"

CMD=(python -u train_net.py --config-file "$CONFIG" --num-gpus "$NUM_GPUS")

if [[ "$MODE" == "eval" ]]; then
  CMD+=(--eval-only MODEL.WEIGHTS "$WEIGHTS")
else
  CMD+=(SOLVER.IMS_PER_BATCH "$IMS_PER_BATCH" SOLVER.BASE_LR "$BASE_LR")
fi

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi

CMD+=(OUTPUT_DIR "$OUTPUT_DIR")

if [[ ${#EXTRA_OPTS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OPTS[@]}")
fi

build_command_string() {
  local cmd_string=""
  printf -v cmd_string '%q ' "${CMD[@]}"
  echo "${cmd_string% }"
}

emit_run_config() {
  local command_string
  command_string="$(build_command_string)"

  echo "==== Launch $(date '+%F %T %Z') ===="
  echo "Run config:"
  echo "  model:       $MODEL"
  echo "  variant:     $VARIANT"
  echo "  mode:        $MODE"
  echo "  config:      $CONFIG"
  echo "  gpus:        $GPUS (num_gpus=$NUM_GPUS)"
  if [[ "$MODE" == "train" ]]; then
    echo "  ims/batch:   $IMS_PER_BATCH"
    echo "  base_lr:     $BASE_LR"
  else
    echo "  weights:     $WEIGHTS"
  fi
  echo "  resume:      $RESUME"
  echo "  output_dir:  $OUTPUT_DIR"
  echo "  log_file:    $LOG_FILE"
  echo "  warnings:    ${PYTHONWARNINGS}"
  echo "  cuda_alloc:  ${PYTORCH_CUDA_ALLOC_CONF}"
  echo "Command:"
  echo "  $command_string"
  echo
}

if [[ "$BACKGROUND" == "1" ]]; then
  emit_run_config | tee -a "$LOG_FILE"
  nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
  PID=$!
  {
    echo "Started in background: PID=$PID"
    echo "Monitor log: tail -f $LOG_FILE"
    echo "Check process: ps -fp $PID"
    echo
  } | tee -a "$LOG_FILE"
else
  emit_run_config | tee -a "$LOG_FILE"
  set +e
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  CMD_STATUS=${PIPESTATUS[0]}
  set -e
  echo "==== Exit $(date '+%F %T %Z') status=${CMD_STATUS} ====" | tee -a "$LOG_FILE"
  exit "$CMD_STATUS"
fi
