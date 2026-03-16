#!/usr/bin/env bash
set -euo pipefail

# Default settings (override by CLI flags)
CONFIG="configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml"
GPUS="2,3"
IMS_PER_BATCH="4"
BASE_LR="0.00005"
LOG_DIR="logs"
RUN_NAME=""
OUTPUT_DIR=""
BACKGROUND="0"
RESUME="0"
EVAL_ONLY="0"

# Keep logs cleaner; avoid AMP deprecation spam in training logs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

EXTRA_OPTS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/train_boundary.sh [options] [-- <extra detectron2 opts>]

Options:
  --config <path>          Config file path.
  --gpus <ids>             CUDA_VISIBLE_DEVICES, e.g. "2" or "2,3".
  --ims-per-batch <int>    SOLVER.IMS_PER_BATCH (training only).
  --base-lr <float>        SOLVER.BASE_LR (training only).
  --log-dir <dir>          Log directory. Default: logs
  --name <run_name>        Run name prefix for log file.
  --output-dir <dir>       Override OUTPUT_DIR in config.
  --resume                 Resume from last checkpoint.
  --eval-only              Eval only mode.
  --background             Run with nohup in background.
  -h, --help               Show help.

Examples:
  scripts/train_boundary.sh
  scripts/train_boundary.sh --gpus 2 --ims-per-batch 2 --base-lr 0.000025
  scripts/train_boundary.sh --background --name r50_ba_gpu23
  scripts/train_boundary.sh --config configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary.yaml -- --num-machines 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
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
    --eval-only)
      EVAL_ONLY="1"
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

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in current environment." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "mask2former" ]]; then
  echo "Warning: current conda env is '${CONDA_DEFAULT_ENV:-<none>}', expected 'mask2former'." >&2
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"

mkdir -p "$LOG_DIR"
timestamp="$(date +%F_%H-%M-%S)"
gpu_tag="${GPUS//,/_}"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="boundary_gpu${gpu_tag}"
fi
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${timestamp}.log"

CMD=(python -u train_net.py --config-file "$CONFIG" --num-gpus "$NUM_GPUS")

if [[ "$EVAL_ONLY" == "1" ]]; then
  CMD+=(--eval-only)
else
  CMD+=(SOLVER.IMS_PER_BATCH "$IMS_PER_BATCH" SOLVER.BASE_LR "$BASE_LR")
fi

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(OUTPUT_DIR "$OUTPUT_DIR")
fi

if [[ ${#EXTRA_OPTS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OPTS[@]}")
fi

echo "Run config:"
echo "  config:      $CONFIG"
echo "  gpus:        $GPUS (num_gpus=$NUM_GPUS)"
echo "  eval_only:   $EVAL_ONLY"
echo "  resume:      $RESUME"
echo "  log_file:    $LOG_FILE"
echo "  warnings:    ${PYTHONWARNINGS}"
echo "  cuda_alloc:  ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "$BACKGROUND" == "1" ]]; then
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PID=$!
  echo "Started in background: PID=$PID"
  echo "Monitor log: tail -f $LOG_FILE"
  echo "Check process: ps -fp $PID"
else
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
