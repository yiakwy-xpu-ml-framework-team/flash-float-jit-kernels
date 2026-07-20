#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${SCRIPT_DIR}/target_symm_gemm_cuda.py"
OUT_DIR="${SCRIPT_DIR}/results/ncu"

PYTHON="${PYTHON:-python}"
NCU="${NCU:-ncu}"
M="${M:-4096}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-1}"
DEVICE="${DEVICE:-cuda}"
PROFILED="${PROFILED:-0}"
SET="${SET:-full}"
KERNEL_NAME="${KERNEL_NAME:-regex:hopper_symm_gemm_kernel_entry}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
OUT="${OUT:-${OUT_DIR}/symm_gemm_cuda_m${M}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

prewarm_args=(--m "${M}" --warmup 1 --iters 1 --device "${DEVICE}" --no-nvtx)
target_args=(
    --m "${M}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --device "${DEVICE}"
    --cuda-profiler-api
)

if [[ "${PROFILED}" == "1" ]]; then
    prewarm_args+=(--profiled)
    target_args+=(--profiled)
fi

echo "[ncu] prewarming JIT/cache outside Nsight Compute..."
"${PYTHON}" "${TARGET}" "${prewarm_args[@]}"

echo "[ncu] writing ${OUT}.ncu-rep"
"${NCU}" \
    --force-overwrite \
    --target-processes all \
    --profile-from-start off \
    --set "${SET}" \
    --kernel-name "${KERNEL_NAME}" \
    --launch-skip "${LAUNCH_SKIP}" \
    --launch-count "${LAUNCH_COUNT}" \
    -o "${OUT}" \
    "${PYTHON}" "${TARGET}" "${target_args[@]}" "$@"
