#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${SCRIPT_DIR}/target_symm_gemm_cuda.py"
OUT_DIR="${SCRIPT_DIR}/results/nsys"

PYTHON="${PYTHON:-python}"
NSYS="${NSYS:-nsys}"
M="${M:-4096}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
DEVICE="${DEVICE:-cuda}"
PROFILED="${PROFILED:-0}"
NSYS_STATS="${NSYS_STATS:-1}"
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

echo "[nsys] prewarming JIT/cache outside Nsight..."
"${PYTHON}" "${TARGET}" "${prewarm_args[@]}"

echo "[nsys] writing ${OUT}.nsys-rep"
"${NSYS}" profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    -o "${OUT}" \
    "${PYTHON}" "${TARGET}" "${target_args[@]}" "$@"

if [[ "${NSYS_STATS}" == "1" ]]; then
    echo "[nsys] summary stats"
    "${NSYS}" stats "${OUT}.nsys-rep" || true
fi
