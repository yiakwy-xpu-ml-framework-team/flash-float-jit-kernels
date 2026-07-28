#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TARGET="${SCRIPT_DIR}/target_symm_gemm_cuda.py"
OUT_DIR="${SCRIPT_DIR}/results/ncu"

PYTHON="${PYTHON:-python}"
NCU="${NCU:-ncu}"
NCU_USE_SUDO="${NCU_USE_SUDO:-0}"
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

ncu_cmd=("${NCU}")
if [[ "${NCU_USE_SUDO}" == "1" ]]; then
    sudo_env=("PATH=${PATH}" "HOME=${HOME}")
    [[ -n "${VIRTUAL_ENV:-}" ]] && sudo_env+=("VIRTUAL_ENV=${VIRTUAL_ENV}")
    [[ -n "${PYTHONPATH:-}" ]] && sudo_env+=("PYTHONPATH=${PYTHONPATH}")
    [[ -n "${LD_LIBRARY_PATH:-}" ]] && sudo_env+=("LD_LIBRARY_PATH=${LD_LIBRARY_PATH}")
    [[ -n "${CUDA_HOME:-}" ]] && sudo_env+=("CUDA_HOME=${CUDA_HOME}")
    [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]] && sudo_env+=("TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}")
    ncu_cmd=(sudo env "${sudo_env[@]}" "${NCU}")
fi

echo "[ncu] prewarming JIT/cache outside Nsight Compute..."
"${PYTHON}" "${TARGET}" "${prewarm_args[@]}"

echo "[ncu] writing ${OUT}.ncu-rep"
"${ncu_cmd[@]}" \
    --force-overwrite \
    --target-processes all \
    --profile-from-start off \
    --set "${SET}" \
    --kernel-name "${KERNEL_NAME}" \
    --launch-skip "${LAUNCH_SKIP}" \
    --launch-count "${LAUNCH_COUNT}" \
    -o "${OUT}" \
    "${PYTHON}" "${TARGET}" "${target_args[@]}" "$@"
