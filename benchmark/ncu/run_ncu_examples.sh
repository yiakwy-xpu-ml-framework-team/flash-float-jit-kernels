#!/usr/bin/env bash
# Example Nsight Compute commands for the symmetric GEMM providers.
#
# Run from the repository root on the CUDA VM:
#
#   bash benchmark/ncu/run_ncu_examples.sh
#
# The target script calls cudaProfilerStart/Stop, so these commands use
# --profile-from-start off to keep warmup and Triton compilation out of the
# profiled region.

set -euo pipefail

M="${M:-4096}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-1}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-benchmark/ncu/reports}"
NCU="${NCU:-ncu}"
NCU_SUDO="${NCU_SUDO:-auto}"
EXPORT_CSV="${EXPORT_CSV:-1}"

should_reexec_with_sudo() {
  if [[ "${NCU_SUDO}" == "0" || "${NCU_SUDO}" == "false" ]]; then
    return 1
  fi
  if [[ "${EUID}" -eq 0 || "${FFJK_NCU_SUDO_REEXEC:-0}" == "1" ]]; then
    return 1
  fi
  if [[ "${NCU_SUDO}" == "1" || "${NCU_SUDO}" == "true" ]]; then
    return 0
  fi
  [[ -r /proc/driver/nvidia/params ]] &&
    grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params
}

if should_reexec_with_sudo; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Nsight Compute counters appear to require admin privileges, but sudo was not found." >&2
  else
    CUDA_HOME_VALUE="${CUDA_HOME:-}"
    SUDO_PATH="${PWD}/.venv/bin:${PATH}"
    SUDO_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    if [[ -n "${CUDA_HOME_VALUE}" ]]; then
      SUDO_PATH="${PWD}/.venv/bin:${CUDA_HOME_VALUE}/bin:${PATH}"
      SUDO_LD_LIBRARY_PATH="${CUDA_HOME_VALUE}/lib64:${CUDA_HOME_VALUE}/extras/CUPTI/lib64:${SUDO_LD_LIBRARY_PATH}"
    fi

    echo "Nsight Compute counters require admin privileges; re-running with sudo."
    exec sudo -E env \
      PATH="${SUDO_PATH}" \
      CUDA_HOME="${CUDA_HOME_VALUE}" \
      LD_LIBRARY_PATH="${SUDO_LD_LIBRARY_PATH}" \
      FFJK_NCU_SUDO_REEXEC=1 \
      bash "$0" "$@"
  fi
fi

COMMON_NCU_ARGS=(
  --set detailed
  --target-processes all
  --profile-from-start off
  --launch-count 1
  --kernel-name-base demangled
  --force-overwrite
)

run_one() {
  local scenario="$1"
  local output_name="$2"
  mkdir -p "${OUT_DIR}"
  "${NCU}" "${COMMON_NCU_ARGS[@]}" \
    -o "${OUT_DIR}/${output_name}" \
    "${PYTHON}" benchmark/ncu/ncu_target.py \
      --scenario "${scenario}" \
      --m "${M}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --poison-output
}

export_report_csv() {
  local output_name="$1"
  local report_path="${OUT_DIR}/${output_name}.ncu-rep"
  if [[ "${EXPORT_CSV}" != "1" && "${EXPORT_CSV}" != "true" ]]; then
    return 0
  fi
  if [[ ! -f "${report_path}" ]]; then
    echo "Skipping CSV export; report not found: ${report_path}" >&2
    return 0
  fi

  echo "Exporting Nsight Compute CSV pages for ${output_name}."
  "${NCU}" -i "${report_path}" --page raw --csv > "${OUT_DIR}/${output_name}_raw.csv" ||
    echo "Failed to export raw page for ${report_path}" >&2
  "${NCU}" -i "${report_path}" --page source --csv > "${OUT_DIR}/${output_name}_source.csv" ||
    echo "Failed to export source page for ${report_path}" >&2
}

restore_output_ownership() {
  if [[ -n "${SUDO_UID:-}" && -n "${SUDO_GID:-}" && -d "${OUT_DIR}" ]]; then
    chown -R "${SUDO_UID}:${SUDO_GID}" "${OUT_DIR}" ||
      echo "Could not restore ownership for ${OUT_DIR}" >&2
  fi
}

run_one cuda_warm "cuda_m${M}"
run_one triton_symm_native_warm "triton_symm_m${M}"
export_report_csv "cuda_m${M}"
export_report_csv "triton_symm_m${M}"
restore_output_ownership

cat <<EOF

Reports:
  ${OUT_DIR}/cuda_m${M}.ncu-rep
  ${OUT_DIR}/triton_symm_m${M}.ncu-rep

Raw/source CSV exports:
  ${NCU} -i ${OUT_DIR}/cuda_m${M}.ncu-rep --page raw --csv > ${OUT_DIR}/cuda_m${M}_raw.csv
  ${NCU} -i ${OUT_DIR}/cuda_m${M}.ncu-rep --page source --csv > ${OUT_DIR}/cuda_m${M}_source.csv
  ${NCU} -i ${OUT_DIR}/triton_symm_m${M}.ncu-rep --page raw --csv > ${OUT_DIR}/triton_symm_m${M}_raw.csv
  ${NCU} -i ${OUT_DIR}/triton_symm_m${M}.ncu-rep --page source --csv > ${OUT_DIR}/triton_symm_m${M}_source.csv

CSV export is enabled by default. To skip it:

  EXPORT_CSV=0 bash benchmark/ncu/run_ncu_examples.sh

If NVIDIA performance counters require admin privileges, this script attempts to
re-run itself with sudo while preserving the virtualenv and CUDA paths. To force
that behavior, run:

  NCU_SUDO=1 bash benchmark/ncu/run_ncu_examples.sh

To disable automatic sudo re-exec, run:

  NCU_SUDO=0 bash benchmark/ncu/run_ncu_examples.sh
EOF
