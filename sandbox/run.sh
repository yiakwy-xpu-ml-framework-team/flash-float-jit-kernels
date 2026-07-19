#!/bin/bash
# Quick launcher for flash-float-jit-kernels sbx sandbox.
# Usage:
#   ./sandbox/run.sh                          # Launch opencode in sandbox
#   ./sandbox/run.sh --name my-exp            # Named sandbox
#   ./sandbox/run.sh --clone                  # Clone mode (safer)
#   ./sandbox/run.sh shell                    # Shell inside sandbox
#   ./sandbox/run.sh agent topk               # Run kernel agent on TopK
#   ./sandbox/run.sh agent gemm --dry         # Dry-run kernel agent on GEMM
#   ./sandbox/run.sh modal topk               # Run on Modal cloud GPU
#   ./sandbox/run.sh stop                     # Stop sandbox
#   ./sandbox/run.sh rm                       # Remove sandbox
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
KIT_PATH="$REPO_ROOT/sbx-kits/flash-kernel-kit"

SBX_NAME="${SBX_NAME:-flash-dev}"
MODE="opencode"
CLONE_FLAG=""

# ─── Arg parsing ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)     SBX_NAME="$2"; shift 2 ;;
        --clone)    CLONE_FLAG="--clone"; shift ;;
        stop)       MODE="stop"; shift ;;
        rm)         MODE="remove"; shift ;;
        shell)      MODE="shell"; shift ;;
        agent)      MODE="agent"; shift ;;
        modal)      MODE="modal"; shift ;;
        --dry)      DRY_RUN="--dry-run"; shift ;;
        topk)       KERNEL_FILE="jit_kernel/csrc/topk_indexer/topk_indexer_radix.cu"; shift ;;
        gemm)       KERNEL_FILE="jit_kernel/csrc/thunder_moun/symm_gemm.cu"; shift ;;
        tritongemm) KERNEL_FILE="jit_kernel/triton3_4/symm_gemm.py"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Default kernel for agent mode
KERNEL_FILE="${KERNEL_FILE:-jit_kernel/triton3_4/symm_gemm.py}"

echo "=== Flash Float sbx Sandbox ==="

case "$MODE" in
    opencode)
        echo "Launching opencode in sandbox..."
        echo "Kit: $KIT_PATH"
        echo "Name: $SBX_NAME"
        echo ""
        cd "$REPO_ROOT"
        sbx run $CLONE_FLAG opencode --kit "$KIT_PATH" --name "$SBX_NAME"
        ;;

    stop)
        echo "Stopping sandbox: $SBX_NAME"
        sbx stop "$SBX_NAME" 2>/dev/null || echo "  (not running)"
        ;;

    remove)
        echo "Removing sandbox: $SBX_NAME"
        sbx rm "$SBX_NAME" 2>/dev/null || echo "  (not found)"
        ;;

    shell)
        echo "Opening shell in sandbox: $SBX_NAME"
        sbx exec -it "$SBX_NAME" bash
        ;;

    agent)
        echo "Running kernel agent in sandbox..."
        echo "Kernel: $KERNEL_FILE"
        echo "Dry run: ${DRY_RUN:-no}"
        echo ""
        sbx exec "$SBX_NAME" -- python tools/kernel_agent.py \
            --kernel-file "$KERNEL_FILE" \
            --gpu h100 \
            --max-iterations 40 \
            --output-dir experiments/ \
            --no-git \
            ${DRY_RUN:-}
        ;;

    modal)
        echo "Running on Modal cloud GPU..."
        echo "Kernel: $KERNEL_FILE"
        echo ""
        python "$REPO_ROOT/tools/modal_sandbox.py" \
            --kernel-file "$KERNEL_FILE" \
            --gpu h100 \
            --max-iterations 20
        ;;
esac
