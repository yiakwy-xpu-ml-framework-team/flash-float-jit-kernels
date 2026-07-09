"""Run a small default CUPTI profile.

From the repository root:

    python benchmark/profiling/init.py
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import cupti_profile


DEFAULT_ARGS = [
    "--m",
    "4096",
    "--scenarios",
    "cuda_warm,triton_symm_native_warm,triton_moun_native_warm",
    "--warmup",
    "10",
    "--iters",
    "5",
    "--check",
    "--range-counters",
]


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_ARGS)
    else:
        sys.argv[1:1] = DEFAULT_ARGS
    cupti_profile.main()


if __name__ == "__main__":
    main()
