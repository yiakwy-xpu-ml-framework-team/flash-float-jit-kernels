# ThunderMoun CUDA Path Nsight Targets

This folder contains external-profiler targets for the ThunderMoun CUDA path.
The default target runs the normal non-embedded-profiler path so Nsight Systems
and Nsight Compute can focus on `hopper_symm_gemm_kernel_entry`.

## Quick Start

```bash
bash profiling/cuda_path/run_nsys.sh
bash profiling/cuda_path/run_ncu.sh
```

The scripts write reports under:

```text
profiling/cuda_path/results/nsys
profiling/cuda_path/results/ncu
```

Open the reports with:

```bash
nsys-ui profiling/cuda_path/results/nsys/<report>.nsys-rep
ncu-ui profiling/cuda_path/results/ncu/<report>.ncu-rep
```

## Common Parameters

Use environment variables to change the default run:

```bash
M=4096 WARMUP=5 ITERS=50 bash profiling/cuda_path/run_nsys.sh
M=4096 ITERS=1 SET=full bash profiling/cuda_path/run_ncu.sh
```

To profile the embedded-profiler route as well:

```bash
PROFILED=1 bash profiling/cuda_path/run_nsys.sh
PROFILED=1 bash profiling/cuda_path/run_ncu.sh
```

When the embedded profiler is enabled, the decoded event summary and Chrome
trace describe the final profiled launch. Use `ITERS=1` when you want the timing
loop and embedded trace to refer to exactly the same launch. Profiled timing also
includes the profiler buffer reset/header copy that happens before each kernel.

The target preallocates the output tensor by default. This keeps PyTorch tensor
allocation and zero-fill kernels out of the measured loop and makes the Nsight
reports cleaner for kernel analysis.

## Direct Target Run

```bash
python profiling/cuda_path/target_symm_gemm_cuda.py --m 4096 --warmup 5 --iters 20
python profiling/cuda_path/target_symm_gemm_cuda.py --m 4096 --warmup 5 --iters 20 --profiled
```

The `--cuda-profiler-api` flag is used by the Nsight scripts. It calls
`cudaProfilerStart` after warmup and `cudaProfilerStop` after the measured loop,
so the profiler capture range excludes setup and warmup work.
