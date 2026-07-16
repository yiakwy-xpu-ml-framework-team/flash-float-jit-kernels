import inspect

import triton
import triton.language as tl


def is_constexpr_param(param):
    ann = param.annotation
    if ann is inspect._empty:
        return False
    return ann is tl.constexpr or getattr(ann, "__name__", "") == "constexpr" or "constexpr" in str(ann)


def cpp_host_type(name: str) -> str:
    if "ptr" in name:
        return "TensorView"
    if "use_" in name or "is_" in name:
        return "bool"
    return "int32_t"


def kernel_arg_type(name: str) -> str:
    if "ptr" in name:
        return "void*"
    if "use_" in name or "is_" in name:
        return "bool"
    return "int32_t"


def kernel_arg_assignment(name: str) -> str:
    if "ptr" in name:
        return f"kargs.{name} = {name}.data_ptr();\n"
    return f"kargs.{name} = {name};\n"


def runtime_arg_decl(name: str) -> str:
    var = f"{name}_arg"
    if "ptr" in name:
        return f"void* {var} = {name}.data_ptr();\n"
    if "use_" in name or "is_" in name:
        return f"bool {var} = {name};\n"
    return f"int32_t {var} = {name};\n"


def generate_tvm_ffi_source(compiled_kernel, kernel_name, debug=False):
    """Generate a TVM-FFI CUBIN launcher for Triton 3.5 NVIDIA kernels.

    Triton 3.5 appends both global_scratch and profile_scratch hidden kernel
    arguments. This launcher currently supports the common zero-scratch case.
    """
    if debug:
        print(compiled_kernel.metadata)
        print(compiled_kernel.asm["ptx"].split(".entry", 1)[1].split(")", 1)[0])

    arg_names = compiled_kernel.src.fn.arg_names
    signature = compiled_kernel.src.fn.signature

    constants = {}
    constants_set = set()
    for key, val in compiled_kernel.src.constants.items():
        name = arg_names[key[0]]
        constants[name] = (key[0], val)
        constants_set.add(name)

    if debug:
        print("arg_names : ", arg_names)
        print("constants : ", constants)

    cpp_arg_names = []
    launch_arg_names = []
    for name, param in signature.parameters.items():
        if is_constexpr_param(param):
            continue

        cpp_arg_names.append(name)
        if name not in constants_set:
            launch_arg_names.append(name)

    cpp_params = [f"{cpp_host_type(name)} {name}" for name in cpp_arg_names]
    cpp_params_str = ", ".join(cpp_params + ["int32_t grid_x", "int32_t grid_y", "int32_t grid_z"])

    launch_args_def = [f"{kernel_arg_type(name)} {name};\n" for name in launch_arg_names]
    launch_args = [kernel_arg_assignment(name) for name in launch_arg_names]

    launch_args_def.append("CUdeviceptr global_scratch;\n")
    launch_args.append("kargs.global_scratch = 0;\n")
    launch_args_def.append("CUdeviceptr profile_scratch;\n")
    launch_args.append("kargs.profile_scratch = 0;\n")

    launch_args_def_str = "".join(launch_args_def)
    launch_args_str = "".join(launch_args)

    runtime_arg_decls = [runtime_arg_decl(name) for name in launch_arg_names]
    runtime_arg_ptrs = [f"&{name}_arg" for name in launch_arg_names]

    runtime_arg_decls.append("CUdeviceptr global_scratch_arg = 0;\n")
    runtime_arg_ptrs.append("&global_scratch_arg")
    runtime_arg_decls.append("CUdeviceptr profile_scratch_arg = 0;\n")
    runtime_arg_ptrs.append("&profile_scratch_arg")

    runtime_arg_decls_str = "".join(runtime_arg_decls)
    runtime_arg_ptrs_str = ",\n        ".join(runtime_arg_ptrs)

    if debug:
        print(f"cpp_params_str : {cpp_params_str}")
        print(f"launch_args_def_str : {launch_args_def_str}")
        print(f"launch_args_str : {launch_args_str}")

    META = compiled_kernel.metadata

    global_scratch_size = getattr(META, "global_scratch_size", 0) or 0
    global_scratch_align = getattr(META, "global_scratch_align", 1) or 1
    profile_scratch_size = getattr(META, "profile_scratch_size", 0) or 0
    profile_scratch_align = getattr(META, "profile_scratch_align", 1) or 1
    if global_scratch_size != 0:
        raise NotImplementedError(
            f"Triton 3.5 TVM-FFI launcher does not allocate global scratch yet "
            f"(size={global_scratch_size}, align={global_scratch_align})."
        )
    if profile_scratch_size != 0:
        raise NotImplementedError(
            f"Triton 3.5 TVM-FFI launcher does not allocate profile scratch yet "
            f"(size={profile_scratch_size}, align={profile_scratch_align})."
        )

    num_warps = META.num_warps
    shared_mem_size = META.shared

    WARP_SIZE = META.target.warp_size

    source = f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define USE_TVM_FFI_LAUNCH_CONVENTION 0

TVM_FFI_EMBED_CUBIN(triton_cubin);

namespace triton_loader {{
using namespace tvm::ffi;

struct KernelArgs {{
    {launch_args_def_str};
}};

void {kernel_name}_launcher({cpp_params_str}) {{
    static auto launcher = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_cubin, "{kernel_name}");

    KernelArgs kargs;
    {launch_args_str};

    {runtime_arg_decls_str}
    void* args[] = {{
        {runtime_arg_ptrs_str}
    }};

    cudaGetLastError();

    int shared_mem_bytes = {shared_mem_size};
    auto kernel = launcher.GetHandle();

    DLDevice device = {arg_names[0]}.device();
    auto raw_stream = TVMFFIEnvGetStream(device.device_type, device.device_id);

    auto device_handle =
        ::tvm::ffi::cuda_api::GetDeviceHandle(device.device_id);

    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
        ::tvm::ffi::cuda_api::SetKernelMaxDynamicSharedMem(
            kernel,
            shared_mem_bytes,
            device_handle)
    );

    tvm::ffi::dim3 grid((unsigned int)grid_x, (unsigned int)grid_y, (unsigned int)grid_z);
    tvm::ffi::dim3 block({num_warps * WARP_SIZE}, 1, 1);

#if USE_TVM_FFI_LAUNCH_CONVENTION
    ::tvm::ffi::cuda_api::StreamHandle stream =
        static_cast<::tvm::ffi::cuda_api::StreamHandle>(raw_stream);
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(launcher.Launch(args, grid, block, stream, shared_mem_bytes));
#else
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
    CUstream stream = static_cast<CUstream>(raw_stream);
    size_t kargs_size = sizeof(KernelArgs);
    void* config[] = {{
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &kargs_size,
        CU_LAUNCH_PARAM_END
    }};

    CUresult result = cuLaunchKernel(
        reinterpret_cast<CUfunction>(kernel),
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        (unsigned int)shared_mem_bytes,
        stream,
        nullptr,
        config);

    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
#else
    cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    cudaError_t result = cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel),
        ::dim3(grid.x, grid.y, grid.z),
        ::dim3(block.x, block.y, block.z),
        args,
        (size_t)shared_mem_bytes,
        stream);

    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
#endif
#endif
}}

}} // namespace triton_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC({kernel_name}, triton_loader::{kernel_name}_launcher);
"""
    return source, constants
