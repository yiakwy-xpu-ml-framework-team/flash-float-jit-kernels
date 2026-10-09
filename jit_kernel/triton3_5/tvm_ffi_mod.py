import inspect

import triton.language as tl


def is_constexpr_param(param):
    ann = param.annotation
    if ann is inspect._empty:
        return False
    return (
        ann is tl.constexpr
        or getattr(ann, "__name__", "") == "constexpr"
        or "constexpr" in str(ann)
    )


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


# Triton 3.5
def generate_tvm_ffi_source(compiled_kernel, kernel_name, debug=False):
    if debug:
        print(compiled_kernel.metadata)
        print(compiled_kernel.asm["ptx"].split(".entry", 1)[1].split(")", 1)[0])

    arg_names = compiled_kernel.src.fn.arg_names
    signature = compiled_kernel.src.fn.signature

    # TVM-FFI args list
    cpp_params = []

    # cuLaunchKernel arguemnt definition
    launch_args_def = []
    # cuLaunchKernel argsument assignment
    launch_args = []

    constants = {}
    constants_set = set()
    for key, val in compiled_kernel.src.constants.items():
        name = arg_names[key[0]]

        constants[name] = (key[0], val)
        constants_set.add(name)

    if debug:
        print("arg_names : ", arg_names)
        print("constants : ", constants)

    for name, param in signature.parameters.items():
        # const param should not be added into cpp_params
        if is_constexpr_param(param):
            continue

        cpp_params.append(f"{cpp_host_type(name)} {name}")

        # const in a compiled kernel should not be added into launch_args_def and launch_args
        if name not in constants_set:
            launch_args_def.append(f"{kernel_arg_type(name)} {name};\n")
            launch_args.append(kernel_arg_assignment(name))

    cpp_params_str = ", ".join(
        cpp_params + ["int32_t grid_x", "int32_t grid_y", "int32_t grid_z"]
    )

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

    launch_args_def.append("CUdeviceptr global_scratch;\n")
    launch_args.append("kargs.global_scratch = 0;\n")
    launch_args_def.append("CUdeviceptr profile_scratch;\n")
    launch_args.append("kargs.profile_scratch = 0;\n")

    launch_args_def_str = "".join(launch_args_def)
    launch_args_str = "".join(launch_args)

    if debug:
        print(f"cpp_params_str : {cpp_params_str}")
        print(f"launch_args_def_str : {launch_args_def_str}")
        print(f"launch_args_str : {launch_args_str}")

    num_warps = META.num_warps  # compiled_kernel.num_warps
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


// NOTE (yiakwy) : for CUresult
#define CHECK_CUDA_DRIVER_ERROR(call) \
    do {{ \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) {{ \
            const char *err_name, *err_str; \
            cuGetErrorName(err, &err_name); \
            cuGetErrorString(err, &err_str); \
            fprintf(stderr, "CUDA Driver Error at %s:%d: %s (%s)\\n", \
                    __FILE__, __LINE__, err_name, err_str); \
            fprintf(stderr, "Error code: %d\\n", err); \
            cuCtxSynchronize(); \
            throw std::runtime_error(std::string(err_name) + ", " + err_str); \
        }} \
    }} while(0)


#define CHECK_CUDA_RUNTIME_ERROR(call) \
    do {{ \
        cudaError_t err = call; \
        if (err != cudaSuccess) {{ \
            const char *err_name, *err_str; \
            err_name = cudaGetErrorName(err); \
            err_str = cudaGetErrorString(err); \
            fprintf(stderr, "CUDA Runtime Error at %s:%d: %s (%s)\\n", \
                    __FILE__, __LINE__, err_name, err_str); \
            fprintf(stderr, "Error code: %d\\n", err); \
            cudaDeviceSynchronize(); \
            throw std::runtime_error(std::string(err_name) + ", " + err_str); \
        }} \
    }} while(0)


TVM_FFI_EMBED_CUBIN(triton_cubin);

namespace triton_loader {{
using namespace tvm::ffi;

// NOTE (yiakwy) : TVM's official method does not handle alignment issue, hence I use this method to escape alignment problem.
// We can also consider to modify TVM's official method to support alignment in the future if necessary.
struct KernelArgs {{
    {launch_args_def_str};
}};

void {kernel_name}_launcher({cpp_params_str}) {{
    static auto launcher = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_cubin, "{kernel_name}");

    // construct Triton arguments list
    KernelArgs kargs;

    {launch_args_str};

    DLDevice device = {arg_names[0]}.device();

    int shared_mem_bytes = {shared_mem_size};
    shared_mem_bytes = (shared_mem_bytes + 7) & ~7;

    // NOTE (yiakwy) : use CUDA driver api
    int max_shared_mem_bytes;
    CUresult get_attr_result = cuDeviceGetAttribute(&max_shared_mem_bytes, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device.device_id);
    CHECK_CUDA_DRIVER_ERROR(get_attr_result);

    // NOTE (yiakwy) : Hopper allows to set maximum share memory >= 483232 bytes
    if (shared_mem_bytes > max_shared_mem_bytes) {{
        max_shared_mem_bytes = shared_mem_bytes;
    }}

    CUfunction kernel = *(reinterpret_cast<CUfunction*>(&launcher));

    // NOTE (yiakwy) : use cuda runtime api
    cudaError_t set_attr_result = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_bytes);
    CHECK_CUDA_RUNTIME_ERROR(set_attr_result);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

    tvm::ffi::dim3 grid((unsigned int)grid_x, (unsigned int)grid_y, (unsigned int)grid_z);
    tvm::ffi::dim3 block({num_warps * WARP_SIZE}, 1, 1);

    size_t kargs_size = sizeof(KernelArgs);
    void* config[] = {{
        CU_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &kargs_size,
        CU_LAUNCH_PARAM_END
    }};

    CUresult result = cuLaunchKernel(
          kernel,
          grid.x, grid.y, grid.z,
          block.x, block.y, block.z,
          (unsigned int)shared_mem_bytes, stream, nullptr, config);

    if (result != CUDA_SUCCESS) {{
        CHECK_CUDA_DRIVER_ERROR(result);
    }}
}}

}} // namespace triton_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC({kernel_name}, triton_loader::{kernel_name}_launcher);
"""
    return source, constants
