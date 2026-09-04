import inspect
import re

from triton.experimental.gluon import language as gl


TMA_DTYPE_DEVICE_TO_HOST = {i: i for i in range(16)}
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9


def is_constexpr_param(param):
    ann = param.annotation
    if ann is inspect._empty:
        return False
    return (
        ann is gl.constexpr
        or getattr(ann, "__name__", "") == "constexpr"
        or "constexpr" in str(ann)
    )


def tensor_param_name(name):
    if name.endswith("_desc"):
        return name[: -len("_desc")]
    return f"{name}_tensor"


def cpp_scalar_type(sig):
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
    }[sig]


def arg_index(key):
    return key[0] if isinstance(key, tuple) else key


def tensordesc_rank(sig, fallback_rank):
    match = re.match(r"tensordesc<([^[>]*)\[([^]]*)\]", sig)
    if match is None:
        return fallback_rank
    return match.group(2).count(",") + 1


def generate_desc_meta_by_name(compiled_kernel, desc_names, abi_sig_by_name):
    desc_meta = compiled_kernel.metadata.tensordesc_meta
    assert len(desc_meta) == len(desc_names)

    desc_meta_by_name = {}
    for name, meta in zip(desc_names, desc_meta):
        block_size = list(meta["block_size"])
        rank = tensordesc_rank(abi_sig_by_name[name], len(block_size))
        desc_meta_by_name[name] = {
            "name": name,
            "rank": rank,
            "swizzle": int(meta["swizzle"]),
            "elem_size": int(meta["elem_size"]),
            "elem_type": TMA_DTYPE_DEVICE_TO_HOST[int(meta["elem_type"])],
            "block_size": [int(x) for x in block_size],
        }
    return desc_meta_by_name


def generate_tensordesc_kernel_fields(name, rank):
    fields = [f"    alignas(64) CUtensorMap {name};\n"]
    fields.extend(f"    uint32_t {name}_shape_{i};\n" for i in range(rank))
    fields.extend(f"    uint64_t {name}_stride_{i};\n" for i in range(rank))
    return "".join(fields)


def generate_tensordesc_kernel_param_ptrs(name, rank):
    params = [f"&kargs.{name}"]
    params.extend(f"&kargs.{name}_shape_{i}" for i in range(rank))
    params.extend(f"&kargs.{name}_stride_{i}" for i in range(rank))
    return params


def generate_scalar_kernel_field(name, sig):
    return f"    {cpp_scalar_type(sig)} {name};\n"


def generate_tensordesc_cpp_params(name, rank):
    tensor_name = tensor_param_name(name)
    params = [f"TensorView {tensor_name}"]
    params.extend(f"int32_t {name}_shape_{i}" for i in range(rank))
    params.extend(f"int64_t {name}_stride_{i}" for i in range(rank))
    return params


def generate_tensordesc_assignment(meta):
    name = meta["name"]
    rank = meta["rank"]
    tensor_name = tensor_param_name(name)

    shape_values = ", ".join(
        f"static_cast<uint64_t>({name}_shape_{i})" for i in range(rank)
    )
    stride_values = ", ".join(
        f"static_cast<uint64_t>({name}_stride_{i})" for i in range(rank)
    )
    block_values = ", ".join(str(x) for x in meta["block_size"])
    shape_assignments = "".join(
        f"    kargs.{name}_shape_{i} = static_cast<uint32_t>({name}_shape_{i});\n"
        for i in range(rank)
    )
    stride_assignments = "".join(
        f"    kargs.{name}_stride_{i} = static_cast<uint64_t>({name}_stride_{i});\n"
        for i in range(rank)
    )

    return f"""
    uint64_t {name}_shape[{rank}] = {{{shape_values}}};
    uint64_t {name}_strides[{rank}] = {{{stride_values}}};
    uint32_t {name}_block_size[{rank}] = {{{block_values}}};
    EncodeTmaDescriptor(
        &kargs.{name},
        {tensor_name}.data_ptr(),
        {meta["swizzle"]},
        {meta["elem_size"]},
        {meta["elem_type"]},
        {rank},
        {name}_block_size,
        {name}_shape,
        {name}_strides);
{shape_assignments}{stride_assignments}"""


def generate_scalar_assignment(name, sig):
    return f"    kargs.{name} = static_cast<{cpp_scalar_type(sig)}>({name});\n"


def flatten_tvm_ffi_args(compiled_kernel, kernel_kwargs, grid):
    signature = compiled_kernel.src.fn.signature
    flat_args = []

    for name, param in signature.parameters.items():
        if is_constexpr_param(param):
            continue

        value = kernel_kwargs[name]
        if "desc" in name:
            flat_args.append(value.base)
            flat_args.extend(int(x) for x in value.shape)
            flat_args.extend(int(x) for x in value.strides)
        else:
            flat_args.append(int(value))

    flat_args.extend(
        [
            int(grid[0]),
            int(grid[1]) if len(grid) > 1 else 1,
            int(grid[2]) if len(grid) > 2 else 1,
        ]
    )
    return flat_args


# Triton 3.6 Gluon
def generate_tvm_ffi_source(compiled_kernel, kernel_name, debug=False):
    if debug:
        print(compiled_kernel.metadata)
        print(compiled_kernel.asm["ptx"].split(".entry", 1)[1].split(")", 1)[0])

    arg_names = compiled_kernel.src.fn.arg_names
    signature = compiled_kernel.src.fn.signature

    constants = {}
    constants_set = set()
    for key, val in compiled_kernel.src.constants.items():
        idx = arg_index(key)
        name = arg_names[idx]
        constants[name] = (idx, val)
        constants_set.add(name)

    abi_sig_by_name = {}
    for key, sig in compiled_kernel.src.signature.items():
        abi_sig_by_name[key] = sig

    desc_names = [
        name
        for name, param in signature.parameters.items()
        if not is_constexpr_param(param) and "desc" in name
    ]
    desc_meta_by_name = generate_desc_meta_by_name(
        compiled_kernel, desc_names, abi_sig_by_name
    )

    if debug:
        print("arg_names : ", arg_names)
        print("constants : ", constants)
        print("abi_sig_by_name : ", abi_sig_by_name)

    cpp_params = []
    launch_args_def = []
    launch_args = []
    kernel_param_ptrs = []
    first_tensor_param = None

    for name, param in signature.parameters.items():
        # const param should not be added into cpp_params
        if is_constexpr_param(param):
            continue

        sig = abi_sig_by_name[name]
        if "desc" in name:
            meta = desc_meta_by_name[name]
            cpp_params.extend(generate_tensordesc_cpp_params(name, meta["rank"]))
            if name in constants_set:
                continue
            launch_args_def.append(generate_tensordesc_kernel_fields(name, meta["rank"]))
            launch_args.append(generate_tensordesc_assignment(meta))
            kernel_param_ptrs.extend(
                generate_tensordesc_kernel_param_ptrs(name, meta["rank"])
            )
            if first_tensor_param is None:
                first_tensor_param = tensor_param_name(name)
        else:
            cpp_params.append(f"{cpp_scalar_type(sig)} {name}")
            if name in constants_set:
                continue
            launch_args_def.append(generate_scalar_kernel_field(name, sig))
            launch_args.append(generate_scalar_assignment(name, sig))
            kernel_param_ptrs.append(f"&kargs.{name}")

    META = compiled_kernel.metadata
    global_scratch_size = getattr(META, "global_scratch_size", 0) or 0
    global_scratch_align = getattr(META, "global_scratch_align", 1) or 1
    profile_scratch_size = getattr(META, "profile_scratch_size", 0) or 0
    profile_scratch_align = getattr(META, "profile_scratch_align", 1) or 1
    if global_scratch_size != 0:
        raise NotImplementedError(
            f"Gluon TVM-FFI launcher does not allocate global scratch yet "
            f"(size={global_scratch_size}, align={global_scratch_align})."
        )
    if profile_scratch_size != 0:
        raise NotImplementedError(
            f"Gluon TVM-FFI launcher does not allocate profile scratch yet "
            f"(size={profile_scratch_size}, align={profile_scratch_align})."
        )

    launch_args_def.append("    CUdeviceptr global_scratch;\n")
    launch_args.append("    kargs.global_scratch = 0;\n")
    kernel_param_ptrs.append("&kargs.global_scratch")
    launch_args_def.append("    CUdeviceptr profile_scratch;\n")
    launch_args.append("    kargs.profile_scratch = 0;\n")
    kernel_param_ptrs.append("&kargs.profile_scratch")

    cpp_params_str = ",\n    ".join(
        cpp_params + ["int32_t grid_x", "int32_t grid_y", "int32_t grid_z"]
    )
    launch_args_def_str = "".join(launch_args_def)
    launch_args_str = "".join(launch_args)
    kernel_param_ptrs_str = ",\n        ".join(kernel_param_ptrs)

    num_warps = META.num_warps
    warp_size = META.target.warp_size
    shared_mem_size = META.shared

    source = f"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>


#define CHECK_CUDA_DRIVER_ERROR(call) \\
    do {{ \\
        CUresult err = call; \\
        if (err != CUDA_SUCCESS) {{ \\
            const char *err_name = nullptr; \\
            const char *err_str = nullptr; \\
            cuGetErrorName(err, &err_name); \\
            cuGetErrorString(err, &err_str); \\
            fprintf(stderr, "CUDA Driver Error at %s:%d: %s (%s)\\n", \\
                    __FILE__, __LINE__, err_name, err_str); \\
            fprintf(stderr, "Error code: %d\\n", err); \\
            cuCtxSynchronize(); \\
            throw std::runtime_error(std::string(err_name) + ", " + err_str); \\
        }} \\
    }} while(0)


#define CHECK_CUDA_RUNTIME_ERROR(call) \\
    do {{ \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) {{ \\
            const char *err_name = cudaGetErrorName(err); \\
            const char *err_str = cudaGetErrorString(err); \\
            fprintf(stderr, "CUDA Runtime Error at %s:%d: %s (%s)\\n", \\
                    __FILE__, __LINE__, err_name, err_str); \\
            fprintf(stderr, "Error code: %d\\n", err); \\
            cudaDeviceSynchronize(); \\
            throw std::runtime_error(std::string(err_name) + ", " + err_str); \\
        }} \\
    }} while(0)


TVM_FFI_EMBED_CUBIN(triton_cubin);

namespace triton_gluon_loader {{
using namespace tvm::ffi;

struct KernelArgs {{
{launch_args_def_str}}};

static void EncodeTmaDescriptor(
    CUtensorMap* desc,
    void* base_ptr,
    int swizzle,
    int elem_size,
    int elem_type,
    int rank,
    const uint32_t* block_size,
    const uint64_t* shape,
    const uint64_t* strides) {{
    if (strides[rank - 1] != 1) {{
        throw std::runtime_error("TMA descriptor expects the innermost stride to be 1.");
    }}

    uint64_t global_dim[5] = {{0, 0, 0, 0, 0}};
    uint64_t global_strides[5] = {{0, 0, 0, 0, 0}};
    uint32_t box_dim[5] = {{1, 1, 1, 1, 1}};
    uint32_t element_strides[5] = {{1, 1, 1, 1, 1}};

    for (int i = 0; i < rank; ++i) {{
        global_dim[rank - i - 1] = shape[i];
        box_dim[rank - i - 1] = block_size[i];
    }}
    for (int i = 0; i + 1 < rank; ++i) {{
        global_strides[rank - i - 2] =
            static_cast<uint64_t>(elem_size) * strides[i];
    }}
    global_strides[rank - 1] =
        global_dim[rank - 1] *
        (rank == 1 ? static_cast<uint64_t>(elem_size)
                   : global_strides[rank - 2]);

    CUresult result = cuTensorMapEncodeTiled(
        desc,
        static_cast<CUtensorMapDataType>(elem_type),
        static_cast<cuuint32_t>(rank),
        base_ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        static_cast<CUtensorMapSwizzle>(swizzle),
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CHECK_CUDA_DRIVER_ERROR(result);
}}

void {kernel_name}_launcher(
    {cpp_params_str}) {{
    static auto launcher = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_cubin, "{kernel_name}");

    KernelArgs kargs;

{launch_args_str}
    DLDevice device = {first_tensor_param}.device();

    int shared_mem_bytes = {shared_mem_size};
    shared_mem_bytes = (shared_mem_bytes + 7) & ~7;

    int max_shared_mem_bytes;
    CUresult get_attr_result = cuDeviceGetAttribute(
        &max_shared_mem_bytes,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device.device_id);
    CHECK_CUDA_DRIVER_ERROR(get_attr_result);

    if (shared_mem_bytes > max_shared_mem_bytes) {{
        max_shared_mem_bytes = shared_mem_bytes;
    }}

    CUfunction kernel = *(reinterpret_cast<CUfunction*>(&launcher));

    cudaError_t set_attr_result = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_shared_mem_bytes);
    CHECK_CUDA_RUNTIME_ERROR(set_attr_result);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(device.device_type, device.device_id));

    tvm::ffi::dim3 grid(
        static_cast<unsigned int>(grid_x),
        static_cast<unsigned int>(grid_y),
        static_cast<unsigned int>(grid_z));
    tvm::ffi::dim3 block({num_warps * warp_size}, 1, 1);

    void* params[] = {{
        {kernel_param_ptrs_str}
    }};

    CUresult result = cuLaunchKernel(
        kernel,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        static_cast<unsigned int>(shared_mem_bytes),
        stream,
        params,
        nullptr);
    if (result != CUDA_SUCCESS) {{
        CHECK_CUDA_DRIVER_ERROR(result);
    }}
}}

}} // namespace triton_gluon_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    {kernel_name},
    triton_gluon_loader::{kernel_name}_launcher);
"""
    return source, constants
