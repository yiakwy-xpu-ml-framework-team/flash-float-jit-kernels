#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <cupti_callbacks.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t kBufferSize = 8 * 1024 * 1024;
constexpr size_t kBufferAlign = 8;

struct ActivityRow {
    std::string record_type;
    std::string name;
    uint32_t activity_kind = 0;
    uint32_t cbid = 0;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint64_t queued_ns = 0;
    uint64_t submitted_ns = 0;
    uint32_t device_id = 0;
    uint32_t context_id = 0;
    uint32_t stream_id = 0;
    uint32_t correlation_id = 0;
    int64_t grid_id = 0;
    int32_t grid_x = 0;
    int32_t grid_y = 0;
    int32_t grid_z = 0;
    int32_t block_x = 0;
    int32_t block_y = 0;
    int32_t block_z = 0;
    uint32_t cluster_x = 0;
    uint32_t cluster_y = 0;
    uint32_t cluster_z = 0;
    int32_t static_smem = 0;
    int32_t dynamic_smem = 0;
    uint32_t shared_memory_executed = 0;
    uint16_t registers_per_thread = 0;
    uint64_t bytes = 0;
    uint32_t copy_kind = 0;
    uint32_t src_kind = 0;
    uint32_t dst_kind = 0;
    uint32_t memset_value = 0;
    uint32_t return_value = 0;
};

std::mutex g_mutex;
std::vector<ActivityRow> g_rows;
std::string g_last_error;
std::string g_output_path;
bool g_callbacks_registered = false;
bool g_active = false;

std::string csv_escape(const std::string& value) {
    bool needs_quotes = false;
    for (char c : value) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        return value;
    }
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

const char* cupti_error_name(CUptiResult result) {
    const char* name = nullptr;
    cuptiGetResultString(result, &name);
    return name ? name : "UNKNOWN_CUPTI_ERROR";
}

void set_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error = message;
}

int cupti_call(CUptiResult result, const char* expr) {
    if (result == CUPTI_SUCCESS) {
        return 0;
    }
    std::ostringstream oss;
    oss << expr << " failed: " << cupti_error_name(result);
    set_error(oss.str());
    return -1;
}

#define CUPTI_CHECK(expr)              \
    do {                               \
        if (cupti_call((expr), #expr)) \
            return -1;                 \
    } while (0)

void add_row(ActivityRow row) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_active) {
        g_rows.emplace_back(std::move(row));
    }
}

std::string callback_name(CUpti_ActivityKind kind, uint32_t cbid) {
    CUpti_CallbackDomain domain;
    if (kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        domain = CUPTI_CB_DOMAIN_RUNTIME_API;
    } else if (kind == CUPTI_ACTIVITY_KIND_DRIVER) {
        domain = CUPTI_CB_DOMAIN_DRIVER_API;
    } else {
        return "";
    }

    const char* name = nullptr;
    CUptiResult result = cuptiGetCallbackName(domain, cbid, &name);
    if (result == CUPTI_SUCCESS && name != nullptr) {
        return name;
    }
    return "";
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
    void* raw = std::aligned_alloc(kBufferAlign, kBufferSize);
    if (raw == nullptr) {
        *buffer = nullptr;
        *size = 0;
        *max_num_records = 0;
        return;
    }
    *buffer = static_cast<uint8_t*>(raw);
    *size = kBufferSize;
    *max_num_records = 0;
}

void handle_api_record(const CUpti_ActivityAPI* api) {
    ActivityRow row;
    row.record_type = (api->kind == CUPTI_ACTIVITY_KIND_RUNTIME) ? "runtime" :
                      (api->kind == CUPTI_ACTIVITY_KIND_DRIVER) ? "driver" :
                      "internal_launch_api";
    row.activity_kind = static_cast<uint32_t>(api->kind);
    row.cbid = static_cast<uint32_t>(api->cbid);
    row.name = callback_name(api->kind, row.cbid);
    row.start_ns = api->start;
    row.end_ns = api->end;
    row.correlation_id = api->correlationId;
    row.return_value = api->returnValue;
    add_row(std::move(row));
}

void handle_kernel_record(const CUpti_ActivityKernel9* kernel_record) {
    ActivityRow row;
    row.record_type = "kernel";
    row.activity_kind = static_cast<uint32_t>(kernel_record->kind);
    row.name = kernel_record->name ? kernel_record->name : "";
    row.start_ns = kernel_record->start;
    row.end_ns = kernel_record->end;
    row.queued_ns = kernel_record->queued;
    row.submitted_ns = kernel_record->submitted;
    row.device_id = kernel_record->deviceId;
    row.context_id = kernel_record->contextId;
    row.stream_id = kernel_record->streamId;
    row.correlation_id = kernel_record->correlationId;
    row.grid_id = kernel_record->gridId;
    row.grid_x = kernel_record->gridX;
    row.grid_y = kernel_record->gridY;
    row.grid_z = kernel_record->gridZ;
    row.block_x = kernel_record->blockX;
    row.block_y = kernel_record->blockY;
    row.block_z = kernel_record->blockZ;
    row.static_smem = kernel_record->staticSharedMemory;
    row.dynamic_smem = kernel_record->dynamicSharedMemory;
    row.registers_per_thread = kernel_record->registersPerThread;
    add_row(std::move(row));
}

void handle_memcpy_record(const CUpti_ActivityMemcpy6* memcpy_record) {
    ActivityRow row;
    row.record_type = "memcpy";
    row.activity_kind = static_cast<uint32_t>(memcpy_record->kind);
    row.start_ns = memcpy_record->start;
    row.end_ns = memcpy_record->end;
    row.device_id = memcpy_record->deviceId;
    row.context_id = memcpy_record->contextId;
    row.stream_id = memcpy_record->streamId;
    row.correlation_id = memcpy_record->correlationId;
    row.bytes = memcpy_record->bytes;
    row.copy_kind = memcpy_record->copyKind;
    row.src_kind = memcpy_record->srcKind;
    row.dst_kind = memcpy_record->dstKind;
    add_row(std::move(row));
}

void handle_memset_record(const CUpti_ActivityMemset4* memset_record) {
    ActivityRow row;
    row.record_type = "memset";
    row.activity_kind = static_cast<uint32_t>(memset_record->kind);
    row.start_ns = memset_record->start;
    row.end_ns = memset_record->end;
    row.device_id = memset_record->deviceId;
    row.context_id = memset_record->contextId;
    row.stream_id = memset_record->streamId;
    row.correlation_id = memset_record->correlationId;
    row.bytes = memset_record->bytes;
    row.memset_value = memset_record->value;
    add_row(std::move(row));
}

void CUPTIAPI buffer_completed(
    CUcontext /*context*/,
    uint32_t /*stream_id*/,
    uint8_t* buffer,
    size_t /*size*/,
    size_t valid_size) {
    if (valid_size > 0) {
        CUpti_Activity* record = nullptr;
        while (true) {
            CUptiResult status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
            if (status == CUPTI_SUCCESS) {
                switch (record->kind) {
                    case CUPTI_ACTIVITY_KIND_RUNTIME:
                    case CUPTI_ACTIVITY_KIND_DRIVER:
#ifdef CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API
                    case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API:
#endif
                        handle_api_record(reinterpret_cast<const CUpti_ActivityAPI*>(record));
                        break;
                    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                    case CUPTI_ACTIVITY_KIND_KERNEL:
                        handle_kernel_record(reinterpret_cast<const CUpti_ActivityKernel9*>(record));
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMCPY:
                        handle_memcpy_record(reinterpret_cast<const CUpti_ActivityMemcpy6*>(record));
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMSET:
                        handle_memset_record(reinterpret_cast<const CUpti_ActivityMemset4*>(record));
                        break;
                    default:
                        break;
                }
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED || status == CUPTI_ERROR_INVALID_KIND) {
                break;
            } else {
                set_error(std::string("cuptiActivityGetNextRecord failed: ") + cupti_error_name(status));
                break;
            }
        }
    }
    std::free(buffer);
}

int enable_kind(CUpti_ActivityKind kind) {
    CUptiResult result = cuptiActivityEnable(kind);
    if (result == CUPTI_SUCCESS || result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        return 0;
    }
    std::ostringstream oss;
    oss << "cuptiActivityEnable(" << static_cast<int>(kind) << ") failed: "
        << cupti_error_name(result);
    set_error(oss.str());
    return -1;
}

void disable_kind(CUpti_ActivityKind kind) {
    cuptiActivityDisable(kind);
}

int write_rows_to_csv(const std::string& path) {
    std::vector<ActivityRow> rows;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        rows = g_rows;
    }

    std::ofstream out(path);
    if (!out) {
        set_error("Could not open CUPTI output CSV: " + path);
        return -1;
    }
    out << "record_type,name,activity_kind,cbid,start_ns,end_ns,duration_us,"
        << "queued_ns,submitted_ns,device_id,context_id,stream_id,correlation_id,"
        << "grid_id,grid_x,grid_y,grid_z,block_x,block_y,block_z,"
        << "cluster_x,cluster_y,cluster_z,static_smem,dynamic_smem,"
        << "shared_memory_executed,registers_per_thread,bytes,copy_kind,"
        << "src_kind,dst_kind,memset_value,return_value\n";

    for (const ActivityRow& row : rows) {
        double duration_us = 0.0;
        if (row.end_ns >= row.start_ns) {
            duration_us = static_cast<double>(row.end_ns - row.start_ns) / 1000.0;
        }
        out << row.record_type << ','
            << csv_escape(row.name) << ','
            << row.activity_kind << ','
            << row.cbid << ','
            << row.start_ns << ','
            << row.end_ns << ','
            << duration_us << ','
            << row.queued_ns << ','
            << row.submitted_ns << ','
            << row.device_id << ','
            << row.context_id << ','
            << row.stream_id << ','
            << row.correlation_id << ','
            << row.grid_id << ','
            << row.grid_x << ','
            << row.grid_y << ','
            << row.grid_z << ','
            << row.block_x << ','
            << row.block_y << ','
            << row.block_z << ','
            << row.cluster_x << ','
            << row.cluster_y << ','
            << row.cluster_z << ','
            << row.static_smem << ','
            << row.dynamic_smem << ','
            << row.shared_memory_executed << ','
            << row.registers_per_thread << ','
            << row.bytes << ','
            << row.copy_kind << ','
            << row.src_kind << ','
            << row.dst_kind << ','
            << row.memset_value << ','
            << row.return_value << '\n';
    }
    return 0;
}

}  // namespace

extern "C" const char* ffjk_cupti_last_error() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_last_error.c_str();
}

extern "C" int ffjk_cupti_start(
    const char* output_path,
    int enable_runtime,
    int enable_driver,
    int enable_memcpy,
    int enable_memset,
    int enable_latency_timestamps) {
    if (output_path == nullptr) {
        set_error("output_path is null");
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_rows.clear();
        g_last_error.clear();
        g_output_path = output_path;
        g_active = true;
    }

    if (!g_callbacks_registered) {
        CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
        g_callbacks_registered = true;
    }

    size_t buffer_size = kBufferSize;
    size_t buffer_size_attr_size = sizeof(buffer_size);
    CUPTI_CHECK(cuptiActivitySetAttribute(
        CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
        &buffer_size_attr_size,
        &buffer_size));

    (void)enable_latency_timestamps;

    if (enable_runtime && enable_kind(CUPTI_ACTIVITY_KIND_RUNTIME)) return -1;
    if (enable_driver && enable_kind(CUPTI_ACTIVITY_KIND_DRIVER)) return -1;
    if (enable_memcpy && enable_kind(CUPTI_ACTIVITY_KIND_MEMCPY)) return -1;
    if (enable_memset && enable_kind(CUPTI_ACTIVITY_KIND_MEMSET)) return -1;
    if (enable_kind(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) return -1;

    return 0;
}

extern "C" int ffjk_cupti_stop() {
    cudaDeviceSynchronize();
    CUptiResult flush_result = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

    disable_kind(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    disable_kind(CUPTI_ACTIVITY_KIND_RUNTIME);
    disable_kind(CUPTI_ACTIVITY_KIND_DRIVER);
    disable_kind(CUPTI_ACTIVITY_KIND_MEMCPY);
    disable_kind(CUPTI_ACTIVITY_KIND_MEMSET);

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_active = false;
    }

    if (flush_result != CUPTI_SUCCESS) {
        std::ostringstream oss;
        oss << "cuptiActivityFlushAll failed: " << cupti_error_name(flush_result);
        set_error(oss.str());
        return -1;
    }

    return write_rows_to_csv(g_output_path);
}
