#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>

#include <algorithm>
#include <cctype>
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

struct RangeProfilerState {
    CUcontext context = nullptr;
    CUpti_RangeProfiler_Object* range_profiler_object = nullptr;
    std::vector<std::string> metric_names;
    std::vector<const char*> metric_name_ptrs;
    std::vector<uint8_t> config_image;
    std::vector<uint8_t> counter_data_image;
    std::string range_name;
    std::string chip_name;
    size_t num_passes = 0;
    int all_passes_submitted = 0;
    bool initialized = false;
    bool enabled = false;
    bool started = false;
    bool range_pushed = false;
};

std::mutex g_mutex;
RangeProfilerState g_state;
std::string g_last_error;

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

const char* cuda_error_name(CUresult result) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    return name ? name : "UNKNOWN_CUDA_ERROR";
}

void set_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error = message;
}

int set_cupti_error(CUptiResult result, const char* expr) {
    if (result == CUPTI_SUCCESS) {
        return 0;
    }
    std::ostringstream oss;
    oss << expr << " failed: " << cupti_error_name(result);
    set_error(oss.str());
    return -1;
}

int set_cuda_error(CUresult result, const char* expr) {
    if (result == CUDA_SUCCESS) {
        return 0;
    }
    std::ostringstream oss;
    oss << expr << " failed: " << cuda_error_name(result);
    set_error(oss.str());
    return -1;
}

#define CUPTI_CHECK(expr)                    \
    do {                                     \
        if (set_cupti_error((expr), #expr))  \
            return -1;                       \
    } while (0)

#define CUDA_CHECK(expr)                   \
    do {                                   \
        if (set_cuda_error((expr), #expr)) \
            return -1;                     \
    } while (0)

std::vector<std::string> split_csv(const char* csv) {
    std::vector<std::string> values;
    if (csv == nullptr) {
        return values;
    }

    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char c) {
            return !std::isspace(c);
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char c) {
            return !std::isspace(c);
        }).base(), item.end());
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    return values;
}

void refresh_metric_ptrs() {
    g_state.metric_name_ptrs.clear();
    g_state.metric_name_ptrs.reserve(g_state.metric_names.size());
    for (const std::string& metric_name : g_state.metric_names) {
        g_state.metric_name_ptrs.push_back(metric_name.c_str());
    }
}

void reset_state_vectors() {
    g_state.metric_names.clear();
    g_state.metric_name_ptrs.clear();
    g_state.config_image.clear();
    g_state.counter_data_image.clear();
    g_state.range_name.clear();
    g_state.chip_name.clear();
    g_state.num_passes = 0;
    g_state.all_passes_submitted = 0;
}

void cleanup_no_throw() {
    if (g_state.range_pushed) {
        CUpti_RangeProfiler_PopRange_Params pop_params{
            CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
        pop_params.pRangeProfilerObject = g_state.range_profiler_object;
        cuptiRangeProfilerPopRange(&pop_params);
        g_state.range_pushed = false;
    }

    if (g_state.started) {
        CUpti_RangeProfiler_Stop_Params stop_params{
            CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
        stop_params.pRangeProfilerObject = g_state.range_profiler_object;
        cuptiRangeProfilerStop(&stop_params);
        g_state.started = false;
    }

    if (g_state.enabled) {
        CUpti_RangeProfiler_Disable_Params disable_params{
            CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
        disable_params.pRangeProfilerObject = g_state.range_profiler_object;
        cuptiRangeProfilerDisable(&disable_params);
        g_state.enabled = false;
    }

    if (g_state.initialized) {
        CUpti_Profiler_DeInitialize_Params deinit_params{
            CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        cuptiProfilerDeInitialize(&deinit_params);
        g_state.initialized = false;
    }

    g_state.context = nullptr;
    g_state.range_profiler_object = nullptr;
}

int initialize_current_context() {
    CUDA_CHECK(cuInit(0));
    cudaError_t runtime_status = cudaFree(nullptr);
    if (runtime_status != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaFree(nullptr) failed: " << cudaGetErrorString(runtime_status);
        set_error(oss.str());
        return -1;
    }

    CUcontext context = nullptr;
    CUDA_CHECK(cuCtxGetCurrent(&context));
    if (context == nullptr) {
        set_error("No current CUDA context is available for CUPTI Range Profiler");
        return -1;
    }
    g_state.context = context;
    return 0;
}

int initialize_range_profiler() {
    CUpti_Profiler_Initialize_Params profiler_initialize_params{
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_CHECK(cuptiProfilerInitialize(&profiler_initialize_params));
    g_state.initialized = true;

    CUdevice device = 0;
    CUDA_CHECK(cuCtxGetDevice(&device));

    CUpti_Device_GetChipName_Params chip_name_params{
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip_name_params.deviceIndex = static_cast<size_t>(device);
    CUPTI_CHECK(cuptiDeviceGetChipName(&chip_name_params));
    g_state.chip_name = chip_name_params.pChipName ? chip_name_params.pChipName : "";

    CUpti_RangeProfiler_Enable_Params enable_params{
        CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    enable_params.ctx = g_state.context;
    CUPTI_CHECK(cuptiRangeProfilerEnable(&enable_params));
    g_state.enabled = true;
    g_state.range_profiler_object = enable_params.pRangeProfilerObject;
    return 0;
}

int sync_device() {
    cudaError_t runtime_status = cudaDeviceSynchronize();
    if (runtime_status != cudaSuccess) {
        std::ostringstream oss;
        oss << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(runtime_status);
        set_error(oss.str());
        return -1;
    }
    return 0;
}

int create_config_image() {
    CUpti_Profiler_Host_Initialize_Params host_initialize_params{
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    host_initialize_params.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    host_initialize_params.pChipName = g_state.chip_name.c_str();
    CUPTI_CHECK(cuptiProfilerHostInitialize(&host_initialize_params));

    CUpti_Profiler_Host_ConfigAddMetrics_Params add_metrics_params{
        CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    add_metrics_params.pHostObject = host_initialize_params.pHostObject;
    add_metrics_params.ppMetricNames = g_state.metric_name_ptrs.data();
    add_metrics_params.numMetrics = g_state.metric_name_ptrs.size();
    if (set_cupti_error(cuptiProfilerHostConfigAddMetrics(&add_metrics_params),
                        "cuptiProfilerHostConfigAddMetrics")) {
        CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
        cuptiProfilerHostDeinitialize(&host_deinitialize_params);
        return -1;
    }

    CUpti_Profiler_Host_GetConfigImageSize_Params config_size_params{
        CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    config_size_params.pHostObject = host_initialize_params.pHostObject;
    if (set_cupti_error(cuptiProfilerHostGetConfigImageSize(&config_size_params),
                        "cuptiProfilerHostGetConfigImageSize")) {
        CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
        cuptiProfilerHostDeinitialize(&host_deinitialize_params);
        return -1;
    }

    g_state.config_image.resize(config_size_params.configImageSize);

    CUpti_Profiler_Host_GetConfigImage_Params config_image_params{
        CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    config_image_params.pHostObject = host_initialize_params.pHostObject;
    config_image_params.pConfigImage = g_state.config_image.data();
    config_image_params.configImageSize = g_state.config_image.size();
    if (set_cupti_error(cuptiProfilerHostGetConfigImage(&config_image_params),
                        "cuptiProfilerHostGetConfigImage")) {
        CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
        cuptiProfilerHostDeinitialize(&host_deinitialize_params);
        return -1;
    }

    CUpti_Profiler_Host_GetNumOfPasses_Params num_passes_params{
        CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    num_passes_params.pConfigImage = g_state.config_image.data();
    num_passes_params.configImageSize = g_state.config_image.size();
    if (set_cupti_error(cuptiProfilerHostGetNumOfPasses(&num_passes_params),
                        "cuptiProfilerHostGetNumOfPasses")) {
        CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
        cuptiProfilerHostDeinitialize(&host_deinitialize_params);
        return -1;
    }
    g_state.num_passes = num_passes_params.numOfPasses;

    CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
    CUPTI_CHECK(cuptiProfilerHostDeinitialize(&host_deinitialize_params));
    return 0;
}

int create_counter_data_image(uint32_t max_ranges) {
    CUpti_RangeProfiler_GetCounterDataSize_Params size_params{
        CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    size_params.pRangeProfilerObject = g_state.range_profiler_object;
    size_params.pMetricNames = g_state.metric_name_ptrs.data();
    size_params.numMetrics = g_state.metric_name_ptrs.size();
    size_params.maxNumOfRanges = max_ranges;
    size_params.maxNumRangeTreeNodes = max_ranges;
    CUPTI_CHECK(cuptiRangeProfilerGetCounterDataSize(&size_params));

    g_state.counter_data_image.resize(size_params.counterDataSize);

    CUpti_RangeProfiler_CounterDataImage_Initialize_Params initialize_params{
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initialize_params.pRangeProfilerObject = g_state.range_profiler_object;
    initialize_params.pCounterData = g_state.counter_data_image.data();
    initialize_params.counterDataSize = g_state.counter_data_image.size();
    CUPTI_CHECK(cuptiRangeProfilerCounterDataImageInitialize(&initialize_params));
    return 0;
}

int set_config() {
    if (g_state.started) {
        set_error("CUPTI range profiler pass is already started");
        return -1;
    }
    if (!g_state.enabled || g_state.range_profiler_object == nullptr) {
        set_error("CUPTI range profiler is not prepared");
        return -1;
    }

    CUpti_RangeProfiler_SetConfig_Params set_config_params{
        CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    set_config_params.pRangeProfilerObject = g_state.range_profiler_object;
    set_config_params.pConfig = g_state.config_image.data();
    set_config_params.configSize = g_state.config_image.size();
    set_config_params.passIndex = 0;
    set_config_params.minNestingLevel = 1;
    set_config_params.numNestingLevels = 1;
    set_config_params.targetNestingLevel = 0;
    set_config_params.pCounterDataImage = g_state.counter_data_image.data();
    set_config_params.counterDataImageSize = g_state.counter_data_image.size();
    set_config_params.range = CUPTI_UserRange;
    set_config_params.replayMode = CUPTI_UserReplay;
    set_config_params.maxRangesPerPass = 1;
    CUPTI_CHECK(cuptiRangeProfilerSetConfig(&set_config_params));
    return 0;
}

int start_profiler() {
    if (g_state.started) {
        set_error("CUPTI range profiler pass is already started");
        return -1;
    }
    if (!g_state.enabled || g_state.range_profiler_object == nullptr) {
        set_error("CUPTI range profiler is not prepared");
        return -1;
    }

    CUpti_RangeProfiler_Start_Params start_params{
        CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
    start_params.pRangeProfilerObject = g_state.range_profiler_object;
    CUPTI_CHECK(cuptiRangeProfilerStart(&start_params));
    g_state.started = true;

    CUpti_RangeProfiler_PushRange_Params push_params{
        CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
    push_params.pRangeProfilerObject = g_state.range_profiler_object;
    push_params.pRangeName = g_state.range_name.c_str();
    CUPTI_CHECK(cuptiRangeProfilerPushRange(&push_params));
    g_state.range_pushed = true;
    return 0;
}

int stop_profiler() {
    if (!g_state.started) {
        return 0;
    }

    if (sync_device()) {
        return -1;
    }

    if (g_state.range_pushed) {
        CUpti_RangeProfiler_PopRange_Params pop_params{
            CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
        pop_params.pRangeProfilerObject = g_state.range_profiler_object;
        CUPTI_CHECK(cuptiRangeProfilerPopRange(&pop_params));
        g_state.range_pushed = false;
    }

    CUpti_RangeProfiler_Stop_Params stop_params{
        CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    stop_params.pRangeProfilerObject = g_state.range_profiler_object;
    CUPTI_CHECK(cuptiRangeProfilerStop(&stop_params));
    g_state.all_passes_submitted = stop_params.isAllPassSubmitted ? 1 : 0;
    g_state.started = false;
    return 0;
}

int decode_counter_data() {
    CUpti_RangeProfiler_DecodeData_Params decode_params{
        CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decode_params.pRangeProfilerObject = g_state.range_profiler_object;
    CUPTI_CHECK(cuptiRangeProfilerDecodeData(&decode_params));
    return 0;
}

int write_metrics_csv(const std::string& output_path) {
    CUpti_Profiler_Host_Initialize_Params host_initialize_params{
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    host_initialize_params.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    host_initialize_params.pChipName = g_state.chip_name.c_str();
    CUPTI_CHECK(cuptiProfilerHostInitialize(&host_initialize_params));

    CUpti_RangeProfiler_GetCounterDataInfo_Params info_params{
        CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
    info_params.pCounterDataImage = g_state.counter_data_image.data();
    info_params.counterDataImageSize = g_state.counter_data_image.size();
    if (set_cupti_error(cuptiRangeProfilerGetCounterDataInfo(&info_params),
                        "cuptiRangeProfilerGetCounterDataInfo")) {
        CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
        cuptiProfilerHostDeinitialize(&host_deinitialize_params);
        return -1;
    }

    std::ofstream out(output_path);
    if (!out) {
        set_error("Could not open CUPTI range output CSV: " + output_path);
        CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
        cuptiProfilerHostDeinitialize(&host_deinitialize_params);
        return -1;
    }
    out << "range_index,range_name,metric,value,chip_name,num_passes,"
        << "all_passes_submitted\n";

    for (size_t range_index = 0; range_index < info_params.numTotalRanges; ++range_index) {
        CUpti_RangeProfiler_CounterData_GetRangeInfo_Params range_info_params{
            CUpti_RangeProfiler_CounterData_GetRangeInfo_Params_STRUCT_SIZE};
        range_info_params.pCounterDataImage = g_state.counter_data_image.data();
        range_info_params.counterDataImageSize = g_state.counter_data_image.size();
        range_info_params.rangeIndex = range_index;
        range_info_params.rangeDelimiter = "/";
        if (set_cupti_error(cuptiRangeProfilerCounterDataGetRangeInfo(&range_info_params),
                            "cuptiRangeProfilerCounterDataGetRangeInfo")) {
            CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
                CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
            host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
            cuptiProfilerHostDeinitialize(&host_deinitialize_params);
            return -1;
        }

        std::string range_name =
            range_info_params.rangeName ? range_info_params.rangeName : "";
        if (range_name.empty() || range_name == "0") {
            range_name = g_state.range_name;
        }

        std::vector<double> metric_values(g_state.metric_names.size(), 0.0);
        CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate_params{
            CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evaluate_params.pHostObject = host_initialize_params.pHostObject;
        evaluate_params.pCounterDataImage = g_state.counter_data_image.data();
        evaluate_params.counterDataImageSize = g_state.counter_data_image.size();
        evaluate_params.rangeIndex = range_index;
        evaluate_params.ppMetricNames = g_state.metric_name_ptrs.data();
        evaluate_params.numMetrics = g_state.metric_name_ptrs.size();
        evaluate_params.pMetricValues = metric_values.data();
        if (set_cupti_error(cuptiProfilerHostEvaluateToGpuValues(&evaluate_params),
                            "cuptiProfilerHostEvaluateToGpuValues")) {
            CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
                CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
            host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
            cuptiProfilerHostDeinitialize(&host_deinitialize_params);
            return -1;
        }

        for (size_t metric_index = 0; metric_index < g_state.metric_names.size(); ++metric_index) {
            out << range_index << ','
                << csv_escape(range_name) << ','
                << csv_escape(g_state.metric_names[metric_index]) << ','
                << metric_values[metric_index] << ','
                << csv_escape(g_state.chip_name) << ','
                << g_state.num_passes << ','
                << g_state.all_passes_submitted << '\n';
        }
    }

    CUpti_Profiler_Host_Deinitialize_Params host_deinitialize_params{
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    host_deinitialize_params.pHostObject = host_initialize_params.pHostObject;
    CUPTI_CHECK(cuptiProfilerHostDeinitialize(&host_deinitialize_params));
    return 0;
}

}  // namespace

extern "C" const char* ffjk_cupti_range_last_error() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_last_error.c_str();
}

extern "C" size_t ffjk_cupti_range_num_passes() {
    return g_state.num_passes;
}

extern "C" void ffjk_cupti_range_abort() {
    cleanup_no_throw();
    reset_state_vectors();
}

extern "C" int ffjk_cupti_range_prepare(const char* metric_csv, const char* range_name) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_last_error.clear();
    }
    cleanup_no_throw();
    reset_state_vectors();

    g_state.metric_names = split_csv(metric_csv);
    if (g_state.metric_names.empty()) {
        set_error("No CUPTI range metrics were provided");
        return -1;
    }
    refresh_metric_ptrs();
    g_state.range_name = (range_name && std::strlen(range_name) > 0) ? range_name : "range";

    if (initialize_current_context()) {
        cleanup_no_throw();
        return -1;
    }
    if (initialize_range_profiler()) {
        cleanup_no_throw();
        return -1;
    }
    if (create_config_image()) {
        cleanup_no_throw();
        return -1;
    }
    if (create_counter_data_image(1)) {
        cleanup_no_throw();
        return -1;
    }
    if (set_config()) {
        cleanup_no_throw();
        return -1;
    }
    return 0;
}

extern "C" int ffjk_cupti_range_start_pass() {
    return start_profiler();
}

extern "C" int ffjk_cupti_range_stop_pass() {
    if (stop_profiler()) {
        return -1;
    }
    return g_state.all_passes_submitted ? 1 : 0;
}

extern "C" int ffjk_cupti_range_finish(const char* output_path) {
    if (output_path == nullptr) {
        set_error("output_path is null");
        cleanup_no_throw();
        return -1;
    }

    if (decode_counter_data()) {
        cleanup_no_throw();
        return -1;
    }

    int write_result = write_metrics_csv(output_path);
    cleanup_no_throw();
    reset_state_vectors();
    return write_result;
}
