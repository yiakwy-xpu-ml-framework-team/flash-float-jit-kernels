/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace ffjk {

enum CudaProfilerEventKind : uint32_t {
    kProfilerEventInstant = 0,
    kProfilerEventBegin = 1,
    kProfilerEventEnd = 2,
};

enum CudaProfilerEventId : uint32_t {
    kProfilerEventKernelEnter = 1,
    kProfilerEventPipelineEnter = 2,
    kProfilerEventTask = 10,
    kProfilerEventTaskMap = 11,
    kProfilerEventPrefetchTma = 20,
    kProfilerEventScaleXLoad = 21,
    kProfilerEventScaleWLoad = 22,
    kProfilerEventTmaWait = 30,
    kProfilerEventMmaIssue = 31,
    kProfilerEventProducerLoadOnce = 32,
    kProfilerEventWgmmaWait = 33,
    kProfilerEventScaleApplyAccum = 34,
    kProfilerEventEpilogueSmemStore = 40,
    kProfilerEventSplitKReduce = 41,
    kProfilerEventStoreLower = 42,
    kProfilerEventMirrorTranspose = 43,
    kProfilerEventMirrorStore = 44,
    kProfilerEventTaskDone = 50,
};

struct alignas(32) CudaProfilerHeader {
    uint32_t capacity;
    uint32_t grid_xy;
    uint32_t records_per_cta;
    uint32_t records_per_task;
    uint32_t max_tasks_per_cta;
    uint32_t max_k_tiles_per_task;
    uint32_t cta_slots;
    uint32_t per_k_slots;
};

struct alignas(16) CudaProfilerRecord {
    uint64_t timestamp;
    uint32_t payload;
    uint16_t tag;
    uint16_t smid;
};

struct alignas(32) CudaProfilerBuffer {
    CudaProfilerHeader header;
};

struct CudaProfilerLayout {
    CudaProfilerBuffer* buffer;
    uint32_t cta_id;
    uint32_t records_per_cta;
    uint32_t records_per_task;
    uint32_t max_tasks_per_cta;
    uint32_t max_k_tiles_per_task;
};

static constexpr uint32_t kCudaProfilerCtaSlots = 2;
static constexpr uint32_t kCudaProfilerTaskSlots = 20;
static constexpr uint32_t kCudaProfilerPerKSlots = 10;

static constexpr int32_t kCudaProfilerInvalidSlot = -1;
static constexpr uint32_t kCudaProfilerTagEventBits = 8;
static constexpr uint32_t kCudaProfilerTagKindShift = 8;
static constexpr uint32_t kCudaProfilerTagFlagsShift = 10;

static constexpr uint32_t kProfilerCtaSlotKernelEnter = 0;
static constexpr uint32_t kProfilerCtaSlotPipelineEnter = 1;

static constexpr uint32_t kProfilerTaskSlotTaskBegin = 0;
static constexpr uint32_t kProfilerTaskSlotTaskMap = 1;
static constexpr uint32_t kProfilerTaskSlotPrefetchTmaBegin = 2;
static constexpr uint32_t kProfilerTaskSlotPrefetchTmaEnd = 3;
static constexpr uint32_t kProfilerTaskSlotScaleXLoadBegin = 4;
static constexpr uint32_t kProfilerTaskSlotScaleXLoadEnd = 5;
static constexpr uint32_t kProfilerTaskSlotScaleWLoadBegin = 6;
static constexpr uint32_t kProfilerTaskSlotScaleWLoadEnd = 7;
static constexpr uint32_t kProfilerTaskSlotEpilogueSmemStoreBegin = 8;
static constexpr uint32_t kProfilerTaskSlotEpilogueSmemStoreEnd = 9;
static constexpr uint32_t kProfilerTaskSlotSplitKReduceBegin = 10;
static constexpr uint32_t kProfilerTaskSlotSplitKReduceEnd = 11;
static constexpr uint32_t kProfilerTaskSlotStoreLowerBegin = 12;
static constexpr uint32_t kProfilerTaskSlotStoreLowerEnd = 13;
static constexpr uint32_t kProfilerTaskSlotMirrorTransposeBegin = 14;
static constexpr uint32_t kProfilerTaskSlotMirrorTransposeEnd = 15;
static constexpr uint32_t kProfilerTaskSlotMirrorStoreBegin = 16;
static constexpr uint32_t kProfilerTaskSlotMirrorStoreEnd = 17;
static constexpr uint32_t kProfilerTaskSlotTaskEnd = 18;
static constexpr uint32_t kProfilerTaskSlotTaskDone = 19;

static constexpr uint32_t kProfilerKSlotTmaWaitBegin = 0;
static constexpr uint32_t kProfilerKSlotTmaWaitEnd = 1;
static constexpr uint32_t kProfilerKSlotMmaIssueBegin = 2;
static constexpr uint32_t kProfilerKSlotMmaIssueEnd = 3;
static constexpr uint32_t kProfilerKSlotProducerLoadOnceBegin = 4;
static constexpr uint32_t kProfilerKSlotProducerLoadOnceEnd = 5;
static constexpr uint32_t kProfilerKSlotWgmmaWaitBegin = 6;
static constexpr uint32_t kProfilerKSlotWgmmaWaitEnd = 7;
static constexpr uint32_t kProfilerKSlotScaleApplyAccumBegin = 8;
static constexpr uint32_t kProfilerKSlotScaleApplyAccumEnd = 9;

static constexpr uint32_t kCudaProfilerHeaderU64Words =
    sizeof(CudaProfilerHeader) / sizeof(uint64_t);
static constexpr uint32_t kCudaProfilerRecordU64Words =
    sizeof(CudaProfilerRecord) / sizeof(uint64_t);

static_assert(sizeof(CudaProfilerHeader) % sizeof(uint64_t) == 0,
              "CudaProfilerHeader must be uint64_t-aligned for Python tensor allocation.");
static_assert(sizeof(CudaProfilerHeader) == 32,
              "CudaProfilerHeader must stay 32 bytes to keep records aligned.");
static_assert(sizeof(CudaProfilerRecord) == 16,
              "CudaProfilerRecord must stay 16 bytes for low-overhead host parsing.");

__host__ __device__ __forceinline__ uint32_t cuda_profiler_pack_u16(
    uint32_t lo,
    uint32_t hi) {
    return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

__host__ __device__ __forceinline__ uint16_t cuda_profiler_make_tag(
    uint32_t event_id,
    CudaProfilerEventKind kind,
    uint32_t flags = 0) {
    return static_cast<uint16_t>(
        (event_id & ((1u << kCudaProfilerTagEventBits) - 1u)) |
        ((static_cast<uint32_t>(kind) & 0x3u) << kCudaProfilerTagKindShift) |
        ((flags & 0x3fu) << kCudaProfilerTagFlagsShift));
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_compact_payload(
    uint32_t event_id,
    uint32_t payload0,
    uint32_t payload1) {
    switch (event_id) {
    case kProfilerEventKernelEnter:
        return cuda_profiler_pack_u16(payload0, payload1);
    case kProfilerEventTask:
        return payload0;
    default:
        return payload1 != 0 ? payload1 : payload0;
    }
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_records_per_task(
    uint32_t max_k_tiles_per_task) {
    return kCudaProfilerTaskSlots + max_k_tiles_per_task * kCudaProfilerPerKSlots;
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_records_per_cta(
    uint32_t max_tasks_per_cta,
    uint32_t max_k_tiles_per_task) {
    return kCudaProfilerCtaSlots +
           max_tasks_per_cta * cuda_profiler_records_per_task(max_k_tiles_per_task);
}

__host__ __device__ __forceinline__ uint32_t cuda_profiler_required_records(
    uint32_t num_ctas,
    uint32_t max_tasks_per_cta,
    uint32_t max_k_tiles_per_task) {
    return num_ctas *
           cuda_profiler_records_per_cta(max_tasks_per_cta, max_k_tiles_per_task);
}

inline cudaError_t cuda_profiler_init(
    void* raw_buffer,
    uint32_t capacity,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t max_tasks_per_cta,
    uint32_t max_k_tiles_per_task,
    cudaStream_t stream) {
    if (raw_buffer == nullptr || capacity == 0) {
        return cudaSuccess;
    }

    cudaError_t memset_state = cudaMemsetAsync(
        raw_buffer,
        0,
        sizeof(CudaProfilerHeader) + static_cast<size_t>(capacity) * sizeof(CudaProfilerRecord),
        stream);
    if (memset_state != cudaSuccess) {
        return memset_state;
    }

    CudaProfilerHeader header{};
    header.capacity = capacity;
    header.grid_xy = cuda_profiler_pack_u16(grid_x, grid_y);
    header.max_tasks_per_cta = max_tasks_per_cta;
    header.max_k_tiles_per_task = max_k_tiles_per_task;
    header.records_per_task = cuda_profiler_records_per_task(max_k_tiles_per_task);
    header.records_per_cta = cuda_profiler_records_per_cta(
        max_tasks_per_cta, max_k_tiles_per_task);
    header.cta_slots = kCudaProfilerCtaSlots;
    header.per_k_slots = kCudaProfilerPerKSlots;

    return cudaMemcpyAsync(
        raw_buffer, &header, sizeof(header), cudaMemcpyHostToDevice, stream);
}

#if defined(__CUDA_ARCH__)

__device__ __forceinline__ uint64_t cuda_profiler_read_timestamp() {
    uint64_t timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
    return timestamp;
}

__device__ __forceinline__ uint32_t cuda_profiler_read_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ CudaProfilerRecord* cuda_profiler_records(
    CudaProfilerBuffer* buffer) {
    return reinterpret_cast<CudaProfilerRecord*>(
        reinterpret_cast<unsigned char*>(buffer) + sizeof(CudaProfilerHeader));
}

__device__ __forceinline__ CudaProfilerLayout cuda_profiler_make_layout(
    CudaProfilerBuffer* buffer,
    uint32_t total_symmetric_tiles,
    uint32_t max_k_tiles_per_task) {
    CudaProfilerLayout layout{};
    layout.buffer = buffer;
    if (buffer == nullptr) {
        return layout;
    }

    layout.cta_id = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    layout.max_tasks_per_cta = (total_symmetric_tiles + gridDim.y - 1) / gridDim.y;
    layout.max_k_tiles_per_task = max_k_tiles_per_task;
    layout.records_per_task = cuda_profiler_records_per_task(max_k_tiles_per_task);
    layout.records_per_cta = cuda_profiler_records_per_cta(
        layout.max_tasks_per_cta, max_k_tiles_per_task);
    return layout;
}

__device__ __forceinline__ int32_t cuda_profiler_cta_slot(
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    if (kind != kProfilerEventInstant) {
        return kCudaProfilerInvalidSlot;
    }

    switch (event_id) {
    case kProfilerEventKernelEnter:
        return kProfilerCtaSlotKernelEnter;
    case kProfilerEventPipelineEnter:
        return kProfilerCtaSlotPipelineEnter;
    default:
        return kCudaProfilerInvalidSlot;
    }
}

__device__ __forceinline__ int32_t cuda_profiler_task_slot(
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    switch (event_id) {
    case kProfilerEventTask:
        if (kind == kProfilerEventBegin) {
            return kProfilerTaskSlotTaskBegin;
        }
        if (kind == kProfilerEventEnd) {
            return kProfilerTaskSlotTaskEnd;
        }
        return kCudaProfilerInvalidSlot;
    case kProfilerEventTaskMap:
        return kind == kProfilerEventInstant
                   ? kProfilerTaskSlotTaskMap
                   : kCudaProfilerInvalidSlot;
    case kProfilerEventPrefetchTma:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotPrefetchTmaBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotPrefetchTmaEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventScaleXLoad:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotScaleXLoadBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotScaleXLoadEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventScaleWLoad:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotScaleWLoadBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotScaleWLoadEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventEpilogueSmemStore:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotEpilogueSmemStoreBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotEpilogueSmemStoreEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventSplitKReduce:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotSplitKReduceBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotSplitKReduceEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventStoreLower:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotStoreLowerBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotStoreLowerEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventMirrorTranspose:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotMirrorTransposeBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotMirrorTransposeEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventMirrorStore:
        return kind == kProfilerEventBegin
                   ? kProfilerTaskSlotMirrorStoreBegin
                   : (kind == kProfilerEventEnd ? kProfilerTaskSlotMirrorStoreEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventTaskDone:
        return kind == kProfilerEventInstant
                   ? kProfilerTaskSlotTaskDone
                   : kCudaProfilerInvalidSlot;
    default:
        return kCudaProfilerInvalidSlot;
    }
}

__device__ __forceinline__ int32_t cuda_profiler_k_slot(
    uint32_t event_id,
    CudaProfilerEventKind kind) {
    switch (event_id) {
    case kProfilerEventTmaWait:
        return kind == kProfilerEventBegin
                   ? kProfilerKSlotTmaWaitBegin
                   : (kind == kProfilerEventEnd ? kProfilerKSlotTmaWaitEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventMmaIssue:
        return kind == kProfilerEventBegin
                   ? kProfilerKSlotMmaIssueBegin
                   : (kind == kProfilerEventEnd ? kProfilerKSlotMmaIssueEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventProducerLoadOnce:
        return kind == kProfilerEventBegin
                   ? kProfilerKSlotProducerLoadOnceBegin
                   : (kind == kProfilerEventEnd ? kProfilerKSlotProducerLoadOnceEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventWgmmaWait:
        return kind == kProfilerEventBegin
                   ? kProfilerKSlotWgmmaWaitBegin
                   : (kind == kProfilerEventEnd ? kProfilerKSlotWgmmaWaitEnd
                                                : kCudaProfilerInvalidSlot);
    case kProfilerEventScaleApplyAccum:
        return kind == kProfilerEventBegin
                   ? kProfilerKSlotScaleApplyAccumBegin
                   : (kind == kProfilerEventEnd ? kProfilerKSlotScaleApplyAccumEnd
                                                : kCudaProfilerInvalidSlot);
    default:
        return kCudaProfilerInvalidSlot;
    }
}

__device__ __forceinline__ void cuda_profiler_record_slot(
    const CudaProfilerLayout& layout,
    uint64_t slot,
    uint32_t event_id,
    CudaProfilerEventKind kind,
    uint32_t payload0,
    uint32_t payload1) {
    if (layout.buffer == nullptr || slot >= layout.buffer->header.capacity) {
        return;
    }

    CudaProfilerRecord* records = cuda_profiler_records(layout.buffer);
    CudaProfilerRecord& record = records[slot];
    record.timestamp = cuda_profiler_read_timestamp();
    record.payload = cuda_profiler_compact_payload(event_id, payload0, payload1);
    record.tag = cuda_profiler_make_tag(event_id, kind);
    record.smid = static_cast<uint16_t>(cuda_profiler_read_smid());
}

__device__ __forceinline__ void cuda_profiler_record_cta_event(
    const CudaProfilerLayout& layout,
    uint32_t event_id,
    CudaProfilerEventKind kind,
    uint32_t payload0,
    uint32_t payload1) {
    int32_t slot_in_cta = cuda_profiler_cta_slot(event_id, kind);
    if (slot_in_cta < 0) {
        return;
    }

    uint64_t slot = static_cast<uint64_t>(layout.cta_id) * layout.records_per_cta +
                    static_cast<uint32_t>(slot_in_cta);
    cuda_profiler_record_slot(layout, slot, event_id, kind, payload0, payload1);
}

__device__ __forceinline__ void cuda_profiler_record_task_event(
    const CudaProfilerLayout& layout,
    uint32_t task_iter,
    int32_t k_iter,
    uint32_t event_id,
    CudaProfilerEventKind kind,
    uint32_t payload0,
    uint32_t payload1) {
    int32_t slot_in_task = cuda_profiler_task_slot(event_id, kind);
    if (slot_in_task < 0) {
        int32_t slot_in_k = cuda_profiler_k_slot(event_id, kind);
        if (slot_in_k < 0 || k_iter < 0 ||
            static_cast<uint32_t>(k_iter) >= layout.max_k_tiles_per_task) {
            return;
        }
        slot_in_task = static_cast<int32_t>(
            kCudaProfilerTaskSlots +
            static_cast<uint32_t>(k_iter) * kCudaProfilerPerKSlots +
            static_cast<uint32_t>(slot_in_k));
    }

    if (task_iter >= layout.max_tasks_per_cta) {
        return;
    }

    uint64_t cta_base = static_cast<uint64_t>(layout.cta_id) * layout.records_per_cta;
    uint64_t task_base = static_cast<uint64_t>(task_iter) * layout.records_per_task;
    uint64_t slot = cta_base + kCudaProfilerCtaSlots + task_base +
                    static_cast<uint32_t>(slot_in_task);
    cuda_profiler_record_slot(layout, slot, event_id, kind, payload0, payload1);
}

#endif // defined(__CUDA_ARCH__)

} // namespace ffjk

#ifdef FFJK_ENABLE_CUDA_PROFILER

#define FFJK_PROFILER_KERNEL_PARAMS , ffjk::CudaProfilerBuffer* ffjk_profiler
#define FFJK_PROFILER_KERNEL_ARGS , ffjk_profiler
#define FFJK_PROFILER_LAUNCH_ARG(profiler) , profiler

#define FFJK_PROFILER_DEFINE_LAYOUT(total_symmetric_tiles, max_k_tiles_per_task)    \
    ffjk::CudaProfilerLayout ffjk_prof_layout = ffjk::cuda_profiler_make_layout(    \
        ffjk_profiler,                                                              \
        static_cast<uint32_t>(total_symmetric_tiles),                               \
        static_cast<uint32_t>(max_k_tiles_per_task))

#define FFJK_PROF_CTA_EVENT_PAYLOAD(event_id, payload0, payload1)                  \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_cta_event(                                   \
                ffjk_prof_layout, event_id, ffjk::kProfilerEventInstant,            \
                static_cast<uint32_t>(payload0), static_cast<uint32_t>(payload1));  \
        }                                                                           \
    } while (0)

#define FFJK_PROF_EVENT(event_id)                                                   \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_task_event(                                  \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                static_cast<int32_t>(ffjk_prof_k_iter), event_id,                   \
                ffjk::kProfilerEventInstant, 0, 0);                                 \
        }                                                                           \
    } while (0)

#define FFJK_PROF_EVENT_PAYLOAD(event_id, payload0, payload1)                       \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_task_event(                                  \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                static_cast<int32_t>(ffjk_prof_k_iter), event_id,                   \
                ffjk::kProfilerEventInstant,                                        \
                static_cast<uint32_t>(payload0), static_cast<uint32_t>(payload1));  \
        }                                                                           \
    } while (0)

#define FFJK_PROF_BEGIN(event_id, payload0, payload1)                               \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_task_event(                                  \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                static_cast<int32_t>(ffjk_prof_k_iter), event_id,                   \
                ffjk::kProfilerEventBegin,                                          \
                static_cast<uint32_t>(payload0), static_cast<uint32_t>(payload1));  \
        }                                                                           \
    } while (0)

#define FFJK_PROF_END(event_id, payload0, payload1)                                 \
    do {                                                                            \
        if (threadIdx.x == 0) {                                                     \
            ffjk::cuda_profiler_record_task_event(                                  \
                ffjk_prof_layout, static_cast<uint32_t>(ffjk_prof_task_iter),       \
                static_cast<int32_t>(ffjk_prof_k_iter), event_id,                   \
                ffjk::kProfilerEventEnd,                                            \
                static_cast<uint32_t>(payload0), static_cast<uint32_t>(payload1));  \
        }                                                                           \
    } while (0)

#else

#define FFJK_PROFILER_KERNEL_PARAMS
#define FFJK_PROFILER_KERNEL_ARGS
#define FFJK_PROFILER_LAUNCH_ARG(profiler)
#define FFJK_PROFILER_DEFINE_LAYOUT(total_symmetric_tiles, max_k_tiles_per_task)
#define FFJK_PROF_CTA_EVENT_PAYLOAD(event_id, payload0, payload1) do {} while (0)
#define FFJK_PROF_EVENT(event_id) do {} while (0)
#define FFJK_PROF_EVENT_PAYLOAD(event_id, payload0, payload1) do {} while (0)
#define FFJK_PROF_BEGIN(event_id, payload0, payload1) do {} while (0)
#define FFJK_PROF_END(event_id, payload0, payload1) do {} while (0)

#endif // FFJK_ENABLE_CUDA_PROFILER
