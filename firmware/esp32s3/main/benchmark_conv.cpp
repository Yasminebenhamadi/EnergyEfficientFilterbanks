#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>
#include <initializer_list>
#include <memory>
#include <random>
#include <type_traits>
#include <cinttypes>


#include "esp_heap_caps.h"

#include "esp_micro_profiler.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "esp_sleep.h"
#include "esp_nn.h"
#include "audio.h"

namespace tflite {
namespace {

using Profiler = ::benchmark::MicroProfiler;

// Seed used for the random input. Input data shouldn't affect invocation
// timing so randomness isn't really needed.
constexpr uint32_t kRandomSeed = 0xFB;

constexpr size_t kTensorArenaSize = 0;
uint8_t* tensor_arena;

static ::benchmark::MicroProfiler profiler;


}  // namespace
}  // namespace tflite


#define BATCH 1
#define H 1
#define W 10240
#define IN_CH 1
#define K_H 1

// Helper to initialize random data for a given buffer sizes
static void init_data_ptrs(int8_t* input_ptr, size_t input_count,
                           int8_t* filter_ptr, size_t filter_count,
                           int32_t* bias_ptr, size_t bias_count) {
    for (size_t i = 0; i < input_count; i++)
        input_ptr[i] = (rand() % 255) - 128;

    for (size_t i = 0; i < filter_count; i++)
        filter_ptr[i] = (rand() % 255) - 128;

    for (size_t i = 0; i < bias_count; i++)
        bias_ptr[i] = rand() % 10;
}

// Run a single benchmark configuration
int conv_benchmark_config(int K_W, int OUT_CH, int STRIDE_W) {
    if (K_W <= 0 || OUT_CH <= 0 || STRIDE_W <= 0) {
        MicroPrintf("Invalid config: K_W=%d OUT_CH=%d STRIDE_W=%d", K_W, OUT_CH, STRIDE_W);
        return -1;
    }

    const int W_OUT = ((W - K_W) / STRIDE_W) + 1;

    size_t input_size = BATCH * H * W * IN_CH;
    size_t filter_size = (size_t)OUT_CH * K_H * K_W * IN_CH;
    size_t bias_size = OUT_CH;
    size_t output_size = (size_t)BATCH * H * W_OUT * OUT_CH;

    int8_t* input = (int8_t*)heap_caps_malloc(input_size, MALLOC_CAP_INTERNAL);
    int8_t* filter = (int8_t*)heap_caps_malloc(filter_size, MALLOC_CAP_INTERNAL);
    int32_t* bias = (int32_t*)heap_caps_malloc(bias_size * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    int8_t* output = (int8_t*)heap_caps_malloc(output_size, MALLOC_CAP_INTERNAL);

    if (!input || !filter || !bias || !output) {
        MicroPrintf("Allocation failed for config K_W=%d OUT_CH=%d STRIDE_W=%d", K_W, OUT_CH, STRIDE_W);
        heap_caps_free(input); heap_caps_free(filter); heap_caps_free(bias); heap_caps_free(output);
        return -1;
    }

    // Tensor dimensions
    data_dims_t input_dims = {
        .width = W,
        .height = H,
        .channels = IN_CH,
        .extra = BATCH
    };

    data_dims_t filter_dims = {
        .width = K_W,
        .height = K_H,
        .channels = IN_CH,
        .extra = OUT_CH
    };

    data_dims_t output_dims = {
        .width = W_OUT,
        .height = H,
        .channels = OUT_CH,
        .extra = BATCH
    };

    data_2d_t stride = {(int)STRIDE_W, 1};
    data_2d_t padding = {0, 0};
    data_2d_t dilation = {1, 1};

    act_params_t activation = {
        .min = -128,
        .max = 127
    };

    conv_params_t conv_params = {
        .in_offset = 0,
        .out_offset = 0,
        .stride = stride,
        .padding = padding,
        .dilation = dilation,
        .activation = activation
    };

    // Quant
    int32_t *mult = (int32_t*)heap_caps_malloc(OUT_CH * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    int32_t *shift = (int32_t*)heap_caps_malloc(OUT_CH * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    if (!mult || !shift) {
        MicroPrintf("Quant alloc failed");
        heap_caps_free(input); heap_caps_free(filter); heap_caps_free(bias); heap_caps_free(output);
        heap_caps_free(mult); heap_caps_free(shift);
        return -1;
    }

    for (int i = 0; i < OUT_CH; i++) {
        mult[i] = 1;
        shift[i] = 0;
    }

    quant_data_t quant_data = {
        .shift = shift,
        .mult = mult
    };

    // Allocate scratch
    int scratch_size = esp_nn_get_conv_scratch_size(
        &input_dims, &filter_dims, &output_dims, &conv_params);

    MicroPrintf("scratch_size: %d for config K_W=%d OUT_CH=%d STRIDE_W=%d", scratch_size, K_W, OUT_CH, STRIDE_W);

    if (scratch_size < 0) {
        MicroPrintf("Invalid scratch size: %d", scratch_size);
        // Free the previously allocated buffers here.
        return -1;
    }

    void *scratch_buf = nullptr;

    if (scratch_size > 0) {
        scratch_buf = heap_caps_malloc(scratch_size, MALLOC_CAP_INTERNAL);
        if (!scratch_buf) {
            MicroPrintf("Scratch allocation failed");
            // Free the previously allocated buffers here.
            return -1;
        }
    }

    esp_nn_set_conv_scratch_buf(scratch_buf);

    if (!scratch_buf) {
        MicroPrintf("Scratch alloc failed");
        heap_caps_free(input); heap_caps_free(filter); heap_caps_free(bias); heap_caps_free(output);
        heap_caps_free(mult); heap_caps_free(shift);
        return -1;
    }

    esp_nn_set_conv_scratch_buf(scratch_buf);

    init_data_ptrs(input, input_size, filter, filter_size, bias, bias_size);

    memset(output, 0x55, output_size);

    // Warm-up
    esp_nn_conv_s8(
        &input_dims, input,
        &filter_dims, filter,
        bias,
        &output_dims, output,
        &conv_params,
        &quant_data
    );

    const int runs = 20;

    esp_sleep_enable_timer_wakeup(100000);
    uint32_t handle = tflite::profiler.BeginEvent("conv");

    for (int i = 0; i < runs; i++) {
        esp_nn_conv_s8(
            &input_dims, input,
            &filter_dims, filter,
            bias,
            &output_dims, output,
            &conv_params,
            &quant_data
        );
    }

    tflite::profiler.EndEvent(handle);

    int64_t total_ticks = tflite::profiler.LogTicksPerTagCsv();
    tflite::profiler.ClearEvents();

    // Normalize results
    int64_t ticks_per_run = total_ticks / runs;
    MicroPrintf(
        "Config K_W=%d OUT_CH=%d STRIDE_W=%d Ticks/run: %" PRId64,
        K_W, OUT_CH, STRIDE_W, ticks_per_run
    );
    // Print CSV line for easy parsing
    MicroPrintf(
        "csv,%d,%d,%d,%" PRId64,
        K_W, OUT_CH, STRIDE_W, ticks_per_run
    );

    heap_caps_free(scratch_buf);
    heap_caps_free(input);
    heap_caps_free(filter);
    heap_caps_free(bias);
    heap_caps_free(output);
    heap_caps_free(mult);
    heap_caps_free(shift);

    return 0;
}



// Run a series of benchmark configurations.
int conv1D_benchmark() {
    const int KWs[] = {21, 31, 61, 91, 101};
    const int OUTs[] = {4, 8, 16, 32, 64};
    const int strides[] = {24, 44, 64, 84};

    int status = 0;

    for (size_t i = 0; i < sizeof(KWs) / sizeof(KWs[0]); i++) {
        for (size_t j = 0; j < sizeof(OUTs) / sizeof(OUTs[0]); j++) {
            for (size_t k = 0; k < sizeof(strides) / sizeof(strides[0]); k++) {
                if (conv_benchmark_config(KWs[i], OUTs[j], strides[k]) != 0) {
                    MicroPrintf(
                        "Benchmark failed: K_W=%d OUT_CH=%d STRIDE_W=%d",
                        KWs[i], OUTs[j], strides[k]
                    );
                    status = -1;
                }
            }
        }
    }

    return status;
}

int conv_benchmark_layers()
{
    MicroPrintf("Starting Conv1D layer benchmarks");

    srand(0xFB);

    int status = conv1D_benchmark();

    MicroPrintf("Benchmarks complete");

    return status;
}