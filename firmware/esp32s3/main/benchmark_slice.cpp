#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "esp_heap_caps.h"
#include "esp_nn.h"
#include "esp_timer.h"

#include "audio.h"

#include "tensorflow/lite/micro/micro_log.h"


// -----------------------------------------------------------------------------
// Benchmark parameters
// -----------------------------------------------------------------------------

#define BATCH       1
#define H           1
#define W           10240
#define IN_CH       1
#define OUT_CH      32
#define K_H         1
#define K_W         41
#define STRIDE_W    64

#define SIGNAL_LENGTH   30720
#define N_FRAMES        (SIGNAL_LENGTH / W)

#define N_RUNS      200

#define W_OUT       (((W - K_W) / STRIDE_W) + 1)


// -----------------------------------------------------------------------------
// Input, filter and output buffers
// -----------------------------------------------------------------------------

int8_t input[BATCH * H * W * IN_CH];
int8_t filter[OUT_CH * K_H * K_W * IN_CH];
int32_t bias[OUT_CH];

int8_t output[BATCH * H * W_OUT * OUT_CH];


// -----------------------------------------------------------------------------
// Split and reconstruction buffers
// -----------------------------------------------------------------------------

int8_t audio_low[W];
int8_t audio_high[W];

int8_t output_high[W_OUT * OUT_CH];
int8_t output_low[W_OUT * OUT_CH];

int16_t output_recon[W_OUT * OUT_CH];


// -----------------------------------------------------------------------------
// Initialize random input, filter and bias data
// -----------------------------------------------------------------------------

void init_data()
{
    for (int i = 0; i < BATCH * H * W * IN_CH; i++) {
        input[i] = (rand() % 255) - 128;
    }

    for (int i = 0; i < OUT_CH * K_H * K_W * IN_CH; i++) {
        filter[i] = (rand() % 255) - 128;
    }

    for (int i = 0; i < OUT_CH; i++) {
        bias[i] = rand() % 10;
    }
}


// -----------------------------------------------------------------------------
// Split INT16 input into high and low INT8 components
// -----------------------------------------------------------------------------

void split_int16(
    const int16_t* x_int16,
    int8_t* x_hi,
    int8_t* x_low,
    size_t n
)
{
    for (size_t i = 0; i < n; ++i) {
        int16_t v = x_int16[i];

        x_hi[i] = (int8_t)(v >> 8);
        x_low[i] = (int8_t)((v >> 1) & 0x7F);
    }
}


// -----------------------------------------------------------------------------
// Reconstruct INT16 output from high and low INT8 components
// -----------------------------------------------------------------------------

void reconstruct_int16(
    const int8_t* x_high,
    const int8_t* x_low,
    int16_t* x_recon,
    size_t n
)
{
    for (size_t i = 0; i < n; ++i) {
        x_recon[i] =
            ((int16_t)x_high[i] << 8) +
            (int8_t)(x_low[i] << 1);
    }
}


// -----------------------------------------------------------------------------
// Run convolution benchmark
// -----------------------------------------------------------------------------

int slice_benchmark()
{
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


    // Convolution parameters
    data_2d_t stride = {STRIDE_W, 1};
    data_2d_t padding = {1, 1};
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


    // Quantization parameters
    static int32_t mult[OUT_CH];
    static int32_t shift[OUT_CH];

    for (int i = 0; i < OUT_CH; i++) {
        mult[i] = 1;
        shift[i] = 0;
    }

    quant_data_t quant_data = {
        .shift = shift,
        .mult = mult
    };


    // Allocate scratch buffer
    int scratch_size = esp_nn_get_conv_scratch_size(
        &input_dims,
        &filter_dims,
        &output_dims,
        &conv_params
    );

    MicroPrintf("Scratch size: %d bytes", scratch_size);

    void* scratch_buf = heap_caps_malloc(
        scratch_size,
        MALLOC_CAP_INTERNAL
    );

    if (scratch_buf == nullptr) {
        MicroPrintf("Failed to allocate convolution scratch buffer");
        return -1;
    }

    esp_nn_set_conv_scratch_buf(scratch_buf);


    // Print tensor dimensions
    MicroPrintf(
        "Input: n=%d h=%d w=%d c=%d",
        input_dims.extra,
        input_dims.height,
        input_dims.width,
        input_dims.channels
    );

    MicroPrintf(
        "Filter: n=%d h=%d w=%d c=%d",
        filter_dims.extra,
        filter_dims.height,
        filter_dims.width,
        filter_dims.channels
    );

    MicroPrintf(
        "Output: n=%d h=%d w=%d c=%d",
        output_dims.extra,
        output_dims.height,
        output_dims.width,
        output_dims.channels
    );


    // Initialize data
    init_data();

    const size_t output_size = BATCH * H * W_OUT * OUT_CH;

    memset(output, 0x55, output_size);


    // -------------------------------------------------------------------------
    // Warm-up
    // -------------------------------------------------------------------------

    split_int16(
        audio_int16,
        audio_high,
        audio_low,
        W
    );

    reconstruct_int16(
        output_high,
        output_low,
        output_recon,
        W_OUT * OUT_CH
    );

    esp_nn_conv_s8(
        &input_dims,
        input,
        &filter_dims,
        filter,
        bias,
        &output_dims,
        output,
        &conv_params,
        &quant_data
    );


    // -------------------------------------------------------------------------
    // Measure split
    // -------------------------------------------------------------------------

    int64_t start_us = esp_timer_get_time();

    for (int i = 0; i < N_RUNS; ++i) {
        split_int16(
            audio_int16,
            audio_high,
            audio_low,
            W
        );
    }

    int64_t split_total_us = esp_timer_get_time() - start_us;


    // -------------------------------------------------------------------------
    // Measure reconstruction
    // -------------------------------------------------------------------------

    start_us = esp_timer_get_time();

    for (int i = 0; i < N_RUNS; ++i) {
        reconstruct_int16(
            output_high,
            output_low,
            output_recon,
            W_OUT * OUT_CH
        );
    }

    int64_t recon_total_us = esp_timer_get_time() - start_us;


    // -------------------------------------------------------------------------
    // Measure convolution
    // -------------------------------------------------------------------------

    start_us = esp_timer_get_time();

    for (int i = 0; i < N_RUNS; ++i) {
        esp_nn_conv_s8(
            &input_dims,
            input,
            &filter_dims,
            filter,
            bias,
            &output_dims,
            output,
            &conv_params,
            &quant_data
        );
    }

    int64_t conv_total_us = esp_timer_get_time() - start_us;


    // -------------------------------------------------------------------------
    // Calculate average time per run
    // -------------------------------------------------------------------------

    double split_per_run_us =
        static_cast<double>(split_total_us) / N_RUNS;

    double recon_per_run_us =
        static_cast<double>(recon_total_us) / N_RUNS;

    double conv_per_run_us =
        static_cast<double>(conv_total_us) / N_RUNS;


    // -------------------------------------------------------------------------
    // Scale timings to the complete signal
    // -------------------------------------------------------------------------

    double split_signal_us = split_per_run_us * N_FRAMES;
    double recon_signal_us = recon_per_run_us * N_FRAMES;
    double conv_signal_us = conv_per_run_us * N_FRAMES;


    // -------------------------------------------------------------------------
    // Print results
    // -------------------------------------------------------------------------

    MicroPrintf("");
    MicroPrintf("========== Benchmark results ==========");

    MicroPrintf("Signal length: %d samples", SIGNAL_LENGTH);
    MicroPrintf("Frame length: %d samples", W);
    MicroPrintf("Number of frames: %d", N_FRAMES);
    MicroPrintf("Runs: %d", N_RUNS);

    MicroPrintf("");

    printf("Split:\n");
    printf("  Time per run:    %.2f us\n", split_per_run_us);
    printf("  Full signal:     %.2f us\n", split_signal_us);

    printf("Reconstruction:\n");
    printf("  Time per run:    %.2f us\n", recon_per_run_us);
    printf("  Full signal:     %.2f us\n", recon_signal_us);

    printf("Convolution:\n");
    printf("  Time per run:    %.2f us\n", conv_per_run_us);
    printf("  Full signal:     %.2f us\n", conv_signal_us);

    double bit_slicing_us = split_signal_us + recon_signal_us + conv_signal_us*2;

    printf("Bit-slicing:     %.2f us\n", bit_slicing_us);

    printf("=======================================\n");


    // Check whether convolution output changed
    bool changed = false;

    for (size_t i = 0; i < output_size; i++) {
        if (output[i] != 0x55) {
            changed = true;
            break;
        }
    }

    MicroPrintf("Output changed: %d", changed);


    // Free scratch buffer
    heap_caps_free(scratch_buf);

    return 0;
}