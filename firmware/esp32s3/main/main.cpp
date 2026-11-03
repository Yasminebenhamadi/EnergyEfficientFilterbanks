#include "tensorflow/lite/micro/micro_log.h"

// Benchmark selection
#define BENCHMARK_SLICE          1
#define BENCHMARK_CONV1D   2

// Select benchmark here
#define SELECTED_BENCHMARK BENCHMARK_SLICE


// Benchmark declarations
int slice_benchmark();
int conv_benchmark_layers();


extern "C" void app_main()
{
    MicroPrintf("\n================================");
    MicroPrintf("      ESP-NN Benchmark");
    MicroPrintf("================================");

#if SELECTED_BENCHMARK == BENCHMARK_SLICE

    MicroPrintf("Running slice benchmark...");
    slice_benchmark();

#elif SELECTED_BENCHMARK == BENCHMARK_CONV1D

    MicroPrintf("Running configurable benchmark...");
    conv_benchmark_layers();

#else

    MicroPrintf("Invalid benchmark selection!");

#endif

    MicroPrintf("Benchmark finished.");
}