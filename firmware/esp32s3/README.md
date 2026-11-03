# ESP32-S3 Conv1D profiling and Bit-sliced Conv1D

## 1. Environment Setup

This project uses ESP-IDF v5.5.2. 

After installing ESP-IDF, set up the ESP-IDF environment:

```bash
export IDF_PATH=~/.espressif/v5.5.2/esp-idf
source ~/.espressif/tools/activate_idf_v5.5.2.sh
```

Then run:
```bash
idf.py flash monitor
```

## 2. Benchmark Selection

The repository contains two main benchmark implementations:

1. **Bit-sliced benchmark:** gives the execusion time of bit-sliced convolution.
2. **Conv1D profling benchmark:** Evaluates multiple convolution configurations by varying:
   - Kernel width (`K_W`)
   - Number of output channels (`OUT_CH`)
   - Stride (`STRIDE_W`)

The benchmark implementation can be selected in `main.cpp`.

For example:

```cpp
#define BENCHMARK_SLICE          1
#define BENCHMARK_CONV1D   2

#define SELECTED_BENCHMARK BENCHMARK_CONV1D
```

runs the Conv1D profling. 

## Acknowledgements and License

This repository contains benchmark code adapted from the BioDCASE-Tiny-2026 project.

Original repository: [BirdNET – BioDCASE-Tiny-2026](https://github.com/birdnet-team/BioDCASE-Tiny-2026)