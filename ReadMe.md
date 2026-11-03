# Learning Energy-Efficient Filterbanks for On-device Bioacoustic Classification

This repository contains the Python code and firmware implementations for the **Apollo510** and **ESP32-S3** platforms associated with the following paper:

Yasmine Benhamadi, Vincent Lostanlen, Mathieu Lagrange.  
*Learning Energy-Efficient Filterbanks for On-device Bioacoustic Classification.*  
IEEE International Symposium on the Internet of Sounds (IS2), October 2026, Cannes, France.

[Paper (HAL)](https://hal.science/hal-05758524v2)

## Training and Quantization

Model training, quantization, and evaluation are performed in Python using TensorFlow and TensorFlow Lite.

- `quantize_gabor.py` contains the implementation of the bit-slicing procedure.
- `filterlayers.py` contains the implementation of the learnable Gabor layers.
- `prune.py` contains the code for recovering the effective Gabor kernels after structured pruning.

## Profiling on Apollo510 and ESP32-S3

We provide the complete inference pipeline implementation for the Apollo510 platform in [**TinyBioacAI-Apollo5**](https://github.com/Yasminebenhamadi/TinyBioacAI-Apollo5), using **NeuralSPOT**. Profiling on the Apollo510 was performed using the `ns_deploy` tool. For the ESP32-S3, we provide only the frontend profiling code, available in `firmware/esp32s3`.