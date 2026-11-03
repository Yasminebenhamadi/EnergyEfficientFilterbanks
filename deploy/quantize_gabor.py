
import os

import cmsisdsp as dsp
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models

import deploy.quantize as quantize
from deploy.tflite_utils import get_tf_results
from deploy.convert_utils import extract_gabor_kernels, copy_layer_weights


EPSILON = 1e-6


def model_path(folder, name, suffix):
    return os.path.join(folder, f"{name}{suffix}")


def save_conv_gabor(folder_models, name, input_shape):
    path = model_path(folder_models, name, ".keras")
    gabor_model = tf.keras.models.load_model(path)

    conv1d, _ = extract_gabor_kernels(gabor_model, layer_idx=1)

    gabor_model.trainable = False

    input_layer = layers.Input(shape=input_shape)
    output = conv1d(input_layer)

    for layer in gabor_model.layers[2:]:
        output = copy_layer_weights(layer, output.shape)(output)

    new_model = models.Model(inputs=input_layer, outputs=output)
    new_model.trainable = False

    new_model.save(model_path(folder_models, name, "_1D.keras"))

    # INT16 block: Gabor convolution
    input_int16 = layers.Input(shape=input_shape)
    output_int16 = copy_layer_weights(
        new_model.layers[1],
        input_int16.shape,
    )(input_int16)

    model_int16 = models.Model(
        inputs=input_int16,
        outputs=output_int16,
    )
    model_int16.save(
        model_path(folder_models, name, "_block_int16.keras")
    )

    # INT8 block: layers after the log layer
    # The log operation is applied separately during inference.
    input_int8 = layers.Input(
        shape=(output_int16.shape[-2], output_int16.shape[-1])
    )
    output_int8 = input_int8

    for layer in new_model.layers[3:]:
        output_int8 = copy_layer_weights(
            layer,
            output_int8.shape,
        )(output_int8)

    model_int8 = models.Model(
        inputs=input_int8,
        outputs=output_int8,
    )
    model_int8.save(
        model_path(folder_models, name, "_block_int8.keras")
    )


def split_input(X):
    """Decompose float input into high and low INT8 components."""

    X_q15 = dsp.arm_float_to_q15(X)

    X_high = (X_q15 >> 8).astype(np.int8)
    X_low = ((X_q15 >> 1) & 0x7F).astype(np.int8)

    X_high = dsp.arm_q7_to_float(X_high).reshape(X.shape)
    X_low = dsp.arm_q7_to_float(X_low).reshape(X.shape)

    return X_high, X_low


def reconstruct_features(X, folder_models, name):
    """Reconstruct Gabor features from high and low INT8 outputs."""

    X_high, X_low = split_input(X)

    low_path = model_path(
        folder_models, name, "_low_int8_model.tflite"
    )
    high_path = model_path(
        folder_models, name, "_hi_int8_model.tflite"
    )

    _, low_pre, low_scale, low_zero_point = get_tf_results(
        X_low, low_path
    )
    _, high_pre, high_scale, high_zero_point = get_tf_results(
        X_high, high_path
    )

    low_pre = np.squeeze(low_pre).astype(np.int8) - low_zero_point
    high_pre = np.squeeze(high_pre).astype(np.int8) - high_zero_point

    high_features = dsp.arm_q15_to_float(
        high_pre.astype(np.int16) << 8
    ) / high_scale

    low_features = dsp.arm_q15_to_float(
        (low_pre.astype(np.int8) << 1).astype(np.int16)
    ) / low_scale

    return (high_features + low_features).reshape(high_pre.shape)


def predict_from_features(X_features, folder_models, name):
    """Apply log compression and run the INT8 classification model."""

    X_log = np.log(np.abs(X_features) + EPSILON)

    path = model_path(
        folder_models, name, "_block_int8_model.tflite"
    )

    scores, _, _, _ = get_tf_results(X_log, path)

    return np.squeeze(scores)


def quantize_gabor(folder_models, name, data_yh):
    X_train, _ = data_yh.load_audio(binary=False, norm=True)
    input_shape = X_train[0].shape

    os.makedirs(folder_models, exist_ok=True)

    save_conv_gabor(folder_models, name, input_shape)

    # Quantize Gabor convolution
    int16_path = model_path(
        folder_models, name, "_block_int16.keras"
    )

    quantize.convert_model_int16(
        folder_models,
        int16_path,
        name + "_block",
        X_train,
    )

    quantize.convert_model_full_int(
        folder_models,
        int16_path,
        name + "_test",
        X_train,
    )

    # Quantize high/low input components
    X_high, X_low = split_input(X_train)

    quantize.convert_model_full_int(
        folder_models,
        int16_path,
        name + "_low",
        X_low,
    )

    quantize.convert_model_full_int(
        folder_models,
        int16_path,
        name + "_hi",
        X_high,
    )

    quantize.convert_model_full_int(
        folder_models,
        int16_path,
        name + "_split",
        np.concatenate([X_high, X_low]),
    )

    # Generate features for INT8 classification model
    X_features, _, _, _ = get_tf_results(
        X_train,
        model_path(folder_models, name, "_block_int16_model.tflite"),
    )

    X_features = np.squeeze(X_features)
    X_log = np.log(np.abs(X_features) + EPSILON)

    int8_model_path = model_path(
        folder_models, name, "_block_int8.keras"
    )

    quantize.convert_model_full_int(
        folder_models,
        int8_model_path,
        name + "_block",
        X_log,
    )


def evaluate_gabor(X_test, folder_models, name):
    """Inference using the INT16 Gabor block."""

    int16_path = model_path(
        folder_models, name, "_block_int16_model.tflite"
    )

    X_features, _, _, _ = get_tf_results(X_test, int16_path)

    return predict_from_features(
        np.squeeze(X_features),
        folder_models,
        name,
    )


def evaluate_split(X_test, folder_models, name):
    """Inference using high/low INT8 input decomposition."""

    X_features = reconstruct_features(
        X_test,
        folder_models,
        name,
    )

    return predict_from_features(
        X_features,
        folder_models,
        name,
    )