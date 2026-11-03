
"""Utilities for Gabor kernel extraction and Keras model inspection."""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def extract_gabor_kernels(
    model: keras.Model,
    layer_idx: int,
    sampling_rate: int = 20480,
    threshold: float = 0.1,
) -> tuple[keras.layers.Conv1D, np.ndarray]:
    """
    Reconstruct Gabor convolution kernels from a trained model.

    The kernels are generated using the learned center frequencies
    and bandwidth parameters of the selected layer.

    Args:
        model: Trained Keras model.
        layer_idx: Index of the layer containing Gabor parameters.
        sampling_rate: Sampling frequency in Hz.
        threshold: Magnitude threshold for kernel sparsification.

    Returns:
        A Conv1D layer initialized with the reconstructed kernels,
        and the corresponding kernel weights.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if not 0 <= threshold:
        raise ValueError("Threshold must be non-negative.")

    center_freq, bandwidth = model.layers[layer_idx].get_weights()

    reference_layer = model.layers[1]

    kernel_size = (
        reference_layer.kernel_size[0]
        if isinstance(reference_layer.kernel_size, tuple)
        else reference_layer.kernel_size
    )
    stride = reference_layer.stride
    padding = reference_layer.padding

    out_channels = len(center_freq)

    # Construct the temporal sampling grid.
    time = tf.linspace(
        -(kernel_size // 2),
        kernel_size // 2,
        kernel_size,
    )

    time = tf.cast(time, tf.float32) / sampling_rate
    time = tf.reshape(time, [kernel_size, 1])

    # Reshape learned Gabor parameters for broadcasting.
    center_freq = tf.reshape(center_freq, [1, out_channels])
    bandwidth = tf.reshape(bandwidth, [1, out_channels])

    pi = tf.constant(np.pi, dtype=tf.float32)

    # Generate Gaussian-windowed sinusoidal kernels.
    sinusoid = tf.cos(2.0 * pi * center_freq * time)
    gaussian = tf.exp(-(pi * bandwidth * time) ** 2)

    kernels = sinusoid * gaussian

    # Convert to Conv1D kernel format: (kernel_size, 1, channels).
    kernels = tf.reshape(
        kernels,
        [kernel_size, 1, out_channels],
    ).numpy()

    # Apply magnitude-based sparsification.
    kernels[np.abs(kernels) < threshold] = 0.0

    conv_layer = keras.layers.Conv1D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        kernel_initializer=keras.initializers.Constant(kernels),
        name="conv1d_gabor",
    )

    return conv_layer, kernels


def copy_layer_weights(
    old_layer: keras.layers.Layer,
    input_shape: tuple,
) -> keras.layers.Layer:
    """
    Recreate a Keras layer from its configuration and copy its weights.

    Args:
        old_layer: Original Keras layer.
        input_shape: Expected input shape excluding the batch dimension.

    Returns:
        A new layer with the same configuration and weights.
    """

    new_layer = type(old_layer).from_config(
        old_layer.get_config()
    )

    new_layer.build(input_shape)
    new_layer.set_weights(old_layer.get_weights())

    return new_layer


def create_activation_model(
    model: keras.Model,
    layer_names: list[str],
) -> keras.Model:
    """
    Create a model exposing intermediate layer activations.

    Args:
        model: Original Keras model.
        layer_names: Names of layers whose outputs are required.

    Returns:
        A Keras model returning the selected intermediate activations.
    """

    outputs = [
        model.get_layer(name).output
        for name in layer_names
    ]

    return keras.Model(
        inputs=model.inputs,
        outputs=outputs,
    )