import os
import numpy as np
import tensorflow as tf

def convert_model(outfolder, model_path, name):
    
    os.makedirs(outfolder, exist_ok=True)

    # Code taken from students
    model = tf.keras.models.load_model(model_path)

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_1 = converter.convert()

    # Save the converted model to a .tflite file
    with open(os.path.join(outfolder,name+'_trained_model.tflite'), 'wb') as f:
        f.write(tflite_model_1)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dynamic_model = converter.convert()

    # Save the converted model to a .tflite file
    with open(os.path.join(outfolder,name+'_dynamic_model.tflite'), 'wb') as f:
        f.write(dynamic_model)

def _representative_dataset(X_train, size=2000, add_channel_dimension=False):
    """Generate representative samples for TFLite quantization.

    Samples are drawn exclusively from the training dataset.
    """
    X = np.asarray(X_train)

    if add_channel_dimension:
        X = X[..., np.newaxis]

    if len(X) == 0:
        raise ValueError("X_train must contain at least one sample.")

    n_samples = min(size, len(X))

    for sample in X[:n_samples]:
        sample = np.expand_dims(sample, axis=0)
        yield [sample.astype(np.float32)]

def _save_tflite_model(model_bytes, outfolder, filename):
    """Save a converted TFLite model."""
    os.makedirs(outfolder, exist_ok=True)

    output_path = os.path.join(outfolder, filename)

    with open(output_path, "wb") as f:
        f.write(model_bytes)

    return output_path

def _convert_quantized(
    outfolder,
    model_path,
    name,
    X_train,
    input_type,
    output_type,
    supported_ops,
    quantization_name,
    representative_size=2000,
    add_channel_dimension=False,
):
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = lambda: _representative_dataset(
        X_train,
        size=representative_size,
        add_channel_dimension=add_channel_dimension,
    )

    converter.target_spec.supported_ops = [supported_ops]
    converter.inference_input_type = input_type
    converter.inference_output_type = output_type

    tflite_model = converter.convert()

    filename = f"{name}_{quantization_name}_model.tflite"

    return _save_tflite_model(tflite_model, outfolder, filename)


def convert_model_full_int(outfolder, model_path, name, X_train,
                           keep_data=True):
    return _convert_quantized(
        outfolder, model_path, name, X_train,
        tf.int8, tf.int8,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        quantization_name="int8",
        add_channel_dimension=not keep_data,
    )


def convert_model_full_uint(outfolder, model_path, name, X_train,
                            keep_data=True):
    return _convert_quantized(
        outfolder, model_path, name, X_train,
        tf.uint8, tf.uint8,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        quantization_name="uint8",
        add_channel_dimension=not keep_data,
    )


def convert_model_int16(outfolder, model_path, name, X_train):
    return _convert_quantized(
        outfolder, model_path, name, X_train,
        tf.int16, tf.int16,
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        quantization_name="int16",
        representative_size=4620,
    )
