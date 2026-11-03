import numpy as np
import tensorflow as tf


def get_tf_results(X_test, model_path):
    """Run inference on a TFLite model.

    Returns:
        output_X: Dequantized model predictions.
        output_pre: Raw model outputs.
        output_scale: Output quantization scale.
        output_zero_point: Output quantization zero point.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    # Check input quantization parameters when using integer inputs
    if np.issubdtype(input_dtype, np.integer) and input_scale <= 0:
        raise ValueError("Invalid input quantization scale.")

    output_pre = []
    output_X = []

    for x in X_test:
        test_input = np.asarray(x, dtype=np.float32)
        test_input = np.expand_dims(test_input, axis=0)

        # Quantize input if the model expects integer values
        if np.issubdtype(input_dtype, np.integer):
            test_input = np.round(
                test_input / input_scale + input_zero_point
            )

            limits = np.iinfo(input_dtype)
            test_input = np.clip(
                test_input, limits.min, limits.max
            ).astype(input_dtype)
        else:
            test_input = test_input.astype(input_dtype)

        interpreter.set_tensor(input_details["index"], test_input)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details["index"])
        output_pre.append(output_data)

        # Dequantize output if the model returns integer values
        if np.issubdtype(output_dtype, np.integer):
            output_data = (
                output_data.astype(np.float32) - output_zero_point
            ) * output_scale
        else:
            output_data = output_data.astype(np.float32)

        output_X.append(output_data)

    return (
        np.array(output_X),
        np.array(output_pre),
        output_scale,
        output_zero_point,
    )