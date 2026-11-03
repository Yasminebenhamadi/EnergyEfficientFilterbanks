
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers, losses, metrics

from dataset import YellowhammerData, augment_distances
import backends as yh_models
import deploy.quantize as quantize
import deploy.quantize_gabor as qt_gabor


# Parameters
SPLIT_NAME = "full_ID_Set_split"
DURATION = 1.5

N_RUNS = 20
BASE_SEED = 42
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 0.2

MODEL_NAME = "gabor_conv1d"
OUTPUT_FOLDER = "Models"
RESULTS_FILE = "yellowhammer_results.csv"
EXPERIMENT = "Gabor"


def evaluate_f1(y_true, scores, n_classes):
    scores = np.asarray(scores)

    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0, :]

    if scores.shape != (len(y_true), n_classes):
        raise ValueError(f"Unexpected prediction shape: {scores.shape}")

    predictions = np.argmax(scores, axis=1)
    return f1_score(y_true, predictions, average="macro")


def save_results(results):
    results_path = os.path.join(OUTPUT_FOLDER, RESULTS_FILE)
    new_results = pd.DataFrame(results)

    if os.path.exists(results_path):
        previous_results = pd.read_csv(results_path)
        previous_results = previous_results[
            previous_results["experiment"] != EXPERIMENT
        ]
        new_results = pd.concat(
            [previous_results, new_results],
            ignore_index=True,
        )

    new_results.to_csv(results_path, index=False)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load data
    data_train = YellowhammerData(
        SPLIT_NAME,
        key="train",
        duration=DURATION,
        augment=augment_distances,
        transform=None,
    )
    X_train, y_train = data_train.load_audio(binary=False, norm=True)

    data_test = YellowhammerData(
        SPLIT_NAME,
        key="test",
        duration=DURATION,
        augment=None,
        transform=None,
    )
    X_test, y_test = data_test.load_audio(binary=False, norm=True)

    # Encode labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    n_classes = len(encoder.classes_)
    y_train_one = tf.keras.utils.to_categorical(y_train, n_classes)
    input_shape = X_train[0].shape

    print("Train:", X_train.shape, y_train_one.shape)
    print("Test:", X_test.shape, y_test.shape)

    # Fixed stratified validation split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train_one,
        test_size=VALIDATION_SIZE,
        random_state=BASE_SEED,
        stratify=y_train,
    )

    results = []

    for run_idx in range(N_RUNS):
        run_number = run_idx + 1
        seed = BASE_SEED + run_idx

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        run_name = f"{MODEL_NAME}_run_{run_number}"
        model_path = os.path.join(OUTPUT_FOLDER, f"{run_name}.keras")

        print(f"\nRun {run_number}/{N_RUNS}")

        model = yh_models.create_song_gabor_model(
            input_shape,
            n_classes,
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.F1Score(average="macro")],
        )

        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            shuffle=True,
            verbose=1,
        )

        validation_loss = min(history.history["val_loss"])
        model.save(model_path)

        # Quantization
        qt_gabor.quantize_gabor(OUTPUT_FOLDER, run_name, data_train)

        quantize.convert_model_full_int(
            OUTPUT_FOLDER,
            model_path,
            run_name,
            X_train,
        )

        # Gabor inference
        scores = qt_gabor.evaluate_gabor(
            X_test, OUTPUT_FOLDER, run_name
        )
        gabor_f1 = evaluate_f1(y_test, scores, n_classes)

        # INT8 TFLite inference
        int8_path = os.path.join(
            OUTPUT_FOLDER,
            f"{run_name}_int8_model.tflite",
        )
        scores, _, _, _ = qt_gabor.get_tf_results(X_test, int8_path)
        int8_f1 = evaluate_f1(y_test, scores, n_classes)

        # Slice inference
        scores = qt_gabor.evaluate_split(
            X_test, OUTPUT_FOLDER, run_name
        )
        slice_f1 = evaluate_f1(y_test, scores, n_classes)

        for inference, f1 in [
            ("GABOR", gabor_f1),
            ("INT8", int8_f1),
            ("SLICE", slice_f1),
        ]:
            results.append({
                "experiment": EXPERIMENT,
                "inference": inference,
                "run": run_number,
                "seed": seed,
                "validation_loss": validation_loss,
                "f1_macro": f1,
            })

        save_results(results)

        print(
            f"GABOR F1={gabor_f1:.4f} | "
            f"INT8 F1={int8_f1:.4f} | "
            f"SLICE F1={slice_f1:.4f} | "
            f"Validation loss={validation_loss:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\nFinal results")
    for inference, group in results_df.groupby("inference"):
        print(
            f"{inference}: "
            f"{group['f1_macro'].mean():.4f} "
            f"± {group['f1_macro'].std():.4f}"
        )


if __name__ == "__main__":
    main()