
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers, losses, metrics

from dataset import YellowhammerData, augment_distances
import frontends.features as features
from backends import create_mel_backend
import deploy.quantize as quantize
import deploy.tflite_utils as tflite_utils


# Parameters
SPLIT_NAME = "full_ID_Set_split"
SAMPLE_DURATION = 1.5

SAMPLE_RATE = 20480
WINDOW_LENGTH = 512
WINDOW_STRIDE = 320
N_MELS = 32
FMIN = 2000
FMAX = 10000

N_RUNS = 20
BASE_SEED = 42
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 0.2
PATIENCE = 10

MODEL_NAME = "mel_2DCNN"
OUTPUT_FOLDER = "Models"
RESULTS_FILE = "yellowhammer_results.csv"
EXPERIMENT = "Mel"


def load_data():
    frontend = features.MelSpecCMSIS(
        samplerate=SAMPLE_RATE,
        window_len=WINDOW_LENGTH,
        window_stride=WINDOW_STRIDE,
        nb_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )

    train_data = YellowhammerData(
        SPLIT_NAME,
        key="train",
        duration=SAMPLE_DURATION,
        augment=augment_distances,
        transform=frontend,
    )
    X_train, y_train = train_data.load_feature(binary=False)

    test_data = YellowhammerData(
        SPLIT_NAME,
        key="test",
        duration=SAMPLE_DURATION,
        augment=None,
        transform=frontend,
    )
    X_test, y_test, _ = test_data.load_feature(
        binary=False,
        with_dist=True,
    )

    return (
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_test),
        np.asarray(y_test),
    )


def encode_labels(y_train, y_test):
    classes = np.unique(y_train)
    class_to_index = {label: i for i, label in enumerate(classes)}

    unknown_classes = set(y_test) - set(classes)
    if unknown_classes:
        raise ValueError(f"Unknown test classes: {unknown_classes}")

    y_train_idx = np.array([class_to_index[label] for label in y_train])
    y_test_idx = np.array([class_to_index[label] for label in y_test])

    y_train_one = tf.keras.utils.to_categorical(
        y_train_idx,
        num_classes=len(classes),
    )

    return y_train_one, y_test_idx, len(classes)


def train_model(X_fit, y_fit, X_val, y_val, n_classes, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = create_mel_backend(X_fit.shape[1:], n_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.CategoricalFocalCrossentropy(),
        metrics=[metrics.F1Score(average="macro")],
    )

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    return model, min(history.history["val_loss"])


def evaluate_model(model_path, X_test, y_test, n_classes):
    scores, _, _, _ = tflite_utils.get_tf_results(X_test, model_path)
    scores = np.asarray(scores).reshape(len(X_test), -1)

    if scores.shape[1] != n_classes:
        raise ValueError("Model output does not match the number of classes.")

    predictions = np.argmax(scores, axis=1)
    return f1_score(y_test, predictions, average="macro")


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

    X_train, y_train, X_test, y_test = load_data()
    y_train_one, y_test_idx, n_classes = encode_labels(y_train, y_test)

    print("Train:", X_train.shape, y_train_one.shape)
    print("Test:", X_test.shape, y_test_idx.shape)

    # Fixed stratified validation split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train_one,
        test_size=VALIDATION_SIZE,
        random_state=BASE_SEED,
        stratify=np.argmax(y_train_one, axis=1),
    )

    results = []

    for run_idx in range(N_RUNS):
        run_number = run_idx + 1
        seed = BASE_SEED + run_idx

        print(f"\nRun {run_number}/{N_RUNS}")

        model, validation_loss = train_model(
            X_fit,
            y_fit,
            X_val,
            y_val,
            n_classes,
            seed,
        )

        run_name = f"{MODEL_NAME}_run_{run_number}"
        model_path = os.path.join(
            OUTPUT_FOLDER,
            f"{run_name}.keras",
        )
        model.save(model_path)

        quantize.convert_model_full_int(
            OUTPUT_FOLDER,
            model_path,
            run_name,
            X_train,
        )

        int8_path = os.path.join(
            OUTPUT_FOLDER,
            f"{run_name}_int8_model.tflite",
        )

        f1 = evaluate_model(
            int8_path,
            X_test,
            y_test_idx,
            n_classes,
        )

        results.append({
            "experiment": EXPERIMENT,
            "inference": "INT8",
            "run": run_number,
            "seed": seed,
            "validation_loss": validation_loss,
            "f1_macro": f1,
        })

        save_results(results)

        print(
            f"INT8 F1={f1:.4f} | "
            f"Validation loss={validation_loss:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\nFinal results")
    print(
        f"INT8: {results_df['f1_macro'].mean():.4f} "
        f"± {results_df['f1_macro'].std():.4f}"
    )


if __name__ == "__main__":
    main()
