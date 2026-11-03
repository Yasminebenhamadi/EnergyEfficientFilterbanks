
import os
import numpy as np
import tensorflow as tf
from frontends.filterlayers import *


# Parameters
MODEL_FOLDER = "Models"
MODEL_NAME = "gabor_conv1d"
RUN_START = 1
RUN_END = 19


def get_prune_idx(kernels):
    prune_start = []
    prune_end = []

    for kernel in kernels:
        non_zero_idx = np.flatnonzero(kernel)

        if len(non_zero_idx) == 0:
            prune_start.append(0)
            prune_end.append(-1)
        else:
            prune_start.append(non_zero_idx.min())
            prune_end.append(non_zero_idx.max())

    return np.array(prune_start), np.array(prune_end)


def main():
    for run_idx in range(RUN_START, RUN_END + 1):
        model_path = os.path.join(
            MODEL_FOLDER,
            f"{MODEL_NAME}_run_{run_idx}_1D.keras",
        )

        model = tf.keras.models.load_model(model_path)

        kernels = np.squeeze(model.get_weights()[0]).T

        idx_start, idx_end = get_prune_idx(kernels)
        effective_lengths = idx_end - idx_start + 1

        print(
            f"Run {run_idx}: "
            f"Pruned kernel length = {np.max(effective_lengths)} "
        )
        print(f"Effective kernel lengths = {np.sort(effective_lengths)}")


if __name__ == "__main__":
    main()
