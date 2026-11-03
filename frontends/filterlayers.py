
import librosa
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class MinConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value):
        self.min_value = float(min_value)

    def __call__(self, w):
        return tf.maximum(w, self.min_value)

    def get_config(self):
        return {"min_value": self.min_value}


@tf.keras.utils.register_keras_serializable()
class GaborConv1D(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        fs,
        stride=1,
        padding="SAME",
        fmin=2000.0,
        fmax=10000.0,
        band_min=500.0,
        bandwidth_init_min=50.0,
        bandwidth_init_max=1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.fs = float(fs)
        self.stride = int(stride)
        self.padding = padding

        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.band_min = float(band_min)

        self.bandwidth_init_min = float(bandwidth_init_min)
        self.bandwidth_init_max = float(bandwidth_init_max)

    def build(self, input_shape):
        mel_freqs, _ = self.init_from_mels()

        self.center_hz = self.add_weight(
            name="center_hz",
            shape=(self.out_channels,),
            initializer=tf.keras.initializers.Constant(mel_freqs),
            trainable=True,
            dtype=tf.float32,
        )

        self.bandwidth = self.add_weight(
            name="bandwidth",
            shape=(self.out_channels,),
            initializer=tf.keras.initializers.RandomUniform(
                minval=self.bandwidth_init_min,
                maxval=self.bandwidth_init_max,
            ),
            trainable=True,
            dtype=tf.float32,
            constraint=MinConstraint(self.band_min),
        )

        super().build(input_shape)

    def init_from_mels(self):
        mel_edges = librosa.mel_frequencies(
            n_mels=self.out_channels + 2,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        mel_freqs = mel_edges[1:-1]
        bandwidths = mel_edges[2:] - mel_edges[:-2]

        return mel_freqs, bandwidths

    def call(self, x):
        t = tf.linspace(
            -(self.kernel_size // 2),
            self.kernel_size // 2,
            self.kernel_size,
        )
        t = tf.cast(t, tf.float32) / self.fs
        t = tf.reshape(t, [self.kernel_size, 1])

        f0 = tf.reshape(self.center_hz, [1, self.out_channels])
        bw = tf.reshape(self.bandwidth, [1, self.out_channels])

        pi = tf.constant(3.141592653589793, dtype=tf.float32)

        sinusoid = tf.cos(2 * pi * f0 * t)
        gaussian = tf.exp(-tf.square(pi * bw * t))

        kernels = sinusoid * gaussian
        kernels = tf.reshape(
            kernels,
            [self.kernel_size, 1, self.out_channels],
        )

        return tf.nn.conv1d(
            x,
            kernels,
            stride=self.stride,
            padding=self.padding,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "fs": self.fs,
            "stride": self.stride,
            "padding": self.padding,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "band_min": self.band_min,
            "bandwidth_init_min": self.bandwidth_init_min,
            "bandwidth_init_max": self.bandwidth_init_max,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class LogLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, x):
        return tf.math.log(tf.abs(x) + self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config