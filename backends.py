import tensorflow as tf
from frontends.filterlayers import GaborConv1D, LogLayer

def create_mel_backend(input_shape, nb_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_main = tf.keras.layers.Conv2D(filters=8, kernel_size=7, strides=(1,4), activation='relu')(input_layer)
    output_main = tf.keras.layers.MaxPooling2D(pool_size=2) (output_main)

    output_main = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu') (output_main)
    output_main = tf.keras.layers.MaxPooling2D(pool_size=2) (output_main)

    output_main = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu') (output_main)

    output_main = tf.keras.layers.GlobalAveragePooling2D()(output_main)
    output_main = tf.keras.layers.Dropout(0.2)(output_main)
    output_main = tf.keras.layers.Dense(units=128, activation='relu') (output_main)
    output_main = tf.keras.layers.Dense(units=nb_classes, activation='softmax', name='output_main') (output_main)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_main)
    return model


def create_song_gabor_model(input_shape, nb_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    conv1d_1 = GaborConv1D(out_channels=32, kernel_size=101, stride=64, fs=20480)(input_layer)

    conv1d_1 = LogLayer(name='log_layer')(conv1d_1)
    conv1d_1 = tf.keras.layers.AveragePooling1D(pool_size=4, name="average_pooling1d")(conv1d_1)

    output_main = tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=2, activation='relu')(conv1d_1)
    output_main = tf.keras.layers.MaxPooling1D(pool_size=2)(output_main)
    
    output_main = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=2, activation='relu')(output_main)
    output_main = tf.keras.layers.MaxPooling1D(pool_size=8)(output_main)
    output_main = tf.keras.layers.Flatten()(output_main)
    output_main = tf.keras.layers.Dense(units=64, activation='relu')(output_main)
    output_main = tf.keras.layers.Dense(units=nb_classes, activation='softmax')(output_main)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_main)
    return model