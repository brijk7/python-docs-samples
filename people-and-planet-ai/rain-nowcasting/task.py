# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""This training script trains binary classifier on Sentinel-2 satellite images.
The model is a fully convolutional neural network that predicts whether a power
plant is turned on or off.

A Sentinel-2 image consists of 13 bands. Each band contains the data for a
specific range of the electromagnetic spectrum.

A JPEG image consists of three channels: Red, Green, and Blue. For Sentinel-2
images, these correspond to Band 4 (red), Band 3 (green), and Band 2 (blue).
These bands contain the raw pixel data directly from the satellite sensors.
For more information on the Sentinel-2 dataset:
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
"""

import argparse
import tensorflow as tf

GOES_BANDS = [f'CMI_C{i:02d}' for i in range(1,17)]
BANDS = [f'T{t}_CMI_C{i:02d}' for t in [0,1,2] for i in range(1,17)]
LABEL1 = 'HQprecipitation'
MODEL_LABEL1 = 'rainAmt'
LABEL2 = 'probabilityLiquidPrecipitation'
MODEL_LABEL2 = 'rainChance'
GOES_PATCH_SIZE = 65
GPM_PATCH_SIZE = 65
CONFIG = 'p64_s11132'
USE_CATEGORICAL_LABEL1 = False


def get_args():
    """Parses args."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, type=str, help="GCS Bucket")
    args = parser.parse_args()
    return args


def parse_tfrecord(example_proto, features_dict):
    """Parses a single tf.train.Example."""

    return tf.io.parse_single_example(example_proto, features_dict)


def create_features_dict():
    """Creates dict of features."""

    features_dict = {
      band_name: tf.io.FixedLenFeature(shape=[GOES_PATCH_SIZE, GOES_PATCH_SIZE], dtype=tf.float32)
      for band_name in BANDS
    }
    features_dict[LABEL1] = tf.io.FixedLenFeature(shape=[GPM_PATCH_SIZE, GPM_PATCH_SIZE], dtype=tf.float32)
    features_dict[LABEL2] = tf.io.FixedLenFeature(shape=[GPM_PATCH_SIZE, GPM_PATCH_SIZE], dtype=tf.float32)
    return features_dict


def split_inputs_and_labels(values: dict):
    """Splits a TFRecord value dictionary into input and label tensors for the model."""
    inputs = []
    for t in ['T0','T1','T2']:
      inputs.append([values[f'{t}_CMI_C{b:02d}'] for b in range(1,17)])
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.transpose(inputs, [0,2,3,1])

    # Crop label1 array to even size
    # Clip precipitation range to 0-20, and quantize to 2 mm/hr ranges.
    tlabel1 = values.pop(LABEL1)
    if USE_CATEGORICAL_LABEL1:
      tlabel1 = tf.math.divide(tf.clip_by_value(tlabel1, 0, 20.0), tf.constant([2.0]))
      tlabel1 = tf.one_hot(tf.cast(tlabel1, tf.uint8), 10)
    else:
      tlabel1 = tf.expand_dims(tlabel1, -1)
    #tlabel1 = tf.image.resize_with_crop_or_pad(tlabel1, GPM_PATCH_SIZE-1, GPM_PATCH_SIZE-1)

    # Take the mean probability of the central 8x8 square for label2
    tlabel2 = tf.expand_dims(values.pop(LABEL2), -1)
    tlabel2 = tf.image.resize_with_crop_or_pad(tlabel2, 8, 8)
    tlabel2 = tf.divide(tf.math.reduce_mean(tlabel2), tf.constant([100.0]))

    labels = {}
    labels[MODEL_LABEL1] = tlabel1
    labels[MODEL_LABEL2] = tlabel2

    return inputs, labels


def create_datasets(bucket):
    """Creates training and validation datasets."""

    train_data_file = f"gs://{bucket}/nowcasting/{CONFIG}/training.tfrecord.gz"
    eval_data_file = f"gs://{bucket}/nowcasting/{CONFIG}/validation.tfrecord.gz"
    features_dict = create_features_dict()

    training_dataset = (
        tf.data.TFRecordDataset(train_data_file, compression_type="GZIP")
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict))
        .map(split_inputs_and_labels)
        .shuffle(10)
        .batch(64)
    )

    validation_dataset = (
        tf.data.TFRecordDataset(eval_data_file, compression_type="GZIP")
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict))
        .map(split_inputs_and_labels)
        .shuffle(10)
        .batch(64)
    )

    return training_dataset, validation_dataset


def create_model(training_dataset):
    """Creates model."""

    feature_ds = training_dataset.map(lambda x, y: x)
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(feature_ds)

    layer0 = tf.keras.Input(shape=(3, GOES_PATCH_SIZE, GOES_PATCH_SIZE, len(GOES_BANDS)))
    layer1 = normalizer(layer0)

    # core layers
    # an initial Conv2D layer for low level features
    layerC1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')
    # time-distributed layer forces Conv2D params to be fixed across the 3 time dimensions
    # this is in-lieu of a more complex LSTM architecture
    layerC1 = tf.keras.layers.TimeDistributed(layerC1)(layer1)
    # remove the time dimension
    layerC1 = tf.keras.layers.MaxPooling3D(pool_size=(3,1,1))(layerC1)
    layerC1 = tf.keras.layers.Reshape((GOES_PATCH_SIZE, GOES_PATCH_SIZE, 32))(layerC1)

    # branch for LABEL1 output
    #layerM1 = tf.keras.layers.UpSampling2D(2)(layerC1)
    layerC3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(layerC1)
    if USE_CATEGORICAL_LABEL1:
      label1_loss_fn = 'categorical_crossentropy'
      layerO1 = tf.keras.layers.Dense(units=10, activation='softmax', name=MODEL_LABEL1)(layerC3)
    else:
      label1_loss_fn = 'mean_squared_error'
      layerO1 = tf.keras.layers.Dense(units=1, name=MODEL_LABEL1)(layerC3)

    # branch for LABEL2 output
    layerC3b = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(layerC1)
    layerF3b = tf.keras.layers.Flatten()(layerC3b)
    layerO2 = tf.keras.layers.Dense(units=1, name=MODEL_LABEL2)(layerF3b)

    model = tf.keras.Model(inputs=layer0, outputs=[layerO1,layerO2])

    model.compile(
        optimizer='adam',
        loss={MODEL_LABEL1: label1_loss_fn, MODEL_LABEL2: 'mean_squared_error'},
        metrics=['accuracy', 'mean_squared_error'],
    )
    return model


def main():
    args = get_args()
    training_dataset, validation_dataset = create_datasets(args.bucket)
    model = create_model(training_dataset)
    print(model.summary())
    model.fit(training_dataset, validation_data=validation_dataset, epochs=10)
    model.save(f"gs://{args.bucket}/nowcasting/{CONFIG}/model")


if __name__ == "__main__":
    main()
