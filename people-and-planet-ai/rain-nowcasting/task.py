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

BANDS = [f'T{t}_CMI_C{i:02d}' for t in [0,1,2] for i in range(1,17)]
LABEL1 = 'HQprecipitation'
MODEL_LABEL1 = 'rainAmt'
LABEL2 = 'probabilityLiquidPrecipitation'
MODEL_LABEL2 = 'rainChance'
BATCH_SIZE = 64
GOES_PATCH_SIZE = 65
GPM_PATCH_SIZE = 33
CONFIG = 'p64_s11132'


def get_args():
    """Parses args."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, type=str, help="GCS Bucket")
    args = parser.parse_args()
    return args


def parse_tfrecord(example_proto, features_dict):
    """Parses a single tf.train.Example."""

    d = tf.io.parse_single_example(example_proto, features_dict)
    return d


def create_features_dict():
    """Creates dict of features."""

    features_dict = {
      band_name: tf.io.FixedLenFeature(shape=[GOES_PATCH_SIZE, GOES_PATCH_SIZE], dtype=tf.float32)
      for band_name in BANDS
    }
    features_dict[LABEL1] = tf.io.FixedLenFeature(shape=[GPM_PATCH_SIZE, GPM_PATCH_SIZE], dtype=tf.float32)
    features_dict[LABEL2] = tf.io.FixedLenFeature(shape=[GPM_PATCH_SIZE, GPM_PATCH_SIZE], dtype=tf.float32)
    return features_dict


def convert_pct_to_binary(probability):
    """Maps a continuous probability percentage to a binary 0 or 1."""

    t50 = tf.constant([50.0])
    return tf.cast(tf.math.greater(probability, t50), tf.float32)


def split_inputs_and_labels(values: dict):
    '''
    inputs = []
    for t in ['T0','T1','T2']:
      inputs.append([values[f'{t}_CMI_C{b:02d}'] for b in range(1,17)])
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.transpose(inputs, [2,3,0,1])
    '''
    inputs = tf.stack([values[b] for b in BANDS], axis=-1)
    inputs = tf.image.resize_with_crop_or_pad(inputs, GOES_PATCH_SIZE-1, GOES_PATCH_SIZE-1)

    labels = {}
    tlabel1 = tf.expand_dims(values.pop(LABEL1), -1)
    # limit precipitation values to a max of 20 mm/hr
    tf.clip_by_value(tlabel1, 0, 20)
    tlabel1 = tf.image.resize_with_crop_or_pad(tlabel1, GPM_PATCH_SIZE-1, GPM_PATCH_SIZE-1)
    tlabel2 = tf.expand_dims(convert_pct_to_binary(values.pop(LABEL2)), -1)
    tlabel2 = tf.image.resize_with_crop_or_pad(tlabel2, GPM_PATCH_SIZE-1, GPM_PATCH_SIZE-1)
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
        .shuffle(100)
        .batch(BATCH_SIZE)
    )

    validation_dataset = (
        tf.data.TFRecordDataset(eval_data_file, compression_type="GZIP")
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict))
        .map(split_inputs_and_labels)
        .shuffle(100)
        .batch(BATCH_SIZE)
    )

    return training_dataset, validation_dataset


def create_model(training_dataset):
    """Creates model."""

    feature_ds = training_dataset.map(lambda x, y: x)
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(feature_ds)

    layer0 = tf.keras.Input(shape=(GOES_PATCH_SIZE-1, GOES_PATCH_SIZE-1, len(BANDS)))
    layer1 = normalizer(layer0)

    layerC1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(layer1)
    layerMP1 = tf.keras.layers.MaxPooling2D(pool_size=2)(layerC1)
    layerC2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(layerMP1)
    layerC3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(layerC2)
    layerO1 = tf.keras.layers.Dense(units=1, name=MODEL_LABEL1)(layerC3)
    #layerMP2 = tf.keras.layers.MaxPooling2D(pool_size=2)(layerC2)
    #layerMP2 = tf.keras.layers.MaxPooling2D(pool_size=2)(layerC2)
    #layerUP1 = tf.keras.layers.UpSampling2D(size=2)(layerC3)
    # layer2t = tf.keras.layers.TimeDistributed(layer2)
    layerC3b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(layerC2)
    layerO2 = tf.keras.layers.Dense(units=1, name=MODEL_LABEL2, activation='sigmoid')(layerC3b)

    model = tf.keras.Model(inputs=layer0, outputs=[layerO1,layerO2])

    model.compile(
        optimizer='adam',
        loss={MODEL_LABEL1: tf.keras.losses.LogCosh(), MODEL_LABEL2: 'mean_squared_error'},
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
