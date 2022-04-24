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


"""This training script trains a DNN model to forecast precipitation (rainfall) amounts
based on GOES-16 satellite imagery.
Input: GOES-16 MCMIPF images (https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPF)
Output/Labels: GPM precipitation values (https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06)
Both the input and output are represented as 2D regions.

Input data is read from a GCS bucket specified as a command line parameter.
The script produces a Keras TF model, which is saved to the same GCS bucket.
"""

import argparse
import tensorflow as tf
import numpy

GOES_BANDS = [f'CMI_C{i:02d}' for i in range(1,17)]
BANDS = [f'T{t}_CMI_C{i:02d}' for t in [0,1,2] for i in range(1,17)]
LABEL1 = 'HQprecipitation'
MODEL_LABEL1 = 'rainAmt'
GOES_PATCH_SIZE = 65
GPM_PATCH_SIZE = 65
USE_CATEGORICAL_LABEL1 = True


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
    return features_dict


def split_inputs_and_labels(values: dict):
    """Splits a TFRecord value dictionary into input and label tensors."""
    inputs = []
    for t in ['T0','T1','T2']:
      inputs.append([values[f'{t}_CMI_C{b:02d}'] for b in range(1,17)])
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.transpose(inputs, [0,2,3,1])

    tlabel1 = values.pop(LABEL1)
    if USE_CATEGORICAL_LABEL1:
      # Clip precipitation range to 0-10, and quantize to 1 mm/hr ranges.
      tlabel1 = tf.math.divide(tf.clip_by_value(tlabel1, 0, 10.0), tf.constant([1.0]))
      tlabel1 = tf.one_hot(tf.cast(tlabel1, tf.uint8), 10)
    else:
      tlabel1 = tf.expand_dims(tlabel1, -1)
    return inputs, tlabel1


def create_datasets(bucket):
    """Creates training and validation datasets."""
    train_data_file = f'gs://{bucket}/nowcasting/training.tfrecord.gz'
    eval_data_file = f'gs://{bucket}/nowcasting/validation.tfrecord.gz'
    features_dict = create_features_dict()

    training_dataset = tf.data.TFRecordDataset(train_data_file, compression_type="GZIP")
    validation_dataset = tf.data.TFRecordDataset(eval_data_file, compression_type="GZIP")
    print('Training dataset size: ', sum(1 for _ in training_dataset))
    print('Validation dataset size: ', sum(1 for _ in validation_dataset))

    training_dataset = (training_dataset
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict))
        .map(split_inputs_and_labels)
        .shuffle(10)
        .batch(64))

    validation_dataset = (validation_dataset
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict))
        .map(split_inputs_and_labels)
        .shuffle(10)
        .batch(64))

    return training_dataset, validation_dataset


def create_model(training_dataset):
    """Creates model."""
    feature_ds = training_dataset.map(lambda x, y: x)
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(feature_ds)

    layer0 = tf.keras.Input(shape=(3, GOES_PATCH_SIZE, GOES_PATCH_SIZE, len(GOES_BANDS)))
    layer1 = normalizer(layer0)

    # time-distributed layer forces Conv2D params to be fixed across the 3 time dimensions
    # this is in-lieu of a more complex LSTM architecture
    layerC1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=(3,3),
                                       kernel_initializer='he_normal',
                                       strides=(1,1),
                                       activation='relu',
                                       padding='same'))(layer1)
    # pool the time dimension
    layerC1 = tf.keras.layers.MaxPooling3D(pool_size=(3,1,1))(layerC1)
    layerC1 = tf.keras.layers.Reshape((GOES_PATCH_SIZE, GOES_PATCH_SIZE, 32))(layerC1)

    layerC2 = tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3,3),
                                     kernel_initializer='he_normal',
                                     strides=(1,1),
                                     activation='relu',
                                     padding='same')(layerC1)

    layerC3 = tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3,3),
                                     kernel_initializer='he_normal',
                                     strides=(1,1),
                                     activation='relu',
                                     padding='same')(layerC2)

    if USE_CATEGORICAL_LABEL1:
      layerO1 = tf.keras.layers.Dense(units=10, activation='softmax', name=MODEL_LABEL1)(layerC3)
    else:
      layerO1 = tf.keras.layers.Dense(units=1, name=MODEL_LABEL1)(layerC3)

    model = tf.keras.Model(inputs=layer0, outputs=layerO1)

    model.compile(
        optimizer='adam',
        #loss='mean_squared_error',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'mean_squared_error'],
    )
    return model


def main():
    args = get_args()
    training_dataset, validation_dataset = create_datasets(args.bucket)
    model = create_model(training_dataset)
    print(model.summary())
    model.fit(training_dataset, validation_data=validation_dataset, epochs=10)
    model.save(f'gs://{args.bucket}/nowcasting/model')


if __name__ == "__main__":
    main()
