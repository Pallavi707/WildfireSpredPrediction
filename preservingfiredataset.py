import argparse
import re
from typing import Dict, List, Text, Tuple
import pickle
import numpy as np
import tensorflow as tf
import os

# Make sure to download the dataset from kaggle and extract it to the same directory where this script will be ran.
# https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread

"""Constants for the data reader."""

INPUT_FEATURES = ['elevation', 'fws', 'population', 'pdsi', 'pr', 'sph', 'slope', 'PrevFireMask',
                  'erc', 'NDVI', 'fpr', 'ftemp', 'th', 'EVI', 'vs', 'tmmx', 'fwd',
                  'aspect', 'tmmn']

OUTPUT_FEATURES = ['FireMask',]

# Data statistics
# For each variable, the statistics are ordered in the form:
# (min_clip, max_clip, mean, standard deviation)
DATA_STATS = {
    'elevation': (0.0, 3492.0, 973.8651650565012, 848.2582623382642),
    'pdsi': (-6.929506301879883, 6.933391571044922, -0.5917681081888462, 2.631040721581213),
    'NDVI': (-1030.5584442138675, 8642.758791992237, 5164.3233936349225, 1702.2336274971083),
    'pr': (-0.10937719792127609, 16.0850124359130864, 0.2079030104452133, 1.120996442992493),
    'sph': (0., 0.019455255940556526, 0.006173964228252022, 0.003562594725196875),
    'th': (0., 349.893463134765, 203.51448736597345, 75.17527113547739),
    'tmmn': (0.0, 300.07110595703125, 281.0813526978816, 26.776763832658812),
    'tmmx': (0.0, 316.160400390625, 296.7791755110431, 28.21604481568473),
    'vs': (0.0, 10.161438941955566, 3.670841557634326, 1.376505541638688),
    'erc': (0.0, 110.70502471923828, 58.42723711884473, 26.448045709188296),
    'population': (0., 3464.451171875, 32.06895734368147, 214.94144945265535),
    'fws': (-7.611932770252228, 16.34676742553711, 0.8698552858092746, 2.808712593586319),
    'slope': (0.0,26.122961044311523, 3.7763974131349465, 4.6385693305945805),
    'fpr': (0.0005380672519095242, 0.01970484294369823, 0.006215652860019671, 0.003481157915072828),
    'ftemp': (-1.0703978538513184, 40.34566116333008, 24.022050817701835, 6.891479893238092),
    'EVI': (0.0, 6330.7783208007895, 2782.431770790949, 935.4603390827173),
    'fwd': (-9.776235580444336, 13.015328407287598, 0.9124042024569722, 2.907059560489641),
    'aspect': (-0.0, 358.8988952636719, 170.92333794728714, 102.2358012656133),

    # We don't want to normalize the FireMasks.
    # 1 indicates fire, 0 no fire, -1 unlabeled data
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}


"""Library of common functions used in deep learning neural networks.
"""

def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Randomly axis-align crop input and output image tensors."""
  combined = tf.concat([input_img, output_img], axis=2)
  combined = tf.image.random_crop(
      combined,
      [sample_size, sample_size, num_in_channels + num_out_channels])
  input_img = combined[:, :, 0:num_in_channels]
  output_img = combined[:, :, -num_out_channels:]
  return input_img, output_img


def center_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Center crops input and output image tensors."""
  central_fraction = sample_size / input_img.shape[0]
  input_img = tf.image.central_crop(input_img, central_fraction)
  output_img = tf.image.central_crop(output_img, central_fraction)
  return input_img, output_img


# -------- NEW: Full-tile downscale that preserves fire -------- #
def _downscale_preserve_fire(input_img: tf.Tensor,
                             output_img: tf.Tensor,
                             sample_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Downscale full HWC tensors to sample_size while ensuring that fire pixels are not lost:
      - Inputs: tf.image.resize with AREA (averaging)
      - PrevFireMask (if present): after AREA resize, binarize (>0 -> 1 else 0)
      - Outputs (FireMask): AREA resize then threshold > 0 (any-fire -> 1)
    """
    # Resize inputs (HWC) with AREA for all channels
    inputs_resized = tf.image.resize(
        input_img, [sample_size, sample_size],
        method=tf.image.ResizeMethod.AREA
    )

    # If PrevFireMask is an input channel, binarize it after AREA resize
    if 'PrevFireMask' in INPUT_FEATURES:
        prev_idx = INPUT_FEATURES.index('PrevFireMask')
        prev = inputs_resized[:, :, prev_idx:prev_idx+1]
        prev_bin = tf.cast(prev > 0.0, tf.float32)
        inputs_resized = tf.concat(
            [inputs_resized[:, :, :prev_idx], prev_bin, inputs_resized[:, :, prev_idx+1:]],
            axis=2
        )

    # Resize FireMask with AREA then threshold (>0 means any fire becomes 1)
    fire_area = tf.image.resize(
        output_img, [sample_size, sample_size],
        method=tf.image.ResizeMethod.AREA
    )
    fire_bin = tf.cast(fire_area > 0.0, tf.float32)

    return inputs_resized, fire_bin
# -------------------------------------------------------------- #


"""Dataset reader for Earth Engine data."""
def _get_base_key(key: Text) -> Text:
  match = re.match(r'([a-zA-Z]+)', key)
  if match:
    return match.group(1)
  raise ValueError(
      'The provided key does not match the expected pattern: {}'.format(key))


def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
  base_key = _get_base_key(key)
  if base_key not in DATA_STATS:
    raise ValueError(
        'No data statistics available for the requested key: {}.'.format(key))
  min_val, max_val, _, _ = DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
  base_key = _get_base_key(key)
  if base_key not in DATA_STATS:
    raise ValueError(
        'No data statistics available for the requested key: {}.'.format(key))
  min_val, max_val, mean, std = DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  inputs = inputs - mean
  return tf.math.divide_no_nan(inputs, std)

def _get_features_dict(
    sample_size: int,
    features: List[Text],
) -> Dict[Text, tf.io.FixedLenFeature]:
  sample_shape = [sample_size, sample_size]
  features = set(features)
  columns = [
      tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
      for _ in features
  ]
  return dict(zip(features, columns))


def _parse_fn(
    example_proto: tf.train.Example, data_size: int, sample_size: int,
    num_in_channels: int, clip_and_normalize: bool,
    clip_and_rescale: bool, random_crop: bool, center_crop: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Reads a serialized example and downsizes full tiles while preserving fire."""
  if (random_crop and center_crop):
    raise ValueError('Cannot have both random_crop and center_crop be True')
  input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
  feature_names = input_features + output_features
  features_dict = _get_features_dict(data_size, feature_names)
  features = tf.io.parse_single_example(example_proto, features_dict)

  if clip_and_normalize:
    inputs_list = [
        _clip_and_normalize(features.get(key), key) for key in input_features
    ]
  elif clip_and_rescale:
    inputs_list = [
        _clip_and_rescale(features.get(key), key) for key in input_features
    ]
  else:
    inputs_list = [features.get(key) for key in input_features]

  inputs_stacked = tf.stack(inputs_list, axis=0)
  input_img = tf.transpose(inputs_stacked, [1, 2, 0])

  outputs_list = [features.get(key) for key in output_features]
  assert outputs_list, 'outputs_list should not be empty'
  outputs_stacked = tf.stack(outputs_list, axis=0)
  outputs_stacked_shape = outputs_stacked.get_shape().as_list()
  assert len(outputs_stacked.shape) == 3, ('outputs_stacked should be rank 3'
                                            'but dimensions of outputs_stacked'
                                            f' are {outputs_stacked_shape}')
  output_img = tf.transpose(outputs_stacked, [1, 2, 0])

  # ---- Replace cropping with full-tile downscale that preserves fire ----
  if sample_size < data_size:
      input_img, output_img = _downscale_preserve_fire(input_img, output_img, sample_size)
  elif sample_size == data_size:
      pass
  else:
      raise ValueError("sample_size cannot exceed data_size")

  return input_img, output_img


def get_dataset(file_pattern: Text, data_size: int, sample_size: int,
                num_in_channels: int, compression_type: Text,
                clip_and_normalize: bool, clip_and_rescale: bool,
                random_crop: bool, center_crop: bool) -> tf.data.Dataset:
  """Gets the dataset from the file pattern."""
  if (clip_and_normalize and clip_and_rescale):
    raise ValueError('Cannot have both normalize and rescale.')
  dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = dataset.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      lambda x: _parse_fn(  # pylint: disable=g-long-lambda
          x, data_size, sample_size, num_in_channels, clip_and_normalize,
          clip_and_rescale, random_crop, center_crop),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

def remove_bad_samples(dataset_np):
    """
    Remove samples where FireMask has any negative value (unlabeled),
    which can propagate through AREA resize.
    dataset_np shape: [N, 20, H, W], with channel 19 = FireMask
    """
    fire_masks_array = np.array(dataset_np[:, 19, :, :])
    good_indices = []
    for img_num in range(len(fire_masks_array)):
        if not np.any(fire_masks_array[img_num] < 0):
            good_indices.append(img_num)
    return dataset_np[good_indices]


def main(dataset_path, tile_size):
    data_size = 64
    sample_size = tile_size  # e.g., 32 or 18

    train_dataset = get_dataset(
        file_pattern=os.path.join(dataset_path, 'archive/next_day_wildfire_spread_train*'),
        data_size=data_size,
        sample_size=sample_size,
        num_in_channels=19,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,     # full-tile downscale
        center_crop=False)

    test_dataset = get_dataset(
        file_pattern=os.path.join(dataset_path, 'archive/next_day_wildfire_spread_test*'),
        data_size=data_size,
        sample_size=sample_size,
        num_in_channels=19,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=False)

    validation_dataset = get_dataset(
        file_pattern=os.path.join(dataset_path, 'archive/next_day_wildfire_spread_eval*'),
        data_size=data_size,
        sample_size=sample_size,
        num_in_channels=19,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=False)

    print("Sucessfully created the tensorflow datasets!")

    # Convert TF dataset to numpy and stack inputs + outputs along channel dim
    x_train = np.moveaxis(np.array([np.concatenate((x[0].numpy(), x[1].numpy()), axis=2) for x in train_dataset]), 3, 1)
    x_test = np.moveaxis(np.array([np.concatenate((x[0].numpy(), x[1].numpy()), axis=2) for x in test_dataset]), 3, 1)
    x_validation = np.moveaxis(np.array([np.concatenate((x[0].numpy(), x[1].numpy()), axis=2) for x in validation_dataset]), 3, 1)

    # Remove samples with any unlabeled (-) pixels in FireMask
    x_train = remove_bad_samples(x_train)
    x_test = remove_bad_samples(x_test)
    x_validation = remove_bad_samples(x_validation)

    print("The tensorflow datasets were sucessfully converted into numpy arrays")

    out_root = f'data{sample_size}/next-day-wildfire-spread/'
    if not os.path.exists(out_root):
        print(f"Creating {out_root} folders")
        os.makedirs(out_root, exist_ok=True)

    with open(os.path.join(out_root, 'train.data'), 'wb') as handle:
        pickle.dump(x_train[:, :19, :, :], handle)
    with open(os.path.join(out_root, 'train.labels'), 'wb') as handle:
        pickle.dump(x_train[:, 19, :, :], handle)

    with open(os.path.join(out_root, 'test.data'), 'wb') as handle:
        pickle.dump(x_test[:, :19, :, :], handle)
    with open(os.path.join(out_root, 'test.labels'), 'wb') as handle:
        pickle.dump(x_test[:, 19, :, :], handle)

    with open(os.path.join(out_root, 'validation.data'), 'wb') as handle:
        pickle.dump(x_validation[:, :19, :, :], handle)
    with open(os.path.join(out_root, 'validation.labels'), 'wb') as handle:
        pickle.dump(x_validation[:, 19, :, :], handle)

    print(f"The numpy arrays were successfully pickled in ./{out_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='.')
    parser.add_argument('--tile_size', type=int, default=32, help='Output tile size (e.g., 32 or 18)')
    args = parser.parse_args()
    main(args.dataset, args.tile_size)
