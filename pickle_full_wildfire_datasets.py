import re
from typing import Dict, List, Optional, Text, Tuple
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

OUTPUT_FEATURES = ['FireMask', ]

# Data statistics 
# For each variable, the statistics are ordered in the form:
# (min_clip, max_clip, mean, standard deviation)
DATA_STATS = {
    # Elevation in m.
    # 0.1 percentile, 99.9 percentile
    'elevation': (0.0, 3492.0, 973.8651650565012, 848.2582623382642),
    
    # Drought Index (Palmer Drought Severity Index)
    # 0.1 percentile, 99.9 percentile
    'pdsi': (-6.929506301879883, 6.933391571044922, -0.5917681081888462, 2.631040721581213),
    
    #Vegetation index (times 10,000 maybe, since it's supposed to be b/w -1 and 1?)
    'NDVI': (-1030.5584442138675, 8642.758791992237, 5164.3233936349225, 1702.2336274971083),  # min, max
   
    # Precipitation in mm.
    # Negative values do not make sense, so min is set to 0.
    # 0., 99.9 percentile
    'pr': (-0.10937719792127609, 16.0850124359130864, 0.2079030104452133, 1.120996442992493),
   
    # Specific humidity.
    # Negative values do not make sense, so min is set to 0.
    # The range of specific humidity is up to 100% so max is 1.
    'sph': (0., 0.019455255940556526, 0.006173964228252022, 0.003562594725196875),
    
    # Wind direction in degrees clockwise from north.
    # Thus min set to 0 and max set to 360.
    'th': (0., 349.893463134765, 203.51448736597345, 75.17527113547739),
    
    # Min/max temperature in Kelvin.
    
    #Min temp
    # -20 degree C, 99.9 percentile
    'tmmn': (0.0, 300.07110595703125, 281.0813526978816, 26.776763832658812),
    
    #Max temp
    # -20 degree C, 99.9 percentile
    'tmmx': (0.0, 316.160400390625, 296.7791755110431, 28.21604481568473),
    
    # Wind speed in m/s.
    # Negative values do not make sense, given there is a wind direction.
    # 0., 99.9 percentile
    'vs': (0.0, 10.161438941955566, 3.670841557634326, 1.376505541638688),
    
    # NFDRS fire danger index energy release component expressed in BTU's per
    # square foot.
    # Negative values do not make sense. Thus min set to zero.
    # 0., 99.9 percentile
    'erc': (0.0, 110.70502471923828, 58.42723711884473, 26.448045709188296),
    
    # Population density
    # min, 99.9 percentile
    'population': (0., 3464.451171875, 32.06895734368147, 214.94144945265535),
    
    #FWS
    'fws': (-7.611932770252228, 16.34676742553711, 0.8698552858092746, 2.808712593586319),
    
    #SLOPE
    'slope': (0.0,26.122961044311523, 3.7763974131349465, 4.6385693305945805),
    
    #ERC
    'erc': (0.0, 110.70502471923828, 58.42723711884473, 26.448045709188296),
    
    'fpr': (0.0005380672519095242, 0.01970484294369823, 0.006215652860019671, 0.003481157915072828),
    
    #FTEMP
    'ftemp': (-1.0703978538513184, 40.34566116333008, 24.022050817701835, 6.891479893238092),

    #EVI
    'EVI': (0.0, 6330.7783208007895, 2782.431770790949, 935.4603390827173),
    #FWD
    'fwd': (-9.776235580444336, 13.015328407287598, 0.9124042024569722, 2.907059560489641),

    #ASPECT
    'aspect': (-0.0, 358.8988952636719, 170.92333794728714, 102.2358012656133),


    # We don't want to normalize the FireMasks.
    # 1 indicates fire, 0 no fire, -1 unlabeled data
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}


"""Library of common functions used in deep learning neural networks.
"""
#YOU PROBABLY WILL NOT USE THESE. (Narrator: They were used...)

def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Randomly axis-align crop input and output image tensors.

  Args:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
    sample_size: side length (square) to crop to.
    num_in_channels: number of channels in input_img.
    num_out_channels: number of channels in output_img.
  Returns:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
  """
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
  """Center crops input and output image tensors.

  Args:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
    sample_size: side length (square) to crop to.
  Returns:
    input_img: tensor with dimensions HWC.
    output_img: tensor with dimensions HWC.
  """
  central_fraction = sample_size / input_img.shape[0]
  input_img = tf.image.central_crop(input_img, central_fraction)
  output_img = tf.image.central_crop(output_img, central_fraction)
  return input_img, output_img


"""Dataset reader for Earth Engine data."""


def _get_base_key(key: Text) -> Text:
  """Extracts the base key from the provided key.

  Earth Engine exports TFRecords containing each data variable with its
  corresponding variable name. In the case of time sequences, the name of the
  data variable is of the form 'variable_1', 'variable_2', ..., 'variable_n',
  where 'variable' is the name of the variable, and n the number of elements
  in the time sequence. Extracting the base key ensures that each step of the
  time sequence goes through the same normalization steps.
  The base key obeys the following naming pattern: '([a-zA-Z]+)'
  For instance, for an input key 'variable_1', this function returns 'variable'.
  For an input key 'variable', this function simply returns 'variable'.

  Args:
    key: Input key.

  Returns:
    The corresponding base key.

  Raises:
    ValueError when `key` does not match the expected pattern.
  """
  match = re.match(r'([a-zA-Z]+)', key)
  if match:
    return match.group(1)
  raise ValueError(
      'The provided key does not match the expected pattern: {}'.format(key))


def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
  """Clips and rescales inputs with the stats corresponding to `key`.

  Args:
    inputs: Inputs to clip and rescale.
    key: Key describing the inputs.

  Returns:
    Clipped and rescaled input.

  Raises:
    ValueError if there are no data statistics available for `key`.
  """
  base_key = _get_base_key(key)
  if base_key not in DATA_STATS:
    raise ValueError(
        'No data statistics available for the requested key: {}.'.format(key))
  min_val, max_val, _, _ = DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
  """Clips and normalizes inputs with the stats corresponding to `key`.

  Args:
    inputs: Inputs to clip and normalize.
    key: Key describing the inputs.

  Returns:
    Clipped and normalized input.

  Raises:
    ValueError if there are no data statistics available for `key`.
  """
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
  """Creates a features dictionary for TensorFlow IO.

  Args:
    sample_size: Size of the input tiles (square).
    features: List of feature names.

  Returns:
    A features dictionary for TensorFlow IO.
  """
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
  """Reads a serialized example.

  Args:
    example_proto: A TensorFlow example protobuf.
    data_size: Size of tiles (square) as read from input files.
    sample_size: Size the tiles (square) when input into the model.
    num_in_channels: Number of input channels.
    clip_and_normalize: True if the data should be clipped and normalized.
    clip_and_rescale: True if the data should be clipped and rescaled.
    random_crop: True if the data should be randomly cropped.
    center_crop: True if the data should be cropped in the center.

  Returns:
    (input_img, output_img) tuple of inputs and outputs to the ML model.
  """
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

  if random_crop:
    input_img, output_img = random_crop_input_and_output_images(
        input_img, output_img, sample_size, num_in_channels, 1)
  if center_crop:
    input_img, output_img = center_crop_input_and_output_images(
        input_img, output_img, sample_size)
  return input_img, output_img


def get_dataset(file_pattern: Text, data_size: int, sample_size: int,
                num_in_channels: int, compression_type: Text,
                clip_and_normalize: bool, clip_and_rescale: bool,
                random_crop: bool, center_crop: bool) -> tf.data.Dataset:
  """Gets the dataset from the file pattern.

  Args:
    file_pattern: Input file pattern.
    data_size: Size of tiles (square) as read from input files.
    sample_size: Size the tiles (square) when input into the model.
    batch_size: Batch size.
    num_in_channels: Number of input channels.
    compression_type: Type of compression used for the input files.
    clip_and_normalize: True if the data should be clipped and normalized, False
      otherwise.
    clip_and_rescale: True if the data should be clipped and rescaled, False
      otherwise.
    random_crop: True if the data should be randomly cropped.
    center_crop: True if the data shoulde be cropped in the center.

  Returns:
    A TensorFlow dataset loaded from the input file pattern, with features
    described in the constants, and with the shapes determined from the input
    parameters to this function.
  """
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

def main():

    # The archive.zip from Kaggle is expected to be unzipped into the directory where this script will be ran.

    train_dataset = get_dataset(
        file_pattern='archive/next_day_wildfire_spread_train*',
        data_size=64,
        sample_size=64,
        num_in_channels=19,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False)

    test_dataset = get_dataset(
        file_pattern='archive/next_day_wildfire_spread_test*',
        data_size=64,
        sample_size=64,
        num_in_channels=19,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False)

    validation_dataset = get_dataset(
        file_pattern='archive/next_day_wildfire_spread_eval*',
        data_size=64,
        sample_size=64,
        num_in_channels=19,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False)

    print("Sucessfully created the tensorflow datasets!")

    x_train = np.moveaxis(np.array([np.concatenate((x[0].numpy(), x[1].numpy()), axis=2) for x in train_dataset]), 3, 1)
    x_test = np.moveaxis(np.array([np.concatenate((x[0].numpy(), x[1].numpy()), axis=2) for x in test_dataset]), 3, 1)
    x_validation = np.moveaxis(np.array([np.concatenate((x[0].numpy(), x[1].numpy()), axis=2) for x in validation_dataset]), 3, 1)

    print("The tensorflow datasets were sucessfully converted into numpy arrays")

    if not os.path.exists('data/next-day-wildfire-spread/'):
        print("Creating data/next-day-wildfire-spread folders")
        os.makedirs('data/next-day-wildfire-spread/')

    with open('data/next-day-wildfire-spread/train.data', 'wb') as handle:
        pickle.dump(x_train[:, :19, :, :], handle)
    with open('data/next-day-wildfire-spread/train.labels', 'wb') as handle:
        pickle.dump(x_train[:, 19, :, :], handle)

    with open('data/next-day-wildfire-spread/test.data', 'wb') as handle:
        pickle.dump(x_test[:, :19, :, :], handle)
    with open('data/next-day-wildfire-spread/test.labels', 'wb') as handle:
        pickle.dump(x_test[:, 19, :, :], handle)

    with open('data/next-day-wildfire-spread/validation.data', 'wb') as handle:
        pickle.dump(x_validation[:, :19, :, :], handle)
    with open('data/next-day-wildfire-spread/validation.labels', 'wb') as handle:
        pickle.dump(x_validation[:, 19, :, :], handle)

    print("The numpy arrays were successfully pickled in ./data/next-day-wildfire-spread/")

if __name__ == '__main__':
    main()
