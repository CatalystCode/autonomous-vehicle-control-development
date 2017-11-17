# Copyright 2016 Google Inc. All Rights Reserved.
# Modified 2017 xiou@microsoft.com. Changed to create a new Dataset about Cars.
#
# Based on image preprocessing code from 
# https://github.com/tensorflow/models/blob/master/research/inception/inception/dataset.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small library that points to the cars data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset

class CarsData(Dataset):
  """Cars data set."""

  def __init__(self, subset):
    super(CarsData, self).__init__('Cars', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 2

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 10178
    if self.subset == 'validation':
      return 200

  def download_message(self):
    """Instruction to find and extract the Cars dataset."""
    print('Failed to find any Cars %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('Please see README.md for instructions on how to build '
          'the cars dataset using prepare_cars_data.\n')