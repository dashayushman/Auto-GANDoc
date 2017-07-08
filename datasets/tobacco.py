# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import numpy

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtype.float32,
                 reshape=[128, 128],
                 seed=None):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                    'Invalid image dtype %r, expected uint8 or float32'.format(
                            dtype))

        assert len(images) == len(labels), (
            'len(images): %s len(labels): %s'format(images.shape, labels.shape))

        random.seed(self.seed)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # self._num_examples = len(labels)

    def shuffle_data(self):
        # Shuffling inspired by
        # https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        images_and_labels = list(zip(images, labels))
        random.shuffle(images_and_labels)
        images, labels = zip(*images_and_labels)

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle_data()

        ret_images = []
        ret_labels = []
        if start + batch_size > self._num_examples:
            # Take the rest of the data
            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle
            if shuffle:
                self.shuffle_data()

            # Start a new epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            ret_images = images_rest_part.append(images_new_part)
            ret_labels = labels_rest_part.append(labels_new_part)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            ret_images = self._images[start:end]
            ret_labels = self._labels[start:end]

        return self.load_image_files(ret_images, ret_labels)

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


def read_data_sets(one_hot=False,
                   reshape=[128, 128],
                   seed=None,
                   dataset_index=dataset_index):
    dataset_path = os.path.join('data', 'datasets', 'tobacco')

    train_list = os.path.join(dataset_path, 'train_{}.txt'.format(dataset_index))
    validation_list = os.path.join(dataset_path,
                'validation_{}.txt'.format(dataset_index))
    test_list = os.path.join(dataset_path, 'test_{}.txt'.format(dataset_index))

    with open(train_list, 'rb') as f:
        train_images, train_labels = parse_data_list(f)

    with open(validation_list, 'rb') as f:
        validation_images, validation_labels = parse_data_list(f)

    with open(test_list, 'rb') as f:
        test_images, test_labels = parse_data_list(f)

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)

def parse_data_list(f):
    # The files are composed by several lines in the format
    # <path_to_file> <class>
    # where `class` is a number, and the mapping between the number and the
    # name of the class is in another file called `labels.txt`
    return zip(*list(csv.reader(f, delimiter=' ')))

    # This is equivalent to:
    #images = []
    #labels = []
    #csv_reader = csv.reader(f, delimiter=' ')
    #for row in csv_reader:
    #    images.append(row[0])
    #    labels.append(row[1])
    #return images, labels

