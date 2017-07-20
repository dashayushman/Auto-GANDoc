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

import os
import csv
import numpy
import random
import numpy as np

import logging

from PIL import Image

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 reshape,
                 classes,
                 one_hot=False,
                 dtype=dtypes.float32,
                 dataset_path='.',
                 seed=None):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                    'Invalid image dtype {}, expected uint8 or float32'.format(
                            dtype))

        assert len(images) == len(labels), (
            'len(images): {} len(labels): {}'.format(images.shape, labels.shape))

        self.seed = seed
        random.seed(self.seed)

        self.dataset_path = dataset_path
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(labels)
        self.reshape = reshape
        self.classes = classes
        self.n_classes = len(classes)
        self.one_hot = one_hot

    def shuffle_data(self):
        # Shuffling inspired by
        # https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
        images_and_labels = list(zip(self._images, self._labels))
        random.shuffle(images_and_labels)
        self._images, self._labels = [list(i) for i in zip(*images_and_labels)]
        #self._images, self._labels = zip(*images_and_labels)
        #print("Images: {}".format(self._images))
        #print("Labels: {}".format(self._labels))

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

            #print("\nINFO: will take elements from {} to {}".
            #        format(start, self._num_examples))
            #print("\nINFO: type(self._images): {}, len(self._images): {}".
            #        format(type(self._images), len(self._images)))

            ret_images = self._images[start:self._num_examples]
            ret_labels = self._labels[start:self._num_examples]

            #logging.info("len(ret_images): {}, type(ret_images): {}".
            #        format(len(ret_images), type(ret_images)))
            #print("\nINFO: len(ret_images): {}, type(ret_images): {}".
            #        format(len(ret_images), type(ret_images)))

            # Shuffle
            if shuffle:
                self.shuffle_data()

            # Start a new epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            ret_images.extend(images_new_part)
            ret_labels.extend(labels_new_part)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            ret_images = self._images[start:end]
            ret_labels = self._labels[start:end]

        if self.one_hot:
            ret_labels = np.array(ret_labels).astype(np.int)
            ret_labels = dense_to_one_hot(ret_labels, self.n_classes)

        #print("======== {}: Epoch {}, Batch {}".format(
        #        'next_batch', self._epochs_completed, self._index_in_epoch))
        return self.load_image_files(ret_images), ret_labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def load_image_files(self, images):
        ret = []
        for i in images:
            #print("current i: {}".format(i))
            image_filename = os.path.join(self.dataset_path, 'data', i)
            img = Image.open(image_filename)
            #w, h = img.size
            img = img.resize((self.reshape[1], self.reshape[0]), Image.BICUBIC)
            w, h = img.size
            img_black_and_white = img.convert('L')
            del img

            img_black_and_white = np.asarray(img_black_and_white,
                                            dtype = np.uint8)
            #img_black_and_white = np.resize(img_black_and_white[:,:], (w, h, 3))
            img_3_channels = np.zeros(shape=(self.reshape[1], self.reshape[0], 3))
            img_3_channels[:,:,0] = img_black_and_white
            img_3_channels[:,:,1] = img_black_and_white
            img_3_channels[:,:,2] = img_black_and_white
            ret.append(img_3_channels)
        return ret

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors.

        E.g., if labels_dense is an nparray of shape (10,), and num_classes
        is 3, then `dense_to_one_hot()` will return a vector of shape (10, 3)
    """
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_data_sets(one_hot=False,
                   reshape=[128,128],
                   seed=None,
                   dataset_index=0):
    dataset_path = os.path.join('data', 'datasets', 'tobacco')

    train_list = os.path.join(dataset_path, 'train_{}.txt'.format(dataset_index))
    validation_list = os.path.join(dataset_path,
                'validate_{}.txt'.format(dataset_index))
    test_list = os.path.join(dataset_path, 'test_{}.txt'.format(dataset_index))
    classes_list = os.path.join(dataset_path, 'labels.txt')

    with open(train_list, 'r') as f:
        train_images, train_labels = parse_data_list(f)

    with open(validation_list, 'r') as f:
        validation_images, validation_labels = parse_data_list(f)

    with open(test_list, 'r') as f:
        test_images, test_labels = parse_data_list(f)

    with open(classes_list, 'r') as f:
        classes = list(list(parse_data_list(f))[0])

    options = dict(reshape=reshape, seed=seed, dataset_path=dataset_path,
                    classes=classes, one_hot=one_hot)


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

