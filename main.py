from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf

from data_utils import prepare_dataset_pair
from utils.arg_parsers import parsers
from utils.logging import hooks_helper

LEARNING_RATE = 1e-4


class Model(tf.keras.Model):
    """Model to recognize digits in the MNIST dataset.

    Network structure is equivalent to:
    https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
    and
    https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

    But written as a tf.keras.Model using the tf.layers API.
    """

    def __init__(self, data_format, image_size):
        """Creates a model for classifying a hand-written digit.

        Args:
          data_format: Either 'channels_first' or 'channels_last'.
            'channels_first' is typically faster on GPUs while 'channels_last' is
            typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        """
        self.image_size = image_size
        super(Model, self).__init__()
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, image_size, image_size]
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, image_size, image_size, 1]

        self.conv1 = tf.layers.Conv2D(
            32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(
            64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10)
        self.dropout = tf.layers.Dropout(0.4)
        self.max_pool2d = tf.layers.MaxPooling2D(
            1, 1, padding='same', data_format=data_format)

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, 10].
        """

        y = tf.reshape(inputs, self._input_shape)
        # y = tf.transpose(y, [0, 3, 1, 2])
        y = self.conv1(y)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        # y = tf.layers.flatten(y)
        # print(y.shape)
        y = tf.reshape(y, [-1, self.image_size ** 2 * 64])
        # y = tf.reshape(y, [-1, self.image_size ** 2 * 32])
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        return self.fc2(y)


def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""
    model = Model(**params)
    image = features
    if isinstance(image, dict):
        image = features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                    tf.metrics.accuracy(
                        labels=labels,
                        predictions=tf.argmax(logits, axis=1)),
            })


def main(argv):
    parser = ArgParser()
    flags = parser.parse_args(args=argv[1:])
    model_function = model_fn
    data_format = flags.data_format
    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=flags.model_dir,
        params={
            'data_format': data_format,
            'image_size': flags.image_size
        })

    def train_input_fn():
        # ds = load_data(os.path.join(os.path.join(flags.data_root, 'data'), 'train'), flags.image_size)

        ds = prepare_dataset_pair(flags.data_root, 'train', 10)
        ds = ds.cache().shuffle(buffer_size=50000).batch(flags.batch_size)
        ds = ds.repeat(flags.epochs_between_evals)
        return ds

    def eval_input_fn():
        testset = prepare_dataset_pair(flags.data_root, 'test', 10)
        return testset.batch(
            flags.batch_size).make_one_shot_iterator().get_next()

    train_hooks = hooks_helper.get_train_hooks(
        flags.hooks, batch_size=flags.batch_size)

    # Train and evaluate model.
    for _ in range(flags.train_epochs // flags.epochs_between_evals):
        classifier.train(input_fn=train_input_fn, hooks=train_hooks)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

    # Export the model
    image = tf.placeholder(tf.float32, [None, flags.image_size, flags.image_size])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    classifier.export_savedmodel(flags.export_dir, input_fn)
    shutil.rmtree(flags.model_dir)


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(parents=[
            parsers.BaseParser(),
            parsers.ImageModelParser()])

        # self.add_argument(
        #     '--export_dir',
        #     type=str)

        self.add_argument(
            '--data_root',
            type=str)

        self.add_argument(
            '--image_size',
            type=int)

        self.set_defaults(
            # data_dir='./data',
            model_dir=os.path.join(sys.path[0], 'fire', 'model'),
            export_dir=os.path.join(sys.path[0], 'fire', 'export'),
            data_root=os.path.join(sys.path[0], 'fire', 'data', 'tfrecords'),
            image_size=28,
            batch_size=5,
            train_epochs=1,
            data_format='channels_first'
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
