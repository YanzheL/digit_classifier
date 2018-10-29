import os

import cv2
import tensorflow as tf


def raw_data_to_tfrecord(root_dir, export_dir, sub='train', raw_img_process_func=lambda x: x):
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    cur_exp = os.path.join(export_dir, sub)
    if not os.path.exists(cur_exp):
        os.mkdir(cur_exp)
    ct = 0
    cur_root = os.path.join(root_dir, sub)
    for label_str in dir_files_gen(cur_root, lambda f: f.isdigit(), True):
        label = int(label_str)
        subdir = os.path.join(cur_root, label_str)
        for img_path in dir_files_gen(subdir, lambda f: f.endswith(('jpg', 'png'))):
            img = cv2.imread(img_path)
            img = raw_img_process_func(img)
            write_record(img, label, ct, os.path.join(export_dir, sub), prefix=sub)
            ct += 1
            print(img_path)


def write_record(img, label, ct, directory, prefix='img'):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(img.tostring()),
        'label': _int64_feature(label)}))
    with tf.python_io.TFRecordWriter(os.path.join(directory, "%s_%d.tfrecord" % (prefix, ct))) as writer:
        writer.write(example.SerializeToString())


def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([1 * 28 * 28])
    image = tf.cast(image, tf.float32)
    # Reshape from [depth * height * width] to [depth, height, width].
    label = tf.cast(features['label'], tf.int32)

    return image, label


def dir_files_gen(dir, name_policy, raw=False):
    for file in os.listdir(dir):
        if name_policy(file):
            rt = file if raw else os.path.join(dir, file)
            yield rt


def transform_datadir(data_root, output_root):
    def _process(img):
        im = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (28, 28))
        cv2.threshold(im, 127, 1, cv2.THRESH_BINARY, im)
        return im

    for tp in ('train', 'test'):
        raw_data_to_tfrecord(data_root, output_root, tp, _process)


def prepare_dataset_pair(tfrecords_root, sub, nthreads=1):
    return tf.data.TFRecordDataset(
        list(dir_files_gen(
            os.path.join(tfrecords_root, sub),
            lambda f: f.endswith('.tfrecord')))
    ).map(parser, nthreads)


if __name__ == '__main__':
    transform_datadir(os.path.expanduser('~/dataset/fire/'), 'fire/data/tfrecords')
