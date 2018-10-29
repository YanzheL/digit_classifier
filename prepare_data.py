import os
import random
from functools import partial
from multiprocessing.pool import ThreadPool

import cv2
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


def load_mnist(pool):
    mnist = fetch_mldata('MNIST original')
    dataset = mnist['data']
    labels = mnist['target']
    images = []
    for i in range(len(dataset)):
        ins = dataset[i, :].reshape(28, 28)
        images.append(ins)
        # plt.imshow(ins.reshape(28,28))
        # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    pool.map(partial(process, flag='train', reverse=True, directory='data'), zip(X_train, y_train))
    pool.map(partial(process, flag='test', reverse=True, directory='data'), zip(X_test, y_test))


def process(pair, directory, flag, reverse=False):
    process_(pair[0], pair[1], directory, flag, reverse)


def process_(img, label, directory, flag, reverse=False):
    dir = os.path.join(directory, flag, str(int(label)))
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, 'f_%d.jpg' % random.randint(1, 1000000))
    # print(dir)
    # cv2.imshow('img',image)
    # if reverse:
    #     cv2.threshold(img, 176, 255, cv2.THRESH_BINARY_INV, img)
    cv2.imwrite(path, img)


if __name__ == '__main__':
    pool = ThreadPool(processes=1000)
    load_mnist(pool)
    pool.terminate()
    pool.join()
