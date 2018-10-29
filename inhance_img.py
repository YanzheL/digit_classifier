import os
import random
from functools import partial
from multiprocessing.pool import ThreadPool

import cv2

from prepare_data import process_


def gensize():
    width = random.randint(30, 100)
    height = random.randint(30, 100)
    return width, height


def inhance(directory, pool):
    for label in range(1, 10):
        subdir = os.path.join(directory, str(label))
        filenames = os.listdir(subdir)
        for filename in filenames:
            if filename.endswith('.jpg'):
                image = cv2.imread(os.path.join(subdir, filename), 0)
                inhanced = []
                for i in range(500):
                    new = cv2.resize(image, dsize=gensize())
                    _, new = cv2.threshold(new, 127, 255, cv2.THRESH_BINARY_INV)
                    # new=new*(1+random.randint(-20,20)/100)
                    inhanced.append(new)

                pool.map(
                    partial(process_, flag='train', reverse=False, directory='data', label=label),
                    inhanced
                )


if __name__ == '__main__':
    pool = ThreadPool(processes=1000)
    inhance('firedigits', pool)
    pool.terminate()
    pool.join()
