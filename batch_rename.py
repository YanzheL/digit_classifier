import os

if __name__ == '__main__':
    path = 'mnist/data/train'
    for dir in os.listdir(path):
        dir = os.path.join(path, dir)
        if os.path.isdir(dir):

            for file in os.listdir(dir):
                newname = 'r_' + file
                os.rename(os.path.join(dir, file), os.path.join(dir, newname))
                print(newname)
