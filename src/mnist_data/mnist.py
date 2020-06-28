import gzip
import numpy as np
import matplotlib.pyplot as plt

def bendi_mnist():
    path ="/root/tmp/pycharm_project_872/src/mnist_data/fashion-mnist/"
    paths = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(path+paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(path+paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(path+paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(path+paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = bendi_mnist()
    print(x_train.shape)

    plt.imshow(x_train[0])
    plt.show()
