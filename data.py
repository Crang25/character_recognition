import numpy as np
import pathlib


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f['x_train'], f['y_train']
    images = images.astype('float32') / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels


# data = pd.read_csv(f"{pathlib.Path(__file__).parent.absolute()}/data/A_Z Handwritten Data.csv").astype('float32')

# x = data.drop('0', axis=1)
# y = data['0']

# print(x)
# print(y)