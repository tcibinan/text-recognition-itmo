import os

import numpy as np
import cv2

from scripts.common.utils import read_file_content, get_alphabet, crop_image
from scripts.neural_network.NeuralNetwork import NeuralNetwork

alphabet = get_alphabet()


def from_symbol_to_category(symbol):
    return [alphabet.index(symbol.lower())]


def from_category_to_symbol(category):
    return alphabet[category[0]]


def main():
    n_classes = len(alphabet)
    data_directory = 'data/train_data'
    unified_image_size = (10, 10)
    test_data_ids = sorted(map(lambda it: int(it.replace('image_', '').replace('.png', '')),
                               filter(lambda it: it.startswith('image'),
                                      os.listdir(data_directory))))
    X = []
    Y = []
    for i in test_data_ids:
        # if i < 20:
        img = crop_image(cv2.imread(os.path.join(data_directory, 'image_%s.png' % i), cv2.IMREAD_GRAYSCALE))
        # else:
        #     img = cv2.imread(os.path.join(data_directory, 'image_%s.png' % i), cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        X.append(np.asarray(resized_img, dtype='uint8').reshape(-1))
        # # resized_img = cv2.resize(img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        # resized_img = img
        # # X.append(np.asarray(resized_img, dtype='uint8').reshape(-1))
        # X.append(np.asarray(resized_img, dtype='uint8'))
        # if i < 20:
        #     if True:
            # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 'RGB')
            # img.save('training/image_%s.png' % i)
        Y.append(from_symbol_to_category(read_file_content(os.path.join(data_directory, 'text_%s.txt' % i))[0]))
    X = np.array(X)
    X = X / 255
    new_y = np.zeros((len(Y), n_classes))
    for i, l in enumerate(Y):
        new_y[i][l] = 1
    Y = new_y

    neural_network = NeuralNetwork(n_classes, 10 * 10)
    neural_network.train(X, Y)


if __name__ == '__main__':
    main()
