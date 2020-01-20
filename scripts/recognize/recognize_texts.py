import os

import cv2
import numpy as np
from Levenshtein._levenshtein import distance

from scripts.common.parsers import parse_letters
from scripts.common.utils import read_file_content, get_alphabet, crop_image
from scripts.model.neural_network import NeuralNetwork

alphabet = get_alphabet()


def from_symbol_to_category(symbol):
    return [alphabet.index(symbol)]


def from_category_to_symbol(category):
    return alphabet[category[0]]


def main():
    n_classes = len(alphabet)
    data_directory = 'data/test'
    unified_image_size = (10, 10)
    test_data_ids = sorted(map(lambda it: int(it.replace('image_', '').replace('.png', '')),
                               filter(lambda it: it.startswith('image'),
                                      os.listdir(data_directory))))
    X = []
    Y = []
    for i in test_data_ids:
        img = cv2.imread(os.path.join(data_directory, 'image_%s.png' % i), cv2.IMREAD_GRAYSCALE)
        X.append(np.asarray(img, dtype='uint8'))
        Y.append(read_file_content(os.path.join(data_directory, 'text_%s.txt' % i)))
    X = np.array(X)
    X = X / 255

    for testing_index, current_X in enumerate(X):
        image_path = os.path.join(data_directory, 'image_%s.png' % testing_index)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        predicted_letters = []
        letter_dimensions = [dim for dim in parse_letters(1 - current_X)]

        i = 0
        for letter_dimension in letter_dimensions:
            i += 1
            letter_img = crop_image(img[letter_dimension[1]:letter_dimension[3], letter_dimension[0]:letter_dimension[2]])
            if not letter_img.shape[0] or not letter_img.shape[1]:
                continue
            resized_letter_img = cv2.resize(letter_img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
            input_img = np.asarray(resized_letter_img, dtype='uint8').reshape(-1)

            neural_network = NeuralNetwork(n_classes, unified_image_size[0] * unified_image_size[1])
            predicted_letter = from_category_to_symbol(neural_network.recognize(np.array([input_img])))
            predicted_letters.append(predicted_letter)
        actual_text = Y[testing_index].replace('\n', ' ')
        predicted_text = ''.join(predicted_letters)
        print('> ' + image_path)
        print('>> Levenshtein distance:', distance(actual_text, predicted_text))
        print('>> Length difference:   ', len(predicted_text) - len(actual_text))
        print('>> Actual:')
        print(actual_text)
        print('>> Predicted:')
        print(predicted_text)
        print()
        testing_index += 1


if __name__ == '__main__':
    main()
