import os

from PIL import Image, ImageDraw
import numpy as np
import cv2

from scripts.common.parsers import parse_letters
from scripts.common.utils import read_file_content


def main():
    data_directory = 'data'
    # unified_image_size = (20, 20)
    unified_image_size = (300, 400)
    # unified_image_size = (150, 100)
    test_data_ids = sorted(map(lambda it: int(it.replace('image_', '').replace('.png', '')),
                               filter(lambda it: it.startswith('image'),
                                      os.listdir(data_directory))))
    X = []
    Y = []
    for i in test_data_ids:
        img = cv2.imread(os.path.join(data_directory, 'image_%s.png' % i), cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        X.append(np.asarray(resized_img, dtype='uint8'))
        Y.append(read_file_content(os.path.join(data_directory, 'text_%s.txt' % i)))
    X = np.array(X)
    X = 1 - X / 255
    Y = np.array(Y)
    input_dim = unified_image_size[0] * unified_image_size[1]

    testing_index = 0
    current_X = X[testing_index]
    letter_dimensions = parse_letters(current_X)

    img = Image.fromarray(cv2.imread(os.path.join(data_directory, 'image_%s.png' % testing_index)), 'RGB')
    draw = ImageDraw.Draw(img)
    recognized_letters = 0
    for letter_dimension in letter_dimensions:
        recognized_letters += 1
        draw.rectangle(list(letter_dimension), outline=(0, 0, 0), width=1)
    print('Actual letters:     %s' % len(Y[testing_index]))
    print('Recognized letters: %s' % recognized_letters)
    img.show()


if __name__ == '__main__':
    main()
