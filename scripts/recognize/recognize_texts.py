import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential

from scripts.common.parsers import parse_letters
from scripts.common.utils import read_file_content, get_alphabet, crop_image

alphabet = get_alphabet()


def from_symbol_to_category(symbol):
    return [alphabet.index(symbol)]


def from_category_to_symbol(category):
    return alphabet[category[0]]


def main():
    n_classes = len(alphabet)
    data_directory = 'data'
    weights_directory = 'weights'
    weights_file = os.path.join(weights_directory, 'letters-recognition.h5')
    unified_image_size = (10, 10)
    test_data_ids = sorted(map(lambda it: int(it.replace('image_', '').replace('.png', '')),
                               filter(lambda it: it.startswith('image'),
                                      os.listdir(data_directory))))
    X = []
    Y = []
    for i in test_data_ids:
        img = cv2.imread(os.path.join(data_directory, 'image_%s.png' % i), cv2.IMREAD_GRAYSCALE)
        # resized_img = cv2.resize(img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        X.append(np.asarray(img, dtype='uint8'))
        Y.append(read_file_content(os.path.join(data_directory, 'text_%s.txt' % i))[0])
    X = np.array(X)
    X = X / 255
    Y = np.array(Y)
    input_dim = unified_image_size[0] * unified_image_size[1]

    testing_index = 0
    current_X = X[testing_index]

    model = Sequential()
    model.add(Dense(512, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.load_weights(weights_file)

    predicted_letters = []
    letter_dimensions = [dim for dim in parse_letters(1 - current_X)]

    # for letter_dimension in letter_dimensions[3:4]:
    #     letter_img = cv2.imread(os.path.join(data_directory, 'image_%s.png' % testing_index))[
    #                  letter_dimension[1]:letter_dimension[3], letter_dimension[0]:letter_dimension[2]]
    #     letter_img = cv2.resize(letter_img, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    #     img = Image.fromarray(letter_img, 'RGB')
    #     img.show()

    i = 0
    for letter_dimension in letter_dimensions:
        i += 1
        letter_img = crop_image(cv2.imread(os.path.join(data_directory, 'image_%s.png' % testing_index), cv2.IMREAD_GRAYSCALE) \
                     [letter_dimension[1]:letter_dimension[3], letter_dimension[0]:letter_dimension[2]])
        # letter_img = current_X[letter_dimension[1]:letter_dimension[3], letter_dimension[0]:letter_dimension[2]]
        # letter_img = current_X[letter_dimension[0]:letter_dimension[2], letter_dimension[1]:letter_dimension[3]]
        # [letter_dimension[0]: letter_dimension[2], letter_dimension[1]: letter_dimension[3]]
        if not letter_img.shape[0] or not letter_img.shape[1]:
            continue
        resized_letter_img = cv2.resize(letter_img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        input_img = np.asarray(resized_letter_img, dtype='uint8').reshape(-1)

        # if i < 20:
        # if True:
        #     img = Image.fromarray(cv2.cvtColor(resized_letter_img, cv2.COLOR_GRAY2RGB), 'RGB')
        #     img.save('detected/image_%s.png' % i)
        # img = Image.fromarray(cv2.cvtColor(resized_letter_img, cv2.COLOR_GRAY2RGB), 'RGB')
        # img.save('detected/image_%s.png' % i)
        # draw = ImageDraw.Draw(img)
        # img.show()
        # draw.rectangle(list(letter_dimension), outline=(0, 0, 0), width=1)

        predicted_letter = from_category_to_symbol([np.argmax(model.predict(np.array([input_img])))])
        predicted_letters.append(predicted_letter)
    predicted_text = ''.join(predicted_letters)
    print(predicted_text)


if __name__ == '__main__':
    main()
