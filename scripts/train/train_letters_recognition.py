import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import numpy as np
import cv2

from scripts.common.utils import read_file_content, get_alphabet

alphabet = get_alphabet()


def from_symbol_to_category(symbol):
    return [alphabet.index(symbol.lower())]


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
        resized_img = cv2.resize(img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        X.append(np.asarray(resized_img, dtype='uint8').reshape(-1))
        # # resized_img = cv2.resize(img, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)
        # resized_img = img
        # # X.append(np.asarray(resized_img, dtype='uint8').reshape(-1))
        # X.append(np.asarray(resized_img, dtype='uint8'))
        Y.append(from_symbol_to_category(read_file_content(os.path.join(data_directory, 'text_%s.txt' % i))[0]))
    X = np.array(X)
    X = X / 255
    Y = np_utils.to_categorical(np.array(Y), n_classes)

    # for i in range(0, X.shape[0]):
    #     x = X[i]
    #     y = Y[i]
    #     actual = from_category_to_symbol([np.argmax(y)])
    #     if actual == 'х' or actual == 'Х':
    #         from PIL import Image
    #         img = Image.fromarray(cv2.cvtColor((x * 255).astype('uint8'), cv2.COLOR_GRAY2RGB), 'RGB')
    #         img.save('detected/image_generated_original.png')
    #         img = Image.fromarray(cv2.cvtColor((cv2.resize(x, dsize=unified_image_size, interpolation=cv2.INTER_CUBIC)).astype('uint8'), cv2.COLOR_GRAY2RGB), 'RGB')
    #         img.save('detected/image_generated_resized.png')


    input_dim = unified_image_size[0] * unified_image_size[1]
    model = Sequential()
    model.add(Dense(512, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adamax')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
    else:
        model.fit(X, Y, epochs=100, verbose=2, batch_size=32, validation_split=0.3)
        model.save_weights(weights_file)
    # X_test = np.array([X[324]])
    # Y_test = Y[324]
    # actual_letter = from_category_to_symbol([np.argmax(Y_test)])
    # predicted_letter = from_category_to_symbol([np.argmax(model.predict(X_test))])
    # print(actual_letter, '->', predicted_letter)


if __name__ == '__main__':
    main()
