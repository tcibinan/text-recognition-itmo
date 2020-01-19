import numpy as np

allowed_symbols_file = 'presets/alphabet.txt'


def read_file_content(path):
    with open(path, 'r') as file:
        return file.read().strip() or ' '


def get_alphabet():
    return read_file_content(allowed_symbols_file)


def crop_image(X, horizontal_multiplier=0, vertical_multiplier=0):
    orig = X
    X = 1 - X / 255
    x = X.shape[0]
    y = X.shape[1]
    line_start = 0
    line_end = x
    column_start = 0
    column_end = y
    for i in np.arange(x):
        horizontal = X[i, :]
        # print(horizontal)
        if np.sum(horizontal) > y * horizontal_multiplier:
            line_start = i
            break
    for i in np.arange(x)[::-1]:
        horizontal = X[i, :]
        if np.sum(horizontal) > y * horizontal_multiplier:
            line_end = i
            break
    for i in np.arange(y):
        vertical = X[:, i]
        if np.sum(vertical) > x * vertical_multiplier:
            column_start = i
            break
    for i in np.arange(y)[::-1]:
        vertical = X[:, i]
        if np.sum(vertical) > x * vertical_multiplier:
            column_end = i
            break
    return orig[line_start:line_end, column_start:column_end]
