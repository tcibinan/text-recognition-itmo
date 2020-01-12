import numpy as np


def parse_letters(X, horizontal_multiplier=0.03, vertical_multiplier=0.03, grain=5):
    y = X.shape[1]
    for line_dimension in parse_lines(X, horizontal_multiplier=horizontal_multiplier, grain=5):
        letter_start = 0
        letter_end = 0
        empty_vertical = True
        for j in np.arange(y):
            vertical = X[line_dimension[0]:line_dimension[1], j]
            if np.sum(vertical) > (line_dimension[1] - line_dimension[0]) * vertical_multiplier:
                if empty_vertical:
                    if letter_end and j - letter_end > grain:
                        yield ([letter_end, line_dimension[0], j - 1, line_dimension[1]])
                empty_vertical = False
            else:
                if empty_vertical:
                    letter_start = j
                else:
                    letter_end = j
                    if letter_end - letter_start > grain:
                        yield ([letter_start, line_dimension[0], letter_end, line_dimension[1]])
                    letter_start = letter_end
                empty_vertical = True


def parse_lines(X, horizontal_multiplier=0.03, grain=5):
    x = X.shape[0]
    y = X.shape[1]
    empty_line = True
    line_start = 0
    for i in np.arange(x):
        horizontal = X[i, :]
        if np.sum(horizontal) > y * horizontal_multiplier:
            empty_line = False
        else:
            if empty_line:
                line_start = i
            else:
                line_end = i
                if line_end - line_start > grain:
                    yield (line_start, line_end)
            empty_line = True
