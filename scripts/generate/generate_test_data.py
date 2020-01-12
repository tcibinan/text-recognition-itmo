import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import requests

from scripts.common.utils import get_alphabet


def generate_test_data(width, height, padding, random_padding, font_name, font_size, data_size, data_type, data_metric):
    localization = 'ru_RU'
    data_folder = 'data'
    fonts_folder = 'fonts'
    default_font_url = 'https://github.com/google/fonts/raw/8143a3e2d9f7656bc7e551f96d6294d47882d907/apache/' \
                       'robotomono/RobotoMono-Regular.ttf'
    image_name_template = 'image_%s.png'
    image_text_name_template = 'text_%s.txt'
    image_size = (width, height)
    background_color = (255, 255, 255)
    text_color = (0, 0, 0)

    font_file = os.path.join(fonts_folder, font_name)
    if not os.path.exists(font_file):
        response = requests.get(default_font_url, stream=True)
        with open(font_file, 'wb') as f:
            for chunk in response.iter_content():
                f.write(chunk)

    font = ImageFont.truetype(font_file, size=font_size)
    font_width, font_height = font.getsize('H')
    line_length = int((image_size[0] - 2 * padding) / font_width)
    text_height = int((image_size[1] - 2 * padding) / font_height)

    fake = Faker(localization)

    alphabet = get_alphabet()

    def generate_alphabet_symbols():
        i = 0
        while True:
            symbol = alphabet[i]
            i += 1
            if i >= len(alphabet):
                i = 0
            yield symbol

    alphabet_generator = generate_alphabet_symbols()

    for index in range(0, data_size):
        image_name = image_name_template % index
        image_path = os.path.join(data_folder, image_name)
        text_name = image_text_name_template % index
        text_path = os.path.join(data_folder, text_name)
        import numpy as np
        actual_padding = padding + np.sign(random_padding) * random.randint(0, np.abs(random_padding))
        if data_type == 'word':
            text = fake.word()
            actual_image_size = min(image_size[0], font_width * len(text)) + 2 * actual_padding, image_size[1]
        elif data_type == 'letter':
            text = alphabet_generator.__next__()
            text = text if random.random() >= 0.5 else text.upper()
            if text.islower():
                actual_font_height = int(font_height * (random.random() * 0.3 + 0.7))
            else:
                actual_font_height = font_height
            actual_image_size = min(image_size[0], font_width * len(text)) + 2 * actual_padding,  min(image_size[1], actual_font_height + actual_padding)
        else:
            text = fake.text(max_nb_chars=line_length * text_height)
            actual_image_size = image_size
        img = Image.new('RGB', actual_image_size, color=background_color)
        draw = ImageDraw.Draw(img)

        text_index = 0
        line_index = 0
        text_position = (0, 0)
        while line_index < text_height:
            next_line_breaker = text.find('\n', text_index + 1)
            if next_line_breaker > 0:
                next_text_index = min(text_index + line_length, next_line_breaker)
            else:
                next_text_index = text_index + line_length
            line_text = text[text_index:next_text_index].replace('\n', '')
            if data_type == 'letter':
                text_position = (actual_padding, actual_padding + line_index * font_height - (font_height - actual_font_height))
            #     text_position = (actual_padding, actual_padding + line_index * font_height - font_height * (random.random() * 0.3))
            else:
                text_position = (actual_padding, actual_padding + line_index * font_height)
            draw.text(text_position, line_text, fill=text_color, font=font)
            text_index = next_text_index
            line_index += 1

        img.save(image_path)
        with open(text_path, 'w') as f:
            actual_text = text[:text_index]
            if data_metric == 'text':
                f.write(actual_text)
            else:
                measurements = []
                element_left_top = text_position
                element_right_bottom = (element_left_top[0] + font_width, element_left_top[1] + font_height)
                for _ in range(len(actual_text)):
                    measurements.append([element_left_top[0], element_left_top[1],
                                         element_right_bottom[0], element_right_bottom[1]])
                    element_left_top = (element_left_top[0] + font_width, element_left_top[1])
                    element_right_bottom = (element_right_bottom[0] + font_width, element_right_bottom[1])
                f.write(str(measurements))


def main():
    parser = argparse.ArgumentParser(description='Generate images with random text')
    parser.add_argument('--width', type=int, default=300, help='Image width')
    parser.add_argument('--height', type=int, default=400, help='Image height')
    parser.add_argument('--padding', type=int, default=10, help='Image padding')
    parser.add_argument('--random-padding', type=int, default=5, help='Image max random padding')
    parser.add_argument('--font-name', type=str, default='RobotoMono-Regular.ttf', help='Font file name')
    parser.add_argument('--font-size', type=int, default=15, help='Font size')
    parser.add_argument('--data-size', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--data-type', type=str, default='text', help='Type of data to generate. '
                                                                      'Either text or word or letter.')
    parser.add_argument('--data-metric', type=str, default='text', help='Metric of data to measure. '
                                                                        'Either text or measurements_number.')
    args = parser.parse_args()
    generate_test_data(width=args.width, height=args.height, padding=args.padding, random_padding=args.random_padding,
                       font_name=args.font_name, font_size=args.font_size, data_size=args.data_size,
                       data_type=args.data_type, data_metric=args.data_metric)


if __name__ == '__main__':
    main()
