import os

from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import requests

localization = 'ru_RU'
data_folder = 'data'
fonts_folder = 'fonts'
default_font_url = 'https://github.com/google/fonts/raw/8143a3e2d9f7656bc7e551f96d6294d47882d907/apache/robotomono/RobotoMono-Regular.ttf'
font_name = 'RobotoMono-Regular.ttf'
image_name_template = 'image_%s.png'
image_text_name_template = 'text_%s.txt'
image_size = (300, 400)
font_size = 15
number_of_images = 10
background_color = (255, 255, 255)
image_margin = 10
text_color = (0, 0, 0)

font_file = os.path.join(fonts_folder, font_name)
if not os.path.exists(font_file):
    response = requests.get(default_font_url, stream=True)
    with open(font_file, 'wb') as f:
        for chunk in response.iter_content():
            f.write(chunk)

fake = Faker(localization)

for index in range(0, number_of_images):
    image_name = image_name_template % index
    image_path = os.path.join(data_folder, image_name)
    text_name = image_text_name_template % index
    text_path = os.path.join(data_folder, text_name)
    img = Image.new('RGB', image_size, color=background_color)
    font = ImageFont.truetype(font_file, size=font_size)
    font_width, font_height = font.getsize('H')
    line_length = int((image_size[0] - 2 * image_margin) / font_width)
    text_height = int((image_size[1] - 2 * image_margin) / font_height)
    draw = ImageDraw.Draw(img)
    text = fake.text(max_nb_chars=line_length * text_height)

    text_index = 0
    line_index = 0

    while line_index < text_height:
        next_line_breaker = text.find('\n', text_index + 1)
        if next_line_breaker > 0:
            next_text_index = min(text_index + line_length, next_line_breaker)
        else:
            next_text_index = text_index + line_length
        line_text = text[text_index:next_text_index].replace('\n', '')
        text_position = (image_margin, image_margin + line_index * font_height)
        draw.text(text_position, line_text, fill=text_color, font=font)
        text_index = next_text_index
        line_index += 1

    img.save(image_path)
    with open(text_path, 'w') as f:
        f.write(text[:text_index])
