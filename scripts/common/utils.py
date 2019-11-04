allowed_symbols_file = 'presets/alphabet.txt'


def read_file_content(path):
    with open(path, 'r') as file:
        return file.read().strip()


def get_alphabet():
    return read_file_content(allowed_symbols_file)
