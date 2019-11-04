## Data preparation

```bash
export PYTHONPATH=$PYTHONPATH:$PWD

# to generate default training dataset for letters recognition
./presets/generate_letters.sh

# to generate default training dataset for symbols detection
./presets/generate_words.sh
```

## Training

```bash
export PYTHONPATH=$PYTHONPATH:$PWD

# to train letters recognition
./presets/train_letters_recognition.sh
```
