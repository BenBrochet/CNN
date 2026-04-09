# digit-identifier

CNN trained on MNIST that reads handwritten digits from images. Point it at a photo and gets a digit back.

Only identifies digits 1–9. Predicting 0 is treated as uncertain by design.

## Setup

```bash
git clone <repo-url>
cd identifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the model first if you don't have weights:
```bash
python3 -m src.train
```

Run on an image:
```bash
python3 -m src.identifier data/five.png
```

Optional flags:
- `--demo` — try the included sample image
- `--debug` — saves each preprocessing step to `debug/`
- `--threshold` — confidence cutoff between 0 and 1 (default: 0.80)

Check accuracy against the MNIST test set:
```bash
python3 -m src.accuracy
```

## Project Structure

```
src/
  model.py       CNN architecture
  train.py       training loop
  pre.py         image preprocessing
  identifier.py  CLI
  config.py      shared constants
models/          saved weights
data/            sample images
```
