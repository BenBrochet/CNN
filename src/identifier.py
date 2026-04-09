

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

from src.model import DigitCNN
from src.pre import preprocess_image
from src.config import MODEL_PATH

THRESHOLD  = 0.80
DEMO_IMAGE = Path(__file__).parent.parent / "data" / "five.png"


def load_model():
    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def confidence_bar(value: float, width: int = 16) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def print_top3(top3: list):
    for rank, (d, c) in enumerate(top3):
        bar    = confidence_bar(c)
        marker = "▶" if rank == 0 else " "
        print(f"  {marker} digit {d}  {bar} {c * 100:5.1f}%")


def predict(image_path: str, debug: bool = False, threshold: float = THRESHOLD):
    filename = os.path.basename(image_path)

    print("loading model...")
    try:
        model = load_model()
    except Exception as e:
        print(f"error loading model: {e}")
        sys.exit(1)

    print("processing image...")
    try:
        tensor = preprocess_image(image_path, debug=debug)
    except FileNotFoundError as e:
        print(f"error: {e}")
        sys.exit(1)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits        = model(tensor)
        probabilities = F.softmax(logits, dim=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    confidence, predicted_class = probabilities.max(dim=1)
    digit      = predicted_class.item()
    confidence = confidence.item()

    top3_conf, top3_idx = probabilities[0].topk(3)
    top3 = [(top3_idx[i].item(), top3_conf[i].item()) for i in range(3)]

    print(f"\n{filename}  ·  {elapsed_ms:.1f}ms\n")

    if digit == 0:
        print("result:  not 1-9")
        print("reason:  predicted 0\n")
        print_top3(top3)
        return None, confidence

    if confidence < threshold:
        print("result:  uncertain\n")
        print_top3(top3)
        return None, confidence

    print(f"digit:   {digit}\n")
    print_top3(top3)

    return digit, confidence


def main():
    parser = argparse.ArgumentParser(
        prog="python3 -m src.identifier",
        description="reads a handwritten digit (1-9) from an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python3 -m src.identifier data/five.png\n"
            "  python3 -m src.identifier photo.jpg --threshold 0.75\n"
            "  python3 -m src.identifier data/seven.png --debug\n"
            "  python3 -m src.identifier --demo\n"
        ),
    )

    parser.add_argument(
        "image_path",
        nargs="?",
        help="jpg, png, or jpeg",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=f"try the included sample ({DEMO_IMAGE.name})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="dump preprocessing steps to debug/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        metavar="FLOAT",
        help=f"confidence cutoff, 0-1 (default: {THRESHOLD})",
    )

    args = parser.parse_args()

    print("\ndigit-identifier  —  reads a handwritten digit from an image\n")

    if args.demo:
        if not DEMO_IMAGE.exists():
            print(f"error: demo image not found at {DEMO_IMAGE}")
            sys.exit(1)
        print(f"sample: {DEMO_IMAGE.name}\n")
        predict(str(DEMO_IMAGE), debug=args.debug, threshold=args.threshold)
        return

    if not args.image_path:
        parser.print_help()
        sys.exit(0)

    predict(args.image_path, debug=args.debug, threshold=args.threshold)


if __name__ == "__main__":
    main()
