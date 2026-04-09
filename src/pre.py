

import os
import cv2
import numpy as np
import torch

from src.config import transform


def preprocess_image(image_path: str, debug: bool = False) -> torch.Tensor:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"could not load image: {image_path}")

    debug_dir = None

    if debug:
        stem      = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = os.path.join("debug", stem)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_original.png"), image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "02_gray.png"), gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "03_threshold.png"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit = thresh
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = thresh[y:y+h, x:x+w]

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "04_cropped.png"), digit)

    size     = max(digit.shape[0], digit.shape[1]) + 20
    canvas   = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - digit.shape[0]) // 2
    x_offset = (size - digit.shape[1]) // 2
    canvas[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit

    final = cv2.resize(canvas, (28, 28))
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "05_final_28x28.png"), final)
        print(f"debug: saved preprocessing steps to {debug_dir}/")

    tensor = transform(final).unsqueeze(0)
    if debug:
        print(f"debug: tensor shape = {list(tensor.shape)}")

    return tensor
