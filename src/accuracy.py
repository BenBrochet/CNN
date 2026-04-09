

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.model import DigitCNN
from src.config import MODEL_PATH, transform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    test_data   = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    accuracy = 100 * correct / total

    print(f"\ntest results")
    print(f"------------")
    print(f"correct:  {correct}/{total}")
    print(f"accuracy: {accuracy:.2f}%\n")


if __name__ == "__main__":
    main()
