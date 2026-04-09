

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.model import DigitCNN
from src.config import MODEL_PATH, transform

EPOCHS        = 10
BATCH_SIZE    = 64
LEARNING_RATE = 0.001


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_data = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    test_data  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    os.makedirs("models", exist_ok=True)

    model     = DigitCNN().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        accuracy = evaluate(model, test_loader, device)

        print(f"epoch {epoch:>2}/{EPOCHS}  |  loss: {avg_loss:.4f}  |  test accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), str(MODEL_PATH))
            print(f"           saved new best → {MODEL_PATH}  ({accuracy * 100:.2f}%)")

    print(f"\ntraining complete — best test accuracy: {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
