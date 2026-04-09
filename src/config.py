from pathlib import Path
from torchvision import transforms

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "digit_cnn.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
