"""3. Clasificación de productos por imagen: CNN sobre Fashion-MNIST.

Descarga Fashion-MNIST automáticamente con torchvision (en data/fashion_mnist/).
Salida: backend/app/models/product_cnn.pt (TorchScript: el backend no necesita la clase)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import MODELS, ROOT

EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProductCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    norm = transforms.Normalize((0.2860,), (0.3530,))
    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    root = ROOT / "data" / "fashion_mnist"
    train = DataLoader(datasets.FashionMNIST(root, train=True, download=True, transform=train_tf),
                       batch_size=128, shuffle=True)
    test = DataLoader(datasets.FashionMNIST(root, train=False, download=True, transform=test_tf),
                      batch_size=512)

    model = ProductCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test:
                correct += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item()
        print(f"Época {epoch}: accuracy test = {correct / len(test.dataset):.4f}")

    torch.jit.script(model.cpu().eval()).save(str(MODELS / "product_cnn.pt"))
    print("Guardado en", MODELS / "product_cnn.pt")


if __name__ == "__main__":
    main()
