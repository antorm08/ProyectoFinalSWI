"""3. Clasificación de productos por imagen: CNN sobre Fashion-MNIST.

Protocolo reproducible:
  - Fashion-MNIST se descarga automáticamente con torchvision en data/fashion_mnist/.
  - Del conjunto oficial de entrenamiento (60 000) se separa una validación estratificada
    (6 000 por defecto). La mejor época se elige SOLO con validación (early stopping).
  - El conjunto oficial de test (10 000) se evalúa una única vez, al final.

Salidas:
  - backend/app/models/product_cnn.pt            TorchScript (el backend no necesita la clase)
  - docs/clasificacion_productos_metricas.json   protocolo, historial y métricas de test

Uso:  python train_products.py [--epochs 20] [--patience 4] [--seed 42]
      python train_products.py --subset 2000 --epochs 1 --output x.pt --metrics x.json  (prueba rápida)
"""
import argparse
import copy
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from .common import MODELS, ROOT
except ImportError:
    from common import MODELS, ROOT

# Mismo orden que las etiquetas de Fashion-MNIST y que CLASES en backend/app/services/product_service.py
CLASES = ["Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]
MEAN, STD = 0.2860, 0.3530          # estadísticos de Fashion-MNIST (el backend usa los mismos)
TARGET_ACCURACY = 0.91              # criterio de aceptación sobre test
DATA_DIR = ROOT / "data" / "fashion_mnist"
DEFAULT_OUTPUT = MODELS / "product_cnn.pt"
DEFAULT_METRICS = ROOT / "docs" / "clasificacion_productos_metricas.json"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(labels: np.ndarray, seed: int, val_size: int, subset: int = 0):
    """Índices (train, validación) estratificados sobre el train oficial. evaluate_products.py la reutiliza."""
    idx = np.arange(len(labels))
    if subset:  # prueba rápida: submuestra estratificada
        idx, _ = train_test_split(idx, train_size=subset, stratify=labels, random_state=seed)
        val_size = min(val_size, len(idx) // 5)
    return train_test_split(idx, test_size=val_size, stratify=labels[idx], random_state=seed)


def build_loaders(args, generator):
    norm = transforms.Normalize((MEAN,), (STD,))
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # Traslación y escala leves: tolera prendas no perfectamente centradas al subir fotos.
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_aug = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=train_tf)
    train_plain = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=eval_tf)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = split_indices(train_aug.targets.numpy(), args.seed, args.val_size, args.subset)
    if args.subset:
        test = Subset(test, np.arange(min(len(test), args.subset)))

    return (
        DataLoader(Subset(train_aug, train_idx), batch_size=args.batch_size, shuffle=True, generator=generator),
        DataLoader(Subset(train_plain, val_idx), batch_size=512),
        DataLoader(test, batch_size=512),
        {"train": len(train_idx), "validation": len(val_idx), "test": len(test)},
    )


def predict(model, loader):
    """Devuelve (etiquetas reales, predicciones, pérdida media) sobre un loader."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    ys, preds, loss = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss += loss_fn(logits, y).item()
            ys.append(y.numpy())
            preds.append(logits.argmax(1).numpy())
    y, p = np.concatenate(ys), np.concatenate(preds)
    return y, p, loss / len(y)


def train(model, loaders, args):
    train_loader, val_loader = loaders
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()
    best = {"accuracy": -1.0, "epoch": 0, "state": None}
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        y_val, p_val, val_loss = predict(model, val_loader)
        val_acc = accuracy_score(y_val, p_val)
        scheduler.step(val_acc)
        history.append({"epoch": epoch, "train_loss": round(total / n, 5), "val_loss": round(val_loss, 5),
                        "val_accuracy": round(val_acc, 5), "lr": opt.param_groups[0]["lr"],
                        "seconds": round(time.perf_counter() - start, 1)})
        mark = ""
        if val_acc > best["accuracy"]:
            best = {"accuracy": val_acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            mark = "  *mejor"
        print(f"Época {epoch:2d}: loss train={total / n:.4f}  val={val_loss:.4f}  "
              f"accuracy val={val_acc:.4f}  ({history[-1]['seconds']}s){mark}")
        if epoch - best["epoch"] >= args.patience:
            print(f"Early stopping: {args.patience} épocas sin mejorar en validación.")
            break

    model.load_state_dict(best["state"])
    return history, best


def test_metrics(y, p):
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=range(len(CLASES)), zero_division=0)
    return {
        "accuracy": round(accuracy_score(y, p), 5),
        "macro_f1": round(f1_score(y, p, average="macro"), 5),
        "per_class": {CLASES[i]: {"precision": round(prec[i], 4), "recall": round(rec[i], 4),
                                  "f1": round(f1[i], 4), "support": int(support[i])}
                      for i in range(len(CLASES))},
        "confusion_matrix": {"labels": CLASES,
                             "rows_true_cols_pred": confusion_matrix(y, p, labels=range(len(CLASES))).tolist()},
    }


def export(model, output: Path, loader) -> str:
    """Guarda TorchScript y verifica que el artefacto recargado predice igual que el modelo."""
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.jit.script(model).save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    x, _ = next(iter(loader))
    with torch.no_grad():
        if not torch.allclose(model(x), reloaded(x), atol=1e-5):
            raise RuntimeError("El artefacto TorchScript no reproduce las salidas del modelo")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(args):
    started = datetime.now(timezone.utc).isoformat()
    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)

    train_loader, val_loader, test_loader, sizes = build_loaders(args, generator)
    print(f"Datos: {sizes}")
    model = ProductCNN(len(CLASES))
    history, best = train(model, (train_loader, val_loader), args)

    y_test, p_test, _ = predict(model, test_loader)
    test = test_metrics(y_test, p_test)
    sha = export(model, args.output, test_loader)

    summary = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": "Fashion-MNIST (torchvision)",
            "seed": args.seed,
            "threads": torch.get_num_threads(),
            "split": {**sizes, "validation_stratified": True, "subset": args.subset},
            "selection": "mejor accuracy de validación; test evaluado una sola vez",
            "parameters": {"epochs_max": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                           "lr": args.lr, "optimizer": "Adam",
                           "scheduler": "ReduceLROnPlateau(max, factor=0.5, patience=1)"},
            "augmentation": ["RandomHorizontalFlip", "RandomAffine(translate=0.08, scale=0.9-1.1)"],
            "normalization": {"mean": MEAN, "std": STD},
            "architecture": "Conv(32)-BN-ReLU-Pool, Conv(64)-BN-ReLU-Pool, Dropout(0.3), FC(128), FC(10)",
            "parameters_count": sum(p.numel() for p in model.parameters()),
            "torch": torch.__version__,
        },
        "history": history,
        "selected_epoch": best["epoch"],
        "validation_accuracy": round(best["accuracy"], 5),
        "test": test,
        "acceptance": {"target_test_accuracy": TARGET_ACCURACY, "passed": test["accuracy"] >= TARGET_ACCURACY},
        "artifact": {"path": rel(args.output), "format": "TorchScript", "sha256": sha},
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nMejor época: {best['epoch']}  accuracy val={best['accuracy']:.4f}")
    print(f"TEST: accuracy={test['accuracy']:.4f}  macro-F1={test['macro_f1']:.4f}  "
          f"(objetivo >= {TARGET_ACCURACY}: {'OK' if summary['acceptance']['passed'] else 'NO alcanzado'})")
    print("Modelo  ->", args.output)
    print("Métricas->", args.metrics)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=20, help="máximo de épocas")
    parser.add_argument("--patience", type=int, default=4, help="épocas sin mejora antes de parar")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=6000)
    parser.add_argument("--threads", type=int, default=0, help="hilos de CPU para torch (0 = por defecto)")
    parser.add_argument("--subset", type=int, default=0, help="usar solo N imágenes (prueba rápida)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
