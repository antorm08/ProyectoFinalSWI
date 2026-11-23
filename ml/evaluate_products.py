"""Evaluación del clasificador de productos (CNN Fashion-MNIST) ya entrenado.

Lee el artefacto backend/app/models/product_cnn.pt (generado por train_products.py) y produce
evidencia para el informe, sin reentrenar:

  1. Métricas en test (10 000) y matriz de confusión; pares de clases más confundidos.
  2. Análisis de confianza: histograma aciertos/errores y tabla de umbrales calculada en
     VALIDACIÓN (misma partición que el entrenamiento), para fijar el aviso de baja confianza
     sin mirar test.
  3. Robustez del pipeline real del backend (product_service.preprocess) ante entradas tipo foto:
     fondo blanco a color, alta resolución y prenda pequeña en un encuadre amplio.

Salidas en docs/resultados_clasificacion_productos/: metricas.json y gráficos PNG.
Uso:  python evaluate_products.py [--robustness-samples 2000]
"""
import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torchvision import datasets

from common import MODELS, ROOT
from train_products import CLASES, DATA_DIR, MEAN, STD, split_indices

sys.path.insert(0, str(ROOT / "backend"))
from app.services.product_service import preprocess  # noqa: E402  (pipeline real de producción)

OUT_DIR = ROOT / "docs" / "resultados_clasificacion_productos"
TRAIN_METRICS = ROOT / "docs" / "clasificacion_productos_metricas.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


# --------------------------------------------------------------------------- inferencia
def to_tensor(images) -> torch.Tensor:
    """PIL 28x28 en gris -> tensor normalizado igual que en el entrenamiento."""
    arr = np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in images])
    return torch.from_numpy(((arr - MEAN) / STD)[:, None])


def probabilities(model, x: torch.Tensor, batch: int = 1000) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(x[i:i + batch]), 1).numpy() for i in range(0, len(x), batch)])


# --------------------------------------------------------------------------- análisis
def threshold_table(y, probs):
    """Cobertura y accuracy por umbral de confianza (en validación)."""
    conf, pred = probs.max(1), probs.argmax(1)
    wrong = pred != y
    rows = []
    for t in THRESHOLDS:
        covered = conf >= t
        rows.append({
            "umbral": t,
            "cobertura": round(covered.mean(), 4),                          # predicciones que NO llevan aviso
            "accuracy_cubiertas": round((pred[covered] == y[covered]).mean(), 4),
            "accuracy_marcadas": round((pred[~covered] == y[~covered]).mean(), 4) if (~covered).any() else None,
            "errores_marcados": round((wrong & ~covered).sum() / max(wrong.sum(), 1), 4),
        })
    return rows


def top_confusions(cm, k=6):
    pairs = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j]
    return [{"real": CLASES[i], "predicha": CLASES[j], "casos": int(n)} for n, i, j in sorted(pairs, reverse=True)[:k]]


# --------------------------------------------------------------------------- robustez
def as_photo(img: Image.Image, size: int, canvas: float = 1.0, background: int = 255,
             noise: float = 0.0, seed: int = 0) -> Image.Image:
    """Simula una foto: prenda oscura sobre fondo claro, RGB, alta resolución.
    canvas > 1 coloca la prenda en un lienzo mayor (encuadre amplio: la prenda ocupa 1/canvas del lado).
    background < 255 da un fondo gris; noise añade ruido gaussiano (desviación en niveles de gris)."""
    garment = np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    side = int(size * canvas)
    off = (side - size) // 2
    alpha = np.zeros((side, side), dtype=np.float32)
    alpha[off:off + size, off:off + size] = garment
    # Negativo de Fashion-MNIST sobre el fondo: con background=255 equivale a ImageOps.invert.
    gray = background * (1 - alpha)
    if noise:
        gray = gray + np.random.default_rng(seed).normal(0, noise, gray.shape)
    return Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")


def encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


SCENARIOS = {
    "original_28px_fondo_negro": lambda img: encode(img, "PNG"),
    "catalogo_280px_fondo_blanco_jpeg": lambda img: encode(as_photo(img, 280), "JPEG"),
    "encuadre_amplio_prenda_50pct": lambda img: encode(as_photo(img, 200, canvas=2.0), "JPEG"),
    "encuadre_amplio_prenda_33pct": lambda img: encode(as_photo(img, 200, canvas=3.0), "JPEG"),
    "fondo_gris_ruido_prenda_50pct": lambda img: encode(
        as_photo(img, 200, canvas=2.0, background=215, noise=6, seed=0), "JPEG"),
}


def robustness(model, test_raw, n: int):
    labels = test_raw.targets.numpy()[:n]
    out = {}
    for name, make in SCENARIOS.items():
        x = torch.from_numpy(np.concatenate([preprocess(make(test_raw[i][0])) for i in range(n)]))
        probs = probabilities(model, x)
        pred = probs.argmax(1)
        out[name] = {"accuracy": round(accuracy_score(labels, pred), 4),
                     "confianza_media": round(float(probs.max(1).mean()), 4),
                     "confianza_media_en_errores": round(float(probs.max(1)[pred != labels].mean()), 4)
                     if (pred != labels).any() else None}
        print(f"  {name:36s} accuracy={out[name]['accuracy']:.4f}")
    return out


# --------------------------------------------------------------------------- gráficos
def plot_confusion(cm, path):
    from matplotlib import pyplot as plt
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "#1f2937")
    ax.set_xticks(range(len(CLASES)), CLASES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASES)), CLASES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión – test Fashion-MNIST (10 000)\ncolor = proporción por fila, número = casos")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_errors(images, y, probs, path, k=40):
    from matplotlib import pyplot as plt
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.flatnonzero(pred != y)
    worst = wrong[np.argsort(-conf[wrong])][:k]     # errores más "seguros"
    cols = 8
    fig, axes = plt.subplots(int(np.ceil(len(worst) / cols)), cols, figsize=(cols * 1.6, len(worst) / cols * 2.0))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, worst):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"R: {CLASES[y[i]]}\nP: {CLASES[pred[i]]} {conf[i]:.0%}", fontsize=7)
    fig.suptitle("Errores con mayor confianza en test (R = real, P = predicha)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confidence(y, probs, path):
    from matplotlib import pyplot as plt
    conf, ok = probs.max(1), probs.argmax(1) == y
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0.1, 1, 37)
    ax.hist(conf[ok], bins=bins, alpha=0.75, label=f"Aciertos ({ok.sum()})", color="#2563eb")
    ax.hist(conf[~ok], bins=bins, alpha=0.85, label=f"Errores ({(~ok).sum()})", color="#dc2626")
    ax.set_yscale("log")
    ax.set_xlabel("Confianza (probabilidad máxima)")
    ax.set_ylabel("Imágenes (escala log)")
    ax.set_title("Confianza del modelo en test: aciertos frente a errores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(args):
    import matplotlib
    matplotlib.use("Agg")
    if not args.model.exists():
        raise SystemExit(f"Falta {args.model}. Ejecuta antes: python train_products.py")
    sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    trained = json.loads(TRAIN_METRICS.read_text(encoding="utf-8")) if TRAIN_METRICS.exists() else {}
    seed = trained.get("protocol", {}).get("seed", 42)
    val_size = trained.get("protocol", {}).get("split", {}).get("validation", 6000)
    if trained and trained["artifact"]["sha256"] != sha:
        print("AVISO: el modelo no coincide con el de clasificacion_productos_metricas.json (sha256 distinto).")

    torch.manual_seed(seed)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    train_raw = datasets.FashionMNIST(DATA_DIR, train=True, download=True)
    test_raw = datasets.FashionMNIST(DATA_DIR, train=False, download=True)

    # 1. Test
    y_test = test_raw.targets.numpy()
    test_imgs = test_raw.data.numpy()
    p_test = probabilities(model, to_tensor(test_imgs))
    pred = p_test.argmax(1)
    cm = confusion_matrix(y_test, pred, labels=range(len(CLASES)))
    print(f"Test: accuracy={accuracy_score(y_test, pred):.4f}")

    # 2. Umbrales en validación (misma partición que el entrenamiento)
    _, val_idx = split_indices(train_raw.targets.numpy(), seed, val_size)
    y_val = train_raw.targets.numpy()[val_idx]
    p_val = probabilities(model, to_tensor(train_raw.data.numpy()[val_idx]))

    # 3. Robustez del pipeline del backend
    print(f"Robustez (preprocess del backend, {args.robustness_samples} imágenes de test):")
    robust = robustness(model, test_raw, args.robustness_samples)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, args.out / "matriz_confusion.png")
    plot_errors(test_imgs, y_test, p_test, args.out / "errores_mayor_confianza.png")
    plot_confidence(y_test, p_test, args.out / "confianza_aciertos_errores.png")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": args.model.resolve().relative_to(ROOT).as_posix(), "sha256": sha,
                     "coincide_con_entrenamiento": bool(trained) and trained["artifact"]["sha256"] == sha},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "test": {
            "n": int(len(y_test)),
            "accuracy": round(accuracy_score(y_test, pred), 5),
            "macro_f1": round(f1_score(y_test, pred, average="macro"), 5),
            "confusion_matrix": {"labels": CLASES, "rows_true_cols_pred": cm.tolist()},
            "top_confusiones": top_confusions(cm),
            "confianza_media_aciertos": round(float(p_test.max(1)[pred == y_test].mean()), 4),
            "confianza_media_errores": round(float(p_test.max(1)[pred != y_test].mean()), 4),
        },
        "umbrales_validacion": {"n": int(len(y_val)), "seed": seed, "tabla": threshold_table(y_val, p_val)},
        "umbrales_test_solo_descriptivo": threshold_table(y_test, p_test),
        "robustez_backend": {"n": args.robustness_samples, "pipeline": "app.services.product_service.preprocess",
                             "escenarios": robust},
        "graficos": sorted(p.name for p in args.out.glob("*.png")),
    }
    (args.out / "metricas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Resultados ->", args.out)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODELS / "product_cnn.pt")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--robustness-samples", type=int, default=2000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
