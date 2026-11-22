"""Clasificación de productos por imagen (CNN entrenada con Fashion-MNIST)."""
import io

import numpy as np
from PIL import Image, ImageOps

from app.core.config import PRODUCT_MODEL
from app.schemas import ProductOutput

CLASES = ["Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]

_model = None


def _get_model():
    global _model
    if _model is None and PRODUCT_MODEL.exists():
        import torch  # import diferido: torch es pesado y opcional
        _model = torch.jit.load(str(PRODUCT_MODEL), map_location="cpu")
        _model.eval()
    return _model


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Convierte una foto cualquiera al formato Fashion-MNIST: 28x28, gris, fondo negro."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    # El fondo se estima con los bordes de la imagen (no con la media global,
    # que falla con prendas claras que ocupan casi todo el encuadre).
    a = np.asarray(img)
    fondo_claro = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean() > 127
    if fondo_claro:               # Fashion-MNIST usa fondo negro y prenda clara
        img = ImageOps.invert(img)
    if img.size != (28, 28):
        img = ImageOps.pad(img, (28, 28), color=0)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - 0.2860) / 0.3530).reshape(1, 1, 28, 28)


def classify(image_bytes: bytes) -> ProductOutput:
    x = preprocess(image_bytes)
    model = _get_model()
    if model is None:
        return ProductOutput(categoria="(modelo no entrenado)", confianza=0.0, top3=[], modelo_entrenado=False)

    import torch
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x)), dim=1)[0].numpy()
    order = probs.argsort()[::-1][:3]
    return ProductOutput(
        categoria=CLASES[order[0]],
        confianza=round(float(probs[order[0]]), 4),
        top3=[{"categoria": CLASES[i], "confianza": round(float(probs[i]), 4)} for i in order],
        modelo_entrenado=True,
    )
