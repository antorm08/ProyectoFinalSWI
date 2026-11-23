"""Clasificación de productos por imagen (CNN entrenada con Fashion-MNIST)."""
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import PRODUCT_CONFIDENCE_THRESHOLD, PRODUCT_MAX_PIXELS, PRODUCT_MODEL
from app.schemas import ProductOutput, ProductScore

CLASES = ["Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]
MEAN, STD = 0.2860, 0.3530      # mismos estadísticos que ml/train_products.py
SIZE = 28
DIFF_FONDO = 30                 # diferencia de gris (0-255) para considerar un píxel parte de la prenda
MIN_PROPORCION = 0.01           # fila/columna con <1 % de píxeles de prenda = ruido, no recorta

_model = None


class InvalidImageError(ValueError):
    """El archivo no se puede interpretar como imagen válida."""


class NoObjectError(ValueError):
    """La imagen es válida pero no contiene ninguna prenda distinguible del fondo."""


def _get_model():
    global _model
    if _model is None and PRODUCT_MODEL.exists():
        import torch  # import diferido: torch es pesado y opcional
        _model = torch.jit.load(str(PRODUCT_MODEL), map_location="cpu")
        _model.eval()
    return _model


def _open(image_bytes: bytes) -> Image.Image:
    """Abre la imagen con orientación EXIF corregida y la transparencia sobre fondo blanco."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width * img.height > PRODUCT_MAX_PIXELS:
            raise InvalidImageError("La imagen tiene demasiados píxeles")
        img = ImageOps.exif_transpose(img)
        img.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("El archivo no es una imagen válida o está dañado") from exc
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        fondo = Image.new("RGBA", rgba.size, "white")
        img = Image.alpha_composite(fondo, rgba)
    return img.convert("L")


def _crop_to_object(a: np.ndarray) -> np.ndarray:
    """Recorta al cuadro que contiene la prenda (fondo = 0)."""
    mask = a > DIFF_FONDO
    filas = np.flatnonzero(mask.sum(1) > max(1, MIN_PROPORCION * a.shape[1]))
    cols = np.flatnonzero(mask.sum(0) > max(1, MIN_PROPORCION * a.shape[0]))
    if len(filas) == 0 or len(cols) == 0:
        raise NoObjectError("No se distingue ninguna prenda del fondo. Usa una foto con la prenda visible "
                            "sobre un fondo liso.")
    return a[filas[0]:filas[-1] + 1, cols[0]:cols[-1] + 1]


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Convierte una foto cualquiera al formato Fashion-MNIST: 28x28, gris, fondo negro, prenda centrada.

    Fashion-MNIST recorta cada producto, escala su lado mayor a 28 px y lo centra; aquí se replica:
    1. gris (+ EXIF y transparencia resueltos en _open);
    2. fondo estimado con los bordes (no con la media global, que falla con prendas claras que ocupan
       casi todo el encuadre); si es claro se invierte, y se resta su nivel para dejarlo en negro;
    3. recorte al cuadro de la prenda, relleno a cuadrado y escalado a 28x28.
    """
    a = np.asarray(_open(image_bytes), dtype=np.int16)
    bordes = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if bordes.mean() > 127:       # Fashion-MNIST usa fondo negro y prenda clara
        a = 255 - a
    nivel_fondo = int(np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])))
    a = np.clip(a - nivel_fondo, 0, 255).astype(np.uint8)

    img = ImageOps.pad(Image.fromarray(_crop_to_object(a)), (SIZE, SIZE), method=Image.LANCZOS, color=0)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).reshape(1, 1, SIZE, SIZE)


def classify(image_bytes: bytes) -> ProductOutput:
    x = preprocess(image_bytes)   # valida la imagen aunque no haya modelo
    model = _get_model()
    if model is None:
        return ProductOutput(categoria="(modelo no entrenado)", confianza=0.0, top3=[],
                             umbral_confianza=PRODUCT_CONFIDENCE_THRESHOLD, modelo_entrenado=False)

    import torch
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x)), dim=1)[0].numpy()
    order = probs.argsort()[::-1][:3]
    confianza = round(float(probs[order[0]]), 4)
    return ProductOutput(
        categoria=CLASES[order[0]],
        confianza=confianza,
        top3=[ProductScore(categoria=CLASES[i], confianza=round(float(probs[i]), 4)) for i in order],
        baja_confianza=confianza < PRODUCT_CONFIDENCE_THRESHOLD,
        umbral_confianza=PRODUCT_CONFIDENCE_THRESHOLD,
        modelo_entrenado=True,
    )
