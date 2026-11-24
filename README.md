# CommerceAI – Sistema inteligente de comercio electrónico y logística

Plataforma web para analizar pedidos, entregas y satisfacción de clientes con 5 componentes de IA:

| # | Componente (enunciado) | Problema real | Técnica | Datos |
|---|---|---|---|---|
| 1 | Predicción | Tiempo estimado de entrega de un pedido | Regresión HGB (scikit-learn) | Olist |
| 2 | Series temporales | Demanda diaria de pedidos | SARIMAX con calendario + banda calibrada (statsmodels) | Olist |
| 3 | Clasificación de imágenes | Categorizar productos por foto | CNN (PyTorch) | Fashion-MNIST |
| 4 | Clustering | Segmentar vendedores por desempeño | K-Means, 3 grupos (scikit-learn) | Olist |
| 5 | Resumen de texto | Resumir opiniones y reclamos | TextRank ponderado + MMR (extractivo) y temas por léxico | Olist (reseñas) |

## Arquitectura

```
┌──────────────────────┐   HTTP/JSON   ┌─────────────────────────────────────────┐
│ Frontend             │ ────────────► │ Backend – FastAPI (Python)              │
│ React + TypeScript   │               │  api/routes  → endpoints REST           │
│ Vite · Tailwind      │ ◄──────────── │  services    → lógica de cada modelo IA │
│ Recharts             │               │  db          → acceso a datos Olist     │
└──────────────────────┘               │  models/     → modelos entrenados       │
                                       └──────────────▲──────────────────────────┘
                                                      │ .joblib / .pt
                                       ┌──────────────┴──────────────┐
                                       │ ml/ – scripts de entrenamiento │ ◄── data/ (Olist, Fashion-MNIST)
                                       └─────────────────────────────┘
```

## Estructura

```
commerceai/
├── backend/            API FastAPI
│   └── app/
│       ├── main.py
│       ├── schemas.py      contratos de entrada/salida
│       ├── core/config.py  rutas y configuración
│       ├── api/routes/     health, delay, demand, products, sellers, reviews
│       ├── services/       un servicio por componente de IA
│       ├── db/olist.py     carga de los CSV de Olist
│       └── models/         artefactos entrenados (generados por ml/)
├── ml/                 entrenamiento: train_delay, train_demand, train_products, train_sellers
│                       evaluación: evaluate_* (p. ej. evaluate_reviews: resumen de reseñas)
├── frontend/           React + TypeScript (una página por componente)
├── data/               datasets (no se suben a git)
├── docs/               manuales e informe técnico
└── docker-compose.yml
```

## Puesta en marcha rápida

**Si el entorno ya está instalado** (`backend/.venv` y `frontend/node_modules`): doble clic en
`iniciar_backend.bat` y `iniciar_frontend.bat`, y abrir http://localhost:5173.

Requisitos: **Python 3.11–3.12** (recomendado para compatibilidad con PyTorch/statsmodels) y **Node.js 20+**.

```bash
# 1) Backend
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uvicorn app.main:app --reload     # http://localhost:8000/docs

# 2) Frontend (desde la raíz, en otra terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

Para la demo, prepara los artefactos locales siguiendo la [auditoría y guía de activación](docs/auditoria_final_modulos.md).
Los modelos y los CSV no se incluyen en Git. ETA no emite una estimación si falta su binario aprobado;
demanda distingue el pronóstico SARIMAX del provisional, e imágenes y vendedores no muestran resultados
entrenados si faltan sus artefactos. TextRank no necesita entrenamiento.

Con Docker: `docker compose up --build` → frontend en http://localhost:8080, API en http://localhost:8000/docs.

## Reparto sugerido (4–5 integrantes)
- Integrante 1: tiempo de entrega (`ml/compare_delivery_regression.py`, `services/delivery_duration_service.py`)
- Integrante 2: series temporales (`ml/train_demand.py`, `services/demand_service.py`)
- Integrante 3: CNN de imágenes (`ml/train_products.py`, `services/product_service.py`)
- Integrante 4: clustering + resumen de texto
- Integrante 5 / todos: frontend, manuales e informe técnico (`docs/`)
