"""CommerceAI – API principal (FastAPI).

Ejecutar desde backend/:  uvicorn app.main:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import delay, demand, health, products, reviews, sellers
from app.core.config import CORS_ORIGINS

app = FastAPI(
    title="CommerceAI API",
    description="Sistema inteligente de comercio electrónico y logística (Olist + Fashion-MNIST)",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

for r in (health, delay, demand, products, sellers, reviews):
    app.include_router(r.router, prefix="/api")
