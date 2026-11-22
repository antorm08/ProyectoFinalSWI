from fastapi import APIRouter

from app.schemas import SellerClusterOutput, SellerMetrics
from app.services import seller_service

router = APIRouter(prefix="/sellers", tags=["4. Clustering de vendedores"])


@router.get("/clusters")
def clusters():
    return seller_service.summary()


@router.post("/assign", response_model=SellerClusterOutput)
def assign(metrics: SellerMetrics):
    return seller_service.assign(metrics)
