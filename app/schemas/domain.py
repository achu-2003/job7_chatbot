"""Read-only domain DTOs used by admin endpoints."""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class ProductDTO(BaseModel):
    sku: str
    name: str
    category: str
    price: Decimal
    stock: int


class HealthStatus(BaseModel):
    status: str
    db: bool
    redis: bool
    vector: bool
