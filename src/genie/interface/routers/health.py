"""Health and readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def ready() -> HealthResponse:
    return HealthResponse(status="ready", version="0.1.0")
