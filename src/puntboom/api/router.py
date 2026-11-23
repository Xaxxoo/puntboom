"""Aggregate all API routers."""

from fastapi import APIRouter

from puntboom.api.health import router as health_router
from puntboom.api.predictions import router as predictions_router
from puntboom.api.matches import router as matches_router
from puntboom.api.performance import router as performance_router
from puntboom.api.bankroll import router as bankroll_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(predictions_router)
api_router.include_router(matches_router)
api_router.include_router(performance_router)
api_router.include_router(bankroll_router)
