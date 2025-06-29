from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from datetime import datetime
from typing import Dict, Any

from app.models.symptom import SymptomRequest, SymptomResponse, ErrorResponse
from app.services.llm_service import LLMService
from app.config import settings

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/v1", tags=["symptoms"])

# Dependency injection
def get_llm_service() -> LLMService:
    return LLMService()

@router.post(
    "/analyze",
    response_model=SymptomResponse,
    responses={
        200: {"description": "Successful symptom analysis"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    },
    summary="Analyze patient symptoms",
    description="""
    Analyze patient symptoms using AI to provide:
    - Possible medical conditions with confidence levels
    - Follow-up questions for better diagnosis
    - Urgency level for medical attention
    - General advice for the patient
    
    **Note**: This is not a substitute for professional medical advice.
    """
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def analyze_symptoms(
    request: Request,
    symptom_request: SymptomRequest,
    llm_service: LLMService = Depends(get_llm_service)
) -> SymptomResponse:
    """
    Analyze symptoms and provide medical insights.
    
    Args:
        symptom_request: Patient symptoms and information
        
    Returns:
        SymptomResponse: Analysis results with conditions, advice, and urgency
        
    Raises:
        HTTPException: Various error conditions
    """
    try:
        logger.info(f"Analyzing symptoms: {symptom_request.symptoms[:50]}...")
        
        # Validate OpenAI API key
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail="OpenAI API key not configured"
            )
        
        # Analyze symptoms
        result = llm_service.analyze_symptoms(symptom_request)
        
        logger.info(f"Analysis completed successfully")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )

@router.get(
    "/health",
    summary="Health check",
    description="Check the health status of the API"
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.version,
        "service": settings.app_name
    }

@router.get(
    "/info",
    summary="API information",
    description="Get API information and capabilities"
)
async def api_info() -> Dict[str, Any]:
    """API information endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": "AI-powered symptom analysis API",
        "features": [
            "Symptom analysis using GPT-4",
            "Condition identification with confidence levels",
            "Urgency assessment",
            "Follow-up question generation",
            "Medical advice provision"
        ],
        "rate_limit": f"{settings.rate_limit_per_minute} requests per minute",
        "caching": "Redis-based caching enabled" if settings.redis_url else "Caching disabled"
    } 