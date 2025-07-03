from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from app.models.symptom import AnalysisMetrics
from app.config import settings
import redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

# Initialize Redis for admin data
redis_client = None
if settings.redis_url:
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
        logger.info("Redis admin panel enabled")
    except Exception as e:
        logger.warning(f"Redis admin panel disabled: {e}")
        redis_client = None

def get_admin_api_key(api_key: str = Query(..., description="Admin API key")):
    """Validate admin API key"""
    if api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")
    return api_key

@router.get("/metrics", response_model=Dict[str, Any])
async def get_analytics(
    days: int = Query(7, description="Number of days to analyze"),
    api_key: str = Depends(get_admin_api_key)
):
    """Get analytics data for the specified time period"""
    try:
        if not redis_client:
            # Return mock analytics data when Redis is not available
            return {
                "period": f"Last {days} days",
                "total_analyses": 15,
                "avg_confidence": 0.85,
                "urgency_distribution": {
                    "low": 8,
                    "medium": 4,
                    "high": 2,
                    "emergency": 1
                },
                "top_symptoms": [
                    {"symptom": "fever", "count": 5},
                    {"symptom": "headache", "count": 4},
                    {"symptom": "fatigue", "count": 3},
                    {"symptom": "cough", "count": 2},
                    {"symptom": "chest pain", "count": 1}
                ],
                "red_flags_count": 3,
                "avg_processing_time": 2.5,
                "cache_hit_rate": 0.0
            }
        
        # Get metrics from Redis
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Get all metrics and filter by date
        all_metrics = []
        if redis_client.exists("analysis_metrics"):
            metrics_list = redis_client.lrange("analysis_metrics", 0, -1)
            for metric_str in metrics_list:
                try:
                    metric = json.loads(metric_str)
                    metric_time = datetime.fromisoformat(metric["timestamp"])
                    if start_time <= metric_time <= end_time:
                        all_metrics.append(metric)
                except Exception as e:
                    logger.warning(f"Failed to parse metric: {e}")
        
        if not all_metrics:
            return {
                "period": f"Last {days} days",
                "total_analyses": 0,
                "avg_confidence": 0.0,
                "urgency_distribution": {},
                "top_symptoms": [],
                "red_flags_count": 0,
                "avg_processing_time": 0.0
            }
        
        # Calculate analytics
        total_analyses = len(all_metrics)
        avg_confidence = sum(m.get("confidence_score", 0) for m in all_metrics) / total_analyses
        avg_processing_time = sum(m.get("processing_time_seconds", 0) for m in all_metrics) / total_analyses
        
        # Urgency distribution
        urgency_counts = {}
        for metric in all_metrics:
            urgency = metric.get("urgency_level", "unknown")
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        # Top symptoms (simplified analysis)
        symptom_counts = {}
        for metric in all_metrics:
            symptoms = metric.get("symptoms", "").lower()
            # Simple word-based analysis
            words = symptoms.split()
            for word in words:
                if len(word) > 3:  # Filter out short words
                    symptom_counts[word] = symptom_counts.get(word, 0) + 1
        
        top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Red flags count
        red_flags_count = sum(1 for m in all_metrics if m.get("has_red_flags", False))
        
        return {
            "period": f"Last {days} days",
            "total_analyses": total_analyses,
            "avg_confidence": round(avg_confidence, 3),
            "urgency_distribution": urgency_counts,
            "top_symptoms": [{"symptom": symptom, "count": count} for symptom, count in top_symptoms],
            "red_flags_count": red_flags_count,
            "avg_processing_time": round(avg_processing_time, 3),
            "cache_hit_rate": sum(1 for m in all_metrics if m.get("cache_hit", False)) / total_analyses if total_analyses > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")

@router.get("/recent-queries", response_model=List[Dict[str, Any]])
async def get_recent_queries(
    limit: int = Query(50, description="Number of recent queries to retrieve"),
    api_key: str = Depends(get_admin_api_key)
):
    """Get recent symptom analysis queries"""
    try:
        if not redis_client:
            # Return mock recent queries when Redis is not available
            return [
                {
                    "request_id": "mock_001",
                    "timestamp": "2025-06-29T09:30:00.000000",
                    "symptoms": "chronic fatigue, recurrent fever, night sweats...",
                    "urgency_level": "high",
                    "confidence_score": 0.95,
                    "conditions_found": 3,
                    "has_red_flags": True,
                    "processing_time": 12.2
                },
                {
                    "request_id": "mock_002",
                    "timestamp": "2025-06-29T09:25:00.000000",
                    "symptoms": "chest pain and shortness of breath...",
                    "urgency_level": "emergency",
                    "confidence_score": 0.95,
                    "conditions_found": 2,
                    "has_red_flags": True,
                    "processing_time": 8.5
                },
                {
                    "request_id": "mock_003",
                    "timestamp": "2025-06-29T09:20:00.000000",
                    "symptoms": "mild headache and fatigue...",
                    "urgency_level": "low",
                    "confidence_score": 0.75,
                    "conditions_found": 2,
                    "has_red_flags": False,
                    "processing_time": 3.2
                }
            ]
        
        recent_queries = []
        if redis_client.exists("analysis_metrics"):
            metrics_list = redis_client.lrange("analysis_metrics", 0, limit - 1)
            for metric_str in metrics_list:
                try:
                    metric = json.loads(metric_str)
                    recent_queries.append({
                        "request_id": metric.get("request_id"),
                        "timestamp": metric.get("timestamp"),
                        "symptoms": metric.get("symptoms", "")[:100] + "..." if len(metric.get("symptoms", "")) > 100 else metric.get("symptoms", ""),
                        "urgency_level": metric.get("urgency_level"),
                        "confidence_score": metric.get("confidence_score"),
                        "conditions_found": metric.get("conditions_found"),
                        "has_red_flags": metric.get("has_red_flags"),
                        "processing_time": metric.get("processing_time_seconds")
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse query: {e}")
        
        return recent_queries
        
    except Exception as e:
        logger.error(f"Error getting recent queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent queries")

@router.get("/system-status", response_model=Dict[str, Any])
async def get_system_status(api_key: str = Depends(get_admin_api_key)):
    """Get system status and health information"""
    try:
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": settings.environment,
            "redis_connected": redis_client is not None,
            "openai_configured": bool(settings.openai_api_key),
            "admin_api_configured": bool(settings.admin_api_key)
        }
        
        # Get basic metrics if Redis is available
        if redis_client:
            try:
                total_queries = redis_client.llen("analysis_metrics")
                status["total_queries_stored"] = total_queries
                
                # Get today's queries
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_queries = 0
                if total_queries > 0:
                    metrics_list = redis_client.lrange("analysis_metrics", 0, min(1000, total_queries))
                    for metric_str in metrics_list:
                        try:
                            metric = json.loads(metric_str)
                            metric_time = datetime.fromisoformat(metric["timestamp"])
                            if metric_time >= today_start:
                                today_queries += 1
                        except:
                            pass
                
                status["queries_today"] = today_queries
                
            except Exception as e:
                logger.warning(f"Failed to get Redis metrics: {e}")
                status["redis_metrics_error"] = str(e)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")

@router.delete("/clear-cache")
async def clear_cache(api_key: str = Depends(get_admin_api_key)):
    """Clear all cached data"""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Cache not available - Redis not configured")
        
        # Clear analysis metrics
        redis_client.delete("analysis_metrics")
        
        # Clear symptom analysis cache (keys starting with "symptom_analysis:")
        cache_keys = redis_client.keys("symptom_analysis:*")
        if cache_keys:
            redis_client.delete(*cache_keys)
        
        return {"message": "Cache cleared successfully", "cleared_keys": len(cache_keys) + 1}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.get("/health-check")
async def admin_health_check(api_key: str = Depends(get_admin_api_key)):
    """Admin-specific health check"""
    return {
        "status": "healthy",
        "admin_access": True,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": redis_client is not None,
            "openai": bool(settings.openai_api_key),
            "admin_api": bool(settings.admin_api_key)
        }
    } 