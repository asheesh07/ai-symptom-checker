import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from datetime import datetime

from app.main import app
from app.models.symptom import SymptomRequest, SymptomResponse, Condition, ConfidenceLevel, UrgencyLevel

client = TestClient(app)

class TestSymptomAnalysis:
    """Test cases for symptom analysis endpoint"""
    
    def test_analyze_symptoms_success(self):
        """Test successful symptom analysis"""
        with patch('app.services.llm_service.openai.ChatCompletion.create') as mock_openai:
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = {'content': json.dumps({
                "conditions": [
                    {
                        "name": "Common Cold",
                        "confidence": "high",
                        "icd10_code": "J00",
                        "description": "Viral upper respiratory infection"
                    }
                ],
                "follow_ups": [
                    "How long have you had these symptoms?",
                    "Do you have a fever?"
                ],
                "urgency": "low",
                "advice": "Rest, stay hydrated, and monitor symptoms.",
                "confidence_score": 0.85
            })}
            mock_openai.return_value = mock_response
            
            # Test request
            response = client.post(
                "/api/v1/analyze",
                json={
                    "symptoms": "fever and headache",
                    "age": 30,
                    "gender": "male"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            assert "conditions" in data
            assert "follow_ups" in data
            assert "urgency" in data
            assert "advice" in data
            assert "confidence_score" in data
            assert "disclaimer" in data
            
            # Validate conditions
            assert len(data["conditions"]) > 0
            assert data["conditions"][0]["name"] == "Common Cold"
            assert data["conditions"][0]["confidence"] == "high"
    
    def test_analyze_symptoms_missing_api_key(self):
        """Test behavior when OpenAI API key is missing"""
        with patch('app.config.settings.openai_api_key', ''):
            response = client.post(
                "/api/v1/analyze",
                json={"symptoms": "fever"}
            )
            
            assert response.status_code == 503
            assert "OpenAI API key not configured" in response.json()["detail"]
    
    def test_analyze_symptoms_empty_symptoms(self):
        """Test validation for empty symptoms"""
        response = client.post(
            "/api/v1/analyze",
            json={"symptoms": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_symptoms_invalid_age(self):
        """Test validation for invalid age"""
        response = client.post(
            "/api/v1/analyze",
            json={
                "symptoms": "fever",
                "age": 150  # Invalid age
            }
        )
        
        assert response.status_code == 422
    
    def test_analyze_symptoms_openai_error(self):
        """Test handling of OpenAI API errors"""
        with patch('app.services.llm_service.openai.ChatCompletion.create') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI API error")
            
            response = client.post(
                "/api/v1/analyze",
                json={"symptoms": "fever"}
            )
            
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]
    
    def test_analyze_symptoms_invalid_json_response(self):
        """Test handling of invalid JSON from OpenAI"""
        with patch('app.services.llm_service.openai.ChatCompletion.create') as mock_openai:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = {'content': "Invalid JSON response"}
            mock_openai.return_value = mock_response
            
            response = client.post(
                "/api/v1/analyze",
                json={"symptoms": "fever"}
            )
            
            assert response.status_code == 500

class TestHealthEndpoints:
    """Test cases for health and info endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_api_info(self):
        """Test API info endpoint"""
        response = client.get("/api/v1/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "AI Symptom Checker"
        assert "version" in data
        assert "features" in data
        assert "rate_limit" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "docs" in data

class TestRateLimiting:
    """Test cases for rate limiting"""
    
    def test_rate_limiting(self):
        """Test that rate limiting is enforced"""
        # Make multiple requests quickly
        responses = []
        for _ in range(65):  # More than the 60/minute limit
            response = client.post(
                "/api/v1/analyze",
                json={"symptoms": "fever"}
            )
            responses.append(response)
        
        # Check that some requests were rate limited
        rate_limited = [r for r in responses if r.status_code == 429]
        assert len(rate_limited) > 0

class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_404_error(self):
        """Test 404 error handling"""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_422_validation_error(self):
        """Test validation error handling"""
        response = client.post(
            "/api/v1/analyze",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 