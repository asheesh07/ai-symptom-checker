from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum

class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"

class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    emergency = "emergency"

class Condition(BaseModel):
    name: str = Field(..., description="Name of the medical condition")
    confidence: Confidence = Field(..., description="Confidence level in this diagnosis")
    description: str = Field(..., description="Description of the condition")
    reasoning: str = Field(..., description="Medical reasoning for this diagnosis")
    urgency: Urgency = Field(..., description="Urgency level for this condition")
    icd10_code: Optional[str] = Field(None, description="ICD-10 code for the condition")

class SymptomRequest(BaseModel):
    symptoms: str = Field(..., description="Description of symptoms")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age of the patient")
    gender: Optional[Gender] = Field(None, description="Gender of the patient")
    medical_history: Optional[str] = Field(None, description="Relevant medical history")
    follow_up_answers: Optional[Dict[str, str]] = Field(None, description="Answers to follow-up questions")
    
    @field_validator('symptoms')
    @classmethod
    def validate_symptoms(cls, v):
        if not v.strip():
            raise ValueError('Symptoms cannot be empty')
        return v.strip()

class SymptomResponse(BaseModel):
    conditions: List[Condition] = Field(..., description="List of possible conditions")
    follow_ups: List[str] = Field(..., description="Follow-up questions for better diagnosis")
    urgency: Urgency = Field(..., description="Overall urgency level")
    advice: str = Field(..., description="Medical advice and recommendations")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    red_flags: List[str] = Field(default_factory=list, description="Red flags that require immediate attention")
    explanation: str = Field(..., description="Explanation of the analysis")
    disclaimer: str = Field(default="This analysis is for informational purposes only and should not replace professional medical advice. Always consult a healthcare provider for proper diagnosis and treatment.", description="Medical disclaimer")

class HealthCheck(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")

class AnalysisMetrics(BaseModel):
    """Model for tracking analysis metrics"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    symptoms: str = Field(..., description="Analyzed symptoms")
    conditions_found: int = Field(..., description="Number of conditions identified")
    urgency_level: str = Field(..., description="Overall urgency level")
    confidence_score: float = Field(..., description="Analysis confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="AI model used for analysis")
    has_red_flags: bool = Field(..., description="Whether red flags were identified") 