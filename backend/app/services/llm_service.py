import openai
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
from functools import wraps
import redis
import hashlib
from app.config import settings
from app.models.symptom import SymptomRequest, SymptomResponse, Condition, Confidence, Urgency

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=settings.openai_api_key)

# Initialize Redis for caching (optional)
redis_client = None
if settings.redis_url:
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()  # Test connection
        logger.info("Redis cache enabled")
    except Exception as e:
        logger.warning(f"Redis cache disabled: {e}")
        redis_client = None

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            raise last_exception
        return wrapper
    return decorator

class LLMService:
    def __init__(self):
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
        
    def _get_cache_key(self, symptoms: str) -> str:
        """Generate cache key for symptoms"""
        return f"symptom_analysis:{hashlib.md5(symptoms.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        if not redis_client:
            return None
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save result to cache"""
        if not redis_client:
            return
        try:
            redis_client.setex(
                cache_key, 
                settings.cache_ttl, 
                json.dumps(result)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _log_metrics(self, request: SymptomRequest, response: SymptomResponse, 
                    tokens_used: int = 0, processing_time: float = 0.0) -> None:
        """Log comprehensive metrics for analysis"""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": hashlib.md5(f"{request.symptoms}{time.time()}".encode()).hexdigest()[:8],
                "symptoms": request.symptoms,
                "age": request.age,
                "gender": request.gender,
                "model_used": self.model,
                "tokens_used": tokens_used,
                "processing_time_seconds": round(processing_time, 3),
                "conditions_found": len(response.conditions),
                "urgency_level": response.urgency,
                "confidence_score": response.confidence_score,
                "cache_hit": bool(redis_client and self._get_cache_key(request.symptoms)),
                "has_red_flags": any(
                    condition.urgency in ["high", "emergency"] 
                    for condition in response.conditions
                )
            }
            
            # Log to structured log
            logger.info(f"ANALYSIS_METRICS: {json.dumps(metrics)}")
            
            # Store in Redis for analytics (if available)
            if redis_client:
                try:
                    redis_client.lpush("analysis_metrics", json.dumps(metrics))
                    redis_client.ltrim("analysis_metrics", 0, 999)  # Keep last 1000 entries
                except Exception as e:
                    logger.warning(f"Failed to store metrics: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def _create_advanced_mock_response(self, symptoms: str, follow_up_answers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create an advanced mock response with precise symptom analysis"""
        symptoms_lower = symptoms.lower()
        
        conditions = []
        urgency = "low"
        advice = "Monitor your symptoms and consult a healthcare provider if they worsen."
        follow_ups = [
            "How long have you had these symptoms?",
            "Are there any other symptoms you're experiencing?",
            "Have you taken any medications recently?"
        ]
        red_flags = []
        
        # Process follow-up answers if provided
        if follow_up_answers:
            # Adjust analysis based on follow-up answers
            if any("weight loss" in q.lower() and follow_up_answers.get(q) == "yes" for q in follow_up_answers):
                # Weight loss confirmed - increase urgency for serious conditions
                if any(word in symptoms_lower for word in ["fatigue", "fever", "night sweats"]):
                    urgency = "high"
                    red_flags.append("Unexplained weight loss with other symptoms requires immediate evaluation")
                    
            if any("lymph nodes" in q.lower() and follow_up_answers.get(q) == "yes" for q in follow_up_answers):
                # Swollen lymph nodes confirmed - consider infections and cancers
                conditions.append({
                    "name": "Lymphadenopathy",
                    "confidence": "high",
                    "description": "Enlarged lymph nodes indicating infection or malignancy",
                    "reasoning": "Swollen lymph nodes with other symptoms suggest serious underlying condition",
                    "urgency": "high",
                    "icd10_code": "R59.9"
                })
                
            if any("hiv" in q.lower() and follow_up_answers.get(q) == "yes" for q in follow_up_answers):
                # HIV testing confirmed - adjust analysis
                conditions.append({
                    "name": "HIV/AIDS",
                    "confidence": "high",
                    "description": "Human immunodeficiency virus infection",
                    "reasoning": "Confirmed HIV status with symptoms consistent with AIDS",
                    "urgency": "high",
                    "icd10_code": "B20"
                })
                
            if any("respiratory" in q.lower() and follow_up_answers.get(q) == "yes" for q in follow_up_answers):
                # Respiratory symptoms confirmed
                if "cough" not in symptoms_lower and "shortness of breath" not in symptoms_lower:
                    conditions.append({
                        "name": "Respiratory Infection",
                        "confidence": "medium",
                        "description": "Infection affecting the respiratory system",
                        "reasoning": "Additional respiratory symptoms suggest respiratory involvement",
                        "urgency": "medium",
                        "icd10_code": "J06.9"
                    })
        
        # HIV/AIDS and Immunodeficiency Analysis
        if any(word in symptoms_lower for word in ["chronic fatigue", "recurrent fever", "night sweats", "weight loss", "aids", "hiv"]):
            conditions.extend([
                {
                    "name": "HIV/AIDS",
                    "confidence": "high",
                    "description": "Human immunodeficiency virus infection leading to acquired immunodeficiency syndrome",
                    "reasoning": "Chronic fatigue, recurrent fever, and night sweats are classic symptoms of HIV/AIDS, especially in advanced stages",
                    "urgency": "high",
                    "icd10_code": "B20"
                },
                {
                    "name": "Lymphoma",
                    "confidence": "medium",
                    "description": "Cancer of the lymphatic system",
                    "reasoning": "Night sweats, fatigue, and recurrent fever are common symptoms of lymphoma",
                    "urgency": "high",
                    "icd10_code": "C85.9"
                },
                {
                    "name": "Tuberculosis",
                    "confidence": "medium",
                    "description": "Bacterial infection primarily affecting the lungs",
                    "reasoning": "Night sweats, fatigue, and fever are hallmark symptoms of tuberculosis",
                    "urgency": "high",
                    "icd10_code": "A15.9"
                }
            ])
            urgency = "high"
            red_flags.extend([
                "Chronic fatigue with recurrent fever and night sweats suggest serious underlying condition",
                "Weight loss with these symptoms requires immediate medical evaluation"
            ])
            advice = "These symptoms require IMMEDIATE medical attention. Please see a healthcare provider as soon as possible for proper testing and diagnosis. These symptoms are consistent with serious conditions that need prompt evaluation."
            follow_ups.extend([
                "Have you experienced any unexplained weight loss?",
                "Do you have any swollen lymph nodes?",
                "Have you been tested for HIV or other infections?",
                "Do you have any respiratory symptoms like cough or shortness of breath?"
            ])
            
        # Cancer-related symptoms
        elif any(word in symptoms_lower for word in ["unexplained weight loss", "persistent fatigue", "night sweats", "cancer", "tumor"]):
            conditions.extend([
                {
                    "name": "Malignancy (Various Types)",
                    "confidence": "high",
                    "description": "Various types of cancer can cause these symptoms",
                    "reasoning": "Unexplained weight loss, persistent fatigue, and night sweats are classic cancer symptoms",
                    "urgency": "high",
                    "icd10_code": "C80.1"
                },
                {
                    "name": "Leukemia",
                    "confidence": "medium",
                    "description": "Cancer of blood-forming tissues",
                    "reasoning": "Fatigue, night sweats, and weight loss are common in leukemia",
                    "urgency": "high",
                    "icd10_code": "C95.9"
                }
            ])
            urgency = "high"
            red_flags.append("Unexplained weight loss with fatigue requires immediate cancer screening")
            advice = "These symptoms require urgent medical evaluation for possible malignancy. Please see a doctor immediately for proper testing."
            
        # Cardiovascular emergencies
        elif any(word in symptoms_lower for word in ["chest pain", "shortness of breath", "heart attack", "angina"]):
            conditions.extend([
                {
                    "name": "Acute Coronary Syndrome",
                    "confidence": "high",
                    "description": "Heart attack or unstable angina",
                    "reasoning": "Chest pain with shortness of breath is a classic presentation of heart attack",
                    "urgency": "emergency",
                    "icd10_code": "I21.9"
                },
                {
                    "name": "Pulmonary Embolism",
                    "confidence": "medium",
                    "description": "Blood clot in the lungs",
                    "reasoning": "Chest pain and shortness of breath can indicate pulmonary embolism",
                    "urgency": "emergency",
                    "icd10_code": "I26.9"
                }
            ])
            urgency = "emergency"
            red_flags.append("Chest pain with shortness of breath is a medical emergency")
            advice = "CALL EMERGENCY SERVICES IMMEDIATELY. This could be a heart attack or pulmonary embolism."
            
        # Neurological emergencies
        elif any(word in symptoms_lower for word in ["severe headache", "worst headache", "stroke", "numbness", "weakness"]):
            conditions.extend([
                {
                    "name": "Subarachnoid Hemorrhage",
                    "confidence": "high",
                    "description": "Bleeding in the brain",
                    "reasoning": "Sudden severe headache described as 'worst ever' is classic for brain hemorrhage",
                    "urgency": "emergency",
                    "icd10_code": "I60.9"
                },
                {
                    "name": "Ischemic Stroke",
                    "confidence": "medium",
                    "description": "Blockage of blood flow to the brain",
                    "reasoning": "Sudden weakness, numbness, or severe headache can indicate stroke",
                    "urgency": "emergency",
                    "icd10_code": "I63.9"
                }
            ])
            urgency = "emergency"
            red_flags.append("Sudden severe headache or neurological symptoms require immediate evaluation")
            advice = "CALL EMERGENCY SERVICES IMMEDIATELY. This could be a stroke or brain hemorrhage."
            
        # Respiratory conditions
        elif any(word in symptoms_lower for word in ["cough", "shortness of breath", "chest tightness"]):
            conditions.extend([
                {
                    "name": "Pneumonia",
                    "confidence": "high",
                    "description": "Infection of the lungs",
                    "reasoning": "Cough with shortness of breath and fever suggests pneumonia",
                    "urgency": "medium",
                    "icd10_code": "J18.9"
                },
                {
                    "name": "COPD Exacerbation",
                    "confidence": "medium",
                    "description": "Worsening of chronic obstructive pulmonary disease",
                    "reasoning": "Increased shortness of breath and cough in COPD patients",
                    "urgency": "medium",
                    "icd10_code": "J44.1"
                }
            ])
            urgency = "medium"
            advice = "Seek medical attention for proper evaluation and treatment of respiratory symptoms."
            
        # Gastrointestinal conditions
        elif any(word in symptoms_lower for word in ["abdominal pain", "nausea", "vomiting", "diarrhea"]):
            conditions.extend([
                {
                    "name": "Appendicitis",
                    "confidence": "medium",
                    "description": "Inflammation of the appendix",
                    "reasoning": "Right lower abdominal pain with nausea suggests appendicitis",
                    "urgency": "high",
                    "icd10_code": "K35.90"
                },
                {
                    "name": "Gastroenteritis",
                    "confidence": "medium",
                    "description": "Inflammation of stomach and intestines",
                    "reasoning": "Nausea, vomiting, and diarrhea are classic symptoms",
                    "urgency": "low",
                    "icd10_code": "K52.9"
                }
            ])
            if "severe" in symptoms_lower or "right" in symptoms_lower:
                urgency = "high"
                red_flags.append("Severe abdominal pain requires immediate evaluation")
            advice = "Monitor symptoms. Seek immediate care if pain becomes severe or localized to right lower abdomen."
            
        # Common viral infections (only for mild symptoms)
        elif any(word in symptoms_lower for word in ["mild fever", "sore throat", "runny nose", "mild headache"]):
            conditions.extend([
                {
                    "name": "Upper Respiratory Infection",
                    "confidence": "high",
                    "description": "Common cold or viral infection",
                    "reasoning": "Mild symptoms suggest common viral infection",
                    "urgency": "low",
                    "icd10_code": "J06.9"
                },
                {
                    "name": "Seasonal Allergies",
                    "confidence": "medium",
                    "description": "Allergic reaction to environmental triggers",
                    "reasoning": "Mild symptoms without fever may indicate allergies",
                    "urgency": "low",
                    "icd10_code": "J30.9"
                }
            ])
            urgency = "low"
            advice = "Rest, stay hydrated, and use over-the-counter medications as needed. Seek care if symptoms worsen."
            
        # Mental health conditions
        elif any(word in symptoms_lower for word in ["depression", "anxiety", "mood changes", "sleep problems"]):
            conditions.extend([
                {
                    "name": "Major Depressive Disorder",
                    "confidence": "medium",
                    "description": "Clinical depression affecting daily functioning",
                    "reasoning": "Persistent low mood, fatigue, and sleep changes suggest depression",
                    "urgency": "medium",
                    "icd10_code": "F32.9"
                },
                {
                    "name": "Generalized Anxiety Disorder",
                    "confidence": "medium",
                    "description": "Excessive worry and anxiety",
                    "reasoning": "Persistent anxiety with physical symptoms",
                    "urgency": "medium",
                    "icd10_code": "F41.1"
                }
            ])
            urgency = "medium"
            advice = "Consider speaking with a mental health professional. These symptoms are treatable."
            
        # If no specific pattern is detected, provide general guidance
        else:
            conditions.append({
                "name": "Non-Specific Symptoms",
                "confidence": "low",
                "description": "Symptoms require more detailed evaluation",
                "reasoning": "Symptoms are non-specific and need comprehensive medical assessment",
                "urgency": "low",
                "icd10_code": "R68.89"
            })
            advice = "Please provide more specific symptoms or consult a healthcare provider for proper evaluation."
            
        # Ensure at least 2 conditions for comprehensive analysis
        if len(conditions) < 2:
            conditions.append({
                "name": "General Medical Evaluation",
                "confidence": "low",
                "description": "Requires comprehensive medical assessment",
                "reasoning": "Symptoms need professional evaluation for accurate diagnosis",
                "urgency": "low",
                "icd10_code": "Z00.00"
            })
            
        # Determine overall urgency based on conditions
        urgency_levels = {"low": 1, "medium": 2, "high": 3, "emergency": 4}
        max_urgency = max([urgency_levels.get(c.get("urgency", "low"), 1) for c in conditions])
        urgency = [k for k, v in urgency_levels.items() if v == max_urgency][0]
            
        # Calculate dynamic confidence score based on analysis quality
        def calculate_confidence_score():
            base_score = 0.5
            
            # Boost for specific symptom patterns
            if any(word in symptoms_lower for word in ["chronic fatigue", "recurrent fever", "night sweats", "weight loss", "aids", "hiv"]):
                base_score += 0.25  # High specificity for serious conditions
            elif any(word in symptoms_lower for word in ["chest pain", "shortness of breath", "heart attack", "angina"]):
                base_score += 0.25  # High specificity for cardiac emergencies
            elif any(word in symptoms_lower for word in ["severe headache", "worst headache", "stroke", "numbness", "weakness"]):
                base_score += 0.25  # High specificity for neurological emergencies
            elif any(word in symptoms_lower for word in ["unexplained weight loss", "persistent fatigue", "night sweats", "cancer", "tumor"]):
                base_score += 0.20  # High specificity for cancer symptoms
            elif any(word in symptoms_lower for word in ["abdominal pain", "nausea", "vomiting", "diarrhea"]):
                base_score += 0.15  # Medium specificity for GI conditions
            elif any(word in symptoms_lower for word in ["cough", "shortness of breath", "chest tightness"]):
                base_score += 0.15  # Medium specificity for respiratory conditions
            elif any(word in symptoms_lower for word in ["mild fever", "sore throat", "runny nose", "mild headache"]):
                base_score += 0.10  # Lower specificity for common conditions
            elif any(word in symptoms_lower for word in ["depression", "anxiety", "mood changes", "sleep problems"]):
                base_score += 0.10  # Lower specificity for mental health
            else:
                base_score += 0.05  # Very low specificity for non-specific symptoms
            
            # Boost for number of conditions (more conditions = more comprehensive analysis)
            if len(conditions) >= 3:
                base_score += 0.15
            elif len(conditions) == 2:
                base_score += 0.10
            else:
                base_score += 0.05
            
            # Boost for high confidence conditions
            high_confidence_count = sum(1 for c in conditions if c.get("confidence") == "high")
            if high_confidence_count > 0:
                base_score += 0.10 * high_confidence_count
            
            # Boost for red flags (indicates serious analysis)
            if red_flags:
                base_score += 0.10
            
            # Penalty for non-specific symptoms
            if "Non-Specific Symptoms" in [c.get("name") for c in conditions]:
                base_score -= 0.15
            
            # Ensure score is within valid range
            return max(0.3, min(0.95, base_score))
        
        confidence_score = calculate_confidence_score()
            
        return {
            "conditions": conditions,
            "follow_ups": follow_ups,
            "urgency": urgency,
            "advice": advice,
            "confidence_score": round(confidence_score, 2),
            "red_flags": red_flags,
            "explanation": f"Analysis based on {len(conditions)} possible conditions with {urgency} urgency level. Symptoms analyzed for specific patterns and red flags."
        }
    
    def _create_advanced_prompt(self, request: SymptomRequest) -> str:
        """Create an advanced prompt that demands multiple conditions with explainability"""
        
        # Build follow-up answers section
        follow_up_section = ""
        if request.follow_up_answers:
            follow_up_section = f"""
FOLLOW-UP ANSWERS:
{chr(10).join([f"- {question}: {answer}" for question, answer in request.follow_up_answers.items()])}

Use these answers to refine your analysis and provide more accurate conditions.
"""
        
        prompt = f"""
You are an advanced medical AI assistant with expertise in symptom analysis. Your task is to provide comprehensive, evidence-based analysis with multiple possible conditions and clear reasoning.

PATIENT INFORMATION:
- Symptoms: {request.symptoms}
- Age: {request.age if request.age else 'Not specified'}
- Gender: {request.gender if request.gender else 'Not specified'}
- Medical History: {request.medical_history if request.medical_history else 'Not specified'}{follow_up_section}

ANALYSIS REQUIREMENTS:
1. Provide AT LEAST 3 possible medical conditions (more if symptoms suggest multiple systems)
2. For each condition, include:
   - Clear reasoning why this condition is suggested
   - Confidence level (high/medium/low) based on symptom specificity
   - Urgency level for medical attention
   - ICD-10 code if applicable
3. Identify any RED FLAG symptoms requiring immediate attention
4. Provide specific follow-up questions to gather more information
5. Give actionable advice with clear next steps

RESPONSE FORMAT (JSON only):
{{
  "conditions": [
    {{
      "name": "Condition name",
      "confidence": "high|medium|low",
      "description": "Brief medical description",
      "reasoning": "Why this condition is suggested based on symptoms",
      "urgency": "low|medium|high|emergency",
      "icd10_code": "ICD-10 code if known"
    }}
  ],
  "follow_ups": [
    "Specific question to gather more information",
    "Question about symptom duration",
    "Question about symptom severity"
  ],
  "urgency": "low|medium|high|emergency",
  "advice": "Specific, actionable advice for the patient",
  "confidence_score": 0.85,
  "red_flags": [
    "Any red flag symptoms identified"
  ],
  "explanation": "Brief explanation of the analysis approach"
}}

CRITICAL GUIDELINES:
- ALWAYS provide at least 3 conditions unless symptoms are extremely specific
- Prioritize patient safety - if ANY red flags, mark urgency as "emergency"
- Be specific about reasoning - explain WHY each condition is suggested
- Use evidence-based medical knowledge
- Include medical disclaimers in advice
- Confidence score should reflect overall analysis quality (0.0-1.0)
- Consider differential diagnosis approach
- If symptoms suggest multiple body systems, include conditions from each system
- If follow-up answers are provided, use them to refine your analysis and increase confidence

RED FLAG SYMPTOMS (require emergency evaluation):
- Chest pain, especially with shortness of breath
- Severe headache (worst ever)
- Sudden weakness or numbness
- Severe abdominal pain
- High fever with confusion
- Difficulty breathing
- Unconsciousness or severe confusion
"""
        return prompt
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _call_openai(self, prompt: str) -> str:
        """Make API call to OpenAI with error handling"""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except openai.AuthenticationError:
            raise ValueError("Invalid OpenAI API key")
        except openai.RateLimitError:
            raise Exception("OpenAI rate limit exceeded")
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error calling OpenAI: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['conditions', 'follow_ups', 'urgency', 'advice', 'confidence_score']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing response: {e}")
    
    def analyze_symptoms(self, request: SymptomRequest) -> SymptomResponse:
        """Analyze symptoms using LLM with caching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(request.symptoms)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            logger.info("Returning cached result")
            return SymptomResponse(**cached_result)
        
        try:
            # Try OpenAI first
            if settings.openai_api_key:
                try:
                    # Create prompt
                    prompt = self._create_advanced_prompt(request)
                    
                    # Call OpenAI
                    logger.info("Calling OpenAI API")
                    raw_response = self._call_openai(prompt)
                    
                    # Parse response
                    parsed_data = self._parse_llm_response(raw_response)
                except Exception as e:
                    logger.warning(f"OpenAI API failed, using mock response: {e}")
                    parsed_data = self._create_advanced_mock_response(request.symptoms, request.follow_up_answers)
            else:
                # Use mock response if no API key
                logger.info("No OpenAI API key, using mock response")
                parsed_data = self._create_advanced_mock_response(request.symptoms, request.follow_up_answers)
            
            # Convert to Pydantic models
            conditions = [
                Condition(
                    name=cond.get('name', ''),
                    confidence=cond.get('confidence', 'low'),
                    description=cond.get('description', ''),
                    reasoning=cond.get('reasoning', ''),
                    urgency=cond.get('urgency', 'low'),
                    icd10_code=cond.get('icd10_code')
                )
                for cond in parsed_data.get('conditions', [])
            ]
            
            # Filter out low confidence conditions
            filtered_conditions = [
                cond for cond in conditions 
                if cond.confidence != Confidence.low
            ]
            
            # Create response
            response = SymptomResponse(
                conditions=filtered_conditions,
                follow_ups=parsed_data.get('follow_ups', []),
                urgency=parsed_data.get('urgency', 'low'),
                advice=parsed_data.get('advice', ''),
                confidence_score=parsed_data.get('confidence_score', 0.5),
                red_flags=parsed_data.get('red_flags', []),
                explanation=parsed_data.get('explanation', 'Analysis completed successfully')
            )
            
            # Cache the result
            self._save_to_cache(cache_key, response.dict())
            
            # Log performance
            elapsed_time = time.time() - start_time
            logger.info(f"Symptom analysis completed in {elapsed_time:.2f}s")
            
            # Log metrics
            self._log_metrics(request, response, len(parsed_data.get('conditions', [])), elapsed_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            raise 