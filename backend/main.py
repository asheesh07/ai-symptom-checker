from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from llm_engine import call_llm
from symptom_logic import parse_llm_output

load_dotenv()

app = FastAPI()

# --- Models ---
class SymptomRequest(BaseModel):
    symptoms: str

# --- Routes ---
@app.post("/analyze")
def analyze_symptoms(req: SymptomRequest):
    try:
        raw_output = call_llm(req.symptoms)
        result = parse_llm_output(raw_output)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))