import json
from pydantic import BaseModel
from typing import List

class Condition(BaseModel):
    name: str
    confidence: str

class SymptomResponse(BaseModel):
    conditions: List[Condition]
    follow_ups: List[str]
    urgency: str
    advice: str

def parse_llm_output(output: str) -> dict:
    data = json.loads(output)
    allowed = [c for c in data["conditions"] if c["confidence"] != "low"]
    return {
        "conditions": allowed,
        "follow_ups": data["follow_ups"],
        "urgency": data["urgency"],
        "advice": data["advice"]
    }
