import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt_template = """
Act as a senior medical assistant. Given the user's symptoms, return possible conditions, confidence score, follow-up questions, urgency level, and advice.

Input:
{symptoms}

Output format (JSON):
{
  "conditions": [
    {"name": "", "confidence": ""}
  ],
  "follow_ups": [""],
  "urgency": "",
  "advice": ""
}
"""

def call_llm(symptoms: str) -> str:
    prompt = prompt_template.format(symptoms=symptoms)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message['content']
