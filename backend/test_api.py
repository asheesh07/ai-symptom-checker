#!/usr/bin/env python3

import requests
import json

def test_api():
    url = "http://localhost:8000/api/v1/analyze"
    
    # Test data
    data = {
        "symptoms": "mild headache and fatigue"
    }
    
    try:
        print("Testing API...")
        response = requests.post(url, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API is working!")
            print(f"Conditions found: {len(result.get('conditions', []))}")
            print(f"Urgency: {result.get('urgency')}")
        else:
            print("❌ API returned an error")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api() 