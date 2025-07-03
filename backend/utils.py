import json

def validate_json_structure(data: str) -> bool:
    try:
        json_obj = json.loads(data)
        return True
    except ValueError:
        return False