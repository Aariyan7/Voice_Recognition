from fastapi import FastAPI, Header, HTTPException, Body
from pydantic import BaseModel
from typing import Any, Dict
from train import check_audio

app = FastAPI()

# Constant API Key for verification
API_KEY = "satchel777"

def verify_api_key(x_api_key: str = Header(...)):
    """
    Verifies that the incoming request has the correct API key in the 'x-api-key' header.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/check")
async def process_data(data: Dict[str, Any] = Body(...), x_api_key: str = Header(...)):
    """
    Endpoint that accepts data in the body, verifies the API key, and returns a JSON response.
    """
    verify_api_key(x_api_key)
    
    # Placeholder for the valid JSON response
    # You can fill this dictionary with the data you want to send back
    response_data = {
        "status": "success",
        "message": "Data received successfully",
        "response": check_audio(data["base64"])
    }
    
    return response_data

