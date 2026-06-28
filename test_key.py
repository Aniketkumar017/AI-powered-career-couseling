import requests
import json
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ GEMINI_API_KEY not found in .env file!")
    exit(1)

print(f"🔑 Key loaded: {API_KEY[:8]}...{API_KEY[-4:]} (length: {len(API_KEY)})")

# Validate key format
if not API_KEY.startswith("AIza"):
    print("⚠️  WARNING: Key does not start with 'AIza' - this may not be a valid Gemini API key!")
    print("   Get a valid key from: https://aistudio.google.com/apikey")

# Models to try
models_to_try = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

for model in models_to_try:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    
    data = {
        "contents": [{
            "parts": [{"text": "Suggest 3 careers for acting skills. Return JSON"}]
        }]
    }
    
    print(f"\n🧪 Trying: {model}")
    response = requests.post(url, json=data, timeout=15)
    
    if response.status_code == 200:
        print(f"✅ Working Model: {model}")
        result = response.json()
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        print(f"Response preview: {text[:200]}")
        break
    elif response.status_code == 429:
        print(f"⏳ {model} rate limited (quota exceeded)")
    elif response.status_code == 400:
        print(f"❌ {model} bad request: {response.json().get('error', {}).get('message', '')}")
    elif response.status_code == 403:
        print(f"❌ {model} unauthorized - API key is INVALID or has no permissions!")
        print(f"   Error: {response.json().get('error', {}).get('message', '')}")
        break
    else:
        print(f"❌ {model} failed: {response.status_code} - {response.text[:200]}")