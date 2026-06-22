import requests
import json
import time

import os
GOOGLE_API_KEY = os.getenv("AQ.Ab8RN6IaQSv9TLsW6fzTDoRxI90PcKUi7rtGurg6vV5wgDDG6A")

# ✅ Naya Stable Model
models_to_try = [
    "gemini-2.5-pro",          # Sabse naya aur stable
    "gemini-2.5-flash",        # Fast alternative
    "gemini-2.0-flash",        # Agar quota reset ho jaye
]

print("⏳ Waiting for quota reset (30 seconds)...")
time.sleep(30)

for model in models_to_try:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    
    data = {
        "contents": [{
            "parts": [{"text": "Suggest 3 careers for acting skills. Return JSON"}]
        }]
    }
    
    print(f"🧪 Trying: {model}")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"✅ Working Model: {model}")
        print("Response:", response.json())
        break
    elif response.status_code == 429:
        print(f"⏳ {model} rate limited, trying next...")
    else:
        print(f"❌ {model} failed: {response.status_code}")