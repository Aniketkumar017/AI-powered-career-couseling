import requests
import json

BASE_URL = "http://localhost:5000"

print("="*60)
print("Testing Google Gemini API Integration")
print("="*60)

# Test 1: Career Suggestions
print("\n1. Getting AI Career Suggestions...")
response = requests.post(f"{BASE_URL}/api/gemini/suggest", json={
    "skills": "python, machine learning, data analysis",
    "education": "graduate",
    "location": "Mumbai",
    "interests": "AI, technology"
})

if response.status_code == 200:
    data = response.json()
    if data.get('success'):
        print(f"   ✅ Got {data.get('count', 0)} suggestions")
        for career in data.get('data', [])[:3]:
            print(f"      - {career.get('career')}: {career.get('reason', '')[:50]}...")
    else:
        print(f"   ❌ Error: {data.get('error')}")
else:
    print(f"   ❌ HTTP Error: {response.status_code}")

# Test 2: Hybrid Recommendations (ML + AI)
print("\n2. Getting Hybrid Recommendations...")
response = requests.post(f"{BASE_URL}/api/hybrid/recommend", json={
    "skills": "teaching, communication, computer",
    "education": "graduate",
    "location": "Delhi"
})

if response.status_code == 200:
    data = response.json()
    print(f"   ✅ ML Recommendations: {len(data.get('ml_recommendations', []))}")
    print(f"   ✅ AI Recommendations: {len(data.get('ai_recommendations', []))}")
else:
    print(f"   ❌ Error: {response.status_code}")

print("\n" + "="*60)