from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import sys
import requests
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    print("⚠️ GEMINI_API_KEY not found in .env file")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.predictor import CareerPredictor
from models.scheme_matcher import SchemeMatcher

app = Flask(__name__)

# Initialize models
predictor = CareerPredictor()
scheme_matcher = SchemeMatcher()


# ============================================
# GOOGLE GEMINI API - AI CAREER SUGGESTIONS
# ============================================

@app.route('/api/gemini/suggest', methods=['POST'])
def gemini_career_suggest():
    """
    Get AI-powered career suggestions using Google Gemini
    """
    try:
        data = request.get_json()
        skills = data.get('skills', '').strip()
        education = data.get('education', '').strip()
        location = data.get('location', '').strip()
        interests = data.get('interests', '').strip()
        
        if not skills:
            return jsonify({'success': False, 'error': 'Skills are required'}), 400
        
        # Check if Gemini API key is configured
        if not GEMINI_API_KEY:
            return jsonify({
                'success': False, 
                'error': 'Gemini API key not configured. Please add GEMINI_API_KEY to .env file'
            }), 500
        
        # Create prompt for Gemini
        prompt = f"""
        You are an AI career counselor for Indian rural youth. Based on the following information, suggest 5 career paths.
        
        User Information:
        - Skills: {skills}
        - Education: {education}
        - Location: {location}
        - Interests: {interests if interests else 'Not specified'}
        
        For each career suggestion, provide:
        1. Career Name
        2. Brief reason why this career fits
        3. Required training duration (in months)
        4. Expected salary range (in INR)
        5. Growth potential (High/Medium/Low)
        
        Format your response as JSON only, no extra text. Use this exact format:
        [
            {{
                "career": "Career Name",
                "reason": "Why this fits",
                "training_months": 6,
                "salary_range": "₹15,000 - ₹25,000",
                "growth": "High"
            }}
        ]
        
        Make sure careers are practical for rural youth in India.
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Parse response
        import json
        try:
            # Clean response text (remove markdown code blocks if any)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            suggestions = json.loads(response_text)
            
            return jsonify({
                'success': True,
                'data': suggestions,
                'count': len(suggestions),
                'source': 'Google Gemini AI'
            })
            
        except json.JSONDecodeError as e:
            # Fallback: return raw text
            return jsonify({
                'success': True,
                'data': [{'career': 'AI Suggestion', 'reason': response.text[:200]}],
                'count': 1,
                'source': 'Google Gemini AI'
            })
            
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/gemini/analyze', methods=['POST'])
def gemini_analyze_profile():
    """
    Deep analyze user profile using Gemini
    """
    try:
        data = request.get_json()
        skills = data.get('skills', '').strip()
        education = data.get('education', '').strip()
        location = data.get('location', '').strip()
        
        if not skills:
            return jsonify({'success': False, 'error': 'Skills are required'}), 400
        
        if not GEMINI_API_KEY:
            return jsonify({'success': False, 'error': 'Gemini API key not configured'}), 500
        
        # First get ML recommendations
        ml_result = predictor.recommend({"skills": skills, "education": education, "location": location})
        ml_careers = ml_result['job_title'].tolist() if not ml_result.empty else []
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze this user profile and provide detailed career guidance.
        
        Profile:
        - Skills: {skills}
        - Education: {education}
        - Location: {location}
        
        ML Model Recommended: {', '.join(ml_careers) if ml_careers else 'No specific recommendations'}
        
        Provide:
        1. Strengths analysis (what user is good at)
        2. Skill gaps (what they need to learn)
        3. Top 3 career recommendations with reasoning
        4. Short-term action plan (next 6 months)
        5. Long-term career roadmap (1-3 years)
        
        Keep response concise and practical for rural youth.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'analysis': response.text,
            'ml_recommendations': ml_careers,
            'source': 'Google Gemini AI'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/gemini/roadmap', methods=['POST'])
def gemini_career_roadmap():
    """
    Generate detailed career roadmap for a specific career
    """
    try:
        data = request.get_json()
        career_name = data.get('career', '').strip()
        skills = data.get('skills', '').strip()
        
        if not career_name:
            return jsonify({'success': False, 'error': 'Career name is required'}), 400
        
        if not GEMINI_API_KEY:
            return jsonify({'success': False, 'error': 'Gemini API key not configured'}), 500
        
        prompt = f"""
        Create a detailed career roadmap for becoming a {career_name}.
        
        User's current skills: {skills if skills else 'Not specified'}
        
        Provide a step-by-step roadmap including:
        1. Required education/certifications
        2. Skills to learn (with timeline)
        3. Training programs in India
        4. Job application strategy
        5. Expected salary progression
        6. Growth opportunities
        
        Format as bullet points. Be practical for Indian context.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return jsonify({
            'success': True,
            'career': career_name,
            'roadmap': response.text,
            'source': 'Google Gemini AI'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# JOB API (External - No key needed)
# ============================================

@app.route('/api/jobs/search', methods=['POST'])
def search_jobs():
    try:
        data = request.get_json()
        job_title = data.get('job_title', '')
        location = data.get('location', 'India')
        
        # Use Remotive.io API (Free, no key needed)
        url = f"https://remotive.io/api/remote-jobs?search={job_title}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            jobs = response.json().get('jobs', [])[:5]
            
            formatted = []
            for job in jobs:
                formatted.append({
                    'title': job.get('title', 'N/A'),
                    'company': job.get('company_name', 'N/A'),
                    'location': job.get('candidate_required_location', 'Remote'),
                    'salary': job.get('salary', 'Not specified'),
                    'url': job.get('url', '#'),
                    'description': job.get('description', '')[:200]
                })
            
            return jsonify({'success': True, 'data': formatted, 'count': len(formatted)})
        else:
            return jsonify({'success': False, 'error': 'Unable to fetch jobs'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# CAREER RECOMMEND API (Your ML Model)
# ============================================

@app.route('/api/careers/recommend', methods=['POST'])
def api_careers_recommend():
    try:
        data = request.get_json()
        
        user_profile = {
            "skills": data.get('skills', ''),
            "education": data.get('education', ''),
            "location": data.get('location', '')
        }
        
        df = predictor.recommend(user_profile)
        
        if df.empty:
            return jsonify({'success': True, 'data': [], 'message': 'No careers found'})
        
        results = df.to_dict('records')
        
        for r in results:
            r['schemes'] = scheme_matcher.get_schemes_for_job(r['job_title'])
            r['match_percent'] = r.get('match_percent', 70)
        
        return jsonify({'success': True, 'data': results, 'count': len(results)})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# COMBINED RECOMMENDATION (ML + AI)
# ============================================

@app.route('/api/hybrid/recommend', methods=['POST'])
def hybrid_recommend():
    """
    Combine ML model + Gemini AI for best recommendations
    """
    try:
        data = request.get_json()
        skills = data.get('skills', '').strip()
        education = data.get('education', '').strip()
        location = data.get('location', '').strip()
        
        if not skills:
            return jsonify({'success': False, 'error': 'Skills are required'}), 400
        
        # Step 1: Get ML recommendations
        ml_result = predictor.recommend({"skills": skills, "education": education, "location": location})
        ml_careers = ml_result.to_dict('records') if not ml_result.empty else []
        
        # Step 2: Get AI recommendations from Gemini
        ai_careers = []
        if GEMINI_API_KEY:
            try:
                prompt = f"""
                Suggest 3 career paths for someone with skills: {skills}, education: {education}.
                Return only JSON array with fields: career, reason, salary_range.
                """
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                
                import json
                response_text = response.text.strip()
                # Clean response
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                ai_careers = json.loads(response_text)
            except Exception as e:
                print(f"Gemini error in hybrid: {e}")
        
        return jsonify({
            'success': True,
            'ml_recommendations': ml_careers,
            'ai_recommendations': ai_careers,
            'count': len(ml_careers) + len(ai_careers)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# HOME ROUTE
# ============================================

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    error = None
    user_input = {}

    if request.method == "POST":
        skills = request.form.get("skills", "").strip()
        education = request.form.get("education", "").strip()
        location = request.form.get("location", "").strip()
        relocate = request.form.get("relocate", "no").strip()

        user_input = {
            "skills": skills,
            "education": education,
            "location": location,
            "relocate": relocate
        }

        if not skills or not education or not location:
            error = "कृपया सभी फ़ील्ड भरें | Please fill all fields"
        else:
            user_profile = {
                "skills": skills,
                "education": education,
                "location": location
            }

            try:
                df = predictor.recommend(user_profile)

                if df.empty:
                    error = "No suitable careers found | कोई उपयुक्त करियर नहीं मिला"
                else:
                    results = df.to_dict("records")

                    for r in results:
                        r["schemes"] = scheme_matcher.get_schemes_for_job(r["job_title"])
                        
                        if "salary" in r and isinstance(r["salary"], str):
                            r["salary_display"] = r["salary"]
                        else:
                            r["salary_display"] = "₹10,000 - ₹20,000"
                        
                        if "match_percent" not in r:
                            r["match_percent"] = round(r.get("combined_score", 70), 1)

            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template("index.html", results=results, error=error, user_input=user_input)


# ============================================
# HEALTH API
# ============================================

@app.route('/api/health')
def health():
    try:
        career_count = len(predictor.careers) if hasattr(predictor, 'careers') and predictor.careers is not None else 20
        return jsonify({
            'success': True, 
            'message': 'Backend running with Gemini AI',
            'stats': {
                'careers_available': career_count,
                'gemini_available': bool(GEMINI_API_KEY)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AI Career Counsellor with Google Gemini")
    print("="*50)
    print(f"✅ Gemini API: {'Configured' if GEMINI_API_KEY else '❌ Missing'}")
    print(f"📡 Server: http://127.0.0.1:5000")
    print("\n📋 Available APIs:")
    print("   POST /api/careers/recommend  - ML-based recommendations")
    print("   POST /api/gemini/suggest     - AI career suggestions")
    print("   POST /api/gemini/analyze     - Deep profile analysis")
    print("   POST /api/hybrid/recommend   - ML + AI combined")
    print("   POST /api/jobs/search        - Live job listings")
    print("="*50 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)