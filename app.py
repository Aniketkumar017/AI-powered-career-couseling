from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import os
import sys
import json
import requests
from dotenv import load_dotenv
# Groq is OpenAI-compatible, no special SDK needed
from functools import wraps

# Load environment variables
load_dotenv()

# ============================================
# GROQ API CONFIGURATION
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    print("[OK] Groq API configured successfully")
else:
    print("[ERR] GROQ_API_KEY not found in .env file")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models and utils
from models.predictor import CareerPredictor
from models.scheme_matcher import SchemeMatcher
from utils.auth import (
    register_user, login_user, get_user, update_user,
    save_career, get_saved_careers, remove_saved_career,
    save_career_recommendation, get_career_history,
    save_feedback
)
from utils.db import get_db_connection

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Register custom template filter for JSON parsing
@app.template_filter('from_json')
def from_json_filter(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}

# Initialize models
predictor = CareerPredictor()
scheme_matcher = SchemeMatcher()

# ============================================
# DECORATORS
# ============================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# DIRECT GEMINI API - GEMINI 2.0 FLASH
# ============================================

def get_gemini_career_suggestions(skills, education="", location=""):
    """Groq API - llama-3.3-70b-versatile (fast & free)"""
    if not GROQ_API_KEY:
        print("[ERR] Groq API key not configured")
        return None
    
    try:
        # Groq uses OpenAI-compatible API
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        You are an AI career counselor for Indian youth.
        
        User skills: {skills}
        Education: {education}
        Location: {location}
        
        Suggest 5 career paths that match these skills.
        Return ONLY a valid JSON array with fields: 
        career, reason, salary_range, training_months, growth.
        
        Example:
        [
            {{
                "career": "Actor/Actress",
                "reason": "Your acting skills are perfect for films and theater",
                "salary_range": "₹30,000 - ₹1,00,000",
                "training_months": 12,
                "growth": "High"
            }}
        ]
        """
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a helpful AI career counselor for Indian youth. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        print(f"[AI] Calling Groq API for skills: {skills}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code != 200:
            print(f"[ERR] Groq API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        print(f"[LOG] Groq Response: {text[:200]}...")
        
        # Clean JSON
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        suggestions = json.loads(text)
        
        formatted = []
        for item in suggestions:
            career_name = item.get('career', 'AI Suggestion')
            formatted.append({
                'job_title': career_name,
                'career': career_name,
                'reason': item.get('reason', f'Based on your skills: {skills[:50]}...'),
                'salary': item.get('salary_range', '₹15,000 - ₹30,000'),
                'training_duration': item.get('training_months', 6),
                'growth': item.get('growth', 'Medium'),
                'source': 'Groq AI',
                'match_percent': 90,
                'is_gemini': True,
                'required_skills': skills,
                'pathway_steps': 'Learn required skills|Get training|Apply for jobs|Start career',
                'schemes': scheme_matcher.get_schemes_for_job(career_name) if career_name else [],
                'colleges': predictor.get_colleges_for_career(career_name) if career_name else []
            })
        
        print(f"[OK] Groq generated {len(formatted)} suggestions")
        return formatted
        
    except json.JSONDecodeError as e:
        print(f"[ERR] Groq JSON Parse Error: {e}")
        return None
        
    except Exception as e:
        print(f"[ERR] Groq API Error: {e}")
        return None

# ============================================
# DATABASE SUGGESTIONS
# ============================================

def get_database_suggestions(skills, education="", location=""):
    """Database se career suggestions"""
    try:
        user_profile = {
            "skills": skills,
            "education": education,
            "location": location
        }
        
        df = predictor.recommend(user_profile)
        
        if df.empty:
            return []
        
        results = df.to_dict('records')
        
        for r in results:
            r['schemes'] = scheme_matcher.get_schemes_for_job(r['job_title'])
            r['match_percent'] = r.get('match_percent', 70)
            r['colleges'] = predictor.get_colleges_for_career(r['job_title'])
            r['source'] = 'ML Model'
            r['is_gemini'] = False
        
        return results
        
    except Exception as e:
        print(f"Database error: {e}")
        return []

# ============================================
# PUBLIC ROUTES
# ============================================

@app.route("/", methods=["GET", "POST"])
def landing():
    """Landing page"""
    if request.method == "POST":
        skills = request.form.get("skills", "").strip()
        education = request.form.get("education", "").strip()
        location = request.form.get("location", "").strip()
        
        session['search_skills'] = skills
        session['search_education'] = education
        session['search_location'] = location
        
        return redirect(url_for('dashboard'))
    
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        user, message = login_user(username, password)
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user.get('full_name', user['username'])
            return redirect(url_for('dashboard'))
        
        return render_template("login.html", error=message)
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Registration page"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        full_name = request.form.get("full_name", "").strip()
        
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match")
        
        success, result = register_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            age=request.form.get("age"),
            gender=request.form.get("gender"),
            location=request.form.get("location"),
            education=request.form.get("education")
        )
        
        if success:
            return redirect(url_for('login'))
        
        return render_template("register.html", error=result)
    
    return render_template("register.html")

@app.route("/logout")
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('landing'))

# ============================================
# PROTECTED ROUTES
# ============================================

@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard"""
    user = get_user(session['user_id'])
    saved = get_saved_careers(session['user_id'])
    history = get_career_history(session['user_id'])
    
    search_skills = session.get('search_skills', '')
    search_education = session.get('search_education', '')
    search_location = session.get('search_location', '')
    
    return render_template("dashboard.html", 
                         user=user, 
                         saved_count=len(saved),
                         history_count=len(history),
                         search_skills=search_skills,
                         search_education=search_education,
                         search_location=search_location)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """User profile page"""
    user = get_user(session['user_id'])
    
    if request.method == "POST":
        updates = {
            'full_name': request.form.get("full_name"),
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
            'location': request.form.get("location"),
            'education': request.form.get("education"),
            'skills': request.form.get("skills"),
            'interests': request.form.get("interests")
        }
        success, message = update_user(session['user_id'], **updates)
        if success:
            return render_template("profile.html", user=user, success=message)
        return render_template("profile.html", user=user, error=message)
    
    return render_template("profile.html", user=user)

@app.route("/saved")
@login_required
def saved_careers():
    """Saved careers page"""
    careers = get_saved_careers(session['user_id'])
    return render_template("saved_careers.html", careers=careers)

@app.route("/career/<path:career_title>")
@login_required
def career_details(career_title):
    """Career details page"""
    import json
    
    df = predictor.careers
    career = df[df['job_title'] == career_title]
    
    if career.empty:
        flash(f'Career "{career_title}" not found.', 'warning')
        return redirect(url_for('dashboard'))
    
    career_data = career.iloc[0].to_dict()
    schemes = scheme_matcher.get_schemes_for_job(career_title)
    colleges = predictor.get_colleges_for_career(career_title)
    career_json = json.dumps(career_data, ensure_ascii=False)
    
    return render_template("career_details.html", 
                         career=career_data, 
                         schemes=schemes,
                         colleges=colleges,
                         career_json=career_json)

@app.route("/jobs")
@login_required
def jobs_page():
    """Jobs page"""
    return render_template("jobs.html")

# ============================================
# API ROUTES - SMART RECOMMEND (DB + GEMINI)
# ============================================

@app.route('/api/careers/smart_recommend', methods=['POST'])
@login_required
def smart_recommend():
    """Smart recommendation - DB first, then Gemini, else No careers found"""
    try:
        data = request.get_json()
        skills = data.get('skills', '').strip()
        education = data.get('education', '').strip()
        location = data.get('location', '').strip()
        
        if not skills:
            return jsonify({'success': False, 'error': 'Skills are required'}), 400
        
        print(f"\n{'='*50}")
        print(f"[SEARCH] Smart Recommendation Request")
        print(f"[LOG] Skills: {skills}")
        print(f"[EDU] Education: {education}")
        print(f"[MAP] Location: {location}")
        print(f"{'='*50}")
        
        final_results = []
        fallback_used = False
        gemini_used = False
        
        # Step 1: Database se match karein
        db_results = get_database_suggestions(skills, education, location)
        
        if db_results:
            print(f"[OK] Found {len(db_results)} careers in database")
            final_results.extend(db_results)
        else:
            print("[ERR] No match found in database")
        
        # Step 2: Agar database mein match nahi mila toh Gemini se lo
        if not db_results:
            fallback_used = True
            print("[AI] Calling Gemini API...")
            
            gemini_results = get_gemini_career_suggestions(skills, education, location)
            
            if gemini_results:
                gemini_used = True
                print(f"[OK] Gemini generated {len(gemini_results)} suggestions")
                final_results.extend(gemini_results)
            else:
                print("[ERR] Gemini also failed")
                # ⚠️ YAHAN PAR "NO CAREERS FOUND" AAYEGA
                # Koi fake suggestion nahi daal rahe
        
        # Save to history (only if results exist)
        if final_results:
            try:
                save_career_recommendation(
                    session['user_id'],
                    skills,
                    education,
                    location,
                    final_results
                )
            except Exception as e:
                print(f"Save history error: {e}")
        
        # [OK] YAHAN SE RESPONSE BHEJ RAHE HAIN
        if final_results:
            response_data = {
                'success': True,
                'data': final_results[:10],
                'count': len(final_results),
                'fallback_used': fallback_used,
                'gemini_used': gemini_used,
                'source': 'Gemini AI' if gemini_used else 'ML Model',
                'message': f'Found {len(final_results)} career recommendations'
            }
            return jsonify(response_data)
        else:
            # [OK] SIRF TAB "NO CAREERS FOUND" AAYEGA JAB DONO FAIL HO
            return jsonify({
                'success': True,
                'data': [],
                'count': 0,
                'fallback_used': True,
                'gemini_used': False,
                'message': 'No careers found. Please try different skills.',
                'suggestion': 'Try adding more specific skills like: programming, teaching, healthcare, farming, etc.'
            })
        
    except Exception as e:
        print(f"[ERR] Smart recommend error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        }), 500

# ============================================
# JOBS API
# ============================================

@app.route('/api/jobs/search', methods=['POST'])
@login_required
def search_jobs():
    try:
        data = request.get_json()
        job_title = data.get('job_title', '').strip()
        location = data.get('location', 'India').strip()
        
        if not job_title:
            return jsonify({'success': False, 'error': 'Job title is required'}), 400
        
        all_jobs = []
        
        try:
            url = f"https://remotive.io/api/remote-jobs?search={job_title}&limit=10"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                jobs_data = response.json()
                for job in jobs_data.get('jobs', [])[:10]:
                    all_jobs.append({
                        'title': job.get('title', 'N/A'),
                        'company': job.get('company_name', 'N/A'),
                        'location': job.get('candidate_required_location', 'Remote'),
                        'salary': job.get('salary', 'Not specified'),
                        'url': job.get('url', '#'),
                        'description': job.get('description', '')[:300],
                        'source': 'Remotive',
                        'posted': job.get('publication_date', '')
                    })
        except Exception as e:
            print(f"Remotive API error: {e}")
        
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = f"{job['title']}_{job['company']}"
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return jsonify({
            'success': True,
            'data': unique_jobs[:20],
            'count': len(unique_jobs),
            'sources': list(set(j['source'] for j in unique_jobs))
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# SAVE CAREER API
# ============================================

@app.route('/api/careers/save', methods=['POST'])
@login_required
def api_save_career():
    try:
        data = request.get_json()
        career_title = data.get('career_title')
        career_data = data.get('career_data', {})
        
        if not career_title:
            return jsonify({'success': False, 'error': 'Career title required'}), 400
        
        success, message = save_career(session['user_id'], career_title, career_data)
        
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# HEALTH CHECK
# ============================================

@app.route('/api/health')
def health():
    try:
        career_count = len(predictor.careers) if hasattr(predictor, 'careers') and predictor.careers is not None else 20
        return jsonify({
            'success': True, 
            'message': 'Backend running with Groq AI',
            'stats': {
                'careers_available': career_count,
                'groq_available': bool(GROQ_API_KEY)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("[START] AI Career Counsellor with Groq AI")
    print("="*50)
    print(f"[OK] Groq API: {'Configured' if GROQ_API_KEY else '[ERR] Missing'}")
    print(f"[SERVER] Server: http://127.0.0.1:5000")
    print("\n[LIST] Available APIs:")
    print("   POST /api/careers/smart_recommend - DB + Groq AI recommendations")
    print("   POST /api/jobs/search            - Live job listings")
    print("="*50 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)