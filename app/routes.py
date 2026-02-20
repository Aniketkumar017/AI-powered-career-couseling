from flask import Blueprint, render_template, request
from models.predictor import CareerPredictor
from models.scheme_matcher import SchemeMatcher

bp = Blueprint('main', __name__)
predictor = CareerPredictor()
scheme_matcher = SchemeMatcher()  # Added SchemeMatcher initialization

@bp.route('/', methods=['GET', 'POST'])
def home():
    error = None
    results = None
    
    if request.method == 'POST':
        try:
            # Get all form data
            skills = request.form.get('skills', '').strip()
            education = request.form.get('education', '').strip()
            location = request.form.get('location', '').strip()
            relocate = request.form.get('relocate', 'no').strip()
            
            if not skills or not education or not location:
                error = "Please fill in all required fields"
            else:
                # Create user profile with all data
                user_profile = {
                    'skills': skills,
                    'education': education,
                    'location': location,
                    'relocate': relocate
                }
                
                results = predictor.recommend(user_profile)
                if results.empty:
                    error = "No suitable careers found for your profile. Try different skills or education level."
                else:
                    results = results.to_dict('records')
                    # Add schemes to each career
                    for career in results:
                        career['schemes'] = scheme_matcher.get_schemes_for_job(career['job_title'])
                    
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('index.html', results=results, error=error)