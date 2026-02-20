import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

class CareerPredictor:
    def __init__(self):
        try:
            # Get absolute paths
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, 'models', 'tfidf_model.joblib')
            careers_path = os.path.join(BASE_DIR, 'data', 'careers_processed.csv')
            
            # Verify files exist with correct sizes
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                raise FileNotFoundError(f"Model file missing or empty at {model_path}")
            if not os.path.exists(careers_path) or os.path.getsize(careers_path) == 0:
                raise FileNotFoundError(f"Careers data missing or empty at {careers_path}")
            
            # Load resources
            self.tfidf = joblib.load(model_path)
            self.careers = pd.read_csv(careers_path)
            
            # Define education hierarchy for filtering
            self.education_levels = {
                '8th_pass': 1,
                '9th_pass': 2,
                '10th_pass':3,
                '12th_pass':5,
                'graduate': 6,
            }
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise

    def recommend(self, user_profile):
        try:
            skills = user_profile['skills']
            user_education = user_profile['education']
            
            # Filter careers by education level first
            if user_education in self.education_levels:
                user_edu_level = self.education_levels[user_education]
                # Filter careers that require equal or lower education level
                filtered_careers = self.careers.copy()
                filtered_careers['min_edu_level'] = filtered_careers['min_education'].map(
                    lambda x: self.education_levels.get(x, 0)
                )
                filtered_careers = filtered_careers[filtered_careers['min_edu_level'] <= user_edu_level]
            else:
                filtered_careers = self.careers
            
            if filtered_careers.empty:
                return pd.DataFrame()
            
            # Calculate similarity on filtered careers
            input_vec = self.tfidf.transform([skills])
            career_text = filtered_careers["required_skills"] + " " + filtered_careers["SDG_category"]
            career_vec = self.tfidf.transform(career_text)
            
            sim_scores = cosine_similarity(input_vec, career_vec)
            top_indices = sim_scores.argsort()[0][-3:][::-1]
            
            return filtered_careers.iloc[top_indices]
            
        except Exception as e:
            print(f"Recommendation failed: {str(e)}")
            return pd.DataFrame()