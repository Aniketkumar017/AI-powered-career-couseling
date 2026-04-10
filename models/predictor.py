import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

class CareerPredictor:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load careers data from CSV
        data_path = os.path.join(BASE_DIR, "data", "careers.csv")
        
        if os.path.exists(data_path):
            self.careers = pd.read_csv(data_path)
            self.careers = self.careers.fillna("")
            print(f" Loaded {len(self.careers)} careers from CSV")
            print(f"Career titles: {list(self.careers['job_title'].head(10))}")
        else:
            print(f"Careers CSV not found at: {data_path}")
            print("Creating sample careers...")
            self.careers = self.create_sample_careers()

        self.education_levels = {
            "8th_pass": 1,
            "10th_pass": 2,
            "12th_pass": 3,
            "graduate": 4
        }

    def create_sample_careers(self):
        """Create sample careers if CSV doesn't exist"""
        return pd.DataFrame([
            {
                'job_title': 'Software Developer',
                'required_skills': 'programming, coding, python, java',
                'salary': '₹30,000 - ₹50,000',
                'SDG_category': 'Industry Innovation',
                'min_education': '12th_pass',
                'training_duration': 6,
                'pathway_steps': 'Learn programming|Build projects|Apply for jobs|Start career',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Web Designer',
                'required_skills': 'html, css, javascript, design',
                'salary': '₹20,000 - ₹35,000',
                'SDG_category': 'Industry Innovation',
                'min_education': '12th_pass',
                'training_duration': 4,
                'pathway_steps': 'Learn HTML/CSS|Learn JavaScript|Build portfolio|Get job',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Data Entry Operator',
                'required_skills': 'typing, computer, ms office',
                'salary': '₹12,000 - ₹18,000',
                'SDG_category': 'Decent Work',
                'min_education': '10th_pass',
                'training_duration': 2,
                'pathway_steps': 'Learn typing|Learn computer basics|Practice|Apply for jobs',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Digital Marketing',
                'required_skills': 'social media, seo, content writing',
                'salary': '₹18,000 - ₹30,000',
                'SDG_category': 'Decent Work',
                'min_education': '12th_pass',
                'training_duration': 3,
                'pathway_steps': 'Learn digital marketing|Get certified|Build portfolio|Apply',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Computer Teacher',
                'required_skills': 'computer, teaching, communication',
                'salary': '₹15,000 - ₹25,000',
                'SDG_category': 'Quality Education',
                'min_education': '12th_pass',
                'training_duration': 3,
                'pathway_steps': 'Learn computer basics|Get teaching cert|Practice|Get job',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Agricultural Technician',
                'required_skills': 'farming, agriculture, crops',
                'salary': '₹15,000 - ₹25,000',
                'SDG_category': 'Zero Hunger',
                'min_education': '10th_pass',
                'training_duration': 3,
                'pathway_steps': 'Learn farming|Get training|Get certified|Start working',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Solar Panel Installer',
                'required_skills': 'solar, electrical, technical',
                'salary': '₹18,000 - ₹30,000',
                'SDG_category': 'Clean Energy',
                'min_education': '10th_pass',
                'training_duration': 4,
                'pathway_steps': 'Learn solar installation|Get certified|Apprenticeship|Start work',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Healthcare Assistant',
                'required_skills': 'healthcare, caregiving, first aid',
                'salary': '₹12,000 - ₹20,000',
                'SDG_category': 'Good Health',
                'min_education': '12th_pass',
                'training_duration': 6,
                'pathway_steps': 'Learn healthcare basics|Get certified|Internship|Apply',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Electrician',
                'required_skills': 'electrical, wiring, repair',
                'salary': '₹15,000 - ₹25,000',
                'SDG_category': 'Clean Energy',
                'min_education': '10th_pass',
                'training_duration': 6,
                'pathway_steps': 'Complete ITI|Apprenticeship|Get license|Start work',
                'training_providers_link': '#'
            },
            {
                'job_title': 'Plumber',
                'required_skills': 'plumbing, pipes, repair',
                'salary': '₹12,000 - ₹22,000',
                'SDG_category': 'Clean Water',
                'min_education': '8th_pass',
                'training_duration': 4,
                'pathway_steps': 'Learn plumbing|Apprenticeship|Get certified|Start work',
                'training_providers_link': '#'
            }
        ])

    def calculate_skill_match(self, user_skills, required_skills):
        """Calculate skill match percentage"""
        if not user_skills or not required_skills:
            return 0
        
        user_skills_list = [s.strip().lower() for s in user_skills.split(',')]
        required_skills_list = [s.strip().lower() for s in required_skills.split(',')]
        
        if not required_skills_list:
            return 0
        
        match_count = 0
        for req_skill in required_skills_list:
            for user_skill in user_skills_list:
                if req_skill in user_skill or user_skill in req_skill:
                    match_count += 1
                    break
        
        return (match_count / len(required_skills_list)) * 100

    def recommend(self, user_profile):
        """Recommend careers based on user profile"""
        skills = user_profile.get("skills", "")
        education = user_profile.get("education", "")

        if not skills or not education:
            return pd.DataFrame()

        print(f"\n Analyzing: Skills='{skills}', Education='{education}'")

        # Get user education level
        user_level = self.education_levels.get(education, 2)
        
        # Filter by education level
        self.careers["edu_level"] = self.careers["min_education"].map(
            lambda x: self.education_levels.get(x, 2)
        )
        
        filtered = self.careers[self.careers["edu_level"] <= user_level].copy()
        
        if filtered.empty:
            print(" No careers match education level")
            return pd.DataFrame()
        
        # Calculate match percentage for each career
        filtered["match_percent"] = filtered["required_skills"].apply(
            lambda x: self.calculate_skill_match(skills, str(x))
        )
        
        # Filter by minimum 20% match
        filtered = filtered[filtered["match_percent"] >= 20]
        
        if filtered.empty:
            print(" No careers match skills (minimum 20% required)")
            return pd.DataFrame()
        
        # Sort by match percentage
        filtered = filtered.sort_values("match_percent", ascending=False)
        
        # Return top 5 matches
        results = filtered.head(5)
        
        print(f" Found {len(results)} matching careers:")
        for _, row in results.iterrows():
            print(f"   - {row['job_title']}: {row['match_percent']:.1f}% match")
        
        return results