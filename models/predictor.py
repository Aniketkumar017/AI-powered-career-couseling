import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

class CareerPredictor:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.tfidf = joblib.load(
            os.path.join(BASE_DIR, "models", "tfidf_model.joblib")
        )

        self.careers = pd.read_csv(
            os.path.join(BASE_DIR, "data", "careers_processed.csv")
        )

        self.education_levels = {
            "8th_pass": 1,
            "10th_pass": 2,
            "12th_pass": 3,
            "graduate": 4
        }

    def recommend(self, user_profile):
        skills = user_profile["skills"]
        education = user_profile["education"]

        user_level = self.education_levels.get(education, 0)

        self.careers["edu_level"] = self.careers["min_education"].map(
            lambda x: self.education_levels.get(x, 0)
        )

        filtered = self.careers[
            self.careers["edu_level"] <= user_level
        ]

        if filtered.empty:
            return pd.DataFrame()

        input_vec = self.tfidf.transform([skills])
        career_vec = self.tfidf.transform(
            filtered["required_skills"] + " " + filtered["SDG_category"]
        )

        similarity = cosine_similarity(input_vec, career_vec)[0]

        filtered = filtered.copy()
        filtered["similarity"] = similarity

        # FINAL HARD STOP (THIS FIXES EVERYTHING)
        filtered = filtered[filtered["similarity"] >= 0.30]

        if filtered.empty:
            return filtered

        filtered = filtered.sort_values(
            by="similarity", ascending=False
        )

        return filtered.head(3)
