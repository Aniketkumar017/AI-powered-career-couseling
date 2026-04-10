import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# BASE DIR = project root (career-counselling-main)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load career data
careers = pd.read_csv(os.path.join(DATA_DIR, "careers.csv"))
careers = careers.fillna("")

# Train TF-IDF on CAREER DATA ONLY
career_text = careers["required_skills"] + " " + careers["SDG_category"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit(career_text)

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(
    tfidf,
    os.path.join(MODEL_DIR, "tfidf_model.joblib")
)

# Save processed careers
careers.to_csv(
    os.path.join(DATA_DIR, "careers_processed.csv"),
    index=False
)

print("TF-IDF model trained successfully (PATH FIXED)")
