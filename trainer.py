import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load career data
careers_path = os.path.join(DATA_DIR, "careers.csv")
if os.path.exists(careers_path):
    careers = pd.read_csv(careers_path)
    careers = careers.fillna("")
    print(f"Loaded {len(careers)} careers")
else:
    print(f"Error: careers.csv not found at {careers_path}")
    exit(1)

# Train TF-IDF on career data
career_text = careers["required_skills"].astype(str) + " " + careers["SDG_category"].astype(str)

tfidf = TfidfVectorizer(stop_words="english", max_features=500)
tfidf.fit(career_text)

# Save model
model_path = os.path.join(MODEL_DIR, "tfidf_model.joblib")
joblib.dump(tfidf, model_path)

# Save processed careers
processed_path = os.path.join(DATA_DIR, "careers_processed.csv")
careers.to_csv(processed_path, index=False)

print(f"TF-IDF model trained successfully and saved to {model_path}")
print(f"Processed careers saved to {processed_path}")