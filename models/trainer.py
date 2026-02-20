import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def train_model():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Input files
        students_path = os.path.join(BASE_DIR, 'data', 'students.csv')
        careers_path = os.path.join(BASE_DIR, 'data', 'careers.csv')
        
        # Output files
        model_path = os.path.join(BASE_DIR, 'models', 'tfidf_model.joblib')
        processed_careers_path = os.path.join(BASE_DIR, 'data', 'careers_processed.csv')
        
        # Verify input files exist
        if not os.path.exists(students_path):
            raise FileNotFoundError(f"Missing: {students_path}")
        if not os.path.exists(careers_path):
            raise FileNotFoundError(f"Missing: {careers_path}")
        
        # Load data
        students = pd.read_csv(students_path)
        careers = pd.read_csv(careers_path)
        
        # Process data - fill NaN values with empty string for new columns
        students["profile"] = students["interests"] + " " + students["skills"]
        careers = careers.fillna('')
        careers["processed_text"] = careers["required_skills"] + " " + careers["SDG_category"]
        
        # Train model
        tfidf = TfidfVectorizer()
        tfidf.fit(students["profile"])
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save outputs
        joblib.dump(tfidf, model_path)
        careers.to_csv(processed_careers_path, index=False)
        
        print("Success! Created:")
        print(f"- Model: {model_path}")
        print(f"- Processed careers: {processed_careers_path}")
        print(f"Processed {len(careers)} careers with new columns")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()