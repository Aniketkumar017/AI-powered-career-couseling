import pandas as pd
import os

class SchemeMatcher:
    def __init__(self):
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            schemes_path = os.path.join(BASE_DIR, 'data', 'schemes.csv')
            
            if os.path.exists(schemes_path):
                self.schemes = pd.read_csv(schemes_path)
            else:
                self.schemes = pd.DataFrame()
                print("Schemes file not found")
                
        except Exception as e:
            print(f"Scheme matcher initialization failed: {str(e)}")
            self.schemes = pd.DataFrame()

    def get_schemes_for_job(self, job_title):
        if self.schemes.empty:
            return []
        
        relevant_schemes = []
        for _, scheme in self.schemes.iterrows():
            linked_jobs = str(scheme['linked_jobs']).split(',')
            if job_title in linked_jobs or 'All' in linked_jobs:
                relevant_schemes.append(scheme.to_dict())
        
        return relevant_schemes