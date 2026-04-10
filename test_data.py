"""
Test script to verify data loading for Career Counsellor
Run this file to check if all data is loading correctly
"""

import os
import pandas as pd

print("="*60)
print(" CAREER COUNSELLOR - DATA VERIFICATION")
print("="*60)

# Get current directory
current_dir = os.getcwd()
print(f"\n Current Directory: {current_dir}")

# Check 1: Check if data/careers.csv exists
print("\n" + "-"*40)
print(" CHECKING CAREERS DATA")
print("-"*40)

careers_path = os.path.join(current_dir, "data", "careers.csv")
if os.path.exists(careers_path):
    print(f" careers.csv found!")
    df = pd.read_csv(careers_path)
    print(f" Total careers: {len(df)}")
    print(f"\n Available Careers:")
    for i, row in df.iterrows():
        print(f"   {i+1}. {row['job_title']} - Skills: {row['required_skills'][:50]}...")
else:
    print(f" careers.csv NOT found at: {careers_path}")

# Check 2: Check if templates/index.html exists
print("\n" + "-"*40)
print(" CHECKING TEMPLATE FILE")
print("-"*40)

template_path = os.path.join(current_dir, "templates", "index.html")
if os.path.exists(template_path):
    print(f" index.html found!")
else:
    print(f" index.html NOT found at: {template_path}")

# Check 3: Test skill matching with different inputs
print("\n" + "-"*40)
print(" TESTING SKILL MATCHING")
print("-"*40)

# Simple skill matching function
def test_skill_match(user_skills, careers_df):
    user_skills_list = [s.strip().lower() for s in user_skills.split(',')]
    results = []
    
    for idx, row in careers_df.iterrows():
        required_skills = str(row['required_skills']).lower()
        match_count = 0
        for user_skill in user_skills_list:
            if user_skill in required_skills:
                match_count += 1
        
        if len(user_skills_list) > 0:
            match_percent = (match_count / len(user_skills_list)) * 100
        else:
            match_percent = 0
        
        if match_percent > 0:
            results.append({
                'job_title': row['job_title'],
                'match_percent': match_percent
            })
    
    results.sort(key=lambda x: x['match_percent'], reverse=True)
    return results[:5]

if os.path.exists(careers_path):
    df = pd.read_csv(careers_path)
    
    test_inputs = [
        "programming, coding, python",
        "farming, agriculture",
        "computer, teaching",
        "healthcare, nursing",
        "electrical, wiring"
    ]
    
    for skills in test_inputs:
        print(f"\n Input: '{skills}'")
        matches = test_skill_match(skills, df)
        if matches:
            print(f"    Matches found:")
            for m in matches:
                print(f"      - {m['job_title']} ({m['match_percent']:.0f}% match)")
        else:
            print(f"    No matches found")
else:
    print("❌ Cannot test - careers.csv not found")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)