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
            print(f"✅ Loaded {len(self.careers)} careers from CSV")
        else:
            print(f"❌ Careers CSV not found at: {data_path}")
            print("Creating sample careers...")
            self.careers = self.create_sample_careers()

        self.education_levels = {
            "8th_pass": 1,
            "10th_pass": 2,
            "12th_pass": 3,
            "graduate": 4
        }
        
        # Complete College Database
        self.college_mapping = self._get_college_database()

    def _get_college_database(self):
        """Complete college database for all careers (India + Abroad)"""
        return {
            # ============ TECHNOLOGY & IT ============
            'Software Developer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                'IIT Roorkee', 'IIT Guwahati', 'IIT Hyderabad', 'IIT Gandhinagar',
                'BITS Pilani', 'NIT Trichy', 'NIT Surathkal', 'NIT Warangal',
                'IIIT Hyderabad', 'IIIT Bangalore', 'IIIT Delhi',
                'DTU Delhi', 'NSUT Delhi', 'VIT Vellore', 'SRM University',
                'PES University', 'RV College of Engineering', 'BMSCE',
                'MIT Manipal', 'Thapar University', 'LNMIIT Jaipur',
                'MIT (USA)', 'Stanford University (USA)', 'Carnegie Mellon University (USA)',
                'University of California-Berkeley (USA)', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'University of Oxford (UK)',
                'ETH Zurich (Switzerland)', 'National University of Singapore',
                'University of Toronto (Canada)', 'University of Waterloo (Canada)',
                'TU Munich (Germany)', 'University of Melbourne (Australia)'
            ],
            
            'Web Developer': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Warangal',
                'BITS Pilani', 'DTU Delhi', 'NSUT Delhi', 'VIT Vellore',
                'MIT Manipal', 'RV College of Engineering', 'BMSCE',
                'Symbiosis Institute of Design', 'National Institute of Design',
                'Pearl Academy', 'MIT Institute of Design',
                'Rhode Island School of Design (USA)', 'Parsons School of Design (USA)',
                'California Institute of the Arts (USA)', 'MIT Media Lab (USA)',
                'Stanford University (USA)', 'University of Washington (USA)',
                'University of Technology Sydney (Australia)', 'RMIT University (Australia)'
            ],
            
            'Full Stack Developer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                'BITS Pilani', 'NIT Trichy', 'NIT Surathkal', 'NIT Warangal',
                'IIIT Hyderabad', 'IIIT Bangalore', 'DTU Delhi', 'NSUT Delhi',
                'VIT Vellore', 'SRM University', 'MIT Manipal',
                'MIT (USA)', 'Stanford University (USA)', 'Carnegie Mellon University (USA)',
                'University of California-Berkeley (USA)', 'University of Cambridge (UK)',
                'University of Oxford (UK)', 'National University of Singapore'
            ],
            
            'Data Scientist': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                'IIT Roorkee', 'BITS Pilani', 'NIT Trichy', 'NIT Surathkal',
                'IIIT Hyderabad', 'IIIT Bangalore', 'DTU Delhi',
                'MIT (USA)', 'Stanford University (USA)', 'Carnegie Mellon University (USA)',
                'University of California-Berkeley (USA)', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'University of Oxford (UK)',
                'ETH Zurich (Switzerland)', 'National University of Singapore'
            ],
            
            'Data Analyst': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'BITS Pilani',
                'NIT Trichy', 'IIIT Hyderabad', 'DTU Delhi', 'Delhi University',
                'University of Mumbai', 'Panjab University', 'MIT (USA)',
                'Stanford University (USA)', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Machine Learning Engineer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                'BITS Pilani', 'NIT Trichy', 'IIIT Hyderabad', 'IIIT Bangalore',
                'MIT (USA)', 'Stanford University (USA)', 'Carnegie Mellon University (USA)',
                'University of California-Berkeley (USA)', 'University of Cambridge (UK)',
                'ETH Zurich (Switzerland)', 'National University of Singapore'
            ],
            
            'AI Engineer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                'BITS Pilani', 'NIT Trichy', 'IIIT Hyderabad', 'IIIT Bangalore',
                'MIT (USA)', 'Stanford University (USA)', 'Carnegie Mellon University (USA)',
                'University of California-Berkeley (USA)', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'University of Oxford (UK)',
                'ETH Zurich (Switzerland)', 'National University of Singapore'
            ],
            
            'Cloud Engineer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'BITS Pilani',
                'NIT Trichy', 'NIT Surathkal', 'DTU Delhi', 'NSUT Delhi',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'DevOps Engineer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'BITS Pilani',
                'NIT Trichy', 'NIT Surathkal', 'DTU Delhi', 'NSUT Delhi',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Cybersecurity Analyst': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'BITS Pilani',
                'NIT Trichy', 'NIT Surathkal', 'IIIT Hyderabad', 'IIIT Bangalore',
                'MIT (USA)', 'Stanford University (USA)', 'Carnegie Mellon University (USA)',
                'University of California-Berkeley (USA)', 'National University of Singapore'
            ],
            
            'Network Engineer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'NIT Trichy',
                'NIT Surathkal', 'DTU Delhi', 'NSUT Delhi', 'VIT Vellore',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Database Administrator': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'NIT Trichy',
                'NIT Surathkal', 'DTU Delhi', 'NSUT Delhi', 'VIT Vellore',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'System Administrator': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'NIT Trichy',
                'NIT Surathkal', 'DTU Delhi', 'NSUT Delhi', 'VIT Vellore',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'IT Support Specialist': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Surathkal', 'DTU Delhi',
                'NSUT Delhi', 'VIT Vellore', 'Amity University', 'Chandigarh University',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)'
            ],
            
            'Frontend Developer': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Warangal', 'BITS Pilani',
                'DTU Delhi', 'NSUT Delhi', 'VIT Vellore', 'MIT Manipal',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Backend Developer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'BITS Pilani',
                'NIT Trichy', 'NIT Surathkal', 'DTU Delhi', 'NSUT Delhi',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Mobile Developer - Android': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Warangal', 'BITS Pilani',
                'DTU Delhi', 'NSUT Delhi', 'VIT Vellore', 'MIT Manipal',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'Carnegie Mellon University (USA)', 'National University of Singapore'
            ],
            
            'Mobile Developer - iOS': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Warangal', 'BITS Pilani',
                'DTU Delhi', 'NSUT Delhi', 'VIT Vellore', 'MIT Manipal',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'Carnegie Mellon University (USA)', 'National University of Singapore'
            ],
            
            'Game Developer': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'BITS Pilani',
                'NIT Trichy', 'NIT Surathkal', 'DTU Delhi', 'VIT Vellore',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'UI/UX Designer': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Warangal', 'BITS Pilani',
                'National Institute of Design', 'Symbiosis Institute of Design',
                'Pearl Academy', 'MIT Institute of Design',
                'Rhode Island School of Design (USA)', 'Parsons School of Design (USA)',
                'California Institute of the Arts (USA)', 'MIT Media Lab (USA)',
                'Stanford University (USA)', 'University of Washington (USA)'
            ],
            
            'QA Tester': [
                'IIT Bombay', 'IIT Delhi', 'NIT Trichy', 'NIT Surathkal', 'BITS Pilani',
                'DTU Delhi', 'NSUT Delhi', 'VIT Vellore', 'Amity University',
                'MIT (USA)', 'Stanford University (USA)', 'University of California-Berkeley (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Technical Writer': [
                'IIT Bombay', 'IIT Delhi', 'Delhi University', 'University of Mumbai',
                'Panjab University', 'JNU Delhi', 'BHU Varanasi', 'AMU Aligarh',
                'MIT (USA)', 'Stanford University (USA)', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'University of Oxford (UK)'
            ],
            
            # ============ BUSINESS & MANAGEMENT ============
            'Digital Marketing Specialist': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'IIM Indore', 'IIM Kozhikode', 'XLRI Jamshedpur', 'SPJIMR Mumbai',
                'NMIMS Mumbai', 'MICA Ahmedabad', 'Symbiosis Pune', 'Xavier\'s Kolkata',
                'Delhi University', 'Mumbai University', 'Harvard Business School (USA)',
                'Stanford Graduate School of Business (USA)', 'Wharton School (USA)',
                'London Business School (UK)', 'INSEAD (France)',
                'Singapore Management University', 'University of Melbourne (Australia)'
            ],
            
            'SEO Specialist': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'Symbiosis Pune', 'MICA Ahmedabad', 'NMIMS Mumbai',
                'Harvard Business School (USA)', 'Stanford Graduate School of Business (USA)',
                'London Business School (UK)', 'Singapore Management University'
            ],
            
            'Social Media Manager': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'Symbiosis Pune', 'MICA Ahmedabad', 'NMIMS Mumbai',
                'Harvard Business School (USA)', 'Stanford Graduate School of Business (USA)',
                'London Business School (UK)', 'Singapore Management University'
            ],
            
            'Content Marketing Manager': [
                'Delhi University', 'JNU Delhi', 'Mumbai University', 'Panjab University',
                'MICA Ahmedabad', 'Symbiosis Pune', 'Harvard Business School (USA)',
                'Stanford Graduate School of Business (USA)', 'London Business School (UK)',
                'Singapore Management University'
            ],
            
            'Sales Executive': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'Symbiosis Pune',
                'Delhi University', 'Mumbai University', 'Panjab University',
                'Harvard Business School (USA)', 'Stanford Graduate School of Business (USA)',
                'Wharton School (USA)', 'London Business School (UK)', 'INSEAD (France)'
            ],
            
            'Business Development Manager': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'Symbiosis Pune',
                'Delhi University', 'Mumbai University', 'Harvard Business School (USA)',
                'Stanford Graduate School of Business (USA)', 'Wharton School (USA)',
                'London Business School (UK)', 'INSEAD (France)', 'Singapore Management University'
            ],
            
            'Marketing Manager': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'MICA Ahmedabad',
                'Symbiosis Pune', 'Delhi University', 'Harvard Business School (USA)',
                'Stanford Graduate School of Business (USA)', 'Wharton School (USA)',
                'London Business School (UK)', 'INSEAD (France)'
            ],
            
            'Product Manager': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'IIT Bombay',
                'IIT Delhi', 'MIT (USA)', 'Stanford University (USA)',
                'Harvard Business School (USA)', 'London Business School (UK)',
                'Singapore Management University'
            ],
            
            'Project Manager': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'IIT Bombay',
                'IIT Delhi', 'MIT (USA)', 'Stanford University (USA)',
                'Harvard Business School (USA)', 'London Business School (UK)',
                'Singapore Management University'
            ],
            
            'Operations Manager': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'IIT Bombay',
                'IIT Delhi', 'MIT (USA)', 'Stanford University (USA)',
                'Harvard Business School (USA)', 'London Business School (UK)',
                'Singapore Management University'
            ],
            
            'Human Resources Manager': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'Symbiosis Pune',
                'Delhi University', 'Harvard Business School (USA)',
                'Stanford Graduate School of Business (USA)', 'London Business School (UK)',
                'Singapore Management University'
            ],
            
            'Recruiter': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'Symbiosis Pune', 'NMIMS Mumbai', 'IIM Ahmedabad', 'IIM Bangalore',
                'Harvard Business School (USA)', 'Stanford Graduate School of Business (USA)',
                'London Business School (UK)'
            ],
            
            'Financial Analyst': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIM Lucknow',
                'XLRI Jamshedpur', 'SPJIMR Mumbai', 'NMIMS Mumbai', 'Delhi University',
                'Harvard Business School (USA)', 'Wharton School (USA)',
                'London Business School (UK)', 'London School of Economics (UK)',
                'Singapore Management University'
            ],
            
            'Accountant': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'Calcutta University', 'Madras University', 'BHU Varanasi',
                'AMU Aligarh', 'Kerala University', 'ICAI', 'IIM Ahmedabad',
                'Harvard Business School (USA)', 'Wharton School (USA)',
                'London School of Economics (UK)'
            ],
            
            'Chartered Accountant': [
                'ICAI (India)', 'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta',
                'Delhi University', 'Mumbai University', 'Harvard Business School (USA)',
                'Wharton School (USA)', 'London School of Economics (UK)'
            ],
            
            'Company Secretary': [
                'ICSI (India)', 'Delhi University', 'Mumbai University', 'IIM Ahmedabad',
                'IIM Bangalore', 'Harvard Business School (USA)',
                'London Business School (UK)'
            ],
            
            'Investment Banker': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIT Bombay',
                'IIT Delhi', 'Harvard Business School (USA)', 'Wharton School (USA)',
                'London Business School (UK)', 'London School of Economics (UK)',
                'MIT (USA)', 'Stanford University (USA)'
            ],
            
            'Management Consultant': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'IIT Bombay',
                'IIT Delhi', 'Harvard Business School (USA)', 'Stanford Graduate School of Business (USA)',
                'Wharton School (USA)', 'London Business School (UK)', 'INSEAD (France)'
            ],
            
            # ============ HEALTHCARE ============
            'Healthcare Assistant': [
                'AIIMS Delhi', 'AIIMS Bhubaneswar', 'AIIMS Bhopal', 'AIIMS Raipur',
                'AIIMS Patna', 'AIIMS Rishikesh', 'CMC Vellore', 'SGPGI Lucknow',
                'PGIMER Chandigarh', 'NIMHANS Bangalore', 'JIPMER Puducherry',
                'KGMU Lucknow', 'BHU Varanasi', 'AMU Aligarh', 'Seth GSMC Mumbai',
                'Johns Hopkins University (USA)', 'Harvard Medical School (USA)',
                'Stanford School of Medicine (USA)', 'University of Oxford (UK)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Nurse': [
                'AIIMS Delhi', 'CMC Vellore', 'PGIMER Chandigarh', 'NIMHANS Bangalore',
                'JIPMER Puducherry', 'KGMU Lucknow', 'BHU Varanasi',
                'Johns Hopkins University (USA)', 'Harvard Medical School (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Pharmacist': [
                'AIIMS Delhi', 'CMC Vellore', 'PGIMER Chandigarh', 'BHU Varanasi',
                'AMU Aligarh', 'KGMU Lucknow', 'Delhi University', 'Mumbai University',
                'Johns Hopkins University (USA)', 'Harvard Medical School (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Physiotherapist': [
                'AIIMS Delhi', 'CMC Vellore', 'PGIMER Chandigarh', 'JIPMER Puducherry',
                'BHU Varanasi', 'Delhi University', 'Mumbai University',
                'Johns Hopkins University (USA)', 'Harvard Medical School (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Medical Lab Technician': [
                'AIIMS Delhi', 'CMC Vellore', 'PGIMER Chandigarh', 'JIPMER Puducherry',
                'BHU Varanasi', 'Delhi University', 'Mumbai University',
                'Johns Hopkins University (USA)', 'Harvard Medical School (USA)',
                'University of Cambridge (UK)'
            ],
            
            'Radiology Technician': [
                'AIIMS Delhi', 'CMC Vellore', 'PGIMER Chandigarh', 'JIPMER Puducherry',
                'BHU Varanasi', 'Delhi University', 'Johns Hopkins University (USA)',
                'Harvard Medical School (USA)', 'University of Cambridge (UK)'
            ],
            
            'Dentist - BDS': [
                'AIIMS Delhi', 'CMC Vellore', 'PGIMER Chandigarh', 'BHU Varanasi',
                'KGMU Lucknow', 'Delhi University', 'Mumbai University',
                'Harvard School of Dental Medicine (USA)', 'University of Cambridge (UK)'
            ],
            
            'Ayurvedic Doctor': [
                'BHU Varanasi', 'Delhi University', 'Gujarat Ayurved University',
                'Tilak Ayurved College Pune', 'Kerala Ayurved University',
                'All India Institute of Ayurveda'
            ],
            
            'Veterinarian': [
                'VCI (India)', 'IARI Delhi', 'GBPUAT Pantnagar', 'PAU Ludhiana',
                'TNAU Coimbatore', 'Cornell University (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Public Health Specialist': [
                'AIIMS Delhi', 'PGIMER Chandigarh', 'IIHMR Jaipur', 'TISS Mumbai',
                'Johns Hopkins University (USA)', 'Harvard School of Public Health (USA)',
                'University of Cambridge (UK)', 'WHO (Switzerland)'
            ],
            
            'Health Educator': [
                'AIIMS Delhi', 'PGIMER Chandigarh', 'Delhi University', 'TISS Mumbai',
                'Johns Hopkins University (USA)', 'Harvard School of Public Health (USA)',
                'University of Cambridge (UK)'
            ],
            
            # ============ EDUCATION ============
            'School Teacher': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'BHU Varanasi', 'AMU Aligarh', 'Calcutta University', 'Madras University',
                'Kerala University', 'Gauhati University', 'NCTE (India)',
                'Harvard University (USA)', 'University of Cambridge (UK)',
                'University of Oxford (UK)', 'National University of Singapore'
            ],
            
            'Computer Teacher': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'BHU Varanasi', 'AMU Aligarh', 'Calcutta University', 'Madras University',
                'IIT Bombay', 'IIT Delhi', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'College Professor': [
                'IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur',
                'IIM Ahmedabad', 'IIM Bangalore', 'Delhi University', 'BHU Varanasi',
                'Harvard University (USA)', 'Stanford University (USA)',
                'University of Cambridge (UK)', 'University of Oxford (UK)',
                'MIT (USA)', 'National University of Singapore'
            ],
            
            'Online Tutor': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'BHU Varanasi', 'AMU Aligarh', 'IIT Bombay', 'IIT Delhi',
                'Harvard University (USA)', 'University of Cambridge (UK)',
                'University of Oxford (UK)', 'National University of Singapore'
            ],
            
            'Education Consultant': [
                'Delhi University', 'Mumbai University', 'JNU Delhi', 'IIM Ahmedabad',
                'IIM Bangalore', 'Harvard University (USA)', 'University of Cambridge (UK)',
                'University of Oxford (UK)', 'National University of Singapore'
            ],
            
            'Special Education Teacher': [
                'Delhi University', 'Mumbai University', 'BHU Varanasi', 'AMU Aligarh',
                'TISS Mumbai', 'Harvard University (USA)', 'University of Cambridge (UK)',
                'National University of Singapore'
            ],
            
            'Early Childhood Educator': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'BHU Varanasi', 'AMU Aligarh', 'NIE (India)', 'Harvard University (USA)',
                'University of Cambridge (UK)', 'National University of Singapore'
            ],
            
            'Corporate Trainer': [
                'IIM Ahmedabad', 'IIM Bangalore', 'IIM Calcutta', 'XLRI Jamshedpur',
                'SPJIMR Mumbai', 'Delhi University', 'Harvard Business School (USA)',
                'Stanford Graduate School of Business (USA)', 'London Business School (UK)'
            ],
            
            # ============ AGRICULTURE ============
            'Agricultural Scientist': [
                'IARI Delhi', 'GBPUAT Pantnagar', 'PAU Ludhiana', 'TNAU Coimbatore',
                'UAS Bangalore', 'MPKV Rahuri', 'BCKV West Bengal', 'OUAT Bhubaneswar',
                'JNKVV Jabalpur', 'Rajasthan Agricultural University', 'SKUAST Kashmir',
                'Wageningen University (Netherlands)', 'Cornell University (USA)',
                'University of California-Davis (USA)', 'University of Reading (UK)',
                'University of Sydney (Australia)', 'University of Guelph (Canada)'
            ],
            
            'Agricultural Technician': [
                'IARI Delhi', 'GBPUAT Pantnagar', 'PAU Ludhiana', 'TNAU Coimbatore',
                'UAS Bangalore', 'MPKV Rahuri', 'Wageningen University (Netherlands)',
                'Cornell University (USA)', 'University of California-Davis (USA)'
            ],
            
            'Farm Manager': [
                'PAU Ludhiana', 'TNAU Coimbatore', 'UAS Bangalore', 'MPKV Rahuri',
                'Wageningen University (Netherlands)', 'Cornell University (USA)',
                'University of California-Davis (USA)'
            ],
            
            'Horticulturist': [
                'IARI Delhi', 'PAU Ludhiana', 'TNAU Coimbatore', 'UAS Bangalore',
                'Wageningen University (Netherlands)', 'Cornell University (USA)',
                'University of California-Davis (USA)'
            ],
            
            'Agronomist': [
                'IARI Delhi', 'GBPUAT Pantnagar', 'PAU Ludhiana', 'TNAU Coimbatore',
                'Wageningen University (Netherlands)', 'Cornell University (USA)',
                'University of California-Davis (USA)'
            ],
            
            'Soil Scientist': [
                'IARI Delhi', 'GBPUAT Pantnagar', 'PAU Ludhiana', 'TNAU Coimbatore',
                'Wageningen University (Netherlands)', 'Cornell University (USA)',
                'University of California-Davis (USA)'
            ],
            
            'Organic Farming Expert': [
                'IARI Delhi', 'GBPUAT Pantnagar', 'PAU Ludhiana', 'TNAU Coimbatore',
                'Wageningen University (Netherlands)', 'Cornell University (USA)',
                'University of California-Davis (USA)'
            ],
            
            'Fishery Manager': [
                'ICAR-CIFE Mumbai', 'TNAU Coimbatore', 'Kerala University',
                'Wageningen University (Netherlands)', 'University of Guelph (Canada)'
            ],
            
            'Poultry Manager': [
                'PAU Ludhiana', 'TNAU Coimbatore', 'UAS Bangalore', 'MPKV Rahuri',
                'Wageningen University (Netherlands)', 'Cornell University (USA)'
            ],
            
            'Dairy Manager': [
                'NDRI Karnal', 'PAU Ludhiana', 'TNAU Coimbatore', 'UAS Bangalore',
                'Wageningen University (Netherlands)', 'Cornell University (USA)'
            ],
            
            # ============ TRADES ============
            'Electrician': [
                'NIT Warangal', 'NIT Trichy', 'NIT Surathkal', 'Anna University',
                'VTU Belgaum', 'IIT Delhi', 'IIT Bombay', 'IIT Kharagpur',
                'DTU Delhi', 'NSUT Delhi', 'VIT Vellore', 'SRM University',
                'MIT Manipal', 'Rajasthan Technical University',
                'MIT (USA)', 'Stanford University (USA)',
                'TU Munich (Germany)', 'University of California-Berkeley (USA)',
                'University of Toronto (Canada)', 'University of Melbourne (Australia)'
            ],
            
            'Plumber': [
                'VTU Belgaum', 'Anna University', 'Panjab University',
                'NIT Warangal', 'IIT Kharagpur', 'Amity University',
                'Chandigarh University', 'Lovely Professional University',
                'University of Toronto (Canada)', 'University of Melbourne (Australia)',
                'TU Munich (Germany)', 'University of California-Berkeley (USA)',
                'University of British Columbia (Canada)'
            ],
            
            'Welder': [
                'NIT Warangal', 'NIT Trichy', 'VTU Belgaum', 'Anna University',
                'IIT Bombay', 'IIT Delhi', 'MIT (USA)', 'Stanford University (USA)',
                'TU Munich (Germany)', 'University of California-Berkeley (USA)'
            ],
            
            'Carpenter': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'AMU Aligarh', 'BHU Varanasi', 'University of Toronto (Canada)',
                'University of Melbourne (Australia)'
            ],
            
            'Mason': [
                'VTU Belgaum', 'Anna University', 'Panjab University',
                'Amity University', 'Chandigarh University'
            ],
            
            'Painter': [
                'Delhi University', 'Mumbai University', 'Panjab University',
                'AMU Aligarh', 'National Institute of Design'
            ],
            
            'Automobile Mechanic': [
                'NIT Warangal', 'NIT Trichy', 'Anna University', 'VTU Belgaum',
                'IIT Bombay', 'IIT Delhi', 'MIT (USA)', 'Stanford University (USA)',
                'TU Munich (Germany)'
            ],
            
            'AC Repair Technician': [
                'VTU Belgaum', 'Anna University', 'NIT Trichy', 'NIT Warangal',
                'Amity University', 'Chandigarh University'
            ],
            
            'Solar Panel Installer': [
                'IIT Delhi', 'IIT Bombay', 'IIT Roorkee', 'NIT Trichy',
                'NIT Warangal', 'Anna University', 'VTU Belgaum',
                'Gujarat Energy Research Institute', 'TERI University',
                'UPES Dehradun', 'Amity University',
                'MIT (USA)', 'Stanford University (USA)',
                'TU Delft (Netherlands)', 'ETH Zurich (Switzerland)',
                'University of California-Berkeley (USA)',
                'University of New South Wales (Australia)'
            ],
            
            # ============ DEFAULT ============
            'default': [
                'IIT Delhi', 'IIT Bombay', 'IIT Kanpur', 'IIT Madras',
                'NIT Trichy', 'NIT Warangal', 'Delhi University',
                'University of Mumbai', 'Panjab University',
                'BHU Varanasi', 'AMU Aligarh', 'Calcutta University',
                'Madras University', 'Kerala University', 'IGNOU',
                'Harvard University (USA)', 'University of Oxford (UK)',
                'University of Cambridge (UK)', 'Stanford University (USA)',
                'MIT (USA)', 'National University of Singapore',
                'University of Melbourne (Australia)', 'University of Toronto (Canada)'
            ]
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

    def get_colleges_for_career(self, career_title):
        """Get recommended colleges for a career (India + Abroad)"""
        return self.college_mapping.get(career_title, self.college_mapping['default'])

    def recommend(self, user_profile):
        """Recommend careers based on user profile"""
        skills = user_profile.get("skills", "")
        education = user_profile.get("education", "")

        if not skills or not education:
            return pd.DataFrame()

        user_level = self.education_levels.get(education, 2)
        
        self.careers["edu_level"] = self.careers["min_education"].map(
            lambda x: self.education_levels.get(x, 2)
        )
        
        filtered = self.careers[self.careers["edu_level"] <= user_level].copy()
        
        if filtered.empty:
            return pd.DataFrame()
        
        filtered["match_percent"] = filtered["required_skills"].apply(
            lambda x: self.calculate_skill_match(skills, str(x))
        )
        
        filtered = filtered[filtered["match_percent"] >= 20]
        
        if filtered.empty:
            return pd.DataFrame()
        
        filtered = filtered.sort_values("match_percent", ascending=False)
        filtered["colleges"] = filtered["job_title"].apply(self.get_colleges_for_career)
        
        # Add Indian and Abroad college categories
        # Note: In a more advanced version, you could split India vs Abroad here
        
        results = filtered.head(10)
        
        return results