"""
Utils Package
"""

# Import all functions from auth and db for easy access
from .auth import *
from .db import *

# Package metadata
__version__ = "1.0.0"
__all__ = [
    # Auth functions
    'register_user',
    'login_user', 
    'get_user',
    'update_user',
    'save_career',
    'get_saved_careers',
    'remove_saved_career',
    'save_career_recommendation',
    'get_career_history',
    'save_feedback',
    'hash_password',
    'verify_password',
    'validate_email',
    'validate_password',
    
    # DB functions
    'get_db_connection',
    'init_db'
]