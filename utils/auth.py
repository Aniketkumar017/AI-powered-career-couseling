import hashlib
import secrets
import re
from datetime import datetime, timedelta
import sqlite3
from utils.db import get_db_connection

def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((salt + password).encode())
    return f"{salt}${hash_obj.hexdigest()}"

def verify_password(password, hashed_password):
    """Verify password against hash"""
    salt, hash_val = hashed_password.split('$')
    hash_obj = hashlib.sha256((salt + password).encode())
    return hash_obj.hexdigest() == hash_val

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def register_user(username, email, password, full_name=None, **kwargs):
    """Register a new user"""
    # Validate input
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters"
    if not validate_email(email):
        return False, "Invalid email format"
    
    valid, msg = validate_password(password)
    if not valid:
        return False, msg
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if username exists
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already exists"
    
    # Check if email exists
    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
    if cursor.fetchone():
        conn.close()
        return False, "Email already registered"
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Insert user
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, age, gender, location, education, skills, interests)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, email, hashed_password, full_name, 
              kwargs.get('age'), kwargs.get('gender'), 
              kwargs.get('location'), kwargs.get('education'),
              kwargs.get('skills'), kwargs.get('interests')))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except Exception as e:
        conn.close()
        return False, str(e)

def login_user(username, password):
    """Authenticate user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, password_hash, full_name, age, gender, 
               location, education, skills, interests, created_at
        FROM users 
        WHERE username = ? OR email = ?
    ''', (username, username))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None, "User not found"
    
    if not verify_password(password, user['password_hash']):
        return None, "Invalid password"
    
    # Update last login
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET updated_at = CURRENT_TIMESTAMP 
        WHERE id = ?
    ''', (user['id'],))
    conn.commit()
    conn.close()
    
    return dict(user), "Login successful"

def get_user(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, full_name, age, gender, 
               location, education, skills, interests, created_at, updated_at
        FROM users 
        WHERE id = ?
    ''', (user_id,))
    
    user = cursor.fetchone()
    conn.close()
    
    return dict(user) if user else None

def update_user(user_id, **kwargs):
    """Update user information"""
    allowed_fields = ['full_name', 'age', 'gender', 'location', 'education', 'skills', 'interests']
    
    updates = []
    values = []
    
    for field in allowed_fields:
        if field in kwargs and kwargs[field] is not None:
            updates.append(f"{field} = ?")
            values.append(kwargs[field])
    
    if not updates:
        return True, "No changes made"
    
    updates.append("updated_at = CURRENT_TIMESTAMP")
    values.append(user_id)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(f'''
            UPDATE users SET {', '.join(updates)}
            WHERE id = ?
        ''', values)
        conn.commit()
        conn.close()
        return True, "Profile updated successfully"
    except Exception as e:
        conn.close()
        return False, str(e)

def save_career_recommendation(user_id, skills, education, location, recommendations):
    """Save career recommendation history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO career_history (user_id, skills, education, location, recommendations)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, skills, education, location, json.dumps(recommendations)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        return False

def get_career_history(user_id, limit=10):
    """Get user's career recommendation history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, skills, education, location, recommendations, created_at
        FROM career_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (user_id, limit))
    
    history = cursor.fetchall()
    conn.close()
    
    return [dict(h) for h in history]

def save_career(user_id, career_title, career_data):
    """Save a career to user's saved list"""
    try:
        import json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if already saved
        cursor.execute('''
            SELECT id FROM saved_careers 
            WHERE user_id = ? AND career_title = ?
        ''', (user_id, career_title))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing
            cursor.execute('''
                UPDATE saved_careers 
                SET career_data = ?, saved_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND career_title = ?
            ''', (json.dumps(career_data), user_id, career_title))
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO saved_careers (user_id, career_title, career_data)
                VALUES (?, ?, ?)
            ''', (user_id, career_title, json.dumps(career_data)))
        
        conn.commit()
        conn.close()
        return True, "Career saved successfully"
        
    except Exception as e:
        print(f"Save career DB error: {str(e)}")
        return False, str(e)
def get_saved_careers(user_id):
    """Get user's saved careers"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, career_title, career_data, saved_at
        FROM saved_careers
        WHERE user_id = ?
        ORDER BY saved_at DESC
    ''', (user_id,))
    
    careers = cursor.fetchall()
    conn.close()
    
    return [dict(c) for c in careers]

def remove_saved_career(user_id, career_title):
    """Remove a saved career"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        DELETE FROM saved_careers
        WHERE user_id = ? AND career_title = ?
    ''', (user_id, career_title))
    conn.commit()
    conn.close()
    
    return True

def save_feedback(user_id, career_title, rating, feedback):
    """Save user feedback"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO user_feedback (user_id, career_title, rating, feedback)
            VALUES (?, ?, ?, ?)
        ''', (user_id, career_title, rating, feedback))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        return False