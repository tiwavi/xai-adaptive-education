import sqlite3
import hashlib
import json
from datetime import datetime

DB_NAME = "edu_xai.db"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'teacher'
        )
    ''')
    
    # Saved Student Records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            student_id TEXT NOT NULL,
            data_json TEXT NOT NULL,
            prediction_status TEXT NOT NULL,
            pass_probability REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    
    # Create default admin account if not exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        admin_pass = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("admin", "admin@eduxai.com", admin_pass, "admin")
        )
        conn.commit()
        print("[Database] Default admin user created (Username: admin, Password: admin123)")
        
    conn.close()

def register_user(username: str, email: str, password: str, role: str = "teacher"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        pass_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, pass_hash, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"success": True, "user_id": user_id, "role": role}
    except sqlite3.IntegrityError:
        conn.close()
        return {"success": False, "error": "Username or email already exists."}

def authenticate_user(username: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    pass_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username, email, role FROM users WHERE username = ? AND password_hash = ?",
        (username, pass_hash)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {"success": True, "user": {"id": user[0], "username": user[1], "email": user[2], "role": user[3]}}
    return {"success": False, "error": "Invalid username or password."}

def save_student_record(user_id: int, student_id: str, data: dict, status: str, prob: float):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO students (user_id, student_id, data_json, prediction_status, pass_probability, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, student_id, json.dumps(data), status, prob, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    conn.close()
    return True

def get_student_records(user_id: int, role: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if role == "admin":
        cursor.execute("SELECT s.id, u.username, s.student_id, s.prediction_status, s.pass_probability, s.created_at FROM students s JOIN users u ON s.user_id = u.id ORDER BY s.id DESC")
    else:
        cursor.execute("SELECT s.id, u.username, s.student_id, s.prediction_status, s.pass_probability, s.created_at FROM students s JOIN users u ON s.user_id = u.id WHERE s.user_id = ? ORDER BY s.id DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for r in rows:
        records.append({
            "id": r[0], "teacher": r[1], "student_id": r[2], 
            "status": r[3], "pass_probability": r[4], "created_at": r[5]
        })
    return records

if __name__ == "__main__":
    init_db()