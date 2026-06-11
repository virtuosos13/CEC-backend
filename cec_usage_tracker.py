import sqlite3
import os

class CECUsageTracker:
    def __init__(self, db_path='cec_users.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create users table
        # trial_limit: max questions for free users
        # questions_asked: current count
        # is_premium: boolean to bypass limits
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                questions_asked INTEGER DEFAULT 0,
                trial_limit INTEGER DEFAULT 5,
                is_premium BOOLEAN DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()

    def get_or_create_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT questions_asked, trial_limit, is_premium FROM users WHERE user_id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            cursor.execute('INSERT INTO users (user_id) VALUES (?)', (user_id,))
            conn.commit()
            user = (0, 5, 0) # Default new user state
        
        conn.close()
        return {
            "questions_asked": user[0],
            "trial_limit": user[1],
            "is_premium": bool(user[2])
        }

    def increment_usage(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET questions_asked = questions_asked + 1 WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()

    def check_access(self, user_id):
        user = self.get_or_create_user(user_id)
        if user['is_premium']:
            return True, "Unlimited Access"
        
        if user['questions_asked'] < user['trial_limit']:
            remaining = user['trial_limit'] - user['questions_asked']
            return True, f"{remaining} trial questions remaining"
        
        return False, "Trial limit reached. Please upgrade to Premium for unlimited electrical code support."

if __name__ == "__main__":
    tracker = CECUsageTracker()
    print("Usage tracker initialized.")
