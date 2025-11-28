#!/usr/bin/env python3
"""
admin_approval_system.py

Admin approval system for user registrations.
Allows admin to view and approve/deny pending user accounts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "db_config.json"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"

class AdminApprovalSystem:
    """Admin approval system for user registrations."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or CONFIG_FILE
        self.config = self.load_config()
        self.engine = None
        self._connect()
    
    def load_config(self) -> Dict:
        """Load database configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Database config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _connect(self):
        """Create database connection."""
        connection_string = (
            f"postgresql://{self.config['username']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
        self.engine = create_engine(connection_string, echo=False)
    
    def create_approval_tables(self):
        """Create tables for admin approval system."""
        try:
            with self.engine.connect() as conn:
                # Create pending_users table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS pending_users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        full_name VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        admin_token VARCHAR(255),
                        status VARCHAR(20) DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        reviewed_at TIMESTAMP,
                        reviewed_by VARCHAR(255),
                        review_notes TEXT
                    )
                """))
                
                # Create approval_logs table for audit trail
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS approval_logs (
                        id SERIAL PRIMARY KEY,
                        user_email VARCHAR(255) NOT NULL,
                        action VARCHAR(20) NOT NULL, -- 'approve', 'deny', 'create'
                        performed_by VARCHAR(255) NOT NULL,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                print("✅ Admin approval tables created successfully!")
                return True
                
        except Exception as e:
            print(f"❌ Error creating approval tables: {e}")
            return False
    
    def add_pending_user(self, email: str, full_name: str, role: str, admin_token: str = None) -> bool:
        """Add a new pending user for admin approval."""
        try:
            with self.engine.connect() as conn:
                # Check if user already exists in pending or approved users
                check_query = text("""
                    SELECT COUNT(*) FROM pending_users WHERE email = :email
                    UNION ALL
                    SELECT COUNT(*) FROM users WHERE email = :email
                """)
                result = conn.execute(check_query, {"email": email})
                counts = [row[0] for row in result.fetchall()]
                
                if any(count > 0 for count in counts):
                    return False, "User already exists or is pending approval"
                
                # Insert pending user
                insert_query = text("""
                    INSERT INTO pending_users (email, full_name, role, admin_token, status)
                    VALUES (:email, :full_name, :role, :admin_token, 'pending')
                """)
                conn.execute(insert_query, {
                    "email": email,
                    "full_name": full_name,
                    "role": role,
                    "admin_token": admin_token
                })
                
                # Log the creation
                log_query = text("""
                    INSERT INTO approval_logs (user_email, action, performed_by, notes)
                    VALUES (:email, 'create', 'system', 'User registration submitted for approval')
                """)
                conn.execute(log_query, {"email": email})
                
                conn.commit()
                return True, "User added to pending approval list"
                
        except Exception as e:
            print(f"❌ Error adding pending user: {e}")
            return False, f"Error: {e}"
    
    def get_pending_users(self) -> List[Dict]:
        """Get all pending users awaiting approval."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT id, email, full_name, role, admin_token, created_at, status
                    FROM pending_users 
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                """)
                result = conn.execute(query)
                rows = result.fetchall()
                # Convert rows to dictionaries properly
                pending_users = []
                for row in rows:
                    pending_users.append({
                        'id': row[0],
                        'email': row[1],
                        'full_name': row[2],
                        'role': row[3],
                        'admin_token': row[4],
                        'created_at': row[5],
                        'status': row[6]
                    })
                return pending_users
        except Exception as e:
            print(f"❌ Error getting pending users: {e}")
            return []
    
    def approve_user(self, user_id: int, admin_email: str, notes: str = "") -> Tuple[bool, str]:
        """Approve a pending user."""
        try:
            with self.engine.connect() as conn:
                # Get user details
                user_query = text("SELECT * FROM pending_users WHERE id = :user_id")
                result = conn.execute(user_query, {"user_id": user_id})
                user = result.fetchone()
                
                if not user:
                    return False, "User not found"
                
                # Update pending user status
                update_query = text("""
                    UPDATE pending_users 
                    SET status = 'approved', reviewed_at = CURRENT_TIMESTAMP, 
                        reviewed_by = :admin_email, review_notes = :notes
                    WHERE id = :user_id
                """)
                conn.execute(update_query, {
                    "user_id": user_id,
                    "admin_email": admin_email,
                    "notes": notes
                })
                
                # Log the approval
                log_query = text("""
                    INSERT INTO approval_logs (user_email, action, performed_by, notes)
                    VALUES (:email, 'approve', :admin_email, :notes)
                """)
                conn.execute(log_query, {
                    "email": user.email,
                    "admin_email": admin_email,
                    "notes": notes
                })
                
                conn.commit()
                return True, f"User {user.email} approved successfully"
                
        except Exception as e:
            print(f"❌ Error approving user: {e}")
            return False, f"Error: {e}"
    
    def deny_user(self, user_id: int, admin_email: str, notes: str = "") -> Tuple[bool, str]:
        """Deny a pending user."""
        try:
            with self.engine.connect() as conn:
                # Get user details
                user_query = text("SELECT * FROM pending_users WHERE id = :user_id")
                result = conn.execute(user_query, {"user_id": user_id})
                user = result.fetchone()
                
                if not user:
                    return False, "User not found"
                
                # Update pending user status
                update_query = text("""
                    UPDATE pending_users 
                    SET status = 'denied', reviewed_at = CURRENT_TIMESTAMP, 
                        reviewed_by = :admin_email, review_notes = :notes
                    WHERE id = :user_id
                """)
                conn.execute(update_query, {
                    "user_id": user_id,
                    "admin_email": admin_email,
                    "notes": notes
                })
                
                # Log the denial
                log_query = text("""
                    INSERT INTO approval_logs (user_email, action, performed_by, notes)
                    VALUES (:email, 'deny', :admin_email, :notes)
                """)
                conn.execute(log_query, {
                    "email": user.email,
                    "admin_email": admin_email,
                    "notes": notes
                })
                
                conn.commit()
                return True, f"User {user.email} denied"
                
        except Exception as e:
            print(f"❌ Error denying user: {e}")
            return False, f"Error: {e}"
    
    def get_approval_history(self, limit: int = 50) -> List[Dict]:
        """Get approval history for audit trail."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT user_email, action, performed_by, notes, created_at
                    FROM approval_logs 
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                result = conn.execute(query, {"limit": limit})
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            print(f"❌ Error getting approval history: {e}")
            return []
    
    def is_user_approved(self, email: str) -> bool:
        """Check if a user is approved to access the system."""
        try:
            with self.engine.connect() as conn:
                # Check if user exists in approved users table
                query = text("SELECT COUNT(*) FROM users WHERE email = :email")
                result = conn.execute(query, {"email": email})
                count = result.scalar()
                return count > 0
        except Exception as e:
            print(f"❌ Error checking user approval: {e}")
            return False
    
    def get_user_approval_status(self, email: str) -> Optional[str]:
        """Get the approval status of a user."""
        try:
            with self.engine.connect() as conn:
                # Check pending users first
                pending_query = text("SELECT status FROM pending_users WHERE email = :email")
                result = conn.execute(pending_query, {"email": email})
                pending_status = result.fetchone()
                
                if pending_status:
                    return pending_status[0]
                
                # Check if user is approved
                approved_query = text("SELECT COUNT(*) FROM users WHERE email = :email")
                result = conn.execute(approved_query, {"email": email})
                if result.scalar() > 0:
                    return "approved"
                
                return None
        except Exception as e:
            print(f"❌ Error getting user approval status: {e}")
            return None
    
    def get_recent_reminders(self, hours: int = 24) -> List[Dict]:
        """Get recent reminders sent by users."""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT user_email, notes, created_at,
                           ROW_NUMBER() OVER (PARTITION BY user_email ORDER BY created_at DESC) as reminder_count
                    FROM approval_logs 
                    WHERE action = 'reminder' 
                    AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                """)
                result = conn.execute(query, (hours,))
                rows = result.fetchall()
                # Convert rows to dictionaries properly
                reminders = []
                for row in rows:
                    reminders.append({
                        'user_email': row[0],
                        'notes': row[1],
                        'created_at': row[2],
                        'reminder_count': row[3]
                    })
                return reminders
        except Exception as e:
            print(f"❌ Error getting recent reminders: {e}")
            return []

# Global instance
_approval_system = None

def get_approval_system() -> AdminApprovalSystem:
    """Get or create admin approval system instance."""
    global _approval_system
    if _approval_system is None:
        _approval_system = AdminApprovalSystem()
    return _approval_system
