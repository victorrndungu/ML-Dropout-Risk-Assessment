#!/usr/bin/env python3
"""
supabase_auth.py

Supabase authentication integration with two-factor authentication (OTP via email)
and Google OAuth sign-in. Syncs with local PostgreSQL for role management.
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

from supabase import create_client, Client
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

ROOT = Path(__file__).parent

class SupabaseAuth:
    """Supabase authentication service with PostgreSQL sync."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file"
            )
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Load local database config for role sync
        self.db_config = self._load_db_config()
    
    def _load_db_config(self) -> Dict:
        """Load local PostgreSQL config."""
        config_file = ROOT / "db_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def sign_up_with_email(
        self, 
        email: str, 
        password: str, 
        full_name: str,
        role: str = "viewer"
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Sign up new user with email and password.
        Supabase automatically sends email verification with OTP.
        
        Returns: (success, message, user_data)
        """
        try:
            # Sign up with Supabase
            app_url = os.getenv("APP_URL", "http://localhost:8504")
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name,
                        "role": role
                    },
                    "email_redirect_to": f"{app_url}/"
                }
            })
            
            if response.user:
                return True, "Signup successful! Please check your email for verification link.", {
                    "id": response.user.id,
                    "email": response.user.email,
                    "full_name": full_name,
                    "role": role
                }
            else:
                return False, "Signup failed. Please try again.", None
                
        except Exception as e:
            error_msg = str(e)
            if "already registered" in error_msg.lower():
                return False, "This email is already registered.", None
            return False, f"Signup error: {error_msg}", None
    
    def sign_in_with_email(self, email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Sign in with email and password.
        Returns user data if successful.
        
        Returns: (success, message, user_data)
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                # Check if email is verified
                if not response.user.email_confirmed_at:
                    return False, "Please verify your email first. Check your inbox for verification link.", None
                
                # Get user metadata
                user_metadata = response.user.user_metadata or {}
                
                user_data = {
                    "id": response.user.id,
                    "email": response.user.email,
                    "full_name": user_metadata.get("full_name", email.split("@")[0]),
                    "role": user_metadata.get("role", "viewer"),
                    "email_verified": response.user.email_confirmed_at is not None
                }
                
                # Sync to local PostgreSQL
                self._sync_user_to_local_db(user_data)
                
                return True, "Login successful!", user_data
            else:
                return False, "Invalid email or password.", None
                
        except Exception as e:
            error_msg = str(e)
            if "invalid" in error_msg.lower():
                return False, "Invalid email or password.", None
            return False, f"Login error: {error_msg}", None
    
    def sign_in_with_google(self) -> Tuple[bool, str]:
        """
        Initiate Google OAuth sign-in.
        Returns the OAuth URL to redirect to.
        
        Returns: (success, oauth_url or error_message)
        """
        try:
            response = self.client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": f"{os.getenv('APP_URL', 'http://localhost:8504')}/auth/callback"
                }
            })
            
            if response.url:
                return True, response.url
            else:
                return False, "Failed to initialize Google sign-in"
                
        except Exception as e:
            return False, f"Google sign-in error: {str(e)}"
    
    def sign_out(self) -> Tuple[bool, str]:
        """Sign out current user."""
        try:
            self.client.auth.sign_out()
            return True, "Signed out successfully"
        except Exception as e:
            return False, f"Sign out error: {str(e)}"
    
    def get_current_user(self) -> Optional[Dict]:
        """Get currently authenticated user."""
        try:
            response = self.client.auth.get_user()
            if response and response.user:
                user_metadata = response.user.user_metadata or {}
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "full_name": user_metadata.get("full_name", response.user.email.split("@")[0]),
                    "role": user_metadata.get("role", "viewer"),
                    "email_verified": response.user.email_confirmed_at is not None
                }
            return None
        except Exception:
            return None
    
    def resend_verification_email(self, email: str) -> Tuple[bool, str]:
        """Resend email verification."""
        try:
            app_url = os.getenv("APP_URL", "http://localhost:8504")
            self.client.auth.resend({
                "type": "signup",
                "email": email,
                "options": {
                    "email_redirect_to": f"{app_url}/"
                }
            })
            return True, "Verification email sent! Please check your inbox."
        except Exception as e:
            return False, f"Failed to resend email: {str(e)}"
    
    def send_password_reset(self, email: str) -> Tuple[bool, str]:
        """Send password reset email."""
        try:
            app_url = os.getenv("APP_URL", "http://localhost:8504")
            self.client.auth.reset_password_email(email, {
                "redirect_to": f"{app_url}/"
            })
            return True, "Password reset email sent! Check your inbox."
        except Exception as e:
            return False, f"Failed to send reset email: {str(e)}"
    
    def verify_otp(self, email: str, otp_code: str) -> Tuple[bool, str]:
        """
        Verify OTP code for email verification.
        
        Returns: (success, message)
        """
        try:
            response = self.client.auth.verify_otp({
                "email": email,
                "token": otp_code,
                "type": "email"
            })
            
            if response.user:
                return True, "Email verified successfully!"
            else:
                return False, "Invalid or expired OTP code"
                
        except Exception as e:
            error_msg = str(e)
            if "expired" in error_msg.lower():
                return False, "OTP code has expired. Please request a new one."
            return False, f"Verification failed: {error_msg}"
    
    def update_user_role(self, user_id: str, new_role: str) -> Tuple[bool, str]:
        """Update user role (admin only)."""
        try:
            self.client.auth.admin.update_user_by_id(
                user_id,
                {"user_metadata": {"role": new_role}}
            )
            return True, f"User role updated to {new_role}"
        except Exception as e:
            return False, f"Failed to update role: {str(e)}"
    
    def _sync_user_to_local_db(self, user_data: Dict):
        """Sync Supabase user to local PostgreSQL for role management."""
        try:
            from sqlalchemy import create_engine, text
            
            if not self.db_config:
                return
            
            connection_string = (
                f"postgresql://{self.db_config['username']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            
            engine = create_engine(connection_string)
            
            with engine.connect() as conn:
                # Check if user exists in local DB
                result = conn.execute(
                    text("SELECT id FROM users WHERE email = :email"),
                    {"email": user_data["email"]}
                )
                
                if result.fetchone():
                    # Update existing user
                    conn.execute(
                        text("""
                            UPDATE users 
                            SET full_name = :full_name, 
                                role = :role::userrole,
                                last_login = :last_login
                            WHERE email = :email
                        """),
                        {
                            "email": user_data["email"],
                            "full_name": user_data["full_name"],
                            "role": user_data["role"],
                            "last_login": datetime.utcnow()
                        }
                    )
                else:
                    # Insert new user
                    conn.execute(
                        text("""
                            INSERT INTO users (email, full_name, role, is_active, is_verified, created_at, last_login)
                            VALUES (:email, :full_name, :role::userrole, true, :verified, :created, :last_login)
                        """),
                        {
                            "email": user_data["email"],
                            "full_name": user_data["full_name"],
                            "role": user_data["role"],
                            "verified": user_data.get("email_verified", True),
                            "created": datetime.utcnow(),
                            "last_login": datetime.utcnow()
                        }
                    )
                
                conn.commit()
                
        except Exception as e:
            print(f"Warning: Failed to sync user to local DB: {e}")

# Global Supabase auth instance
_supabase_auth = None

def get_supabase_auth() -> SupabaseAuth:
    """Get or create Supabase auth instance."""
    global _supabase_auth
    if _supabase_auth is None:
        _supabase_auth = SupabaseAuth()
    return _supabase_auth

