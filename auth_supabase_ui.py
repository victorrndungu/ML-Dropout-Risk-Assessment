#!/usr/bin/env python3
"""
supabase_auth_ui.py

Streamlit UI components for Supabase authentication with real email OTP and Google sign-in.
"""
import streamlit as st
from pathlib import Path
from auth_supabase import get_supabase_auth

ROOT = Path(__file__).parent

def init_supabase_session_state():
    """Initialize session state variables for Supabase authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_stage' not in st.session_state:
        st.session_state.auth_stage = 'login'  # 'login', 'signup', 'verify_email', 'forgot_password'
    if 'pending_email' not in st.session_state:
        st.session_state.pending_email = None
    if 'show_login_button' not in st.session_state:
        st.session_state.show_login_button = False
    if 'show_resend_button' not in st.session_state:
        st.session_state.show_resend_button = False
    if 'resend_email' not in st.session_state:
        st.session_state.resend_email = None

def show_supabase_login_page():
    """Display Supabase login page with email and Google sign-in."""
    st.title("🔐 Login to Dropout Risk Assessment System")
    
    auth = get_supabase_auth()
    
    # Email/Password Login
    st.subheader("Login with Email")
    with st.form("login_form"):
        email = st.text_input("Email Address", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("🔑 Login", use_container_width=True)
        with col2:
            signup_button = st.form_submit_button("📝 Sign Up", use_container_width=True)
        
        if login_button:
            if not email or not password:
                st.error("Please enter both email and password")
            else:
                with st.spinner("Signing in..."):
                    success, message, user_data = auth.sign_in_with_email(email, password)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = user_data
                        st.success("✅ Login successful!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(message)
                        # If email not verified, offer resend option
                        if "verify your email" in message.lower():
                            st.session_state.show_resend_button = True
                            st.session_state.resend_email = email
        
        if signup_button:
            st.session_state.auth_stage = 'signup'
            st.rerun()
    
    # Show resend verification button outside form if needed
    if st.session_state.get('show_resend_button', False):
        if st.button("📧 Resend Verification Email", key="resend_outside_form"):
            auth = get_supabase_auth()
            resend_success, resend_msg = auth.resend_verification_email(st.session_state.resend_email)
            if resend_success:
                st.success(resend_msg)
            else:
                st.error(resend_msg)
            st.session_state.show_resend_button = False
    
    # Divider
    st.markdown("---")
    st.markdown("<center>OR</center>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Google Sign-In
    st.subheader("Sign in with Google")
    if st.button("🔵 Continue with Google", use_container_width=True):
        success, oauth_url = auth.sign_in_with_google()
        if success:
            st.markdown(f"[Click here to sign in with Google]({oauth_url})")
            st.info("You'll be redirected to Google for authentication")
        else:
            st.error(oauth_url)  # Error message
    
    # Additional options
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔑 Forgot Password?"):
            st.session_state.auth_stage = 'forgot_password'
            st.rerun()
    with col2:
        if st.button("📧 Resend Verification"):
            st.session_state.auth_stage = 'resend_verification'
            st.rerun()

def show_supabase_signup_page():
    """Display Supabase signup page."""
    st.title("📝 Sign Up - Dropout Risk Assessment System")
    
    auth = get_supabase_auth()
    
    # Role descriptions
    with st.expander("ℹ️ Role Descriptions", expanded=False):
        st.markdown("""
        **👑 Admin**: Full system access, user management, system configuration
        
        **👨‍💼 Social Worker**: Upload profiles, view needs assessments, manage cases, access recommendations, full workflow access
        
        **👨‍🏫 Teacher**: View student profiles, access risk assessments, limited case management
        
        **👁️ Viewer**: Read-only access to dashboards and reports
        
        **Note**: Admin role requires approval. Please sign up as Social Worker, Teacher, or Viewer.
        """)
    
    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email Address", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", placeholder="Minimum 8 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        
        role = st.selectbox(
            "Select Your Role",
            options=["social_worker", "teacher", "viewer"],
            format_func=lambda x: {
                "social_worker": "👨‍💼 Social Worker",
                "teacher": "👨‍🏫 Teacher",
                "viewer": "👁️ Viewer"
            }[x]
        )
        
        st.info("💡 You'll receive an email verification link. Please check your inbox after signing up.")
        
        col1, col2 = st.columns(2)
        with col1:
            signup_button = st.form_submit_button("📝 Sign Up", use_container_width=True)
        with col2:
            back_button = st.form_submit_button("← Back to Login", use_container_width=True)
        
        if signup_button:
            # Validation
            if not all([full_name, email, password, confirm_password]):
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters")
            else:
                with st.spinner("Creating your account..."):
                    success, message, user_data = auth.sign_up_with_email(
                        email=email,
                        password=password,
                        full_name=full_name,
                        role=role
                    )
                    
                    if success:
                        st.success("✅ " + message)
                        st.info("📧 **Check your email for a verification link!**")
                        st.markdown("---")
                        st.markdown("After verifying your email, return here to login.")
                        st.session_state.pending_email = email
                        # Set a flag to show the button outside the form
                        st.session_state.show_login_button = True
                    else:
                        st.error(message)
        
        if back_button:
            st.session_state.auth_stage = 'login'
            st.rerun()
    
    # Show "Go to Login" button outside form if signup was successful
    if st.session_state.get('show_login_button', False):
        if st.button("Go to Login", key="go_to_login_outside_form"):
            st.session_state.auth_stage = 'login'
            st.session_state.show_login_button = False
            st.rerun()
    
    # Google Sign-Up
    st.markdown("---")
    st.markdown("<center>OR</center>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Sign up with Google")
    if st.button("🔵 Continue with Google", use_container_width=True):
        success, oauth_url = auth.sign_in_with_google()
        if success:
            st.markdown(f"[Click here to sign up with Google]({oauth_url})")
            st.info("You'll be redirected to Google for authentication")
        else:
            st.error(oauth_url)

def show_forgot_password_page():
    """Display forgot password page."""
    st.title("🔑 Reset Password")
    
    auth = get_supabase_auth()
    
    st.info("Enter your email address and we'll send you a password reset link.")
    
    with st.form("forgot_password_form"):
        email = st.text_input("Email Address", placeholder="your.email@example.com")
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("📧 Send Reset Link", use_container_width=True)
        with col2:
            back_button = st.form_submit_button("← Back to Login", use_container_width=True)
        
        if submit_button:
            if not email:
                st.error("Please enter your email address")
            else:
                with st.spinner("Sending reset link..."):
                    success, message = auth.send_password_reset(email)
                    if success:
                        st.success(message)
                        st.info("Check your email for the password reset link")
                    else:
                        st.error(message)
        
        if back_button:
            st.session_state.auth_stage = 'login'
            st.rerun()

def show_resend_verification_page():
    """Display resend verification page."""
    st.title("📧 Resend Verification Email")
    
    auth = get_supabase_auth()
    
    st.info("Enter your email address and we'll resend the verification link.")
    
    with st.form("resend_verification_form"):
        email = st.text_input("Email Address", placeholder="your.email@example.com")
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("📧 Resend Verification", use_container_width=True)
        with col2:
            back_button = st.form_submit_button("← Back to Login", use_container_width=True)
        
        if submit_button:
            if not email:
                st.error("Please enter your email address")
            else:
                with st.spinner("Sending verification email..."):
                    success, message = auth.resend_verification_email(email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        if back_button:
            st.session_state.auth_stage = 'login'
            st.rerun()

def show_supabase_auth_page():
    """Main Supabase authentication page router."""
    init_supabase_session_state()
    
    # If already authenticated, don't show auth page
    if st.session_state.authenticated:
        return True
    
    # Check for existing session
    auth = get_supabase_auth()
    current_user = auth.get_current_user()
    if current_user:
        st.session_state.authenticated = True
        st.session_state.user = current_user
        return True
    
    # Route to appropriate page based on auth stage
    stage = st.session_state.auth_stage
    
    if stage == 'login':
        show_supabase_login_page()
    elif stage == 'signup':
        show_supabase_signup_page()
    elif stage == 'forgot_password':
        show_forgot_password_page()
    elif stage == 'resend_verification':
        show_resend_verification_page()
    
    return False

def supabase_logout():
    """Logout current user."""
    auth = get_supabase_auth()
    auth.sign_out()
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.auth_stage = 'login'
    st.session_state.pending_email = None
    st.rerun()

def show_supabase_user_info():
    """Display current user information in sidebar."""
    if st.session_state.authenticated and st.session_state.user:
        user = st.session_state.user
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Current User")
        st.sidebar.markdown(f"**{user['full_name']}**")
        st.sidebar.markdown(f"📧 {user['email']}")
        
        # Role badge with emoji
        role_emoji = {
            'admin': '👑',
            'social_worker': '👨‍💼',
            'teacher': '👨‍🏫',
            'viewer': '👁️'
        }
        role_display = user['role'].replace('_', ' ').title()
        st.sidebar.markdown(f"{role_emoji.get(user['role'], '👤')} **{role_display}**")
        
        # Verification status
        if user.get('email_verified'):
            st.sidebar.success("✅ Email Verified")
        else:
            st.sidebar.warning("⚠️ Email Not Verified")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            supabase_logout()

