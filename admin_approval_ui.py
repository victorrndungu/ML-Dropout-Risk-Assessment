#!/usr/bin/env python3
"""
admin_approval_ui.py

Streamlit UI for admin approval system.
Allows admin to view, approve, and deny pending user registrations.
"""

import streamlit as st
from datetime import datetime
from admin_approval_system import get_approval_system, ApprovalStatus

def show_admin_approval_page():
    """Show the admin approval page."""
    st.title("🔐 Admin User Approval")
    st.markdown("Review and approve pending user registrations.")
    
    approval_system = get_approval_system()
    
    # Check for recent reminders
    try:
        recent_reminders = approval_system.get_recent_reminders()
        if recent_reminders:
            st.warning(f"🚨 **{len(recent_reminders)} user(s) have sent reminders for approval!**")
            for reminder in recent_reminders[:3]:  # Show first 3
                st.write(f"• {reminder['user_email']} sent reminder #{reminder['reminder_count']} at {reminder['created_at']}")
            if len(recent_reminders) > 3:
                st.write(f"... and {len(recent_reminders) - 3} more")
            st.markdown("---")
    except Exception as e:
        st.write(f"⚠️ Could not load reminders: {e}")
    
    # Create tables if they don't exist
    if st.button("🔧 Initialize Approval System", type="secondary"):
        with st.spinner("Creating approval tables..."):
            success = approval_system.create_approval_tables()
            if success:
                st.success("✅ Approval system initialized!")
            else:
                st.error("❌ Failed to initialize approval system")
    
    st.markdown("---")
    
    # Get pending users
    pending_users = approval_system.get_pending_users()
    
    if not pending_users:
        st.info("📭 No pending user registrations to review.")
        st.write("**Note:** If you just signed up as a user, make sure you completed email verification.")
        return
    
    st.subheader(f"📋 Pending Approvals ({len(pending_users)})")
    
    # Display pending users
    for user in pending_users:
        with st.expander(f"👤 {user['full_name']} ({user['email']}) - {user['role'].title()}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Email:** {user['email']}")
                st.write(f"**Full Name:** {user['full_name']}")
                st.write(f"**Role:** {user['role'].title()}")
                st.write(f"**Registration Date:** {user['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                if user['admin_token']:
                    st.write(f"**Admin Token Used:** ✅ Yes")
                else:
                    st.write(f"**Admin Token Used:** ❌ No")
            
            with col2:
                st.write("**Actions:**")
                
                # Approval form
                with st.form(f"approve_form_{user['id']}"):
                    approval_notes = st.text_area(
                        "Approval Notes (Optional)",
                        placeholder="Add any notes about this approval...",
                        key=f"approval_notes_{user['id']}"
                    )
                    
                    col_approve, col_deny = st.columns(2)
                    
                    with col_approve:
                        approve_clicked = st.form_submit_button("✅ Approve", type="primary")
                    
                    with col_deny:
                        deny_clicked = st.form_submit_button("❌ Deny", type="secondary")
                
                # Handle approval/denial
                if approve_clicked:
                    with st.spinner("Approving user..."):
                        success, message = approval_system.approve_user(
                            user['id'], 
                            st.session_state.get('user', {}).get('email', 'admin'),
                            approval_notes
                        )
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                
                if deny_clicked:
                    with st.spinner("Denying user..."):
                        success, message = approval_system.deny_user(
                            user['id'], 
                            st.session_state.get('user', {}).get('email', 'admin'),
                            approval_notes
                        )
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
    
    st.markdown("---")
    
    # Show approval history
    st.subheader("📜 Approval History")
    
    if st.button("🔄 Refresh History"):
        st.rerun()
    
    history = approval_system.get_approval_history(limit=20)
    
    if history:
        # Create a DataFrame for better display
        import pandas as pd
        
        history_df = pd.DataFrame(history)
        history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Color code actions
        def color_action(val):
            if val == 'approve':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'deny':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'create':
                return 'background-color: #d1ecf1; color: #0c5460'
            return ''
        
        styled_df = history_df.style.applymap(color_action, subset=['action'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("📭 No approval history found.")

def show_pending_approval_page_with_reminder():
    """Show enhanced pending approval page with reminder system."""
    st.title("⏳ Account Pending Approval")
    
    # Add logout button at the top
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            from auth_supabase_ui import supabase_logout
            supabase_logout()
            st.rerun()
    
    # Get user info
    user = st.session_state.get('user', {})
    user_email = user.get('email', 'Unknown')
    user_name = user.get('full_name', user_email.split('@')[0])
    
    # Initialize session state for reminders
    if 'last_reminder_sent' not in st.session_state:
        st.session_state.last_reminder_sent = None
    if 'reminder_count' not in st.session_state:
        st.session_state.reminder_count = 0
    
    # Main message
    st.markdown(f"""
    ## 👋 Hello {user_name}!
    
    **Your account is pending admin approval.**
    
    Your registration has been submitted and is waiting for an administrator to review and approve your account.
    
    You will receive an email notification once your account has been approved.
    """)
    
    # Show approval status
    approval_system = get_approval_system()
    status = approval_system.get_user_approval_status(user_email)
    
    if status == 'pending':
        st.info(f"📋 Status: **Pending Approval**")
        
        # Reminder system
        st.markdown("---")
        st.subheader("📧 Send Admin Reminder")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            **Need faster approval?** 
            
            You can send a reminder to the administrator to review your account.
            Reminders can be sent every 5 minutes to avoid spam.
            """)
        
        with col2:
            # Check if enough time has passed since last reminder
            import time
            current_time = time.time()
            last_reminder = st.session_state.last_reminder_sent
            
            can_send_reminder = (
                last_reminder is None or 
                (current_time - last_reminder) >= 300  # 5 minutes = 300 seconds
            )
            
            if can_send_reminder:
                if st.button("📤 Send Reminder", type="primary", use_container_width=True):
                    # Send reminder (simulate - in real implementation, this would send email/notification)
                    st.session_state.last_reminder_sent = current_time
                    st.session_state.reminder_count += 1
                    
                    # Log the reminder
                    try:
                        approval_system = get_approval_system()
                        # Add to approval logs using the approval system's method
                        import psycopg2
                        from psycopg2.extras import RealDictCursor
                        
                        # Use the correct connection method
                        conn = psycopg2.connect(
                            host=approval_system.config['host'],
                            port=approval_system.config['port'],
                            database=approval_system.config['database'],
                            user=approval_system.config['username'],
                            password=approval_system.config['password']
                        )
                        with conn.cursor(cursor_factory=RealDictCursor) as cur:
                            cur.execute("""
                                INSERT INTO approval_logs (user_email, action, performed_by, notes)
                                VALUES (%s, 'reminder', %s, %s)
                            """, (user_email, user_email, f"User sent reminder #{st.session_state.reminder_count}"))
                        conn.commit()
                        conn.close()
                        
                        st.success("✅ Reminder sent to administrator!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to send reminder: {e}")
                        # Don't show success message if it actually failed
            else:
                # Calculate time remaining
                time_remaining = 300 - (current_time - last_reminder)
                minutes_remaining = int(time_remaining // 60)
                seconds_remaining = int(time_remaining % 60)
                
                st.button(
                    f"⏰ Next reminder in {minutes_remaining}:{seconds_remaining:02d}", 
                    disabled=True,
                    use_container_width=True
                )
        
        # Show reminder count
        if st.session_state.reminder_count > 0:
            st.info(f"📊 You have sent {st.session_state.reminder_count} reminder(s) to the administrator.")
        
        # Contact information
        st.markdown("---")
        st.subheader("📞 Contact Administrator")
        
        st.markdown("""
        **If you need immediate assistance:**
        
        - **Email:** Contact your system administrator directly
        - **Phone:** Use your organization's contact information
        - **In-Person:** Visit your administrator's office
        
        **Please be patient** - approval typically takes 1-2 business days.
        """)
        
        # Auto-refresh every 30 seconds to check approval status
        import time
        if 'last_status_check' not in st.session_state:
            st.session_state.last_status_check = time.time()
        
        current_time = time.time()
        if (current_time - st.session_state.last_status_check) >= 30:  # 30 seconds
            st.session_state.last_status_check = current_time
            st.rerun()
        
        # Show refresh button
        if st.button("🔄 Check Approval Status", type="secondary"):
            st.rerun()
    
    elif status == 'denied':
        st.error(f"❌ Status: **Denied** - Please contact your administrator")
        
        st.markdown("---")
        st.subheader("📞 Contact Administrator")
        
        st.markdown("""
        **Your account has been denied access.**
        
        Please contact your system administrator to:
        - Understand the reason for denial
        - Request reconsideration if appropriate
        - Get guidance on proper registration
        """)
    
    else:
        st.success(f"✅ Status: **Approved** - Redirecting to main application...")
        st.rerun()

def show_pending_approval_message():
    """Show simple message for users waiting for approval (legacy function)."""
    show_pending_approval_page_with_reminder()

def check_user_approval_status(user_email: str) -> str:
    """Check if user is approved to access the system."""
    approval_system = get_approval_system()
    return approval_system.get_user_approval_status(user_email)
