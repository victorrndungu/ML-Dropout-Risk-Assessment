#!/usr/bin/env python3
"""
rbac.py

Role-Based Access Control (RBAC) system for the Dropout Risk Assessment application.
"""
import streamlit as st
from typing import List

class Permission:
    """Permission definitions."""
    # Dashboard and viewing
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_PROFILES = "view_profiles"
    VIEW_RISK_ASSESSMENT = "view_risk_assessment"
    VIEW_NEEDS_OVERVIEW = "view_needs_overview"
    VIEW_METRICS = "view_metrics"
    
    # Data management
    UPLOAD_CASES = "upload_cases"
    EDIT_PROFILES = "edit_profiles"
    DELETE_PROFILES = "delete_profiles"
    
    # Case management
    VIEW_CASES = "view_cases"
    CREATE_CASES = "create_cases"
    UPDATE_CASES = "update_cases"
    CLOSE_CASES = "close_cases"
    ASSIGN_CASES = "assign_cases"
    
    # Interventions and recommendations
    VIEW_RECOMMENDATIONS = "view_recommendations"
    CREATE_INTERVENTIONS = "create_interventions"
    UPDATE_INTERVENTIONS = "update_interventions"
    
    # Batch operations
    BATCH_PROCESSING = "batch_processing"
    
    # System administration
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    CONFIGURE_SYSTEM = "configure_system"
    
    # Database operations
    QUERY_DATABASE = "query_database"
    EXPORT_DATA = "export_data"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    'admin': [
        # Admin has ALL permissions
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILES,
        Permission.VIEW_RISK_ASSESSMENT,
        Permission.VIEW_NEEDS_OVERVIEW,
        Permission.VIEW_METRICS,
        Permission.UPLOAD_CASES,
        Permission.EDIT_PROFILES,
        Permission.DELETE_PROFILES,
        Permission.VIEW_CASES,
        Permission.CREATE_CASES,
        Permission.UPDATE_CASES,
        Permission.CLOSE_CASES,
        Permission.ASSIGN_CASES,
        Permission.VIEW_RECOMMENDATIONS,
        Permission.CREATE_INTERVENTIONS,
        Permission.UPDATE_INTERVENTIONS,
        Permission.BATCH_PROCESSING,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.CONFIGURE_SYSTEM,
        Permission.QUERY_DATABASE,
        Permission.EXPORT_DATA,
    ],
    'social_worker': [
        # Social workers have most permissions except system admin
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILES,
        Permission.VIEW_RISK_ASSESSMENT,
        Permission.VIEW_NEEDS_OVERVIEW,
        Permission.VIEW_METRICS,
        Permission.UPLOAD_CASES,
        Permission.EDIT_PROFILES,
        Permission.VIEW_CASES,
        Permission.CREATE_CASES,
        Permission.UPDATE_CASES,
        Permission.CLOSE_CASES,
        Permission.ASSIGN_CASES,
        Permission.VIEW_RECOMMENDATIONS,
        Permission.CREATE_INTERVENTIONS,
        Permission.UPDATE_INTERVENTIONS,
        Permission.BATCH_PROCESSING,
        Permission.EXPORT_DATA,
        Permission.QUERY_DATABASE,
    ],
    'teacher': [
        # Teachers have viewing and limited case management
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILES,
        Permission.VIEW_RISK_ASSESSMENT,
        Permission.VIEW_NEEDS_OVERVIEW,
        Permission.VIEW_METRICS,
        Permission.VIEW_CASES,
        Permission.CREATE_CASES,
        Permission.VIEW_RECOMMENDATIONS,
        Permission.EXPORT_DATA,
    ],
    'viewer': [
        # Viewers have read-only access
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILES,
        Permission.VIEW_RISK_ASSESSMENT,
        Permission.VIEW_NEEDS_OVERVIEW,
        Permission.VIEW_METRICS,
        Permission.VIEW_CASES,
        Permission.VIEW_RECOMMENDATIONS,
    ]
}

# Page-Permission mapping
PAGE_PERMISSIONS = {
    'Dashboard': [Permission.VIEW_DASHBOARD],
    'Risk Assessment': [Permission.VIEW_RISK_ASSESSMENT],
    'Needs Overview': [Permission.VIEW_NEEDS_OVERVIEW],
    'Upload New Cases': [Permission.UPLOAD_CASES],
    'Model Metrics': [Permission.VIEW_METRICS],
    'Case Management': [Permission.VIEW_CASES],
    'Batch Processing': [Permission.BATCH_PROCESSING],
}

def get_user_permissions(role: str) -> List[str]:
    """Get list of permissions for a role."""
    return ROLE_PERMISSIONS.get(role, [])

def has_permission(user_role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    permissions = get_user_permissions(user_role)
    return permission in permissions

def can_access_page(user_role: str, page_name: str) -> bool:
    """Check if a role can access a specific page."""
    required_permissions = PAGE_PERMISSIONS.get(page_name, [])
    if not required_permissions:
        return True  # Page has no restrictions
    
    user_permissions = get_user_permissions(user_role)
    return any(perm in user_permissions for perm in required_permissions)

def get_accessible_pages(user_role: str) -> List[str]:
    """Get list of pages accessible to a role."""
    accessible = []
    for page_name in PAGE_PERMISSIONS.keys():
        if can_access_page(user_role, page_name):
            accessible.append(page_name)
    return accessible

def filter_pages_by_role(pages: List[str], user_role: str) -> List[str]:
    """Filter page list based on user role permissions."""
    return [page for page in pages if can_access_page(user_role, page)]

def check_feature_access(feature_name: str, permission: str) -> bool:
    """
    Check if user has access to a feature and show UI accordingly.
    
    Returns True if accessible, False otherwise.
    Shows appropriate error messages if not accessible.
    """
    if not st.session_state.get('authenticated', False):
        st.warning(f"🔒 Please login to access {feature_name}")
        return False
    
    user = st.session_state.get('user')
    if not user:
        st.error("User session not found")
        return False
    
    if not has_permission(user['role'], permission):
        st.error(f"🚫 You don't have permission to access {feature_name}")
        st.caption(f"Required permission: {permission}")
        return False
    
    return True

def show_permission_denied(required_permission: str = None):
    """Display permission denied message."""
    st.error("🚫 Access Denied")
    
    user = st.session_state.get('user')
    if user:
        st.warning(f"Your role ({user['role'].replace('_', ' ').title()}) doesn't have access to this feature.")
    
    if required_permission:
        st.info(f"Required permission: {required_permission}")

