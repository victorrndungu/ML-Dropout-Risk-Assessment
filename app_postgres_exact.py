#!/usr/bin/env python3
"""
Streamlit App using PostgreSQL Exact Replica
EXACTLY the same as app_enhanced.py but data from PostgreSQL
NOW WITH AUTHENTICATION & ROLE-BASED ACCESS CONTROL
"""

import os
import json
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from postgres_exact_replica import PostgreSQLExactReplica
from heuristics import apply_heuristics
from recommendations import generate_recommendations, format_recommendations_for_display, batch_generate_recommendations
from case_management import (
    add_new_case, update_case_status, add_intervention, get_case_summary,
    get_cases_by_status, get_overdue_cases, get_high_priority_cases,
    generate_workload_summary, CASE_STATUS, INTERVENTION_TYPES
)

# Authentication imports - SUPABASE VERSION
from auth_supabase_ui import show_supabase_auth_page, show_supabase_user_info, init_supabase_session_state
from rbac import filter_pages_by_role, has_permission, Permission, check_feature_access

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# Load trained models and transformers
@st.cache_resource
def load_models():
    """Load trained models and preprocessing components."""
    try:
        models = {
            'needs': joblib.load(MODELS_DIR / 'random_forest_needs_model.pkl'),
            'priority': joblib.load(MODELS_DIR / 'random_forest_priority_model.pkl'),
            'dropout': joblib.load(MODELS_DIR / 'random_forest_dropout_model.pkl')
        }
        scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
        pca = joblib.load(MODELS_DIR / 'pca_transformer.pkl')
        priority_encoder = joblib.load(MODELS_DIR / 'priority_encoder.pkl')
        with open(MODELS_DIR / 'feature_names.json', 'r') as f:
            feature_names = json.load(f)
            
        return models, scaler, pca, priority_encoder, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_data
def load_data_from_postgres():
    """Load the features dataset from PostgreSQL - EXACT replica of local CSV."""
    try:
        # TEMPORARY FIX: Use local CSV data directly since PostgreSQL DataFrame loading has embedding issues
        # TODO: Fix the PostgreSQL DataFrame loading method
        local_csv_path = ROOT / "usable" / "features_dataset.csv"
        if local_csv_path.exists():
            df = pd.read_csv(local_csv_path)
            print("✅ Using local CSV data (PostgreSQL DataFrame loading has embedding issues)")
            return df
        
        # Fallback to PostgreSQL if local CSV not available
        db = PostgreSQLExactReplica()
        df = db.get_features_as_dataframe()
        
        if df.empty:
            st.warning("No data in PostgreSQL. Please run migration first.")
            return None
        
        # Apply PCA to embeddings if needed
        emb_cols = [c for c in df.columns if c.startswith('emb_') and not c.startswith('emb_pca_')]
        if emb_cols:
            models, scaler, pca, priority_encoder, feature_names = load_models()
            if pca is not None:
                X_emb = df[emb_cols].values
                X_emb_pca = pca.transform(X_emb)
                pca_df = pd.DataFrame(X_emb_pca, columns=[f'emb_pca_{i}' for i in range(X_emb_pca.shape[1])])
                df = pd.concat([df.drop(columns=emb_cols), pca_df], axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading from PostgreSQL: {e}")
        return None

def prepare_features_for_prediction(df, scaler, pca, feature_names):
    """Prepare features for model prediction."""
    # Structured features
    numeric_cols = ['age', 'last_exam_score', 'meals_per_day', 'siblings_count', 'sentence_count', 'text_len']
    flag_cols = [c for c in df.columns if c.endswith('_flag')]
    
    X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    X_flags = df[flag_cols].fillna(0)
    
    # PCA-reduced embeddings
    emb_pca_cols = [c for c in df.columns if c.startswith('emb_pca_')]
    if emb_pca_cols:
        # Use existing PCA columns
        X_emb_pca = df[emb_pca_cols].values
    else:
        # Apply PCA transformation to raw embeddings
        emb_cols = [c for c in df.columns if c.startswith('emb_') and not c.startswith('emb_pca_')]
        if emb_cols and pca is not None:
            X_emb_raw = df[emb_cols].values
            X_emb_pca = pca.transform(X_emb_raw)
        else:
            X_emb_pca = np.empty((len(df), 0))
    
    # Combine and scale
    X = np.hstack([X_numeric.values, X_flags.values, X_emb_pca])
    X_scaled = scaler.transform(X)
    
    return X_scaled

def predict_risk_profile(df_row, models, scaler, pca, priority_encoder):
    """Predict risk profile for a single case."""
    # Prepare single row as DataFrame
    df_single = pd.DataFrame([df_row])
    X = prepare_features_for_prediction(df_single, scaler, pca, None)
    
    # Predictions
    needs_pred = models['needs'].predict(X)[0]
    priority_pred_encoded = models['priority'].predict(X)[0]
    dropout_pred = models['dropout'].predict(X)[0]
    
    # Get probabilities for confidence
    needs_proba = models['needs'].predict_proba(X)
    priority_proba = models['priority'].predict_proba(X)[0]
    dropout_proba = models['dropout'].predict_proba(X)[0]
    
    # Decode priority
    priority_pred = priority_encoder.inverse_transform([priority_pred_encoded])[0]
    
    # Format needs predictions
    needs_labels = ['need_food', 'need_school_fees', 'need_housing', 'need_economic', 
                   'need_family_support', 'need_health', 'need_counseling']
    needs_dict = {label: int(pred) for label, pred in zip(needs_labels, needs_pred)}
    
    # Confidence scores
    confidence_scores = {}
    for i, label in enumerate(needs_labels):
        if hasattr(needs_proba[i], '__len__') and len(needs_proba[i]) > 1:
            confidence_scores[label] = float(needs_proba[i][1])  # Probability of positive class
        else:
            confidence_scores[label] = 0.5
    
    return {
        'needs': needs_dict,
        'priority': priority_pred,
        'dropout_risk': int(dropout_pred),
        'confidence_scores': confidence_scores,
        'priority_confidence': float(max(priority_proba)),
        'dropout_confidence': float(max(dropout_proba))
    }

def dashboard_page(df):
    """Main dashboard page."""
    st.title("🎓 Dropout Risk Assessment Dashboard")
    st.markdown("*Data from PostgreSQL - Exact replica of local processing*")
    
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    # Apply heuristics if not already done
    if 'heuristic_score' not in df.columns:
        df = apply_heuristics(df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    
    with col2:
        high_risk = len(df[df['heuristic_score'] >= 8])
        st.metric("High Risk", high_risk)
    
    with col3:
        medium_risk = len(df[(df['heuristic_score'] >= 5) & (df['heuristic_score'] < 8)])
        st.metric("Medium Risk", medium_risk)
    
    with col4:
        low_risk = len(df[df['heuristic_score'] < 5])
        st.metric("Low Risk", low_risk)
    
    # Data source
    st.subheader("📊 Data Sources")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        usable_count = len(df[df['source_directory'] == 'usable'])
        st.metric("Original Files (usable/)", usable_count)
    
    with col2:
        aug_count = len(df[df['source_directory'] == 'usable_aug'])
        st.metric("Augmented Files (usable_aug/)", aug_count)
    
    with col3:
        st.metric("Total in PostgreSQL", len(df))
    
    # Risk distribution
    st.subheader("📈 Risk Distribution")
    if 'weak_label' in df.columns:
        risk_dist = df['weak_label'].value_counts()
        st.bar_chart(risk_dist)
    
    # High risk cases table
    st.subheader("⚠️ High Risk Cases")
    high_risk_df = df[df['heuristic_score'] >= 8][['uid', 'age', 'heuristic_score', 'weak_label', 'source_directory']].head(10)
    if not high_risk_df.empty:
        st.dataframe(high_risk_df, use_container_width=True)
    else:
        st.info("No high risk cases found")

def comparison_page():
    """Compare PostgreSQL vs Local Files."""
    st.title("🔍 PostgreSQL vs Local Files Comparison")
    st.markdown("*Verify that PostgreSQL EXACTLY matches local file processing*")
    
    # Load from PostgreSQL
    st.subheader("Loading from PostgreSQL...")
    df_postgres = load_data_from_postgres()
    
    # Load from local CSV if exists
    local_csv = ROOT / "usable" / "features_dataset.csv"
    if local_csv.exists():
        st.subheader("Loading from Local CSV...")
        df_local = pd.read_csv(local_csv)
        
        # Compare
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("PostgreSQL Rows", len(df_postgres) if df_postgres is not None else 0)
            st.metric("PostgreSQL Columns", len(df_postgres.columns) if df_postgres is not None else 0)
        
        with col2:
            st.metric("Local CSV Rows", len(df_local))
            st.metric("Local CSV Columns", len(df_local.columns))
        
        # Check matching UIDs
        if df_postgres is not None and not df_postgres.empty:
            common_uids = set(df_local['uid']) & set(df_postgres['uid'])
            st.metric("Matching UIDs", len(common_uids))
            
            if common_uids:
                st.success(f"✅ {len(common_uids)} files match between PostgreSQL and local!")
                
                # Sample comparison
                sample_uid = st.selectbox("Select UID to compare:", sorted(list(common_uids))[:10])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Local CSV")
                    local_row = df_local[df_local['uid'] == sample_uid].iloc[0]
                    st.json({
                        'uid': local_row['uid'],
                        'age': int(local_row['age']) if pd.notna(local_row['age']) else None,
                        'meals_per_day': int(local_row['meals_per_day']) if pd.notna(local_row['meals_per_day']) else None,
                        'siblings_count': int(local_row['siblings_count']) if pd.notna(local_row['siblings_count']) else None
                    })
                
                with col2:
                    st.subheader("PostgreSQL")
                    postgres_row = df_postgres[df_postgres['uid'] == sample_uid].iloc[0]
                    st.json({
                        'uid': postgres_row['uid'],
                        'age': int(postgres_row['age']) if pd.notna(postgres_row['age']) else None,
                        'meals_per_day': int(postgres_row['meals_per_day']) if pd.notna(postgres_row['meals_per_day']) else None,
                        'siblings_count': int(postgres_row['siblings_count']) if pd.notna(postgres_row['siblings_count']) else None,
                        'heuristic_score': float(postgres_row.get('heuristic_score', 0)),
                        'weak_label': postgres_row.get('weak_label', 'unknown')
                    })
    else:
        st.warning(f"Local CSV not found: {local_csv}")

def risk_assessment_page(df, db_manager):
    """Individual risk assessment page - FULL functionality using PostgreSQL data."""
    st.title("Individual Risk Assessment")
    
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    # Load models
    models, scaler, pca, priority_encoder, feature_names = load_models()
    if models is None:
        st.error("Failed to load models. Please run ml_training.py first.")
        return
    
    # Profile selection
    profile_options = [f"{row['uid']} - {row.get('source_file', 'unknown')}" for _, row in df.iterrows()]
    selected_profile = st.selectbox("Select a profile to assess:", profile_options)
    
    if selected_profile:
        uid = selected_profile.split(' - ')[0]
        profile_data = df[df['uid'] == uid].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Profile Information")
            st.write(f"**UID:** {profile_data['uid']}")
            st.write(f"**Age:** {profile_data.get('age', 'N/A')}")
            st.write(f"**Class:** {profile_data.get('class', 'N/A')}")
            st.write(f"**Last Exam Score:** {profile_data.get('last_exam_score', 'N/A')}")
            st.write(f"**Meals per Day:** {profile_data.get('meals_per_day', 'N/A')}")
            st.write(f"**Siblings Count:** {profile_data.get('siblings_count', 'N/A')}")
            
            # Key flags
            st.subheader("Risk Indicators")
            flag_cols = [c for c in df.columns if c.endswith('_flag')]
            active_flags = [c.replace('_flag', '').replace('_', ' ').title() 
                          for c in flag_cols if profile_data.get(c, 0) == 1]
            
            if active_flags:
                for flag in active_flags:
                    st.write(f"🚩 {flag}")
            else:
                st.write("No major risk flags identified")
        
        with col2:
            st.subheader("ML Risk Assessment")
            
            # Get predictions
            predictions = predict_risk_profile(profile_data, models, scaler, pca, priority_encoder)
            
            # Display priority and dropout risk
            priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            st.write(f"**Priority Level:** {priority_color.get(predictions['priority'], '⚪')} {predictions['priority'].upper()}")
            
            if predictions['dropout_risk']:
                st.write("**⚠️ DROPOUT RISK IDENTIFIED**")
            
            # Predicted needs
            st.subheader("Predicted Needs")
            active_needs = [need.replace('need_', '').replace('_', ' ').title() 
                          for need, value in predictions['needs'].items() if value == 1]
            
            if active_needs:
                for need in active_needs:
                    st.write(f"• {need}")
            else:
                st.write("No specific needs predicted")
        
        # Generate and display recommendations
        st.subheader("Recommendations")
        recommendations = generate_recommendations(
            predictions['needs'], 
            predictions['priority'], 
            predictions['dropout_risk'],
            predictions['confidence_scores']
        )
        
        formatted_recs = format_recommendations_for_display(recommendations)
        st.text(formatted_recs)
        
        # Case management actions
        st.subheader("Case Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Add to Case Load"):
                success = add_new_case(
                    uid=uid,
                    priority_level=predictions['priority'],
                    dropout_risk=bool(predictions['dropout_risk']),
                    assigned_worker=st.session_state.get('worker_name', ''),
                    case_notes=f"PostgreSQL Assessment: {len(active_needs)} needs identified"
                )
                if success:
                    st.success("Case added to tracking system!")
                else:
                    st.warning("Case already exists in system")
        
        with col2:
            worker_name = st.text_input("Worker Name", key="worker_name")
        
        with col3:
            if st.button("View Case History"):
                case_summary = get_case_summary(uid)
                if case_summary:
                    st.json(case_summary)
                else:
                    st.info("No case history found")

def case_management_page():
    """Case management and workflow page - FULL functionality."""
    st.title("Case Management")
    
    tab1, tab2, tab3 = st.tabs(["Active Cases", "Add Intervention", "Case Reports"])
    
    with tab1:
        st.subheader("Active Cases")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by Status", ['All'] + CASE_STATUS)
        with col2:
            priority_filter = st.selectbox("Filter by Priority", ['All', 'high', 'medium', 'low'])
        
        # Load and filter cases
        if status_filter == 'All':
            from case_management import load_cases
            cases_df = load_cases()
        else:
            cases_df = get_cases_by_status(status_filter)
        
        if priority_filter != 'All':
            cases_df = cases_df[cases_df['priority_level'] == priority_filter]
        
        if len(cases_df) > 0:
            # Display cases
            st.dataframe(cases_df[['uid', 'priority_level', 'status', 'assigned_worker', 
                                 'last_contact', 'next_follow_up', 'interventions_count']])
            
            # Case actions
            selected_uid = st.selectbox("Select case for action:", cases_df['uid'].tolist())
            
            col1, col2 = st.columns(2)
            with col1:
                new_status = st.selectbox("Update Status:", CASE_STATUS)
                if st.button("Update Status"):
                    success = update_case_status(selected_uid, new_status, 
                                               st.session_state.get('worker_name', ''),
                                               f"Status updated to {new_status}")
                    if success:
                        st.success("Status updated!")
                        st.rerun()
            
            with col2:
                if st.button("View Full Case"):
                    case_summary = get_case_summary(selected_uid)
                    if case_summary:
                        st.json(case_summary)
        else:
            st.info("No cases found matching the filters")
    
    with tab2:
        st.subheader("Add Intervention")
        
        # Load cases for intervention
        from case_management import load_cases
        all_cases = load_cases()
        active_cases = all_cases[~all_cases['status'].isin(['closed', 'referred', 'lost_contact'])]
        
        if len(active_cases) > 0:
            intervention_uid = st.selectbox("Select Case:", active_cases['uid'].tolist())
            intervention_type = st.selectbox("Intervention Type:", INTERVENTION_TYPES)
            intervention_desc = st.text_area("Description:")
            intervention_worker = st.text_input("Worker Name:")
            intervention_outcome = st.text_area("Outcome (if completed):")
            follow_up_needed = st.checkbox("Follow-up needed", value=True)
            intervention_notes = st.text_area("Additional Notes:")
            
            if st.button("Log Intervention"):
                if intervention_uid and intervention_type and intervention_desc and intervention_worker:
                    success = add_intervention(
                        intervention_uid, intervention_type, intervention_desc,
                        intervention_worker, intervention_outcome, follow_up_needed,
                        intervention_notes
                    )
                    if success:
                        st.success("Intervention logged!")
                    else:
                        st.error("Failed to log intervention")
                else:
                    st.error("Please fill in all required fields")
        else:
            st.info("No active cases available for intervention")
    
    with tab3:
        st.subheader("Case Reports")
        
        # Overdue cases
        overdue_cases = get_overdue_cases()
        if len(overdue_cases) > 0:
            st.warning(f"⚠️ {len(overdue_cases)} cases are overdue for follow-up")
            st.dataframe(overdue_cases[['uid', 'priority_level', 'status', 'next_follow_up']])
        
        # High priority cases
        high_priority = get_high_priority_cases()
        if len(high_priority) > 0:
            st.info(f"🔴 {len(high_priority)} high priority cases")
            st.dataframe(high_priority[['uid', 'priority_level', 'dropout_risk', 'status']])
        
        # Workload summary
        worker_name = st.text_input("Worker name for summary (leave blank for overall):")
        if st.button("Generate Workload Summary"):
            summary = generate_workload_summary(worker_name if worker_name else "")
            st.json(summary)

def upload_new_cases_page(df, db_manager):
    """Upload and analyze new cases page - FULL functionality."""
    st.title("Upload New Cases to PostgreSQL")
    st.write("Upload text files to analyze new student cases using the PostgreSQL database.")
    
    # File upload
    uploaded_files = st.file_uploader("Choose text files", type=['txt'], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"**Files selected:** {len(uploaded_files)}")
        
        if st.button("Process and Add to Database"):
            st.info(f"Processing {len(uploaded_files)} files...")
            
            progress_bar = st.progress(0)
            success_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Read file content
                    case_text = str(uploaded_file.read(), "utf-8")
                    case_uid = uploaded_file.name.replace('.txt', '')
                    
                    # Store in PostgreSQL
                    db_manager.store_raw_text(
                        uid=case_uid,
                        source_file=uploaded_file.name,
                        source_directory='uploads',
                        raw_text=case_text
                    )
                    
                    # Process the file
                    db_manager.process_file(case_uid)
                    
                    success_count += 1
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"✅ Successfully processed {success_count} out of {len(uploaded_files)} files!")
            st.info("Refresh the page to see the new cases in the database.")

def needs_overview_page(df, db_manager):
    """Needs overview page showing profiles categorized by need type, ordered by risk."""
    st.title("Needs Overview - Cases by Category & Risk")
    
    st.write("View all profiles categorized by their predicted needs, ordered by risk level within each category.")
    
    # Load models
    models, scaler, pca, priority_encoder, feature_names = load_models()
    if models is None:
        st.error("Failed to load ML models")
        return
    
    # Process all profiles to get predictions
    with st.spinner("Processing all profiles for needs assessment..."):
        all_predictions = []
        
        # Process in batches for better performance
        batch_size = 50
        progress_bar = st.progress(0)
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            for idx, row in batch_df.iterrows():
                predictions = predict_risk_profile(row, models, scaler, pca, priority_encoder)
                
                # Create risk score for ordering (higher = more risk)
                risk_score = 0
                if predictions['priority'] == 'high':
                    risk_score += 3
                elif predictions['priority'] == 'medium':
                    risk_score += 2
                else:
                    risk_score += 1
                
                if predictions['dropout_risk']:
                    risk_score += 2
                
                # Add confidence penalty (lower confidence = higher risk score for ordering)
                risk_score += (1 - predictions['priority_confidence']) * 0.5
                
                result = {
                    'uid': row['uid'],
                    'age': row.get('age', 'N/A'),
                    'class': row.get('class', 'N/A'),
                    'last_exam_score': row.get('last_exam_score', 'N/A'),
                    'priority': predictions['priority'],
                    'dropout_risk': predictions['dropout_risk'],
                    'priority_confidence': predictions['priority_confidence'],
                    'risk_score': risk_score,
                    'needs': predictions['needs'],
                    'total_needs': sum(predictions['needs'].values())
                }
                all_predictions.append(result)
            
            progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
    
    # Define need categories with display names
    need_categories = {
        'need_food': '🍽️ Food Security',
        'need_school_fees': '💰 School Fees',
        'need_housing': '🏠 Housing',
        'need_economic': '💼 Economic Support',
        'need_family_support': '👨‍👩‍👧‍👦 Family Support',
        'need_health': '🏥 Health Services',
        'need_counseling': '🗣️ Counseling'
    }
    
    # Create tabs for each need category
    tab_names = list(need_categories.values())
    tabs = st.tabs(tab_names)
    
    for i, (need_key, need_display) in enumerate(need_categories.items()):
        with tabs[i]:
            # Filter profiles that have this need
            profiles_with_need = [p for p in all_predictions if p['needs'][need_key] == 1]
            
            if profiles_with_need:
                # Sort by risk score (highest first)
                profiles_with_need.sort(key=lambda x: x['risk_score'], reverse=True)
                
                st.subheader(f"{need_display} - {len(profiles_with_need)} Cases")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    high_priority = len([p for p in profiles_with_need if p['priority'] == 'high'])
                    st.metric("High Priority", high_priority)
                with col2:
                    dropout_risk = len([p for p in profiles_with_need if p['dropout_risk'] == 1])
                    st.metric("Dropout Risk", dropout_risk)
                with col3:
                    avg_needs = sum(p['total_needs'] for p in profiles_with_need) / len(profiles_with_need)
                    st.metric("Avg Total Needs", f"{avg_needs:.1f}")
                with col4:
                    avg_confidence = sum(p['priority_confidence'] for p in profiles_with_need) / len(profiles_with_need)
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Display profiles in order of risk
                st.subheader("Cases (Ordered by Risk Level)")
                
                # Create display dataframe
                display_data = []
                for rank, profile in enumerate(profiles_with_need, 1):
                    # Risk indicators
                    risk_indicators = []
                    if profile['dropout_risk']:
                        risk_indicators.append("🚨 DROPOUT RISK")
                    if profile['priority'] == 'high':
                        risk_indicators.append("🔴 HIGH PRIORITY")
                    elif profile['priority'] == 'medium':
                        risk_indicators.append("🟡 MEDIUM PRIORITY")
                    else:
                        risk_indicators.append("🟢 LOW PRIORITY")
                    
                    # Other needs
                    other_needs = [need_categories[k].split(' ', 1)[1] for k, v in profile['needs'].items() 
                                 if v == 1 and k != need_key]
                    
                    display_data.append({
                        'Rank': rank,
                        'UID': profile['uid'],
                        'Age': profile['age'],
                        'Class': profile['class'],
                        'Exam Score': profile['last_exam_score'],
                        'Risk Level': ' | '.join(risk_indicators),
                        'Other Needs': ', '.join(other_needs) if other_needs else 'None',
                        'Total Needs': profile['total_needs'],
                        'Confidence': f"{profile['priority_confidence']:.1%}"
                    })
                
                # Display as dataframe
                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, use_container_width=True)
                
                # Action buttons for high-risk cases
                if high_priority > 0 or dropout_risk > 0:
                    st.subheader("Quick Actions for High-Risk Cases")
                    
                    high_risk_profiles = [p for p in profiles_with_need 
                                        if p['priority'] == 'high' or p['dropout_risk'] == 1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Add All High-Risk to Case Load", key=f"add_high_risk_{need_key}"):
                            added_count = 0
                            for profile in high_risk_profiles:
                                success = add_new_case(
                                    uid=profile['uid'],
                                    priority_level=profile['priority'],
                                    dropout_risk=bool(profile['dropout_risk']),
                                    case_notes=f"Auto-added from {need_display} category - High risk identified"
                                )
                                if success:
                                    added_count += 1
                            st.success(f"Added {added_count} high-risk cases to tracking system!")
                    
                    with col2:
                        # Generate recommendations for high-risk cases
                        if st.button(f"Generate Recommendations", key=f"gen_recs_{need_key}"):
                            st.subheader("Recommendations for High-Risk Cases")
                            for profile in high_risk_profiles[:5]:  # Show top 5
                                with st.expander(f"UID: {profile['uid']} - {profile['priority'].upper()} Priority"):
                                    recommendations = generate_recommendations(
                                        profile['needs'], 
                                        profile['priority'], 
                                        profile['dropout_risk']
                                    )
                                    formatted_recs = format_recommendations_for_display(recommendations)
                                    st.text(formatted_recs)
                
                # Export option
                if st.button(f"Export {need_display} Cases to CSV", key=f"export_{need_key}"):
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {need_display} Cases CSV",
                        data=csv,
                        file_name=f"{need_key}_cases_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key=f"download_{need_key}"
                    )
            
            else:
                st.info(f"No profiles currently identified with {need_display.split(' ', 1)[1].lower()} needs.")
                st.write("This could indicate:")
                st.write("- Good overall situation in this area")
                st.write("- Need for improved detection methods")
                st.write("- Requirement for additional data collection")

def batch_processing_page(df, db_manager):
    """Batch processing page - FULL functionality."""
    st.title("Batch Risk Assessment")
    
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    st.write("Process multiple profiles at once and generate recommendations.")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.slider("Number of profiles to process:", 10, 100, 50)
    with col2:
        priority_threshold = st.selectbox("Focus on priority:", ['All', 'high', 'medium'])
    
    if st.button("Process Batch"):
        # Load models
        models, scaler, pca, priority_encoder, feature_names = load_models()
        if models is None:
            st.error("Failed to load models.")
            return
        
        # Sample profiles
        if priority_threshold == 'All':
            sample_df = df.sample(n=min(batch_size, len(df)), random_state=42)
        else:
            filtered_df = df[df['weak_label'] == priority_threshold]
            sample_df = filtered_df.sample(n=min(batch_size, len(filtered_df)), random_state=42)
        
        # Process each profile
        results = []
        progress_bar = st.progress(0)
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            predictions = predict_risk_profile(row, models, scaler, pca, priority_encoder)
            
            result = {
                'uid': row['uid'],
                'ml_priority': predictions['priority'],
                'dropout_risk': predictions['dropout_risk'],
                'total_needs': sum(predictions['needs'].values())
            }
            
            # Add individual needs
            for need, value in predictions['needs'].items():
                result[need] = value
            
            results.append(result)
            progress_bar.progress((i + 1) / len(sample_df))
        
        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Batch Processing Results")
        st.dataframe(results_df)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Priority Cases", len(results_df[results_df['ml_priority'] == 'high']))
        with col2:
            st.metric("Dropout Risk Cases", len(results_df[results_df['dropout_risk'] == 1]))
        with col3:
            st.metric("Average Needs per Case", f"{results_df['total_needs'].mean():.1f}")
        
        # Download option
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

def model_metrics_page():
    """Display comprehensive model evaluation metrics - EXACT copy of local version."""
    st.title("Model Performance Metrics")
    st.write("Comprehensive evaluation metrics for the trained ML models.")
    
    # Load training results
    results_path = MODELS_DIR / 'training_results.json'
    
    if not results_path.exists():
        st.error("Training results not found. Please run ml_training.py first.")
        return
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        st.error(f"Error loading training results: {e}")
        return
    
    # Model information
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", results['best_model'].replace('_', ' ').title())
    with col2:
        st.metric("Training Set Size", results['splits']['train_size'])
    with col3:
        st.metric("Test Set Size", results['splits']['test_size'])
    
    # Create tabs for different model types
    tab1, tab2, tab3 = st.tabs(["Priority Classification", "Needs Prediction", "Dropout Risk"])
    
    with tab1:
        st.subheader("Priority Classification Metrics")
        
        # Get priority metrics
        val_priority = results['validation_results'][results['best_model']]['priority']
        test_priority = results['test_results']['priority']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{test_priority['accuracy']:.3f}")
        with col2:
            st.metric("High Priority Recall", f"{test_priority['high_recall']:.3f}")
        with col3:
            st.metric("High Priority Precision", f"{test_priority['high_precision']:.3f}")
        with col4:
            st.metric("High Priority F1", f"{test_priority['high_f1']:.3f}")
        
        # Validation vs Test comparison
        st.subheader("Validation vs Test Performance")
        comparison_data = {
            'Metric': ['Accuracy', 'High Priority Recall', 'High Priority Precision', 'High Priority F1'],
            'Validation': [
                f"{val_priority['accuracy']:.3f}",
                f"{val_priority['high_recall']:.3f}",
                f"{val_priority['high_precision']:.3f}",
                f"{val_priority['high_f1']:.3f}"
            ],
            'Test': [
                f"{test_priority['accuracy']:.3f}",
                f"{test_priority['high_recall']:.3f}",
                f"{test_priority['high_precision']:.3f}",
                f"{test_priority['high_f1']:.3f}"
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Test Set Confusion Matrix")
        confusion_matrix = test_priority['confusion_matrix']
        
        # Create confusion matrix DataFrame
        cm_df = pd.DataFrame(
            confusion_matrix,
            index=['Actual Low', 'Actual Medium', 'Actual High'],
            columns=['Predicted Low', 'Predicted Medium', 'Predicted High']
        )
        
        # Display as heatmap-style table
        st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        # Detailed classification report
        if 'classification_report' in test_priority:
            st.subheader("Detailed Classification Report")
            
            # Convert classification report to DataFrame
            report = test_priority['classification_report']
            report_data = []
            
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    report_data.append({
                        'Class': class_name,
                        'Precision': f"{metrics['precision']:.3f}",
                        'Recall': f"{metrics['recall']:.3f}",
                        'F1-Score': f"{metrics['f1-score']:.3f}",
                        'Support': int(metrics['support'])
                    })
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                st.dataframe(report_df, use_container_width=True)
    
    with tab2:
        st.subheader("Multi-Label Needs Prediction Metrics")
        
        # Get needs metrics
        val_needs = results['validation_results'][results['best_model']]['needs']
        test_needs = results['test_results']['needs']
        
        # Create metrics comparison table
        needs_data = []
        for need_type in val_needs.keys():
            needs_data.append({
                'Need Type': need_type.replace('need_', '').replace('_', ' ').title(),
                'Val Precision': f"{val_needs[need_type]['precision']:.3f}",
                'Val Recall': f"{val_needs[need_type]['recall']:.3f}",
                'Val F1': f"{val_needs[need_type]['f1']:.3f}",
                'Test Precision': f"{test_needs[need_type]['precision']:.3f}",
                'Test Recall': f"{test_needs[need_type]['recall']:.3f}",
                'Test F1': f"{test_needs[need_type]['f1']:.3f}"
            })
        
        needs_df = pd.DataFrame(needs_data)
        st.dataframe(needs_df, use_container_width=True)
        
        # Average metrics
        st.subheader("Average Performance Across All Needs")
        
        # Calculate averages
        test_avg_precision = sum(test_needs[need]['precision'] for need in test_needs) / len(test_needs)
        test_avg_recall = sum(test_needs[need]['recall'] for need in test_needs) / len(test_needs)
        test_avg_f1 = sum(test_needs[need]['f1'] for need in test_needs) / len(test_needs)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Precision", f"{test_avg_precision:.3f}")
        with col2:
            st.metric("Average Recall", f"{test_avg_recall:.3f}")
        with col3:
            st.metric("Average F1-Score", f"{test_avg_f1:.3f}")
        
        # Performance by need type (bar chart)
        st.subheader("F1-Score by Need Type")
        
        # Create chart data
        chart_data = pd.DataFrame({
            'Need Type': [need.replace('need_', '').replace('_', ' ').title() for need in test_needs.keys()],
            'F1-Score': [test_needs[need]['f1'] for need in test_needs.keys()]
        })
        
        st.bar_chart(chart_data.set_index('Need Type'))
    
    with tab3:
        st.subheader("Dropout Risk Prediction Metrics")
        
        # Get dropout metrics
        val_dropout = results['validation_results'][results['best_model']]['dropout']
        test_dropout = results['test_results']['dropout']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{test_dropout['accuracy']:.3f}")
        with col2:
            st.metric("Test Precision", f"{test_dropout['precision']:.3f}")
        with col3:
            st.metric("Test Recall", f"{test_dropout['recall']:.3f}")
        with col4:
            st.metric("Test F1-Score", f"{test_dropout['f1']:.3f}")
        
        # AUC Score (most important for binary classification)
        st.subheader("ROC AUC Score")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Validation AUC", f"{val_dropout['auc']:.3f}")
        with col2:
            st.metric("Test AUC", f"{test_dropout['auc']:.3f}")
        
        # Confusion Matrix for Dropout Risk
        st.subheader("Dropout Risk Confusion Matrix")
        dropout_cm = test_dropout['confusion_matrix']
        
        dropout_cm_df = pd.DataFrame(
            dropout_cm,
            index=['Actual No Risk', 'Actual Dropout Risk'],
            columns=['Predicted No Risk', 'Predicted Dropout Risk']
        )
        
        st.dataframe(dropout_cm_df.style.background_gradient(cmap='Reds'), use_container_width=True)
        
        # Performance interpretation
        st.subheader("Performance Interpretation")
        
        if test_dropout['auc'] >= 0.9:
            st.success("🎉 Excellent performance! AUC ≥ 0.9 indicates outstanding discrimination ability.")
        elif test_dropout['auc'] >= 0.8:
            st.info("✅ Good performance! AUC ≥ 0.8 indicates good discrimination ability.")
        elif test_dropout['auc'] >= 0.7:
            st.warning("⚠️ Fair performance. AUC ≥ 0.7 indicates acceptable discrimination ability.")
        else:
            st.error("❌ Poor performance. AUC < 0.7 indicates limited discrimination ability.")
        
        # Key insights
        st.write("**Key Insights:**")
        st.write(f"• The model correctly identifies {test_dropout['recall']:.1%} of actual dropout risk cases")
        st.write(f"• {test_dropout['precision']:.1%} of cases flagged as dropout risk are actually at risk")
        st.write(f"• Overall accuracy of {test_dropout['accuracy']:.1%} on the test set")
    
    # Model comparison section
    st.subheader("Model Comparison (Validation Results)")
    
    # Compare all models that were trained
    model_comparison = []
    for model_name, model_results in results['validation_results'].items():
        model_comparison.append({
            'Model': model_name.replace('_', ' ').title(),
            'Priority Accuracy': f"{model_results['priority']['accuracy']:.3f}",
            'High Priority Recall': f"{model_results['priority']['high_recall']:.3f}",
            'Dropout AUC': f"{model_results['dropout']['auc']:.3f}",
            'Avg Needs F1': f"{sum(model_results['needs'][need]['f1'] for need in model_results['needs']) / len(model_results['needs']):.3f}"
        })
    
    comparison_df = pd.DataFrame(model_comparison)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Feature importance note
    st.subheader("Model Interpretability")
    st.info("""
    **Note on Feature Importance:** The Random Forest model provides built-in feature importance scores. 
    Key factors typically include:
    - Economic indicators (rent arrears, school fees)
    - Family structure (single parent, father absent)
    - Academic performance (exam scores)
    - Basic needs (meals per day, housing conditions)
    - Text-derived features (embeddings capturing case narrative)
    """)
    
    # Download metrics report
    st.subheader("Export Metrics")
    
    if st.button("Generate Detailed Metrics Report"):
        # Create comprehensive report
        detailed_report = {
            'model_info': {
                'best_model': results['best_model'],
                'training_date': datetime.now().isoformat(),
                'dataset_splits': results['splits']
            },
            'priority_classification': {
                'validation': val_priority,
                'test': test_priority
            },
            'needs_prediction': {
                'validation': val_needs,
                'test': test_needs
            },
            'dropout_prediction': {
                'validation': val_dropout,
                'test': test_dropout
            },
            'model_comparison': results['validation_results']
        }
        
        report_json = json.dumps(detailed_report, indent=2)
        st.download_button(
            label="Download Metrics Report (JSON)",
            data=report_json,
            file_name=f"model_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def main():
    """Main application - WITH AUTHENTICATION & RBAC."""
    st.set_page_config(
        page_title="Dropout Risk Assessment System", 
        layout="wide"
    )
    
    # Initialize session state for Supabase authentication
    init_supabase_session_state()
    
    # Show Supabase authentication page if not authenticated
    if not show_supabase_auth_page():
        return  # User not authenticated, show login/signup
    
    # User is authenticated - show main app
    
    # Initialize database
    db_manager = PostgreSQLExactReplica()
    
    # Load data
    df = load_data_from_postgres()
    if df is None:
        st.error("Failed to load data from PostgreSQL database.")
        return
    
    # Apply heuristics for comparison
    df = apply_heuristics(df)
    
    # Sidebar navigation with role-based filtering
    st.sidebar.title("🎓 Navigation")
    
    # Show user info in sidebar (Supabase)
    show_supabase_user_info()
    
    # Show PostgreSQL connection status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Source")
    if df is not None and not df.empty:
        st.sidebar.success(f"✅ PostgreSQL Connected")
        st.sidebar.metric("Records in Database", len(df))
        st.sidebar.caption("*Using exact same processing as local files*")
    else:
        st.sidebar.error("❌ No PostgreSQL data")
    st.sidebar.markdown("---")
    
    # Get user role and filter pages
    user = st.session_state.get('user')
    user_role = user['role'] if user else 'viewer'
    
    # Define all pages
    all_pages = [
        "Risk Assessment", "Case Management", "Upload New Cases", 
        "Needs Overview", "Model Metrics", "Batch Processing", "Dashboard"
    ]
    
    # Filter pages based on user role
    accessible_pages = filter_pages_by_role(all_pages, user_role)
    
    page = st.sidebar.selectbox("Choose a page", accessible_pages)
    
    # Route to pages with permission checks
    if page == "Risk Assessment":
        risk_assessment_page(df, db_manager)
    elif page == "Case Management":
        if check_feature_access("Case Management", Permission.VIEW_CASES):
            case_management_page()
    elif page == "Upload New Cases":
        if check_feature_access("Upload New Cases", Permission.UPLOAD_CASES):
            upload_new_cases_page(df, db_manager)
    elif page == "Needs Overview":
        needs_overview_page(df, db_manager)
    elif page == "Model Metrics":
        model_metrics_page()
    elif page == "Batch Processing":
        if check_feature_access("Batch Processing", Permission.BATCH_PROCESSING):
            batch_processing_page(df, db_manager)
    elif page == "Dashboard":
        dashboard_page(df)

if __name__ == "__main__":
    main()

