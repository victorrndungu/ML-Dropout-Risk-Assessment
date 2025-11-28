#!/usr/bin/env python3
"""
case_management.py

Case management system for tracking case status, interventions, and outcomes.
Provides persistent storage in CSV format for social worker workflow.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

ROOT = Path(__file__).parent
CASES_DIR = ROOT / "case_data"
CASES_DIR.mkdir(parents=True, exist_ok=True)

CASES_FILE = CASES_DIR / "case_tracking.csv"
INTERVENTIONS_FILE = CASES_DIR / "interventions_log.csv"

# Case status options
CASE_STATUS = [
    'new',           # Newly identified case
    'contacted',     # Initial contact made
    'assessed',      # Full assessment completed
    'in_progress',   # Interventions in progress
    'monitoring',    # Regular monitoring phase
    'closed',        # Case closed successfully
    'referred',      # Referred to other services
    'lost_contact'   # Unable to maintain contact
]

# Intervention types
INTERVENTION_TYPES = [
    'food_support', 'school_fees', 'housing_assistance', 'economic_support',
    'family_counseling', 'health_services', 'individual_counseling',
    'academic_support', 'referral', 'follow_up', 'assessment', 'other'
]

def initialize_case_files():
    """Initialize case tracking files if they don't exist."""
    if not CASES_FILE.exists():
        cases_df = pd.DataFrame(columns=[
            'uid', 'date_identified', 'priority_level', 'dropout_risk',
            'status', 'assigned_worker', 'last_contact', 'next_follow_up',
            'interventions_count', 'case_notes', 'outcome'
        ])
        cases_df.to_csv(CASES_FILE, index=False)
    
    if not INTERVENTIONS_FILE.exists():
        interventions_df = pd.DataFrame(columns=[
            'uid', 'date', 'intervention_type', 'description',
            'worker', 'outcome', 'follow_up_needed', 'notes'
        ])
        interventions_df.to_csv(INTERVENTIONS_FILE, index=False)

def load_cases() -> pd.DataFrame:
    """Load existing cases from CSV."""
    initialize_case_files()
    return pd.read_csv(CASES_FILE)

def load_interventions() -> pd.DataFrame:
    """Load interventions log from CSV."""
    initialize_case_files()
    return pd.read_csv(INTERVENTIONS_FILE)

def add_new_case(uid: str, priority_level: str, dropout_risk: bool,
                assigned_worker: str = "", case_notes: str = "") -> bool:
    """Add a new case to the tracking system."""
    cases_df = load_cases()
    
    # Check if case already exists
    if uid in cases_df['uid'].values:
        return False
    
    # Calculate next follow-up based on priority
    if priority_level == 'high' or dropout_risk:
        next_follow_up = datetime.now() + timedelta(days=3)
    elif priority_level == 'medium':
        next_follow_up = datetime.now() + timedelta(days=7)
    else:
        next_follow_up = datetime.now() + timedelta(days=14)
    
    new_case = {
        'uid': uid,
        'date_identified': datetime.now().strftime('%Y-%m-%d'),
        'priority_level': priority_level,
        'dropout_risk': dropout_risk,
        'status': 'new',
        'assigned_worker': assigned_worker,
        'last_contact': '',
        'next_follow_up': next_follow_up.strftime('%Y-%m-%d'),
        'interventions_count': 0,
        'case_notes': case_notes,
        'outcome': ''
    }
    
    cases_df = pd.concat([cases_df, pd.DataFrame([new_case])], ignore_index=True)
    cases_df.to_csv(CASES_FILE, index=False)
    return True

def update_case_status(uid: str, new_status: str, worker: str = "",
                      notes: str = "") -> bool:
    """Update case status and add notes."""
    if new_status not in CASE_STATUS:
        return False
    
    cases_df = load_cases()
    case_idx = cases_df[cases_df['uid'] == uid].index
    
    if len(case_idx) == 0:
        return False
    
    idx = case_idx[0]
    cases_df.loc[idx, 'status'] = new_status
    cases_df.loc[idx, 'last_contact'] = datetime.now().strftime('%Y-%m-%d')
    
    if worker:
        cases_df.loc[idx, 'assigned_worker'] = worker
    
    if notes:
        existing_notes = cases_df.loc[idx, 'case_notes']
        if pd.isna(existing_notes) or existing_notes == '':
            cases_df.loc[idx, 'case_notes'] = notes
        else:
            cases_df.loc[idx, 'case_notes'] = f"{existing_notes}; {notes}"
    
    # Update next follow-up based on new status
    if new_status in ['new', 'contacted']:
        next_follow_up = datetime.now() + timedelta(days=7)
    elif new_status in ['assessed', 'in_progress']:
        next_follow_up = datetime.now() + timedelta(days=14)
    elif new_status == 'monitoring':
        next_follow_up = datetime.now() + timedelta(days=30)
    else:  # closed, referred, lost_contact
        next_follow_up = None
    
    if next_follow_up:
        cases_df.loc[idx, 'next_follow_up'] = next_follow_up.strftime('%Y-%m-%d')
    else:
        cases_df.loc[idx, 'next_follow_up'] = ''
    
    cases_df.to_csv(CASES_FILE, index=False)
    return True

def add_intervention(uid: str, intervention_type: str, description: str,
                    worker: str, outcome: str = "", follow_up_needed: bool = True,
                    notes: str = "") -> bool:
    """Log an intervention for a case."""
    if intervention_type not in INTERVENTION_TYPES:
        return False
    
    interventions_df = load_interventions()
    
    new_intervention = {
        'uid': uid,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'intervention_type': intervention_type,
        'description': description,
        'worker': worker,
        'outcome': outcome,
        'follow_up_needed': follow_up_needed,
        'notes': notes
    }
    
    interventions_df = pd.concat([interventions_df, pd.DataFrame([new_intervention])], ignore_index=True)
    interventions_df.to_csv(INTERVENTIONS_FILE, index=False)
    
    # Update intervention count in cases
    cases_df = load_cases()
    case_idx = cases_df[cases_df['uid'] == uid].index
    if len(case_idx) > 0:
        idx = case_idx[0]
        current_count = cases_df.loc[idx, 'interventions_count']
        if pd.isna(current_count):
            current_count = 0
        cases_df.loc[idx, 'interventions_count'] = int(current_count) + 1
        cases_df.to_csv(CASES_FILE, index=False)
    
    return True

def get_case_summary(uid: str) -> Optional[Dict]:
    """Get comprehensive case summary including interventions."""
    cases_df = load_cases()
    interventions_df = load_interventions()
    
    case_data = cases_df[cases_df['uid'] == uid]
    if len(case_data) == 0:
        return None
    
    case = case_data.iloc[0].to_dict()
    
    # Get interventions for this case
    case_interventions = interventions_df[interventions_df['uid'] == uid]
    case['interventions'] = case_interventions.to_dict('records')
    
    return case

def get_cases_by_status(status: str) -> pd.DataFrame:
    """Get all cases with a specific status."""
    cases_df = load_cases()
    return cases_df[cases_df['status'] == status]

def get_overdue_cases() -> pd.DataFrame:
    """Get cases that are overdue for follow-up."""
    cases_df = load_cases()
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Filter cases with next_follow_up date that has passed
    overdue = cases_df[
        (cases_df['next_follow_up'] != '') & 
        (cases_df['next_follow_up'] < today) &
        (~cases_df['status'].isin(['closed', 'referred', 'lost_contact']))
    ]
    
    return overdue.sort_values('next_follow_up')

def get_high_priority_cases() -> pd.DataFrame:
    """Get all high priority and dropout risk cases."""
    cases_df = load_cases()
    high_priority = cases_df[
        (cases_df['priority_level'] == 'high') | 
        (cases_df['dropout_risk'] == True)
    ]
    return high_priority.sort_values('date_identified', ascending=False)

def generate_workload_summary(worker: str = "") -> Dict:
    """Generate workload summary for a worker or overall."""
    cases_df = load_cases()
    interventions_df = load_interventions()
    
    if worker:
        worker_cases = cases_df[cases_df['assigned_worker'] == worker]
        worker_interventions = interventions_df[interventions_df['worker'] == worker]
    else:
        worker_cases = cases_df
        worker_interventions = interventions_df
    
    summary = {
        'total_cases': len(worker_cases),
        'active_cases': len(worker_cases[~worker_cases['status'].isin(['closed', 'referred', 'lost_contact'])]),
        'high_priority_cases': len(worker_cases[worker_cases['priority_level'] == 'high']),
        'dropout_risk_cases': len(worker_cases[worker_cases['dropout_risk'] == True]),
        'overdue_cases': len(get_overdue_cases() if not worker else 
                           get_overdue_cases()[get_overdue_cases()['assigned_worker'] == worker]),
        'cases_by_status': worker_cases['status'].value_counts().to_dict(),
        'interventions_this_month': len(worker_interventions[
            worker_interventions['date'] >= (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        ]),
        'intervention_types': worker_interventions['intervention_type'].value_counts().to_dict()
    }
    
    return summary

def clear_pending_cases(keep_with_interventions: bool = True) -> Tuple[bool, int, str]:
    """
    Clear pending cases that haven't had interventions or follow-ups.
    
    Args:
        keep_with_interventions: If True, only clear cases with no interventions
        
    Returns:
        Tuple of (success, count_cleared, message)
    """
    try:
        cases_df = load_cases()
        interventions_df = load_interventions()
        
        # Filter active cases
        active_cases_df = cases_df[~cases_df['status'].isin(['closed', 'referred', 'lost_contact'])]
        
        if keep_with_interventions:
            # Get cases that have interventions
            cases_with_interventions = set(interventions_df['uid'].unique())
            
            # Find cases to clear: active cases with no interventions
            cases_to_clear = active_cases_df[
                ~active_cases_df['uid'].isin(cases_with_interventions)
            ]
            
            count_to_clear = len(cases_to_clear)
            
            if count_to_clear == 0:
                return True, 0, "No pending cases without interventions to clear."
            
            # Remove these cases from the cases dataframe
            cases_df = cases_df[~cases_df['uid'].isin(cases_to_clear['uid'])]
            cases_df.to_csv(CASES_FILE, index=False)
            
            return True, count_to_clear, f"Successfully cleared {count_to_clear} pending case(s) without interventions."
        else:
            # Clear all active cases regardless of interventions
            count_to_clear = len(active_cases_df)
            
            if count_to_clear == 0:
                return True, 0, "No active cases to clear."
            
            cases_df = cases_df[cases_df['status'].isin(['closed', 'referred', 'lost_contact'])]
            cases_df.to_csv(CASES_FILE, index=False)
            
            return True, count_to_clear, f"Successfully cleared all {count_to_clear} active case(s)."
            
    except Exception as e:
        return False, 0, f"Error clearing cases: {e}"

def export_case_report(output_path: str, include_interventions: bool = True) -> bool:
    """Export comprehensive case report to CSV."""
    try:
        cases_df = load_cases()
        
        if include_interventions:
            interventions_df = load_interventions()
            
            # Create summary report with intervention counts by type
            intervention_summary = interventions_df.groupby(['uid', 'intervention_type']).size().unstack(fill_value=0)
            intervention_summary.columns = [f'interventions_{col}' for col in intervention_summary.columns]
            
            # Merge with cases
            report_df = cases_df.merge(intervention_summary, left_on='uid', right_index=True, how='left')
            report_df = report_df.fillna(0)
        else:
            report_df = cases_df
        
        report_df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error exporting report: {e}")
        return False
