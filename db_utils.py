#!/usr/bin/env python3
"""
db_utils.py

Database utilities and connection helpers for the Dropout Risk Assessment system.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database_setup import Profile, ProfileEmbedding, RiskAssessment, CaseRecord, Intervention

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "db_config.json"

class DatabaseConnection:
    """Simplified database connection manager."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or CONFIG_FILE
        self.config = self.load_config()
        self.engine = None
        self.Session = None
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
        self.Session = sessionmaker(bind=self.engine)
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.Session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params=params or {})
    
    def get_profiles(self, limit: int = None, include_augmented: bool = True) -> pd.DataFrame:
        """Get profiles as DataFrame."""
        query = "SELECT * FROM profiles"
        if not include_augmented:
            query += " WHERE is_original = true"
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_profile_with_assessment(self, uid: str) -> Dict:
        """Get profile with latest risk assessment."""
        query = """
        SELECT p.*, ra.priority_level, ra.dropout_risk, ra.priority_confidence,
               ra.need_food, ra.need_school_fees, ra.need_housing, ra.need_economic,
               ra.need_family_support, ra.need_health, ra.need_counseling,
               ra.assessment_date
        FROM profiles p
        LEFT JOIN risk_assessments ra ON p.id = ra.profile_id
        WHERE p.uid = %(uid)s
        ORDER BY ra.assessment_date DESC
        LIMIT 1
        """
        
        result = self.execute_query(query, {'uid': uid})
        return result.iloc[0].to_dict() if len(result) > 0 else None
    
    def get_high_risk_profiles(self) -> pd.DataFrame:
        """Get all high-risk profiles with latest assessments."""
        query = """
        SELECT p.uid, p.age, p.last_exam_score, p.meals_per_day,
               ra.priority_level, ra.dropout_risk, ra.priority_confidence, ra.assessment_date
        FROM profiles p
        JOIN risk_assessments ra ON p.id = ra.profile_id
        WHERE ra.priority_level = 'high' OR ra.dropout_risk = true
        ORDER BY ra.assessment_date DESC
        """
        
        return self.execute_query(query)
    
    def get_cases_by_status(self, status: str) -> pd.DataFrame:
        """Get cases by status."""
        query = """
        SELECT cr.*, p.uid, p.age, p.last_exam_score
        FROM case_records cr
        JOIN profiles p ON cr.profile_id = p.id
        WHERE cr.status = %(status)s
        ORDER BY cr.created_at ASC
        """
        
        return self.execute_query(query, {'status': status})
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics."""
        stats = {}
        
        # Profile counts
        stats['total_profiles'] = self.execute_query(
            "SELECT COUNT(*) as count FROM profiles"
        ).iloc[0]['count']
        
        # Since we don't have is_original column, we'll use total profiles
        stats['original_profiles'] = stats['total_profiles']
        
        # Risk assessment counts
        risk_counts = self.execute_query("""
            SELECT priority_level, COUNT(*) as count
            FROM risk_assessments
            GROUP BY priority_level
        """)
        stats['risk_distribution'] = dict(zip(risk_counts['priority_level'], risk_counts['count']))
        
        # Case management stats
        case_counts = self.execute_query("""
            SELECT status, COUNT(*) as count
            FROM case_records
            GROUP BY status
        """)
        stats['case_status_distribution'] = dict(zip(case_counts['status'], case_counts['count']))
        
        return stats
    
    def save_risk_assessment(self, profile_uid: str, assessment_data: Dict) -> bool:
        """Save risk assessment to database."""
        try:
            with self.get_session() as session:
                # Get profile
                profile = session.query(Profile).filter_by(uid=profile_uid).first()
                if not profile:
                    return False
                
                # Create assessment
                assessment = RiskAssessment(
                    profile_id=profile.id,
                    priority_level=assessment_data.get('priority_level'),
                    priority_confidence=assessment_data.get('priority_confidence', 0.0),
                    dropout_risk=assessment_data.get('dropout_risk', False),
                    dropout_confidence=assessment_data.get('dropout_confidence', 0.0),
                    need_food=assessment_data.get('needs', {}).get('need_food', False),
                    need_school_fees=assessment_data.get('needs', {}).get('need_school_fees', False),
                    need_housing=assessment_data.get('needs', {}).get('need_housing', False),
                    need_economic=assessment_data.get('needs', {}).get('need_economic', False),
                    need_family_support=assessment_data.get('needs', {}).get('need_family_support', False),
                    need_health=assessment_data.get('needs', {}).get('need_health', False),
                    need_counseling=assessment_data.get('needs', {}).get('need_counseling', False),
                    confidence_scores=assessment_data.get('confidence_scores', {}),
                    model_version=assessment_data.get('model_version', '1.0'),
                    model_type=assessment_data.get('model_type', 'enhanced')
                )
                
                session.add(assessment)
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error saving assessment: {e}")
            return False
    
    def create_case(self, profile_uid: str, case_data: Dict) -> bool:
        """Create new case record."""
        try:
            with self.get_session() as session:
                # Get profile
                profile = session.query(Profile).filter_by(uid=profile_uid).first()
                if not profile:
                    return False
                
                # Check if case already exists
                existing_case = session.query(CaseRecord).filter_by(profile_id=profile.id).first()
                if existing_case:
                    return False  # Case already exists
                
                # Create case
                case = CaseRecord(
                    profile_id=profile.id,
                    status=case_data.get('status', 'new'),
                    assigned_worker=case_data.get('assigned_worker', ''),
                    case_notes=case_data.get('case_notes', '')
                )
                
                session.add(case)
                session.commit()
                return True
                
        except Exception as e:
            print(f"Error creating case: {e}")
            return False

# Global database connection instance
_db_connection = None

def get_db_connection() -> DatabaseConnection:
    """Get or create database connection."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection

def query_profiles(query: str, params: Dict = None) -> pd.DataFrame:
    """Convenience function to query profiles."""
    db = get_db_connection()
    return db.execute_query(query, params)

def get_profile_by_uid(uid: str) -> Dict:
    """Get single profile by UID."""
    db = get_db_connection()
    return db.get_profile_with_assessment(uid)
