#!/usr/bin/env python3
"""
database_setup.py

PostgreSQL database setup and migration script for the Dropout Risk Assessment system.
This script will:
1. Create database tables with proper schema
2. Migrate data from CSV files to PostgreSQL
3. Set up indexes for performance
4. Provide connection utilities
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Text, DateTime,
    ForeignKey, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
import psycopg2
from psycopg2 import sql

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'dropout_risk_db',
    'username': 'kahindo',
    'password': ''
}

ROOT = Path(__file__).parent
Base = declarative_base()

class Profile(Base):
    """Main profile table storing student case information."""
    __tablename__ = 'profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(100), unique=True, nullable=False, index=True)
    source_file = Column(String(255))
    text_content = Column(Text)
    text_len = Column(Integer)
    
    # Demographic information
    age = Column(Integer)
    class_level = Column(String(50))
    health_status = Column(String(255))
    
    # Academic information
    last_exam_score = Column(Integer)
    
    # Family information
    meals_per_day = Column(Integer)
    siblings_count = Column(Integer)
    siblings_list = Column(Text)
    
    # Housing conditions
    iron_sheets_flag = Column(Boolean, default=False)
    single_room_flag = Column(Boolean, default=False)
    shared_bed_flag = Column(Boolean, default=False)
    no_electric_flag = Column(Boolean, default=False)
    
    # Economic indicators
    rent_arrears_flag = Column(Boolean, default=False)
    landlord_lock_flag = Column(Boolean, default=False)
    works_unstable_flag = Column(Boolean, default=False)
    no_school_fees_flag = Column(Boolean, default=False)
    
    # Family structure
    hunger_flag = Column(Boolean, default=False)
    father_absent_flag = Column(Boolean, default=False)
    mother_hawker_flag = Column(Boolean, default=False)
    single_parent_flag = Column(Boolean, default=False)
    
    # Data quality
    leftover_phone = Column(Boolean, default=False)
    leftover_email = Column(Boolean, default=False)
    sentence_count = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_original = Column(Boolean, default=True)  # False for augmented data
    original_uid = Column(String(100), index=True)  # Reference to original for augmented data
    
    # Relationships
    embeddings = relationship("ProfileEmbedding", back_populates="profile", cascade="all, delete-orphan")
    assessments = relationship("RiskAssessment", back_populates="profile", cascade="all, delete-orphan")
    cases = relationship("CaseRecord", back_populates="profile", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_profile_uid', 'uid'),
        Index('idx_profile_original_uid', 'original_uid'),
        Index('idx_profile_flags', 'rent_arrears_flag', 'hunger_flag', 'no_school_fees_flag'),
        Index('idx_profile_demographics', 'age', 'last_exam_score'),
    )

class ProfileEmbedding(Base):
    """Store text embeddings separately for performance."""
    __tablename__ = 'profile_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey('profiles.id'), nullable=False)
    embedding_vector = Column(ARRAY(Float))  # Store as PostgreSQL array
    embedding_model = Column(String(100), default='all-MiniLM-L6-v2')
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    profile = relationship("Profile", back_populates="embeddings")
    
    # Indexes
    __table_args__ = (
        Index('idx_embedding_profile_id', 'profile_id'),
    )

class RiskAssessment(Base):
    """Store ML model predictions and risk assessments."""
    __tablename__ = 'risk_assessments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey('profiles.id'), nullable=False)
    
    # ML Predictions
    priority_level = Column(String(20))  # high, medium, low
    priority_confidence = Column(Float)
    dropout_risk = Column(Boolean, default=False)
    dropout_confidence = Column(Float)
    
    # Needs predictions (multi-label)
    need_food = Column(Boolean, default=False)
    need_school_fees = Column(Boolean, default=False)
    need_housing = Column(Boolean, default=False)
    need_economic = Column(Boolean, default=False)
    need_family_support = Column(Boolean, default=False)
    need_health = Column(Boolean, default=False)
    need_counseling = Column(Boolean, default=False)
    
    # Confidence scores for needs
    confidence_scores = Column(JSONB)  # Store as JSON
    
    # Model metadata
    model_version = Column(String(50))
    model_type = Column(String(50))  # enhanced_random_forest, etc.
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Heuristic comparison
    heuristic_score = Column(Integer)
    weak_label = Column(String(20))
    weak_confidence = Column(Float)
    
    # Relationship
    profile = relationship("Profile", back_populates="assessments")
    
    # Indexes
    __table_args__ = (
        Index('idx_assessment_profile_id', 'profile_id'),
        Index('idx_assessment_priority', 'priority_level', 'dropout_risk'),
        Index('idx_assessment_date', 'assessment_date'),
    )

class CaseRecord(Base):
    """Case management and intervention tracking."""
    __tablename__ = 'case_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey('profiles.id'), nullable=False)
    
    # Case management
    status = Column(String(50))  # new, contacted, assessed, in_progress, monitoring, closed, etc.
    assigned_worker = Column(String(255))
    date_identified = Column(DateTime, default=datetime.utcnow)
    last_contact = Column(DateTime)
    next_follow_up = Column(DateTime)
    
    # Case details
    case_notes = Column(Text)
    interventions_count = Column(Integer, default=0)
    outcome = Column(String(255))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profile = relationship("Profile", back_populates="cases")
    interventions = relationship("Intervention", back_populates="case", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_case_profile_id', 'profile_id'),
        Index('idx_case_status', 'status'),
        Index('idx_case_worker', 'assigned_worker'),
        Index('idx_case_follow_up', 'next_follow_up'),
    )

class Intervention(Base):
    """Individual interventions and their outcomes."""
    __tablename__ = 'interventions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey('case_records.id'), nullable=False)
    
    # Intervention details
    intervention_type = Column(String(100))  # food_support, school_fees, etc.
    description = Column(Text)
    worker = Column(String(255))
    intervention_date = Column(DateTime, default=datetime.utcnow)
    
    # Outcome tracking
    outcome = Column(Text)
    follow_up_needed = Column(Boolean, default=True)
    notes = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    case = relationship("CaseRecord", back_populates="interventions")
    
    # Indexes
    __table_args__ = (
        Index('idx_intervention_case_id', 'case_id'),
        Index('idx_intervention_type', 'intervention_type'),
        Index('idx_intervention_date', 'intervention_date'),
    )

class DatabaseManager:
    """Database connection and operations manager."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DATABASE_CONFIG
        self.engine = None
        self.Session = None
        
    def create_database_and_user(self):
        """Create database and user (run as postgres superuser)."""
        try:
            # Connect as current user to create database and user
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database='postgres',  # Connect to default database
                user=self.config['username']  # Use current user
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Skip user creation since we're using existing user
            print(f"Using existing user: {self.config['username']}")
            
            # Create database
            try:
                cursor.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(self.config['database']),
                    sql.Identifier(self.config['username'])
                ))
                print(f"Created database: {self.config['database']}")
            except psycopg2.errors.DuplicateDatabase:
                print(f"Database {self.config['database']} already exists")
            
            # Grant privileges
            cursor.execute(sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
                sql.Identifier(self.config['database']),
                sql.Identifier(self.config['username'])
            ))
            
            cursor.close()
            conn.close()
            print("Database and user setup complete!")
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            print("Please ensure PostgreSQL is running and you have superuser access.")
            return False
        
        return True
    
    def connect(self):
        """Create database connection."""
        try:
            connection_string = (
                f"postgresql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print(f"Connected to PostgreSQL database: {self.config['database']}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            print("Database tables created successfully!")
            return True
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False
    
    def migrate_profiles_data(self):
        """Migrate profile data from CSV and text files."""
        print("Starting profile data migration...")
        
        # Load features dataset
        csv_path = ROOT / "usable" / "features_dataset.csv"
        if not csv_path.exists():
            print(f"Features dataset not found: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} profiles from CSV")
        
        session = self.Session()
        try:
            migrated_count = 0
            
            for _, row in df.iterrows():
                # Determine if this is augmented data
                uid = str(row['uid'])
                is_original = not ('_aug' in uid)
                original_uid = uid.split('_aug')[0] if not is_original else uid
                
                # Load text content
                text_content = ""
                if is_original:
                    text_file = ROOT / "usable" / f"{uid}.txt"
                else:
                    text_file = ROOT / "usable_aug" / f"{uid}.txt"
                
                if text_file.exists():
                    text_content = text_file.read_text(encoding='utf-8', errors='ignore')
                
                # Create profile record
                profile = Profile(
                    uid=uid,
                    source_file=row.get('source_file', ''),
                    text_content=text_content,
                    text_len=int(row.get('text_len', 0)),
                    age=int(row['age']) if pd.notna(row.get('age')) else None,
                    class_level=str(row['class']) if pd.notna(row.get('class')) else None,
                    health_status=str(row['health_status']) if pd.notna(row.get('health_status')) else None,
                    last_exam_score=int(row['last_exam_score']) if pd.notna(row.get('last_exam_score')) else None,
                    meals_per_day=int(row['meals_per_day']) if pd.notna(row.get('meals_per_day')) else None,
                    siblings_count=int(row['siblings_count']) if pd.notna(row.get('siblings_count')) else None,
                    siblings_list=str(row['siblings_list']) if pd.notna(row.get('siblings_list')) else None,
                    iron_sheets_flag=bool(row.get('iron_sheets_flag', False)),
                    single_room_flag=bool(row.get('single_room_flag', False)),
                    shared_bed_flag=bool(row.get('shared_bed_flag', False)),
                    no_electric_flag=bool(row.get('no_electric_flag', False)),
                    rent_arrears_flag=bool(row.get('rent_arrears_flag', False)),
                    landlord_lock_flag=bool(row.get('landlord_lock_flag', False)),
                    works_unstable_flag=bool(row.get('works_unstable_flag', False)),
                    no_school_fees_flag=bool(row.get('no_school_fees_flag', False)),
                    hunger_flag=bool(row.get('hunger_flag', False)),
                    father_absent_flag=bool(row.get('father_absent_flag', False)),
                    mother_hawker_flag=bool(row.get('mother_hawker_flag', False)),
                    single_parent_flag=bool(row.get('single_parent_flag', False)),
                    leftover_phone=bool(row.get('leftover_phone', False)),
                    leftover_email=bool(row.get('leftover_email', False)),
                    sentence_count=int(row.get('sentence_count', 0)),
                    is_original=is_original,
                    original_uid=original_uid
                )
                
                session.add(profile)
                
                # Add embeddings
                emb_cols = [c for c in df.columns if c.startswith('emb_')]
                if emb_cols:
                    embedding_vector = [float(row[col]) for col in emb_cols]
                    embedding = ProfileEmbedding(
                        profile=profile,
                        embedding_vector=embedding_vector
                    )
                    session.add(embedding)
                
                migrated_count += 1
                
                if migrated_count % 100 == 0:
                    session.commit()
                    print(f"Migrated {migrated_count} profiles...")
            
            session.commit()
            print(f"Successfully migrated {migrated_count} profiles!")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error during migration: {e}")
            return False
        finally:
            session.close()
    
    def create_sample_assessments(self):
        """Create sample risk assessments using the enhanced model."""
        print("Creating sample risk assessments...")
        
        try:
            from heuristics import apply_heuristics
            
            # Load a sample of profiles
            session = self.Session()
            profiles = session.query(Profile).limit(100).all()
            
            for profile in profiles:
                # Create a DataFrame row for heuristic calculation
                profile_data = {
                    'uid': profile.uid,
                    'age': profile.age,
                    'last_exam_score': profile.last_exam_score,
                    'meals_per_day': profile.meals_per_day,
                    'rent_arrears_flag': profile.rent_arrears_flag,
                    'hunger_flag': profile.hunger_flag,
                    'no_school_fees_flag': profile.no_school_fees_flag,
                    'father_absent_flag': profile.father_absent_flag,
                    'single_parent_flag': profile.single_parent_flag,
                    'no_electric_flag': profile.no_electric_flag,
                    'shared_bed_flag': profile.shared_bed_flag,
                }
                
                df_sample = pd.DataFrame([profile_data])
                df_with_heuristics = apply_heuristics(df_sample)
                heuristic_row = df_with_heuristics.iloc[0]
                
                # Create risk assessment
                assessment = RiskAssessment(
                    profile_id=profile.id,
                    priority_level=heuristic_row['weak_label'],
                    priority_confidence=float(heuristic_row['weak_confidence']),
                    dropout_risk=False,  # Placeholder
                    dropout_confidence=0.5,
                    heuristic_score=int(heuristic_row['heuristic_score']),
                    weak_label=heuristic_row['weak_label'],
                    weak_confidence=float(heuristic_row['weak_confidence']),
                    model_version='1.0',
                    model_type='heuristic_baseline'
                )
                
                session.add(assessment)
            
            session.commit()
            session.close()
            print("Sample risk assessments created!")
            return True
            
        except Exception as e:
            print(f"Error creating sample assessments: {e}")
            return False

def main():
    """Main setup function."""
    print("=== PostgreSQL Database Setup ===")
    
    db_manager = DatabaseManager()
    
    # Step 1: Create database and user
    print("\n1. Creating database and user...")
    if not db_manager.create_database_and_user():
        print("Failed to create database and user. Exiting.")
        return
    
    # Step 2: Connect to database
    print("\n2. Connecting to database...")
    if not db_manager.connect():
        print("Failed to connect to database. Exiting.")
        return
    
    # Step 3: Create tables
    print("\n3. Creating database tables...")
    if not db_manager.create_tables():
        print("Failed to create tables. Exiting.")
        return
    
    # Step 4: Migrate data
    print("\n4. Migrating profile data...")
    if not db_manager.migrate_profiles_data():
        print("Failed to migrate data. Exiting.")
        return
    
    # Step 5: Create sample assessments
    print("\n5. Creating sample risk assessments...")
    db_manager.create_sample_assessments()
    
    print("\n=== Database Setup Complete! ===")
    print(f"Database: {DATABASE_CONFIG['database']}")
    print(f"User: {DATABASE_CONFIG['username']}")
    print(f"Host: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}")
    
    # Save connection config for other scripts
    config_file = ROOT / "db_config.json"
    with open(config_file, 'w') as f:
        json.dump(DATABASE_CONFIG, f, indent=2)
    print(f"Connection config saved to: {config_file}")

if __name__ == "__main__":
    main()
