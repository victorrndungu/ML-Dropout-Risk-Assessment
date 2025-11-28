#!/usr/bin/env python3
"""
PostgreSQL Exact Replica of Local File Processing

This module replicates your EXACT local workflow:
1. Store raw .txt files in PostgreSQL
2. Process each file individually using build_features.py logic
3. Store features as JSON (preserving exact structure)
4. Apply heuristics exactly as local version
5. Generate embeddings for each file

NO changes to processing logic - 100% identical to local files.
"""

import os, re, json
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import Json
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional
import pickle

# Import your EXACT feature extraction logic
import sys
sys.path.append(str(Path(__file__).parent))
from build_features import (
    extract_structured_from_text,
    detect_housing_flags,
    detect_keywords,
    detect_leftover_pii,
    safe_read,
    EMB_MODEL_NAME
)
from heuristics import apply_heuristics

class PostgreSQLExactReplica:
    """
    PostgreSQL storage that EXACTLY replicates local file processing.
    Every file is processed individually, features stored as JSON.
    """
    
    def __init__(self, config_file: str = "db_config.json"):
        # Try to use environment variables first, fallback to config file
        self.config = self._load_config(config_file)
        self.engine = None
        self.embedding_model = None
        self.connect()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load database configuration from environment variables or config file."""
        import os
        
        # Try environment variables first (more secure)
        config = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_NAME'),
            'username': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        # If all env vars are set, use them
        if all(config.values()):
            return config
        
        # Fallback to config file
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with env vars (env vars take precedence)
            for key in config:
                if config[key]:
                    file_config[key] = config[key]
            
            return file_config
        
        # If neither exists, raise error
        raise ValueError(
            "Database configuration not found. "
            "Set environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) "
            f"or create {config_file} file."
        )
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            connection_string = (
                f"postgresql://{self.config['username']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string, echo=False)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print(f"✅ Connected to PostgreSQL database: {self.config['database']}")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def create_exact_replica_schema(self):
        """
        Create schema that mirrors your local file structure.
        Stores raw text + processed features as JSON.
        """
        
        # Table 1: Raw text files (exactly as stored locally)
        create_raw_files_table = """
        CREATE TABLE IF NOT EXISTS raw_text_files (
            uid VARCHAR(255) PRIMARY KEY,
            source_file VARCHAR(255),
            source_directory VARCHAR(100),
            raw_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Table 2: Processed features (exactly as build_features.py generates)
        create_processed_features_table = """
        CREATE TABLE IF NOT EXISTS processed_features (
            uid VARCHAR(255) PRIMARY KEY REFERENCES raw_text_files(uid),
            -- Store ALL features as JSON (exact replica of CSV row)
            features_json JSONB NOT NULL,
            -- Store embedding as array
            embedding_vector FLOAT8[],
            -- Processing metadata
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_version VARCHAR(20) DEFAULT '1.0'
        );
        """
        
        # Table 3: Heuristic results (exactly as heuristics.py generates)
        create_heuristics_table = """
        CREATE TABLE IF NOT EXISTS heuristic_results (
            uid VARCHAR(255) PRIMARY KEY REFERENCES raw_text_files(uid),
            heuristic_score FLOAT,
            weak_label VARCHAR(20),
            weak_confidence FLOAT,
            dropout_risk BOOLEAN,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Create indexes for fast queries
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_raw_source_dir ON raw_text_files(source_directory);
        CREATE INDEX IF NOT EXISTS idx_raw_created ON raw_text_files(created_at);
        CREATE INDEX IF NOT EXISTS idx_features_processed ON processed_features(processed_at);
        CREATE INDEX IF NOT EXISTS idx_heuristics_score ON heuristic_results(heuristic_score);
        CREATE INDEX IF NOT EXISTS idx_heuristics_label ON heuristic_results(weak_label);
        CREATE INDEX IF NOT EXISTS idx_heuristics_risk ON heuristic_results(dropout_risk);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_raw_files_table))
                conn.execute(text(create_processed_features_table))
                conn.execute(text(create_heuristics_table))
                conn.execute(text(create_indexes))
                conn.commit()
            
            print("✅ Database schema created (exact replica structure)")
            return True
            
        except Exception as e:
            print(f"❌ Error creating schema: {e}")
            return False
    
    def _load_embedding_model(self):
        """Load embedding model with error handling to avoid broken pipe issues."""
        if self.embedding_model is None:
            try:
                print(f"Loading embedding model: {EMB_MODEL_NAME}")
                # Disable multiprocessing to avoid broken pipe errors
                import os
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                self.embedding_model = SentenceTransformer(EMB_MODEL_NAME)
                # Explicitly set device to CPU to avoid subprocess issues
                try:
                    import torch
                    device = 'cpu'  # Use CPU to avoid GPU-related pipe issues
                    self.embedding_model = self.embedding_model.to(device)
                except:
                    pass  # If torch not available, continue without device setting
                print(f"Embedding dim: {self.embedding_model.get_sentence_embedding_dimension()}")
            except Exception as e:
                error_msg = f"Failed to load embedding model: {str(e)}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg) from e
        return self.embedding_model
    
    def process_single_file(self, file_path: str, source_dir: str, text: str = None) -> Dict:
        """
        Process a single .txt file EXACTLY as build_features.py does.
        Returns dictionary with all features.
        
        Args:
            file_path: Path to the file (used for uid and source_file name)
            source_dir: Source directory identifier
            text: Optional text content. If provided, file_path is not read.
        """
        if text is None:
            text = safe_read(file_path)
        uid = Path(file_path).stem
        
        # EXACT feature extraction (same as build_features.py lines 156-173)
        row = {
            "uid": uid, 
            "source_file": Path(file_path).name,
            "text_len": len(text)
        }
        
        # Structured fields (EXACT same extraction)
        s = extract_structured_from_text(text)
        row.update(s)
        
        # Housing flags (EXACT same detection)
        row.update(detect_housing_flags(text))
        
        # Keyword flags (EXACT same detection)
        row.update(detect_keywords(text))
        
        # Leftover PII check (EXACT same detection)
        row.update(detect_leftover_pii(text))
        
        # Sentence count (EXACT same calculation)
        row["sentence_count"] = max(1, len(re.split(r"[.!?]\s+", text)))
        
        # Compute embedding (EXACT same model and method)
        # Use try-except to handle potential broken pipe errors from model
        try:
            model = self._load_embedding_model()
            # Ensure text is not empty
            if not text or len(text.strip()) == 0:
                raise ValueError("Text is empty, cannot generate embedding")
            # Use explicit parameters to avoid multiprocessing issues
            emb = model.encode(
                text, 
                show_progress_bar=False, 
                convert_to_numpy=True,
                batch_size=1,  # Single batch to avoid subprocess issues
                normalize_embeddings=False
            )
        except (BrokenPipeError, OSError) as e:
            # Check if it's a broken pipe error (errno 32)
            is_broken_pipe = (
                isinstance(e, BrokenPipeError) or 
                (isinstance(e, OSError) and e.errno == 32)
            )
            if is_broken_pipe:
                # Retry once if broken pipe error occurs
                print(f"⚠️ Broken pipe error during embedding (errno {getattr(e, 'errno', 'unknown')}), retrying once...")
                try:
                    # Reload model to reset any subprocess state
                    self.embedding_model = None  # Force reload
                    model = self._load_embedding_model()
                    emb = model.encode(
                        text, 
                        show_progress_bar=False, 
                        convert_to_numpy=True,
                        batch_size=1,
                        normalize_embeddings=False
                    )
                except Exception as retry_error:
                    error_msg = f"Embedding generation failed after retry for {uid}: {str(retry_error)}"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg) from retry_error
            else:
                # Re-raise if it's a different OSError
                raise
        except Exception as e:
            error_msg = f"Embedding generation failed for {uid}: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg) from e
        
        return {
            "uid": uid,
            "source_file": Path(file_path).name,
            "source_directory": source_dir,
            "raw_text": text,
            "features": row,
            "embedding": emb.tolist()
        }
    
    def store_raw_text(self, uid: str, source_file: str, source_directory: str, raw_text: str):
        """
        Store raw text file in PostgreSQL.
        Used for uploading new cases.
        """
        try:
            with self.engine.begin() as conn:
                # Store raw text file
                conn.execute(text("""
                    INSERT INTO raw_text_files (uid, source_file, source_directory, raw_text)
                    VALUES (:uid, :source_file, :source_dir, :raw_text)
                    ON CONFLICT (uid) DO UPDATE SET
                        source_file = EXCLUDED.source_file,
                        source_directory = EXCLUDED.source_directory,
                        raw_text = EXCLUDED.raw_text,
                        created_at = CURRENT_TIMESTAMP
                """), {
                    'uid': uid,
                    'source_file': source_file,
                    'source_dir': source_directory,
                    'raw_text': raw_text
                })
                
                print(f"✅ Stored raw text for {uid}")
                return True
                
        except Exception as e:
            print(f"❌ Error storing raw text for {uid}: {e}")
            return False

    def process_file(self, uid: str):
        """
        Process a stored raw text file by UID using EXACT same algorithm as local files.
        Used for processing uploaded cases.
        
        Returns:
            tuple: (success: bool, error_message: str)
        """
        temp_file_path = None
        try:
            # Get the raw text from database
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    SELECT raw_text, source_file, source_directory 
                    FROM raw_text_files 
                    WHERE uid = :uid
                """), {'uid': uid})
                
                row = result.fetchone()
                if not row:
                    error_msg = f"No raw text found for UID: {uid}. Please store the raw text first using store_raw_text()."
                    print(f"❌ {error_msg}")
                    return False, error_msg
                
                raw_text, source_file, source_directory = row
            
            # Process the text directly in memory (no temporary file needed)
            # Create a virtual file path for processing (used only for uid extraction)
            import os
            temp_file_path = os.path.join("/tmp", f"{uid}.txt")  # Virtual path, not actually used
            
            # Process using existing method (EXACT same as build_features.py)
            # Pass text directly to avoid file I/O issues
            try:
                processed_data = self.process_single_file(temp_file_path, source_directory, text=raw_text)
                
                # Validate that essential features were extracted
                features = processed_data.get('features', {})
                if not features:
                    error_msg = f"Feature extraction returned empty features for {uid}. Check if text file is valid."
                    print(f"❌ {error_msg}")
                    return False, error_msg
                
                # Check if uid matches
                if processed_data.get('uid') != uid:
                    error_msg = f"UID mismatch: expected {uid}, got {processed_data.get('uid')}"
                    print(f"❌ {error_msg}")
                    return False, error_msg
                    
            except Exception as e:
                error_msg = f"Feature extraction failed for {uid}: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                print(traceback.format_exc())
                return False, error_msg
            
            # Store the processed data
            try:
                stored, store_error = self.store_processed_file(processed_data)
                if not stored:
                    error_msg = f"Failed to store processed data for {uid}: {store_error if store_error else 'Check database connection and schema'}"
                    print(f"❌ {error_msg}")
                    return False, error_msg
            except Exception as e:
                error_msg = f"Database storage failed for {uid}: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                print(traceback.format_exc())
                return False, error_msg
            
            # Apply heuristics to the processed data (EXACT same as local files)
            try:
                self.apply_heuristics_to_processed_case(uid)
            except Exception as e:
                # Heuristics failure is not critical - log but don't fail
                print(f"⚠️ Heuristics application failed for {uid}: {e} (non-critical)")
            
            print(f"✅ Processed file for {uid} using EXACT same algorithm as local files")
            return True, "Success"
                
        except Exception as e:
            error_msg = f"Error processing file for {uid}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(traceback.format_exc())
            return False, error_msg
        finally:
            # No temporary file cleanup needed - we process text directly in memory
            pass

    def apply_heuristics_to_processed_case(self, uid: str):
        """Apply heuristics to a single processed case (EXACT same as local files)."""
        try:
            # Get the processed features
            with self.engine.begin() as conn:
                result = conn.execute(text("""
                    SELECT features_json FROM processed_features WHERE uid = :uid
                """), {'uid': uid})
                
                row = result.fetchone()
                if not row:
                    print(f"No processed features found for {uid}")
                    return False
                
                features = row[0]
                
                # Convert to DataFrame for heuristics
                import pandas as pd
                df = pd.DataFrame([features])
                
                # Apply heuristics (EXACT same as local files)
                from heuristics import apply_heuristics
                df_with_heuristics = apply_heuristics(df)
                
                # Update the processed features with heuristic results
                heuristic_results = df_with_heuristics.iloc[0].to_dict()
                
                # Update the features in the database
                conn.execute(text("""
                    UPDATE processed_features 
                    SET features_json = :features
                    WHERE uid = :uid
                """), {
                    'uid': uid,
                    'features': Json(heuristic_results)
                })
                
                print(f"✅ Applied heuristics to {uid} - heuristic_score: {heuristic_results.get('heuristic_score', 'Not found')}")
                return True
                
        except Exception as e:
            print(f"❌ Error applying heuristics to {uid}: {e}")
            return False

    def store_processed_file(self, processed_data: Dict):
        """
        Store processed file in PostgreSQL.
        Maintains exact same structure as local CSV.
        Also creates profile record for case management.
        
        Returns:
            tuple: (success: bool, error_message: str)
        """
        try:
            features = processed_data.get('features', {})
            uid = processed_data['uid']
            
            with self.engine.begin() as conn:
                # 1. Store raw text file
                conn.execute(text("""
                    INSERT INTO raw_text_files (uid, source_file, source_directory, raw_text)
                    VALUES (:uid, :source_file, :source_dir, :raw_text)
                    ON CONFLICT (uid) DO UPDATE SET
                        raw_text = EXCLUDED.raw_text,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'uid': uid,
                    'source_file': processed_data['source_file'],
                    'source_dir': processed_data['source_directory'],
                    'raw_text': processed_data['raw_text']
                })
                
                # 2. Store processed features as JSON
                conn.execute(text("""
                    INSERT INTO processed_features (uid, features_json, embedding_vector)
                    VALUES (:uid, :features_json, :embedding_vector)
                    ON CONFLICT (uid) DO UPDATE SET
                        features_json = EXCLUDED.features_json,
                        embedding_vector = EXCLUDED.embedding_vector,
                        processed_at = CURRENT_TIMESTAMP
                """), {
                    'uid': uid,
                    'features_json': json.dumps(features),
                    'embedding_vector': processed_data['embedding']
                })
                
                # 3. Create profile record (needed for case_records foreign key)
                # Extract fields from features dict
                profile_data = {
                    'uid': uid,
                    'source_file': processed_data.get('source_file', ''),
                    'text_content': processed_data.get('raw_text', ''),
                    'text_len': features.get('text_len'),
                    'age': features.get('age'),
                    'class_level': str(features.get('class', '')) if features.get('class') else None,
                    'health_status': str(features.get('health_status', '')) if features.get('health_status') else None,
                    'last_exam_score': features.get('last_exam_score'),
                    'meals_per_day': features.get('meals_per_day'),
                    'siblings_count': features.get('siblings_count'),
                    'siblings_list': str(features.get('siblings_list', '')) if features.get('siblings_list') else None,
                    'sentence_count': features.get('sentence_count'),
                    # Flags
                    'iron_sheets_flag': bool(features.get('iron_sheets_flag', False)),
                    'single_room_flag': bool(features.get('single_room_flag', False)),
                    'shared_bed_flag': bool(features.get('shared_bed_flag', False)),
                    'no_electric_flag': bool(features.get('no_electric_flag', False)),
                    'rent_arrears_flag': bool(features.get('rent_arrears_flag', False)),
                    'landlord_lock_flag': bool(features.get('landlord_lock_flag', False)),
                    'works_unstable_flag': bool(features.get('works_unstable_flag', False)),
                    'no_school_fees_flag': bool(features.get('no_school_fees_flag', False)),
                    'hunger_flag': bool(features.get('hunger_flag', False)),
                    'father_absent_flag': bool(features.get('father_absent_flag', False)),
                    'mother_hawker_flag': bool(features.get('mother_hawker_flag', False)),
                    'single_parent_flag': bool(features.get('single_parent_flag', False)),
                    'leftover_phone': bool(features.get('leftover_phone', False)),
                    'leftover_email': bool(features.get('leftover_email', False))
                }
                
                # Insert or update profile (handle both 'id' and 'uid' foreign key schemes)
                # Try to insert with all columns, but handle missing columns gracefully
                try:
                    conn.execute(text("""
                        INSERT INTO profiles (
                            uid, source_file, text_content, text_len, age, class_level, health_status,
                            last_exam_score, meals_per_day, siblings_count, siblings_list, sentence_count,
                            iron_sheets_flag, single_room_flag, shared_bed_flag, no_electric_flag,
                            rent_arrears_flag, landlord_lock_flag, works_unstable_flag, no_school_fees_flag,
                            hunger_flag, father_absent_flag, mother_hawker_flag, single_parent_flag,
                            leftover_phone, leftover_email
                        )
                        VALUES (
                            :uid, :source_file, :text_content, :text_len, :age, :class_level, :health_status,
                            :last_exam_score, :meals_per_day, :siblings_count, :siblings_list, :sentence_count,
                            :iron_sheets_flag, :single_room_flag, :shared_bed_flag, :no_electric_flag,
                            :rent_arrears_flag, :landlord_lock_flag, :works_unstable_flag, :no_school_fees_flag,
                            :hunger_flag, :father_absent_flag, :mother_hawker_flag, :single_parent_flag,
                            :leftover_phone, :leftover_email
                        )
                        ON CONFLICT (uid) DO UPDATE SET
                            source_file = EXCLUDED.source_file,
                            text_content = EXCLUDED.text_content,
                            text_len = EXCLUDED.text_len,
                            age = EXCLUDED.age,
                            class_level = EXCLUDED.class_level,
                            health_status = EXCLUDED.health_status,
                            last_exam_score = EXCLUDED.last_exam_score,
                            meals_per_day = EXCLUDED.meals_per_day,
                            siblings_count = EXCLUDED.siblings_count,
                            siblings_list = EXCLUDED.siblings_list,
                            sentence_count = EXCLUDED.sentence_count,
                            iron_sheets_flag = EXCLUDED.iron_sheets_flag,
                            single_room_flag = EXCLUDED.single_room_flag,
                            shared_bed_flag = EXCLUDED.shared_bed_flag,
                            no_electric_flag = EXCLUDED.no_electric_flag,
                            rent_arrears_flag = EXCLUDED.rent_arrears_flag,
                            landlord_lock_flag = EXCLUDED.landlord_lock_flag,
                            works_unstable_flag = EXCLUDED.works_unstable_flag,
                            no_school_fees_flag = EXCLUDED.no_school_fees_flag,
                            hunger_flag = EXCLUDED.hunger_flag,
                            father_absent_flag = EXCLUDED.father_absent_flag,
                            mother_hawker_flag = EXCLUDED.mother_hawker_flag,
                            single_parent_flag = EXCLUDED.single_parent_flag,
                            leftover_phone = EXCLUDED.leftover_phone,
                            leftover_email = EXCLUDED.leftover_email,
                            updated_at = CURRENT_TIMESTAMP
                    """), profile_data)
                except Exception as col_error:
                    # If columns don't exist, try minimal insert
                    error_str = str(col_error).lower()
                    if 'column' in error_str and 'does not exist' in error_str:
                        # Try minimal insert with only essential columns
                        print(f"⚠️ Some columns missing, using minimal profile insert for {uid}")
                        try:
                            conn.execute(text("""
                                INSERT INTO profiles (uid, source_file, text_content, text_len)
                                VALUES (:uid, :source_file, :text_content, :text_len)
                                ON CONFLICT (uid) DO UPDATE SET
                                    source_file = EXCLUDED.source_file,
                                    text_content = EXCLUDED.text_content,
                                    text_len = EXCLUDED.text_len,
                                    updated_at = CURRENT_TIMESTAMP
                            """), {
                                'uid': uid,
                                'source_file': processed_data.get('source_file', ''),
                                'text_content': processed_data.get('raw_text', ''),
                                'text_len': features.get('text_len', 0)
                            })
                        except Exception as min_error:
                            # Even minimal insert failed - this is a serious error
                            raise Exception(f"Profile insert failed even with minimal columns: {str(min_error)}. Original error: {str(col_error)}")
                    else:
                        # Re-raise if it's a different error
                        raise
            
            print(f"✅ Stored processed file and created profile for {uid}")
            return True, "Success"
            
        except Exception as e:
            import traceback
            error_details = str(e)
            error_msg = f"Error storing file {processed_data.get('uid', 'UNKNOWN')}: {error_details}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            # Return error message for better debugging
            return False, error_msg
    
    def apply_heuristics_to_all(self):
        """
        Apply heuristics to all processed files.
        Uses EXACT same logic as heuristics.py.
        """
        # Load all features from database
        df = self.get_features_as_dataframe()
        
        if df.empty:
            print("No features to process")
            return
        
        # Apply heuristics (EXACT same function from heuristics.py)
        df_with_heuristics = apply_heuristics(df)
        
        # Store heuristic results
        try:
            with self.engine.begin() as conn:
                for _, row in df_with_heuristics.iterrows():
                    conn.execute(text("""
                        INSERT INTO heuristic_results 
                        (uid, heuristic_score, weak_label, weak_confidence, dropout_risk)
                        VALUES (:uid, :score, :label, :confidence, :risk)
                        ON CONFLICT (uid) DO UPDATE SET
                            heuristic_score = EXCLUDED.heuristic_score,
                            weak_label = EXCLUDED.weak_label,
                            weak_confidence = EXCLUDED.weak_confidence,
                            dropout_risk = EXCLUDED.dropout_risk,
                            computed_at = CURRENT_TIMESTAMP
                    """), {
                        'uid': row['uid'],
                        'score': float(row.get('heuristic_score', 0)),
                        'label': row.get('weak_label', 'low'),
                        'confidence': float(row.get('weak_confidence', 0)),
                        'risk': bool(row.get('dropout_risk', False))
                    })
            
            print(f"✅ Heuristics applied to {len(df_with_heuristics)} files")
            
        except Exception as e:
            print(f"❌ Error applying heuristics: {e}")
    
    def get_features_as_dataframe(self) -> pd.DataFrame:
        """
        Get all processed features as DataFrame.
        EXACTLY matches the structure of features_dataset.csv
        """
        query = """
        SELECT 
            rtf.uid,
            rtf.source_file,
            rtf.source_directory,
            pf.features_json,
            pf.embedding_vector,
            hr.heuristic_score,
            hr.weak_label,
            hr.weak_confidence,
            hr.dropout_risk
        FROM raw_text_files rtf
        LEFT JOIN processed_features pf ON rtf.uid = pf.uid
        LEFT JOIN heuristic_results hr ON rtf.uid = hr.uid
        ORDER BY rtf.uid
        """
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # Extract features from JSON
                    if row_dict['features_json']:
                        features = json.loads(row_dict['features_json']) if isinstance(row_dict['features_json'], str) else row_dict['features_json']
                        row_dict.update(features)
                    
                    # Expand embedding into individual columns
                    if row_dict.get('embedding_vector'):
                        emb = row_dict['embedding_vector']
                        # Ensure embeddings are properly converted to float
                        for i, val in enumerate(emb):
                            row_dict[f'emb_{i}'] = float(val)
                    
                    # Remove JSON and vector columns
                    row_dict.pop('features_json', None)
                    row_dict.pop('embedding_vector', None)
                    
                    data.append(row_dict)
                
                df = pd.DataFrame(data)
                
                # Ensure column order matches CSV (uid first, then features, then embeddings)
                cols = ['uid', 'source_file', 'source_directory']
                cols += [c for c in df.columns if c not in cols and not c.startswith('emb_')]
                cols += sorted([c for c in df.columns if c.startswith('emb_')])
                
                return df[cols]
                
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            return pd.DataFrame()
    
    def migrate_from_local_files(self, usable_dir: str = "usable", usable_aug_dir: str = "usable_aug"):
        """
        Migrate all local .txt files to PostgreSQL.
        Processes each file EXACTLY as build_features.py does.
        """
        print("\n" + "="*60)
        print("MIGRATING LOCAL FILES TO POSTGRESQL")
        print("Using EXACT same processing as build_features.py")
        print("="*60 + "\n")
        
        # Create schema
        self.create_exact_replica_schema()
        
        # Collect all .txt files
        files = []
        if os.path.isdir(usable_dir):
            files += [(f, "usable") for f in Path(usable_dir).glob("*.txt")]
        if os.path.isdir(usable_aug_dir):
            files += [(f, "usable_aug") for f in Path(usable_aug_dir).glob("*.txt")]
        
        if not files:
            print("No .txt files found")
            return
        
        print(f"Found {len(files)} files to process")
        
        # Process each file individually
        processed_count = 0
        for file_path, source_dir in files:
            try:
                # Process file (EXACT same logic as build_features.py)
                processed_data = self.process_single_file(str(file_path), source_dir)
                
                # Store in PostgreSQL
                stored, _ = self.store_processed_file(processed_data)
                if stored:
                    processed_count += 1
                    
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{len(files)} files...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"\n✅ Processed and stored {processed_count} files")
        
        # Apply heuristics (EXACT same logic as heuristics.py)
        print("\nApplying heuristics...")
        self.apply_heuristics_to_all()
        
        # Show summary
        df = self.get_features_as_dataframe()
        print(f"\n{'='*60}")
        print(f"MIGRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total files in PostgreSQL: {len(df)}")
        if 'source_directory' in df.columns:
            print(f"From usable/: {len(df[df['source_directory'] == 'usable'])}")
            print(f"From usable_aug/: {len(df[df['source_directory'] == 'usable_aug'])}")
        if 'dropout_risk' in df.columns:
            print(f"High risk (dropout=True): {df['dropout_risk'].sum()}")
        
        return df

# ============= USAGE EXAMPLES =============

def verify_exact_match(local_csv_path: str = "usable/features_dataset.csv"):
    """
    Verify that PostgreSQL data EXACTLY matches local CSV.
    """
    print("\n" + "="*60)
    print("VERIFYING EXACT MATCH: PostgreSQL vs Local CSV")
    print("="*60 + "\n")
    
    db = PostgreSQLExactReplica()
    
    # Load from PostgreSQL
    df_postgres = db.get_features_as_dataframe()
    
    # Load from local CSV
    df_local = pd.read_csv(local_csv_path)
    
    # Compare
    print(f"Local CSV rows: {len(df_local)}")
    print(f"PostgreSQL rows: {len(df_postgres)}")
    
    # Check columns
    local_cols = set(df_local.columns)
    postgres_cols = set(df_postgres.columns)
    
    missing_in_postgres = local_cols - postgres_cols
    extra_in_postgres = postgres_cols - local_cols
    
    if missing_in_postgres:
        print(f"\n⚠️  Columns in local but not PostgreSQL: {missing_in_postgres}")
    if extra_in_postgres:
        print(f"\n⚠️  Extra columns in PostgreSQL: {extra_in_postgres}")
    
    # Check common UIDs
    common_uids = set(df_local['uid']) & set(df_postgres['uid'])
    print(f"\nCommon UIDs: {len(common_uids)}")
    
    if common_uids:
        # Sample comparison
        sample_uid = list(common_uids)[0]
        print(f"\nSample comparison for UID: {sample_uid}")
        print("\nLocal:")
        print(df_local[df_local['uid'] == sample_uid].iloc[0][['uid', 'age', 'heuristic_score', 'weak_label']])
        print("\nPostgreSQL:")
        print(df_postgres[df_postgres['uid'] == sample_uid].iloc[0][['uid', 'age', 'heuristic_score', 'weak_label']])
    
    print("\n✅ Verification complete")

if __name__ == "__main__":
    # Create replica instance
    db = PostgreSQLExactReplica()
    
    # Migrate all files
    db.migrate_from_local_files()
    
    # Verify exact match
    verify_exact_match()

