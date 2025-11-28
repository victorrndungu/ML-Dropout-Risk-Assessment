#!/usr/bin/env python3
"""
Complete Upload Processor
========================

This module implements the FULL feature extraction pipeline for uploaded cases,
ensuring they get the exact same treatment as the original training data.

Key Features:
- Complete feature extraction (age, class, exam score, meals, etc.)
- Text preprocessing and analysis
- Embedding generation using SentenceTransformer
- PCA transformation
- Composite feature creation
- Heuristic scoring
- ML prediction preparation

This ensures uploaded cases are assessed with the same accuracy as training data.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import joblib
import json
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteUploadProcessor:
    """
    Complete upload processor that replicates the exact feature extraction
    pipeline used for training data.
    """
    
    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.embedding_model = None
        self.pca_model = None
        self.scaler = None
        self.priority_encoder = None
        self.feature_names = None
        
        # Load models and components
        self._load_models()
        
        # Define regex patterns for feature extraction
        self._setup_regex_patterns()
    
    def _load_models(self):
        """Load all required models and components."""
        try:
            # Load embedding model with proper error handling to avoid meta tensor issues
            import os
            import torch
            # Disable tokenizer parallelism to avoid issues
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            # Ensure PyTorch uses CPU (prevents GPU auto-detection that might cause meta tensor issues)
            # Save original CUDA setting
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            try:
                # Temporarily disable CUDA to force CPU loading
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                # Load model - should now load on CPU
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            finally:
                # Restore original CUDA setting
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
            # Test encode to ensure model is fully loaded and not in meta mode
            try:
                test_embedding = self.embedding_model.encode("test", convert_to_numpy=True)
                logger.info(f"✅ SentenceTransformer model verified (embedding dim: {len(test_embedding)})")
            except Exception as e:
                logger.warning(f"⚠️ Model test encode failed: {e}. Model may still work for actual encoding.")
            
            # Load PCA model
            pca_path = self.models_dir / 'pca_transformer_proper.pkl'
            if pca_path.exists():
                self.pca_model = joblib.load(pca_path)
                logger.info("✅ Loaded PCA model (proper)")
            else:
                self.pca_model = joblib.load(self.models_dir / 'pca_transformer.pkl')
                logger.info("✅ Loaded PCA model (original)")
            
            # Load scaler
            scaler_path = self.models_dir / 'feature_scaler_proper.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Loaded scaler (proper)")
            else:
                self.scaler = joblib.load(self.models_dir / 'feature_scaler.pkl')
                logger.info("✅ Loaded scaler (original)")
            
            # Load priority encoder
            encoder_path = self.models_dir / 'priority_encoder_proper.pkl'
            if encoder_path.exists():
                self.priority_encoder = joblib.load(encoder_path)
                logger.info("✅ Loaded priority encoder (proper)")
            else:
                self.priority_encoder = joblib.load(self.models_dir / 'priority_encoder.pkl')
                logger.info("✅ Loaded priority encoder (original)")
            
            # Load feature names
            feature_path = self.models_dir / 'feature_names_proper.json'
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info("✅ Loaded feature names (proper)")
            else:
                with open(self.models_dir / 'feature_names.json', 'r') as f:
                    self.feature_names = json.load(f)
                logger.info("✅ Loaded feature names (original)")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _setup_regex_patterns(self):
        """Set up regex patterns for feature extraction."""
        # Age patterns
        self.RE_AGE = re.compile(
            r"\bAGE\s*[:\-]?\s*(\d{1,2})\b|\b(\d{1,2})-year-old\b|\b(\d{1,2})\s+years?\s+old\b",
            re.I
        )
        
        # Class patterns
        self.RE_CLASS = re.compile(
            r"\bCLASS\s*[:\-]?\s*([A-Za-z0-9\-]+)\b|\bclass\s+(\d{1,2})\b|\bin\s+class\s+(\d{1,2})\b",
            re.I
        )
        
        # Health status patterns
        self.RE_HEALTH = re.compile(
            r"\bHEALTH\s*[:\-]?\s*([A-Za-z\s]+?)(?:\n|$)",
            re.I
        )
        
        # Exam score patterns
        self.RE_LAST_SCORE = re.compile(
            r"(LAST\s+EXAM[S]?\s+SCORE|LAST\s+SCORE|LAST\s+EXAMS?)\s*[:\-]?\s*(\d{1,4})|\blast\s+exam\s+score\s+was\s+(\d{1,4})\b|\b(\d{1,4})\s+marks?\b|\bmissed\s+(?:mid-term\s+)?exams?\b|\bperformance\s+has\s+dropped\b|\bperformance\s+dropped\b",
            re.I
        )
        
        # Meals patterns
        self.RE_MEALS_NUM = re.compile(
            r"\bMEALS?\s*[:\-]?\s*(\d{1,2})\b",
            re.I
        )
        
        self.RE_MEALS_TEXT = re.compile(
            r"\b(one|two|three|four|2|1|3)\s+meals?\b|\beats?\s+only\s+(one|two|three|four)\s+meals?\b|\b(one|two|three|four)\s+or\s+(one|two|three|four)\s+meals?\b|\bcomes?\s+to\s+school\s+hungry\b|\boften\s+hungry\b|\bgoes?\s+hungry\b",
            re.I
        )
        
        # Siblings patterns
        self.RE_SIBLINGS_BLOCK = re.compile(
            r"SIBLINGS?\s*[:\-]?\s*([A-Za-z\s,]+?)(?:\n|$)",
            re.I
        )
        
        self.RE_SIB_NAME = re.compile(
            r"\b[A-Z][a-z]+\b"
        )
    
    def extract_structured_features(self, text: str) -> Dict:
        """
        Extract all structured features from text using the exact same
        logic as the original build_features.py script.
        """
        features = {}
        
        # Age extraction
        m = self.RE_AGE.search(text)
        if m:
            age = None
            if m.lastindex:
                for i in range(1, m.lastindex + 1):
                    if m.group(i) is not None:
                        age = int(m.group(i))
                        break
            features["age"] = age
        else:
            features["age"] = None
        
        # Class extraction
        m = self.RE_CLASS.search(text)
        if m:
            class_val = None
            if m.lastindex:
                for i in range(1, m.lastindex + 1):
                    if m.group(i) is not None:
                        class_val = m.group(i).strip()
                        break
            features["class"] = class_val
        else:
            features["class"] = None
        
        # Health status
        m = self.RE_HEALTH.search(text)
        features["health_status"] = m.group(1).strip() if m else None
        
        # Exam score extraction
        m = self.RE_LAST_SCORE.search(text)
        if m:
            score = None
            if m.lastindex:
                for i in range(1, m.lastindex + 1):
                    if m.group(i) is not None:
                        try:
                            score = int(m.group(i))
                            break
                        except ValueError:
                            if "missed" in m.group(i).lower() or "dropped" in m.group(i).lower():
                                score = 0
                                break
            features["last_exam_score"] = score
        else:
            features["last_exam_score"] = None
        
        # Meals extraction
        m = self.RE_MEALS_NUM.search(text)
        if m:
            features["meals_per_day"] = int(m.group(1))
        else:
            m2 = self.RE_MEALS_TEXT.search(text)
            if m2:
                txtnum = None
                if m2.lastindex:
                    for i in range(1, m2.lastindex + 1):
                        if m2.group(i) is not None:
                            txtnum = m2.group(i)
                            break
                if txtnum:
                    if "hungry" in txtnum.lower():
                        features["meals_per_day"] = 0
                    else:
                        features["meals_per_day"] = {"one":1,"two":2,"three":3,"four":4}.get(txtnum.lower(), None)
                else:
                    features["meals_per_day"] = None
            else:
                # Check for hunger indicators
                hunger_indicators = ["hungry", "starving", "no food", "lack of food", "food shortage"]
                if any(indicator in text.lower() for indicator in hunger_indicators):
                    features["meals_per_day"] = 0
                else:
                    features["meals_per_day"] = None
        
        # Critical flags detection
        features["critical_flags"] = []
        
        # Academic crisis
        academic_crisis = ["missed", "dropped", "failed", "absent", "truant", "performance dropped", "grades dropped"]
        if any(indicator in text.lower() for indicator in academic_crisis):
            features["critical_flags"].append("academic_crisis")
            if features["last_exam_score"] is None:
                features["last_exam_score"] = 0
        
        # Food crisis
        food_crisis = ["hungry", "starving", "no food", "lack of food", "food shortage", "malnourished"]
        if any(indicator in text.lower() for indicator in food_crisis):
            features["critical_flags"].append("food_crisis")
            if features["meals_per_day"] is None:
                features["meals_per_day"] = 0
        
        # Family crisis
        family_crisis = ["abandoned", "left", "no support", "single parent", "orphaned", "neglected"]
        if any(indicator in text.lower() for indicator in family_crisis):
            features["critical_flags"].append("family_crisis")
        
        # Housing crisis
        housing_crisis = ["iron sheets", "no electricity", "no water", "poor housing", "overcrowded"]
        if any(indicator in text.lower() for indicator in housing_crisis):
            features["critical_flags"].append("housing_crisis")
        
        # Emotional crisis
        emotional_crisis = ["cries", "withdrawn", "depressed", "anxious", "emotional distress"]
        if any(indicator in text.lower() for indicator in emotional_crisis):
            features["critical_flags"].append("emotional_crisis")
        
        # Economic crisis
        economic_crisis = ["struggles", "cannot afford", "no money", "poverty", "financial difficulties"]
        if any(indicator in text.lower() for indicator in economic_crisis):
            features["critical_flags"].append("economic_crisis")
        
        # Convert critical flags to string
        features["critical_flags"] = ";".join(features["critical_flags"]) if features["critical_flags"] else ""
        
        # Siblings count
        sib_block = self.RE_SIBLINGS_BLOCK.search(text)
        if sib_block:
            names = self.RE_SIB_NAME.findall(sib_block.group(1))
            features["siblings_count"] = len(names)
            features["siblings_list"] = ";".join(names) if names else ""
        else:
            s_cnt = len(re.findall(r"\b(sister|brother)\b", text, re.I))
            features["siblings_count"] = s_cnt
            features["siblings_list"] = ""
        
        return features
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract text-based features."""
        features = {}
        
        # Text length
        features["text_len"] = len(text)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        features["sentence_count"] = len([s for s in sentences if s.strip()])
        
        return features
    
    def _detect_hunger_with_severity(self, text: str) -> int:
        """Detect hunger with severity: 0 = none, 1 = intermittent, 2 = constant."""
        import re
        text_lower = text.lower()
        
        # Constant/severe hunger indicators (weight = 2)
        severe_patterns = [
            'starving', 'malnourished', 'sleep hungry', 'sleep on empty stomach',
            'goes hungry', 'comes hungry', 'often hungry', 'frequently hungry',
            'no food', 'lacks food', 'food shortage', 'food crisis'
        ]
        
        # Intermittent hunger indicators (weight = 1)
        intermittent_patterns = [
            'sometimes skip', 'sometimes skips', 'sometimes goes hungry',
            'when business is poor', 'when business is bad', 'relies on neighbors',
            'skips breakfast', 'skip breakfast', 'occasionally', 'at times',
            'during difficult times', 'when money is short'
        ]
        
        # Check for severe/constant hunger first
        if any(pattern in text_lower for pattern in severe_patterns):
            return 2  # Constant/severe hunger
        elif any(pattern in text_lower for pattern in intermittent_patterns):
            return 1  # Intermittent hunger
        else:
            # Check for general hunger indicators (default to intermittent if not severe)
            general_patterns = [
                'hungry', 'hunger', 'go without meals', 'without meals',
                'lack of food', 'lack sufficient food', 'not enough food',
                'insufficient food', 'food insecurity', 'skips meals'
            ]
            if any(pattern in text_lower for pattern in general_patterns):
                return 1  # Default to intermittent if severity unclear
            return 0  # No hunger detected
    
    def _detect_intermittent_hunger(self, text: str) -> int:
        """Detect if hunger is intermittent (sometimes, when X happens, etc.)."""
        import re
        text_lower = text.lower()
        
        intermittent_keywords = [
            'sometimes', 'occasionally', 'at times', 'when business is poor',
            'when business is bad', 'relies on neighbors', 'when money is short',
            'during difficult times', 'when needed', 'when available'
        ]
        
        return 1 if any(keyword in text_lower for keyword in intermittent_keywords) else 0
    
    def _detect_positive_indicators(self, text: str) -> Dict:
        """Detect positive indicators that suggest case is manageable (reduce priority)."""
        import re
        text_lower = text.lower()
        
        return {
            'attends_regularly': 1 if any(phrase in text_lower for phrase in [
                'attends school', 'attends most days', 'attends regularly',
                'goes to school', 'attends class', 'regular attendance'
            ]) else 0,
            
            'decent_academic': 1 if any(phrase in text_lower for phrase in [
                'scored', 'marks', 'performs well', 'good student', 'hardworking',
                'polite', 'responsible', 'attentive'
            ]) else 0,
            
            'has_support': 1 if any(phrase in text_lower for phrase in [
                'relies on neighbors', 'neighbors help', 'community support',
                'has support', 'receives help', 'family support'
            ]) else 0,
            
            'mother_present': 1 if any(phrase in text_lower for phrase in [
                'lives with mother', 'mother is', 'mother works', 'mother provides',
                'mother cares', 'mother supports'
            ]) else 0
        }
    
    def extract_flag_features(self, text: str) -> Dict:
        """Extract all flag features from text with comprehensive pattern matching for all wording variations."""
        import re
        text_lower = text.lower()
        
        # Helper function to check if any pattern in a list matches
        def has_any_pattern(patterns):
            return 1 if any(pattern in text_lower for pattern in patterns) else 0
        
        # Detect hunger with severity
        hunger_severity = self._detect_hunger_with_severity(text)
        intermittent_hunger = self._detect_intermittent_hunger(text)
        
        # Detect positive indicators
        positive_indicators = self._detect_positive_indicators(text)
        
        flags = {
            # Housing flags - comprehensive patterns
            'iron_sheets_flag': has_any_pattern([
                'iron sheets', 'iron sheet', 'corrugated', 'tin roof', 'metal sheets'
            ]),
            
            'single_room_flag': has_any_pattern([
                'single room', 'single-room', 'one room', 'one-room', 'single roomed',
                'one roomed', 'lives in a single room', 'only one room'
            ]),
            
            'shared_bed_flag': has_any_pattern([
                'shared bed', 'shares bed', 'sleeps together', 'same bed',
                'sleep in the same', 'sharing a bed'
            ]),
            
            'no_electric_flag': has_any_pattern([
                'no electricity', 'no electric', 'without electricity', 'lack of electricity',
                'no power', 'without power', 'no lights'
            ]),
            
            # Economic flags - comprehensive patterns
            'rent_arrears_flag': has_any_pattern([
                'rent arrears', 'rent arrears', 'rent unpaid', 'back rent', 'rent debt',
                'rent owing', 'cannot pay rent', 'rent overdue', 'behind on rent',
                'rent not paid', 'rent payment', 'rent problem'
            ]),
            
            'hunger_flag': 1 if hunger_severity >= 1 else 0,  # Any hunger detected
            'hunger_severity': hunger_severity,  # 0=none, 1=intermittent, 2=constant
            'intermittent_hunger_flag': intermittent_hunger,
            **positive_indicators,  # Add positive indicators to flags
            
            'landlord_lock_flag': has_any_pattern([
                'landlord lock', 'locked out', 'locked by landlord', 'landlord locked',
                'house locked', 'door locked', 'evicted', 'eviction', 'lockout',
                'locked by', 'being locked out', 'got locked out', 'was locked out'
            ]),
            
            'no_school_fees_flag': has_any_pattern([
                # Direct fee issues
                'no school fees', 'cannot afford fees', 'cannot afford school fees',
                'sent home for school fees', 'sent home for fees', 'fees not paid',
                'fees are unpaid', 'unpaid fees', 'fees arrears', 'school fees arrears',
                'cannot pay fees', 'fees owing', 'back fees', 'behind on fees',
                'fees overdue', 'no money for fees', 'no money for school fees',
                'afford fees', 'cannot afford', 'school fees problem', 'fees issue',
                'missing school due to fees', 'fees unpaid', 'fees not paid',
                # School-related expenses (uniforms, shoes, books, supplies)
                'buying school', 'school uniform', 'school shoes', 'school books',
                'school supplies', 'buying uniform', 'buying shoes', 'uniform difficult',
                'shoes difficult', 'cannot afford uniform', 'cannot afford shoes',
                'difficult buying school', 'trouble buying school', 'struggle to buy school',
                'hard to buy school', 'problem buying school', 'unable to buy school',
                'no money for uniform', 'no money for shoes', 'no money for books',
                'school clothes', 'school materials', 'school items', 'school necessities',
                'school requirements', 'afford uniform', 'afford shoes', 'afford books',
                'school expenses', 'educational expenses', 'school costs', 'school-related expenses'
            ]),
            
            'works_unstable_flag': has_any_pattern([
                'unstable work', 'irregular work', 'goes for days without earning',
                'unemployed', 'jobless', 'no work', 'no job', 'lost job', 'lost work',
                'inconsistent work', 'irregular income', 'unstable income', 'no steady income',
                'work is unstable', 'work irregular', 'sometimes no work', 'days without work',
                'cannot find work', 'struggles to find work', 'no employment'
            ]),
            
            # Additional economic stress indicators (can be used in composite features)
            'economic_stress_indicators': has_any_pattern([
                # Financial hardship phrases
                'barely covers', 'struggles', 'cannot afford', 'no money', 'poverty',
                'poor', 'very poor', 'financial difficulties', 'financial problems',
                'financial struggle', 'money problems', 'cannot pay', 'unable to pay',
                'has no money', 'short of money', 'lack of money', 'no income',
                'low income', 'small income', 'meagre income', 'scarcity',
                # Difficulty buying/providing - catch all variations
                'difficult buying', 'difficult to buy', 'buying has been difficult',
                'buying been difficult', 'been difficult buying', 'difficult purchasing',
                'has been difficult', 'been difficult', 'difficult this term',
                'difficult this', 'trouble buying', 'struggle to buy', 'struggles to buy',
                'hard to buy', 'problem buying', 'unable to buy', 'cannot buy',
                'afford buying', 'difficult providing', 'trouble providing',
                'struggle to provide', 'struggles to provide', 'hard to provide',
                'difficult to afford', 'struggles to afford', 'trouble affording',
                'hard to afford', 'difficulty affording', 'cannot provide',
                'unable to provide', 'trouble purchasing', 'problem purchasing',
                # Economic stress phrases
                'money is tight', 'financial constraints', 'economic hardship',
                'financial stress', 'economic difficulties', 'economic problems',
                'financial pressure', 'economic pressure', 'struggling financially',
                'financially struggling', 'hard times', 'difficult times',
                # Direct difficulty mentions
                'been challenging', 'challenging to buy', 'challenging buying',
                'been hard', 'hard this term', 'struggle this term'
            ]),
            
            # Family flags - comprehensive patterns
            'father_absent_flag': has_any_pattern([
                'father absent', 'no father', 'father left', 'father not supporting',
                'father gone', 'father departed', 'no dad', 'dad left', 'father missing',
                'father walked out', 'father abandoned', 'abandoned', 'abandoned by',
                'father not present', 'father not around', 'father disappeared',
                'father not there', 'without father', 'fatherless'
            ]),
            
            'mother_hawker_flag': has_any_pattern([
                'mother hawker', 'hawker', 'sells vegetables', 'vegetable stall',
                'sells food', 'vendor', 'street vendor', 'selling', 'sells goods',
                'sells by roadside', 'roadside seller', 'market seller', 'hawking',
                'small business', 'petty trader', 'informal trader'
            ]),
            
            'single_parent_flag': has_any_pattern([
                'single parent', 'single mother', 'single father', 'single mum', 'single dad',
                'lives with mother', 'lives with mum', 'mother and', 'only mother',
                'only mum', 'mother alone', 'abandoned by both parents',
                'both parents left', 'parents abandoned', 'lives with grandmother',
                'lives with grandparent', 'lives with aunt', 'lives with uncle',
                'orphan', 'orphaned', 'no parents', 'parents not present',
                'no parental support', 'lacks parental support', 'without parental support'
            ]),
            
            # Critical risk flags - comprehensive patterns
            'pregnancy_flag': has_any_pattern([
                'pregnant', 'got pregnant', 'pregnancy', 'expecting', 'expecting a child',
                'with child', 'carrying', 'pregnant girl', 'teenage pregnancy',
                'young pregnancy', 'student pregnancy'
            ]),
            
            'missing_school_flag': has_any_pattern([
                'missing school', 'missing classes', 'frequently absent', 'missed several days',
                'missing school frequently', 'absent from school', 'absent frequently',
                'skips school', 'skips classes', 'truant', 'truancy', 'not attending',
                'not going to school', 'stays away from school', 'absentee', 'absenteeism',
                'misses classes', 'frequently misses', 'often absent', 'regular absence',
                'not attending classes', 'missing days', 'absent days', 'truancy problem',
                'often skips', 'regularly skips', 'does not attend'
            ]),
            
            'withdrawn_flag': has_any_pattern([
                'withdrawn', 'isolated', 'feels hopeless', 'hopeless', 'depressed',
                'sad', 'depression', 'lonely', 'alone', 'keeps to self',
                'does not interact', 'quiet', 'very quiet', 'not talking', 'silent',
                'emotional', 'distressed', 'stressed', 'anxious', 'worried',
                'feels hopeless about', 'hopeless about future', 'no hope',
                'gives up', 'wants to give up', 'feels worthless', 'low self esteem',
                'emotional distress', 'mental health', 'psychological', 'trauma'
            ]),
            
            'elderly_caregiver_flag': has_any_pattern([
                'elderly grandmother', 'elderly grandparent', 'elderly caregiver',
                'old grandmother', 'old grandparent', 'elderly guardian',
                'elderly aunt', 'elderly uncle', 'elderly relative', 'elderly person',
                'grandmother is elderly', 'grandmother is old', 'elderly and',
                'too old to', 'unable to provide', 'cannot provide care',
                'elderly cannot', 'old cannot', 'grandmother elderly'
            ]),
        }
        
        return flags
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for the text."""
        try:
            # Generate raw embedding with explicit numpy conversion to avoid device issues
            raw_embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            
            # Apply PCA transformation
            if self.pca_model is not None:
                pca_embedding = self.pca_model.transform(raw_embedding.reshape(1, -1))
                return pca_embedding.flatten()
            else:
                return raw_embedding
                
        except RuntimeError as e:
            if 'meta tensor' in str(e).lower():
                logger.error(f"❌ Meta tensor error in embedding generation: {e}")
                logger.error("This usually indicates the model was loaded in meta mode. Try restarting the application.")
            else:
                logger.error(f"❌ Error generating embeddings: {e}")
            # Return zero vector as fallback
            if self.pca_model is not None:
                return np.zeros(64)  # PCA reduced dimensions
            else:
                return np.zeros(384)  # Raw embedding dimensions
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            # Return zero vector as fallback
            if self.pca_model is not None:
                return np.zeros(64)  # PCA reduced dimensions
            else:
                return np.zeros(384)  # Raw embedding dimensions
    
    def create_composite_features(self, features: Dict) -> Dict:
        """Create composite features for retrained models."""
        composite = {}
        
        # Housing Risk Score
        composite['housing_risk_score'] = (
            features.get('iron_sheets_flag', 0) * 2 + 
            features.get('single_room_flag', 0) * 1.5 + 
            features.get('shared_bed_flag', 0) * 2 + 
            features.get('no_electric_flag', 0) * 1.5
        )
        
        # Economic Stress Score
        composite['economic_stress_score'] = (
            features.get('rent_arrears_flag', 0) * 2 + 
            features.get('landlord_lock_flag', 0) * 2 + 
            features.get('works_unstable_flag', 0) * 1.5 +
            features.get('no_school_fees_flag', 0) * 1.5
        )
        
        # Family Instability Score
        composite['family_instability_score'] = (
            features.get('father_absent_flag', 0) * 2 + 
            features.get('single_parent_flag', 0) * 1.5 + 
            features.get('mother_hawker_flag', 0) * 1
        )
        
        # Age-based indicators - SAFE NoneType handling
        age = features.get('age')
        if age is None or not isinstance(age, (int, float)):
            age = 15  # Default age
        composite['is_very_young'] = 1 if age < 12 else 0
        composite['is_older_student'] = 1 if age > 16 else 0
        
        # Academic performance indicators - SAFE NoneType handling
        exam_score = features.get('last_exam_score')
        if exam_score is None or not isinstance(exam_score, (int, float)):
            exam_score = 250  # Default score
        composite['low_academic_performance'] = 1 if exam_score < 200 else 0
        composite['high_academic_performance'] = 1 if exam_score > 300 else 0
        
        # Text complexity score - SAFE NoneType handling
        text_len = features.get('text_len')
        if text_len is None or not isinstance(text_len, (int, float)):
            text_len = 0
        sentence_count = features.get('sentence_count')
        if sentence_count is None or not isinstance(sentence_count, (int, float)):
            sentence_count = 0
        composite['text_complexity_score'] = (
            text_len / 1000 + 
            sentence_count / 10
        )
        
        # Nutrition indicators - SAFE NoneType handling
        meals = features.get('meals_per_day')
        if meals is None or not isinstance(meals, (int, float)):
            meals = 3  # Default meals
        composite['poor_nutrition'] = 1 if meals <= 1 else 0
        composite['insufficient_meals'] = 1 if meals <= 2 else 0
        
        return composite
    
    def process_upload(self, text: str, uid: str) -> Dict:
        """
        Process a complete upload case through the full feature extraction pipeline.
        
        Args:
            text: The uploaded text content
            uid: Unique identifier for the case
            
        Returns:
            Dictionary containing all extracted features and predictions
        """
        logger.info(f"🔄 Processing upload: {uid}")
        
        try:
            # Step 1: Extract structured features
            structured_features = self.extract_structured_features(text)
            logger.info(f"✅ Extracted structured features: {len(structured_features)} features")
            
            # Step 2: Extract text features
            text_features = self.extract_text_features(text)
            logger.info(f"✅ Extracted text features: {len(text_features)} features")
            
            # Step 3: Extract flag features
            flag_features = self.extract_flag_features(text)
            logger.info(f"✅ Extracted flag features: {len(flag_features)} features")
            
            # Step 4: Generate embeddings
            embeddings = self.generate_embeddings(text)
            logger.info(f"✅ Generated embeddings: {len(embeddings)} dimensions")
            
            # Step 5: Create composite features
            composite_features = self.create_composite_features({
                **structured_features,
                **text_features,
                **flag_features
            })
            logger.info(f"✅ Created composite features: {len(composite_features)} features")
            
            # Step 6: Combine all features
            all_features = {
                'uid': uid,
                **structured_features,
                **text_features,
                **flag_features,
                **composite_features,
                'raw_text': text,  # Store original text for needs detection
                'text_content': text  # Alias for compatibility
            }
            
            # Step 7: Add embeddings
            for i, emb_val in enumerate(embeddings):
                all_features[f'emb_pca_{i}'] = float(emb_val)
            
            logger.info(f"✅ Total features created: {len(all_features)} features")
            
            return all_features
            
        except Exception as e:
            logger.error(f"❌ Error processing upload {uid}: {e}")
            raise
    
    def prepare_for_ml_prediction(self, features: Dict) -> np.ndarray:
        """
        Prepare features for ML model prediction.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Numpy array ready for ML model input
        """
        try:
            # Create DataFrame from features
            df = pd.DataFrame([features])
            
            # Define feature columns in the exact order expected by the scaler
            numeric_cols = ['age', 'last_exam_score', 'meals_per_day', 'siblings_count', 'sentence_count', 'text_len', 'last_exam_score']
            flag_cols = ['iron_sheets_flag', 'single_room_flag', 'shared_bed_flag', 'no_electric_flag', 'rent_arrears_flag', 'hunger_flag', 'father_absent_flag', 'mother_hawker_flag', 'landlord_lock_flag', 'no_school_fees_flag', 'works_unstable_flag', 'single_parent_flag']
            score_cols = ['housing_risk_score', 'economic_stress_score', 'family_instability_score', 'text_complexity_score']
            indicator_cols = ['is_very_young', 'is_older_student', 'low_academic_performance', 'high_academic_performance', 'poor_nutrition', 'insufficient_meals']
            
            # Get PCA embedding columns
            emb_pca_cols = [c for c in df.columns if c.startswith('emb_pca_')]
            
            # Ensure all required columns exist, create missing ones with defaults
            for col in numeric_cols + flag_cols + score_cols + indicator_cols:
                if col not in df.columns:
                    if col in ['age', 'last_exam_score', 'meals_per_day', 'siblings_count', 'sentence_count', 'text_len']:
                        df[col] = 0  # Default numeric values
                    elif col.endswith('_flag'):
                        df[col] = 0  # Default flag values
                    elif 'score' in col:
                        df[col] = 0.0  # Default score values
                    elif col.startswith('is_') or col.startswith('low_') or col.startswith('high_') or col.startswith('poor_') or col.startswith('insufficient_'):
                        df[col] = 0  # Default indicator values
            
            # Prepare feature matrix
            X_numeric = df[numeric_cols].fillna(0)
            X_flags = df[flag_cols].fillna(0)
            X_scores = df[score_cols].fillna(0)
            X_indicators = df[indicator_cols].fillna(0)
            X_emb_pca = df[emb_pca_cols].values if emb_pca_cols else np.empty((len(df), 0))
            
            # Combine all features
            X = np.hstack([X_numeric.values, X_flags.values, X_scores.values, X_indicators.values, X_emb_pca])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            logger.info(f"✅ Prepared ML input: {X_scaled.shape[1]} features")
            return X_scaled
            
        except Exception as e:
            logger.error(f"❌ Error preparing ML input: {e}")
            raise
    
    def predict_risk_profile(self, features: Dict, ml_models: Dict) -> Dict:
        """
        Predict risk profile using OPTION 1: Heuristics as Primary Assessment.
        
        This approach uses heuristics as the primary method and ML as secondary validation.
        For cases where ML and heuristics disagree, heuristics takes precedence.
        
        Args:
            features: Dictionary of extracted features
            ml_models: Dictionary containing trained ML models
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            # PRIMARY METHOD: Use heuristics (more reliable for critical cases)
            from heuristics import compute_risk_score, score_to_label
            
            # Calculate heuristic score FIRST
            # Adjusted thresholds: LOW (0-3), MEDIUM (4-6), HIGH (7+)
            # But consider positive indicators for borderline cases
            heuristic_score = compute_risk_score(pd.Series(features))
            
            # Calculate needs BEFORE determining priority (so we can use needs count for priority adjustment)
            heuristic_needs = self._estimate_needs_from_heuristics(features, heuristic_score)
            
            # Adjust priority for borderline cases with positive indicators
            positive_count = sum([
                features.get('attends_regularly', 0),
                features.get('decent_academic', 0),
                features.get('has_support', 0),
                features.get('mother_present', 0)
            ])
            
            # Count total needs to ensure cases with multiple needs get appropriate priority
            total_needs = sum([
                heuristic_needs.get('need_food', 0),
                heuristic_needs.get('need_school_fees', 0),
                heuristic_needs.get('need_housing', 0),
                heuristic_needs.get('need_economic', 0),
                heuristic_needs.get('need_family_support', 0),
                heuristic_needs.get('need_health', 0),
                heuristic_needs.get('need_counseling', 0)
            ])
            
            # Check for elderly caregiver
            has_elderly_caregiver = (
                features.get('elderly_caregiver_flag', 0) == 1 or
                any(phrase in str(features.get('text_content', '')).lower() for phrase in [
                    'grandmother', 'grandfather', 'grandparent', 'elderly'
                ]) if 'text_content' in features else False
            )
            
            # Priority adjustment based on needs count and complexity
            # Cases with 3+ needs OR elderly caregiver should be at least MEDIUM
            # HIGH priority if score >= 6
            if total_needs >= 3 or has_elderly_caregiver:
                # Multiple needs or elderly caregiver = minimum MEDIUM priority
                if heuristic_score >= 6:
                    heuristic_priority = 'high'
                elif heuristic_score >= 4:
                    heuristic_priority = 'medium'
                else:
                    # Even low score, but 3+ needs or elderly caregiver = MEDIUM
                    heuristic_priority = 'medium'
            elif total_needs >= 2:
                # 2 needs = minimum MEDIUM priority (cases with 2+ needs need intervention)
                if heuristic_score >= 6:
                    heuristic_priority = 'high'
                else:
                    # 2+ needs = MEDIUM priority (even if score is low)
                    heuristic_priority = 'medium'
            else:
                # 0-1 needs: use normal thresholds, but consider positive indicators for borderline cases
                if heuristic_score >= 6:
                    heuristic_priority = 'high'
                elif heuristic_score >= 4:
                    heuristic_priority = 'medium'
                else:
                    heuristic_priority = 'low'
            heuristic_dropout_risk = 1 if heuristic_score >= 6 else 0
            
            logger.info(f"🎯 Heuristic Assessment: score={heuristic_score}, priority={heuristic_priority}, dropout={heuristic_dropout_risk}")
            
            # SECONDARY METHOD: Use ML models for validation and needs prediction
            ml_priority = None
            ml_dropout_risk = None
            ml_needs = None
            ml_confidence = 0.5
            
            try:
                # Prepare features for ML prediction
                X_scaled = self.prepare_for_ml_prediction(features)
                
                # Make ML predictions
                needs_pred = ml_models['needs'].predict(X_scaled)[0]
                priority_pred_encoded = ml_models['priority'].predict(X_scaled)[0]
                dropout_pred = ml_models['dropout'].predict(X_scaled)[0]
                
                # Get probabilities for confidence
                needs_proba = ml_models['needs'].predict_proba(X_scaled)
                priority_proba = ml_models['priority'].predict_proba(X_scaled)[0]
                dropout_proba = ml_models['dropout'].predict(X_scaled)[0]
                
                # Decode priority
                ml_priority = self.priority_encoder.inverse_transform([priority_pred_encoded])[0]
                ml_dropout_risk = int(dropout_pred)
                
                # Format needs predictions
                needs_labels = ['need_food', 'need_school_fees', 'need_housing', 'need_economic', 
                               'need_family_support', 'need_health', 'need_counseling']
                ml_needs = {label: int(pred) for label, pred in zip(needs_labels, needs_pred)}
                
                # Calculate ML confidence
                ml_confidence = float(max(priority_proba))
                
                logger.info(f"🤖 ML Assessment: priority={ml_priority}, dropout={ml_dropout_risk}, confidence={ml_confidence:.2f}")
                
            except Exception as ml_error:
                logger.warning(f"⚠️ ML prediction failed: {ml_error}")
                ml_priority = None
                ml_dropout_risk = None
                ml_needs = None
                ml_confidence = 0.0
            
            # DECISION LOGIC: Heuristics as Primary, ML as Secondary
            
            # 1. PRIORITY: Use heuristics (catches critical cases ML might miss)
            final_priority = heuristic_priority
            priority_confidence = 0.9  # High confidence in heuristics
            
            # 2. DROPOUT RISK: Use heuristics (more reliable for edge cases)
            final_dropout_risk = heuristic_dropout_risk
            dropout_confidence = 0.9  # High confidence in heuristics
            
            # 3. NEEDS: Use heuristics if critical flags present OR ML confidence is low
            # Critical flags (pregnancy, missing school, etc.) require comprehensive needs assessment
            has_critical_flags = (
                features.get('pregnancy_flag', 0) == 1 or
                features.get('missing_school_flag', 0) == 1 or
                features.get('withdrawn_flag', 0) == 1 or
                heuristic_score >= 10
            )
            
            # Heuristic needs already calculated above for priority adjustment
            # (no need to recalculate - using heuristic_needs from line 777)
            
            if has_critical_flags or ml_confidence < 0.6:
                # Use heuristics for critical cases - more reliable
                final_needs = heuristic_needs
                needs_confidence = 0.85  # High confidence in heuristics for critical cases
                logger.info(f"✅ Using heuristics for needs prediction (critical flags detected or low ML confidence)")
            elif ml_needs is not None:
                # Use ML for less critical cases, BUT override with heuristics for critical indicators
                final_needs = ml_needs.copy()
                # Override housing need with heuristic assessment (ML has bias on single_room)
                final_needs['need_housing'] = heuristic_needs['need_housing']
                # Override school fees and economic needs if flags are detected (ML might miss these)
                if features.get('no_school_fees_flag', 0) == 1 or \
                   features.get('economic_stress_indicators', 0) == 1:
                    final_needs['need_school_fees'] = heuristic_needs['need_school_fees']
                    final_needs['need_economic'] = heuristic_needs['need_economic']
                needs_confidence = ml_confidence
                logger.info(f"✅ Using ML for needs prediction (housing and flagged needs use heuristics)")
            else:
                # Fallback to heuristics
                final_needs = heuristic_needs
                needs_confidence = 0.8
                logger.info(f"✅ Using heuristics for needs prediction (ML unavailable)")
            
            # 4. CONFIDENCE SCORES: Based on method used
            confidence_scores = {}
            if ml_needs is not None:
                # Use ML confidence scores for needs
                for i, label in enumerate(needs_labels):
                    if hasattr(needs_proba[i], '__len__') and len(needs_proba[i]) > 1:
                        confidence_scores[label] = float(needs_proba[i][1])
                    else:
                        confidence_scores[label] = 0.7
            else:
                # Use heuristic-based confidence
                for need in final_needs.keys():
                    confidence_scores[need] = 0.7
            
            # 5. AGREEMENT ANALYSIS: Log when methods disagree
            if ml_priority is not None and ml_priority != heuristic_priority:
                logger.warning(f"⚠️ Priority disagreement: Heuristics={heuristic_priority}, ML={ml_priority} → Using heuristics")
            
            if ml_dropout_risk is not None and ml_dropout_risk != heuristic_dropout_risk:
                logger.warning(f"⚠️ Dropout disagreement: Heuristics={heuristic_dropout_risk}, ML={ml_dropout_risk} → Using heuristics")
            
            # Final predictions
            predictions = {
                'needs': final_needs,
                'priority': final_priority,
                'dropout_risk': final_dropout_risk,
                'confidence_scores': confidence_scores,
                'priority_confidence': priority_confidence,
                'dropout_confidence': dropout_confidence,
                'heuristic_score': heuristic_score,
                'method': 'heuristics_primary_ml_secondary',
                'agreement': {
                    'priority_agrees': ml_priority == heuristic_priority if ml_priority else None,
                    'dropout_agrees': ml_dropout_risk == heuristic_dropout_risk if ml_dropout_risk is not None else None,
                    'ml_confidence': ml_confidence
                }
            }
            
            logger.info(f"✅ Final Assessment: priority={final_priority}, dropout={final_dropout_risk}, needs={len([n for n in final_needs.values() if n == 1])}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating predictions: {e}")
            raise
    
    def _estimate_needs_from_heuristics(self, features: Dict, heuristic_score: int) -> Dict:
        """Estimate needs based on heuristic score and features - comprehensive detection for ALL needs."""
        needs = {
            'need_food': 0,
            'need_school_fees': 0,
            'need_housing': 0,
            'need_economic': 0,
            'need_family_support': 0,
            'need_health': 0,
            'need_counseling': 0
        }
        
        # 1. FOOD NEEDS - comprehensive detection
        # Two meals per day is adequate for needy cases - only flag when truly inadequate
        # Flag food needs when: 1 meal or less, OR clear hunger indicators, OR missing meals at school
        meals = features.get('meals_per_day', 3)
        has_hunger_flag = features.get('hunger_flag', 0) == 1
        
        # Check text for explicit food needs indicators
        text_content = str(features.get('text_content', '')).lower() if 'text_content' in features else ''
        if not text_content:
            # Try to get text from original processing if available
            text_content = str(features.get('raw_text', '')).lower() if 'raw_text' in features else ''
        
        # Explicit food need indicators
        explicit_food_indicators = [
            'feeding program', 'requires feeding', 'needs feeding', 'feeding support',
            'misses lunch', 'miss lunch', 'skip lunch', 'skips lunch',
            'no lunch', 'without lunch', 'goes without lunch',
            'meal support', 'food support', 'nutritional support',
            'sometimes misses', 'occasionally misses', 'often misses',
            'sometimes the pupil misses', 'pupil misses lunch', 'misses lunch at school',
            'sometimes miss lunch', 'occasionally miss lunch'
        ]
        has_explicit_food_need = any(indicator in text_content for indicator in explicit_food_indicators)
        
        # Only flag if: 1 meal or less (not 2 meals), OR clear hunger indicators, OR explicit food need
        # Two meals per day is considered adequate UNLESS there are explicit indicators
        if (meals is not None and meals <= 1) or \
           (has_hunger_flag and (meals is None or meals <= 2)) or \
           has_explicit_food_need:
            # 1 meal or less = inadequate, OR hunger indicators even with 2 meals, OR explicit food need
            needs['need_food'] = 1
        
        # 2. SCHOOL FEES NEEDS - flag when explicitly mentioned or needs assistance
        # Check for explicit mentions or assistance needs
        text_content = str(features.get('text_content', '')).lower() if 'text_content' in features else ''
        if not text_content:
            text_content = str(features.get('raw_text', '')).lower() if 'raw_text' in features else ''
        
        school_fees_indicators = [
            'school fee', 'school fees', 'fee assistance', 'fees assistance',
            'partial school fee', 'partial school fees', 'partial fee assistance',
            'cannot pay school', "can't pay school", 'struggles to cover',
            'fees not paid', 'school fees not paid', 'needs fee assistance',
            'school-related expenses', 'school expenses'
        ]
        has_explicit_fee_need = any(indicator in text_content for indicator in school_fees_indicators)
        
        if features.get('no_school_fees_flag', 0) == 1 or has_explicit_fee_need:
            # Direct flag for school fees OR explicit mention of fee assistance
            needs['need_school_fees'] = 1
        
        # 3. HOUSING NEEDS - comprehensive detection
        # Only flag housing needs when there are ACTUAL housing problems
        # A one-room house with electricity is decent, not a problem
        # Housing needs trigger when: poor roofing (iron sheets), no electricity, overcrowding (shared bed), 
        # eviction/lockout, OR multiple housing issues together
        has_housing_problem = False
        
        # Get text for explicit housing mentions
        text_content = str(features.get('text_content', '')).lower() if 'text_content' in features else ''
        if not text_content:
            text_content = str(features.get('raw_text', '')).lower() if 'raw_text' in features else ''
        
        # Critical housing problems
        if (features.get('iron_sheets_flag', 0) == 1 or  # Poor roofing
            features.get('no_electric_flag', 0) == 1 or  # No electricity
            features.get('shared_bed_flag', 0) == 1 or  # Overcrowding
            features.get('landlord_lock_flag', 0) == 1):  # Eviction/lockout
            has_housing_problem = True
        
        # Single room only becomes a problem if combined with other issues
        # BUT: "rented single room" in slum area (Mukuru) with no amenities = housing need
        if features.get('single_room_flag', 0) == 1:
            # Check if it's in a slum/informal area (higher risk)
            slum_indicators = ['mukuru', 'kibera', 'mathare', 'kawangware', 'dandora', 'kayole']
            is_slum_area = any(indicator in text_content for indicator in slum_indicators)
            
            # Only flag if combined with no electricity, overcrowding, poor roofing, OR in slum area
            if (features.get('no_electric_flag', 0) == 1 or 
                features.get('shared_bed_flag', 0) == 1 or
                features.get('iron_sheets_flag', 0) == 1 or
                is_slum_area):
                has_housing_problem = True
        
        # High housing risk score indicates multiple problems
        if features.get('housing_risk_score', 0) > 3:
            has_housing_problem = True
        
        if has_housing_problem:
            needs['need_housing'] = 1
        
        # 4. ECONOMIC NEEDS - comprehensive detection
        # Any economic hardship, poverty, or financial instability
        if (features.get('rent_arrears_flag', 0) == 1) or \
           (features.get('landlord_lock_flag', 0) == 1) or \
           (features.get('works_unstable_flag', 0) == 1) or \
           (features.get('mother_hawker_flag', 0) == 1) or \
           (features.get('elderly_caregiver_flag', 0) == 1) or \
           (features.get('economic_stress_score', 0) > 0) or \
           (features.get('economic_stress_indicators', 0) == 1) or \
           (features.get('no_school_fees_flag', 0) == 1) or \
           (features.get('hunger_flag', 0) == 1):
            # Small income, elderly caregiver, can't pay fees, hunger = economic stress
            needs['need_economic'] = 1
        
        # 5. FAMILY SUPPORT NEEDS - comprehensive detection
        # Absence of parental support, single parent, elderly caregiver, or family instability
        # Also check text for explicit mentions
        text_content = str(features.get('text_content', '')).lower() if 'text_content' in features else ''
        if not text_content:
            text_content = str(features.get('raw_text', '')).lower() if 'raw_text' in features else ''
        
        # Check for elderly caregiver in text
        elderly_in_text = any(phrase in text_content for phrase in [
            'grandmother', 'grandfather', 'grandparent', 'elderly', 'old caregiver',
            'lives with grandmother', 'lives with grandfather', 'cared for by grandmother',
            'grandmother does', 'grandfather does', 'grandmother works'
        ])
        
        # Check for missing parents (no mention of parents but mention of caregiver)
        parents_absent = any(phrase in text_content for phrase in [
            'lives with grandmother', 'lives with grandfather', 'lives with',
            'no parents', 'parents not', 'without parents'
        ]) and not any(phrase in text_content for phrase in ['mother', 'father', 'parent', 'parents'])
        
        if (features.get('father_absent_flag', 0) == 1) or \
           (features.get('single_parent_flag', 0) == 1) or \
           (features.get('mother_hawker_flag', 0) == 1) or \
           (features.get('elderly_caregiver_flag', 0) == 1) or \
           elderly_in_text or \
           parents_absent or \
           (features.get('pregnancy_flag', 0) == 1) or \
           (features.get('family_instability_score', 0) > 0) or \
           (features.get('withdrawn_flag', 0) == 1):
            # Mother working, elderly caregiver, pregnancy, withdrawn = need family support
            needs['need_family_support'] = 1
        
        # 6. HEALTH NEEDS - only flag when health issues are explicitly mentioned
        # Do NOT flag based on meals alone - that's a food need, not health
        # Do NOT flag based on pregnancy/withdrawn/missing school - those are other needs
        health_status = str(features.get('health_status', '')).lower()
        
        # Only flag if health status explicitly mentions health problems
        if health_status and ('poor' in health_status or 'sick' in health_status or 'ill' in health_status or
            'disease' in health_status or 'condition' in health_status or 'health' in health_status):
            # Health problems explicitly mentioned in health status field
            needs['need_health'] = 1
        
        # 7. COUNSELING NEEDS - only flag for explicit emotional/mental health issues
        # Do NOT flag just because of economic stress, age, or family structure
        # Only flag for clear mental health/emotional distress indicators
        if (features.get('pregnancy_flag', 0) == 1) or \
           (features.get('withdrawn_flag', 0) == 1) or \
           (features.get('missing_school_flag', 0) == 1):
            # Only flag for explicit mental health/emotional distress indicators:
            # - Pregnancy (requires counseling support)
            # - Withdrawn behavior (mental health issue)
            # - Missing school (can indicate emotional distress)
            needs['need_counseling'] = 1
        
        return needs


def test_upload_processor():
    """Test the complete upload processor."""
    print("🧪 TESTING COMPLETE UPLOAD PROCESSOR")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = CompleteUploadProcessor()
        print("✅ Processor initialized successfully")
        
        # Test case
        test_text = """
        NAME: John Doe
        AGE: 12
        CLASS: 6
        HEALTH: Good
        LAST EXAM SCORE: 0 (missed exams)
        MEALS: 0 (comes to school hungry)
        SIBLINGS: Mary, Peter, Sarah
        
        John is a 12-year-old boy in class 6. He lives in a single room with iron sheets.
        His father is absent and his mother works as a hawker. They have rent arrears
        and the landlord has locked them out. John often comes to school hungry and
        has missed his last exams. He struggles with school fees and the family cannot
        afford basic necessities.
        """
        
        print(f"\n📝 Test Case:")
        print(f"  Text length: {len(test_text)} characters")
        print(f"  Contains: age, missed exams, hunger, rent arrears, family issues")
        
        # Process upload
        features = processor.process_upload(test_text, "test_case_001")
        
        print(f"\n📊 Extracted Features:")
        print(f"  Age: {features.get('age')}")
        print(f"  Class: {features.get('class')}")
        print(f"  Exam Score: {features.get('last_exam_score')}")
        print(f"  Meals: {features.get('meals_per_day')}")
        print(f"  Siblings: {features.get('siblings_count')}")
        print(f"  Text Length: {features.get('text_len')}")
        print(f"  Critical Flags: {features.get('critical_flags')}")
        
        # Show flag features
        flag_features = {k: v for k, v in features.items() if k.endswith('_flag')}
        print(f"\n🚩 Flag Features:")
        for flag, value in flag_features.items():
            if value == 1:
                print(f"  {flag}: {value}")
        
        # Show composite features
        composite_features = {k: v for k, v in features.items() if 'score' in k or 'is_' in k}
        print(f"\n🔧 Composite Features:")
        for comp, value in composite_features.items():
            print(f"  {comp}: {value:.2f}")
        
        # Show embedding info
        emb_features = {k: v for k, v in features.items() if k.startswith('emb_pca_')}
        print(f"\n🧠 Embeddings:")
        print(f"  PCA dimensions: {len(emb_features)}")
        print(f"  Sample values: {list(emb_features.values())[:5]}")
        
        print(f"\n✅ Upload processing completed successfully!")
        print(f"✅ Total features extracted: {len(features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_upload_processor()
