#!/usr/bin/env python3
"""
ml_training.py

Comprehensive ML training pipeline for dropout risk assessment with:
- Multi-label needs prediction (food, school_fees, housing, etc.)
- Priority classification (high/medium/low)
- Dropout risk flag (binary)
- Proper train/test/validation splits
- Decision Trees and Random Forest models
- Comprehensive evaluation metrics focusing on high-risk recall

Usage:
    python3 ml_training.py
"""
import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, multilabel_confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

from heuristics import apply_heuristics

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "usable" / "features_dataset.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define needs categories and their corresponding flags
NEEDS_MAPPING = {
    'food': ['hunger_flag', 'meals_per_day'],
    'school_fees': ['no_school_fees_flag'],
    'housing': ['iron_sheets_flag', 'single_room_flag', 'shared_bed_flag', 'no_electric_flag'],
    'economic': ['rent_arrears_flag', 'landlord_lock_flag', 'works_unstable_flag'],
    'family_support': ['father_absent_flag', 'single_parent_flag'],
    'health': ['health_status'],
    'counseling': ['age']  # proxy: very young/old ages may need counseling
}

def load_and_prepare_data() -> pd.DataFrame:
    """Load features and apply heuristic labels."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Features dataset not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    df = apply_heuristics(df)
    print(f"Loaded {len(df)} profiles with heuristic labels")
    return df

def create_needs_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-label needs based on flags and thresholds."""
    needs_df = pd.DataFrame(index=df.index)
    
    # Food need: hunger flag OR meals <= 2
    needs_df['need_food'] = (
        df['hunger_flag'].fillna(0) | 
        (df['meals_per_day'].fillna(3) <= 2)
    ).astype(int)
    
    # School fees need: direct flag
    needs_df['need_school_fees'] = df['no_school_fees_flag'].fillna(0).astype(int)
    
    # Housing need: any housing flag
    housing_flags = ['iron_sheets_flag', 'single_room_flag', 'shared_bed_flag', 'no_electric_flag']
    needs_df['need_housing'] = df[housing_flags].fillna(0).any(axis=1).astype(int)
    
    # Economic need: rent arrears or unstable work
    economic_flags = ['rent_arrears_flag', 'landlord_lock_flag', 'works_unstable_flag']
    needs_df['need_economic'] = df[economic_flags].fillna(0).any(axis=1).astype(int)
    
    # Family support need: father absent or single parent
    family_flags = ['father_absent_flag', 'single_parent_flag']
    needs_df['need_family_support'] = df[family_flags].fillna(0).any(axis=1).astype(int)
    
    # Health need: based on health status (simple heuristic)
    needs_df['need_health'] = (
        df['health_status'].fillna('').str.contains('poor|sick|ill|problem', case=False, na=False)
    ).astype(int)
    
    # Counseling need: very young (<12) or older (>18) students, or high stress indicators
    stress_score = df[['hunger_flag', 'rent_arrears_flag', 'father_absent_flag']].fillna(0).sum(axis=1)
    needs_df['need_counseling'] = (
        (df['age'].fillna(15) < 12) | 
        (df['age'].fillna(15) > 18) |
        (stress_score >= 2)
    ).astype(int)
    
    return needs_df

def create_priority_labels(df: pd.DataFrame, needs_df: pd.DataFrame) -> pd.Series:
    """Create priority labels based on number of needs and severity."""
    # Count total needs per profile
    total_needs = needs_df.sum(axis=1)
    
    # High priority: 4+ needs OR critical combination
    critical_combo = (
        needs_df['need_food'] & needs_df['need_school_fees'] & needs_df['need_economic']
    )
    
    priority = pd.Series('low', index=df.index)
    priority[total_needs >= 2] = 'medium'
    priority[(total_needs >= 4) | critical_combo] = 'high'
    
    return priority

def create_dropout_flag(df: pd.DataFrame, priority: pd.Series, needs_df: pd.DataFrame) -> pd.Series:
    """Create dropout risk flag for high-priority cases with specific risk factors."""
    # Dropout risk: high priority + academic/engagement risk indicators
    academic_risk = (df['last_exam_score'].fillna(400) < 250)  # Low exam scores
    
    # Age risk: significantly older than typical (grade repetition indicator)
    age_risk = (df['age'].fillna(15) > 17)
    
    # Multiple severe needs
    severe_needs = (
        needs_df['need_food'] & needs_df['need_school_fees'] & 
        (needs_df['need_economic'] | needs_df['need_family_support'])
    )
    
    dropout_flag = (
        (priority == 'high') & 
        (academic_risk | age_risk | severe_needs)
    ).astype(int)
    
    return dropout_flag

def prepare_features(df: pd.DataFrame, pca_components: int = 64) -> Tuple[np.ndarray, List[str]]:
    """Prepare feature matrix with structured fields + PCA-reduced embeddings."""
    # Structured features
    numeric_cols = ['age', 'last_exam_score', 'meals_per_day', 'siblings_count', 'sentence_count', 'text_len']
    flag_cols = [c for c in df.columns if c.endswith('_flag')]
    
    # Handle missing values
    X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    X_flags = df[flag_cols].fillna(0)
    
    # PCA-reduced embeddings
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    if emb_cols:
        X_emb = df[emb_cols].values
        pca = PCA(n_components=min(pca_components, len(emb_cols)), random_state=42)
        X_emb_pca = pca.fit_transform(X_emb)
        emb_feature_names = [f'emb_pca_{i}' for i in range(X_emb_pca.shape[1])]
        
        # Save PCA transformer
        joblib.dump(pca, MODELS_DIR / 'pca_transformer.pkl')
    else:
        X_emb_pca = np.empty((len(df), 0))
        emb_feature_names = []
    
    # Combine features
    X = np.hstack([X_numeric.values, X_flags.values, X_emb_pca])
    feature_names = list(numeric_cols) + list(flag_cols) + emb_feature_names
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / 'feature_scaler.pkl')
    
    return X_scaled, feature_names

def train_models(X: np.ndarray, y_needs: pd.DataFrame, y_priority: pd.Series, 
                y_dropout: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    """Train Decision Tree and Random Forest models with proper evaluation."""
    
    # Encode priority labels
    le_priority = LabelEncoder()
    y_priority_encoded = le_priority.fit_transform(y_priority)
    joblib.dump(le_priority, MODELS_DIR / 'priority_encoder.pkl')
    
    # Train/test split (stratified on priority)
    X_train, X_test, y_needs_train, y_needs_test, y_priority_train, y_priority_test, y_dropout_train, y_dropout_test = train_test_split(
        X, y_needs, y_priority_encoded, y_dropout, 
        test_size=0.2, random_state=42, stratify=y_priority_encoded
    )
    
    # Further split train into train/val
    X_train, X_val, y_needs_train, y_needs_val, y_priority_train, y_priority_val, y_dropout_train, y_dropout_val = train_test_split(
        X_train, y_needs_train, y_priority_train, y_dropout_train,
        test_size=0.25, random_state=42, stratify=y_priority_train  # 0.25 * 0.8 = 0.2 of total
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    models = {}
    results = {}
    
    # Model configurations
    model_configs = {
        'decision_tree': {
            'needs': DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42),
            'priority': DecisionTreeClassifier(max_depth=8, min_samples_split=15, min_samples_leaf=8, random_state=42),
            'dropout': DecisionTreeClassifier(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=42)
        },
        'random_forest': {
            'needs': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10, 
                                          min_samples_leaf=5, random_state=42, class_weight='balanced'),
            'priority': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=8,
                                             min_samples_leaf=4, random_state=42, class_weight='balanced'),
            'dropout': RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=6,
                                            min_samples_leaf=3, random_state=42, class_weight='balanced')
        }
    }
    
    for model_name, configs in model_configs.items():
        print(f"\nTraining {model_name}...")
        models[model_name] = {}
        results[model_name] = {}
        
        # Multi-label needs prediction
        needs_model = MultiOutputClassifier(configs['needs'])
        needs_model.fit(X_train, y_needs_train)
        models[model_name]['needs'] = needs_model
        
        # Priority classification
        priority_model = configs['priority']
        priority_model.fit(X_train, y_priority_train)
        models[model_name]['priority'] = priority_model
        
        # Dropout risk classification
        dropout_model = configs['dropout']
        dropout_model.fit(X_train, y_dropout_train)
        models[model_name]['dropout'] = dropout_model
        
        # Evaluate on validation set
        results[model_name] = evaluate_models(
            models[model_name], X_val, y_needs_val, y_priority_val, y_dropout_val, le_priority
        )
        
        print(f"{model_name} validation results:")
        print(f"  Priority accuracy: {results[model_name]['priority']['accuracy']:.3f}")
        print(f"  High priority recall: {results[model_name]['priority']['high_recall']:.3f}")
        print(f"  Dropout AUC: {results[model_name]['dropout']['auc']:.3f}")
    
    # Select best model based on high priority recall
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['priority']['high_recall'])
    print(f"\nBest model: {best_model_name}")
    
    # Final evaluation on test set
    final_results = evaluate_models(
        models[best_model_name], X_test, y_needs_test, y_priority_test, y_dropout_test, le_priority
    )
    
    # Save best models
    for task, model in models[best_model_name].items():
        joblib.dump(model, MODELS_DIR / f'{best_model_name}_{task}_model.pkl')
    
    # Save feature names
    with open(MODELS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    return {
        'best_model': best_model_name,
        'validation_results': results,
        'test_results': final_results,
        'feature_names': feature_names,
        'splits': {
            'train_size': len(X_train),
            'val_size': len(X_val), 
            'test_size': len(X_test)
        }
    }

def evaluate_models(models: Dict[str, Any], X: np.ndarray, y_needs: pd.DataFrame, 
                   y_priority: np.ndarray, y_dropout: np.ndarray, le_priority: LabelEncoder) -> Dict[str, Any]:
    """Comprehensive evaluation of all models."""
    results = {}
    
    # Needs prediction evaluation
    y_needs_pred = models['needs'].predict(X)
    needs_report = {}
    for i, need in enumerate(y_needs.columns):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_needs.iloc[:, i], y_needs_pred[:, i], average='binary', zero_division=0
        )
        needs_report[need] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    results['needs'] = needs_report
    
    # Priority classification evaluation
    y_priority_pred = models['priority'].predict(X)
    priority_accuracy = accuracy_score(y_priority, y_priority_pred)
    
    # Get per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_priority, y_priority_pred, average=None, zero_division=0
    )
    
    priority_classes = le_priority.classes_
    high_idx = np.where(priority_classes == 'high')[0][0] if 'high' in priority_classes else 0
    
    results['priority'] = {
        'accuracy': priority_accuracy,
        'high_recall': recall[high_idx] if high_idx < len(recall) else 0,
        'high_precision': precision[high_idx] if high_idx < len(precision) else 0,
        'high_f1': f1[high_idx] if high_idx < len(f1) else 0,
        'confusion_matrix': confusion_matrix(y_priority, y_priority_pred).tolist(),
        'classification_report': classification_report(y_priority, y_priority_pred, 
                                                     target_names=priority_classes, output_dict=True)
    }
    
    # Dropout risk evaluation
    y_dropout_pred = models['dropout'].predict(X)
    dropout_accuracy = accuracy_score(y_dropout, y_dropout_pred)
    
    # Get probabilities for AUC
    if hasattr(models['dropout'], 'predict_proba'):
        y_dropout_proba = models['dropout'].predict_proba(X)[:, 1]
        dropout_auc = roc_auc_score(y_dropout, y_dropout_proba)
    else:
        dropout_auc = 0.5
    
    dropout_precision, dropout_recall, dropout_f1, _ = precision_recall_fscore_support(
        y_dropout, y_dropout_pred, average='binary', zero_division=0
    )
    
    results['dropout'] = {
        'accuracy': dropout_accuracy,
        'precision': dropout_precision,
        'recall': dropout_recall,
        'f1': dropout_f1,
        'auc': dropout_auc,
        'confusion_matrix': confusion_matrix(y_dropout, y_dropout_pred).tolist()
    }
    
    return results

def save_results(results: Dict[str, Any], output_path: Path):
    """Save training results to JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_clean = convert_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)

def main():
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("Creating labels...")
    needs_df = create_needs_labels(df)
    priority_series = create_priority_labels(df, needs_df)
    dropout_series = create_dropout_flag(df, priority_series, needs_df)
    
    print("Label distribution:")
    print(f"Priority: {priority_series.value_counts().to_dict()}")
    print(f"Dropout risk: {dropout_series.value_counts().to_dict()}")
    print(f"Needs summary: {needs_df.sum().to_dict()}")
    
    print("Preparing features...")
    X, feature_names = prepare_features(df)
    
    print("Training models...")
    results = train_models(X, needs_df, priority_series, dropout_series, feature_names)
    
    print("Saving results...")
    save_results(results, MODELS_DIR / 'training_results.json')
    
    print(f"\nTraining complete! Models saved to {MODELS_DIR}")
    print(f"Best model: {results['best_model']}")
    print(f"Test results - High priority recall: {results['test_results']['priority']['high_recall']:.3f}")
    print(f"Test results - Dropout AUC: {results['test_results']['dropout']['auc']:.3f}")

if __name__ == "__main__":
    main()
