"""
heuristics.py

Utilities to compute a heuristic dropout risk score and weak labels
from the structured fields and keyword flags produced by build_features.py.

Usage:
    import pandas as pd
    from heuristics import apply_heuristics

    df = pd.read_csv("usable/features_dataset.csv")
    df_h = apply_heuristics(df)
    df_h.to_csv("usable/features_with_heuristics.csv", index=False)
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


CORE_FLAG_COLUMNS = [
    "rent_arrears_flag",
    "hunger_flag",
    "no_school_fees_flag",
    "father_absent_flag",
    "single_parent_flag",
    "no_electric_flag",
    "shared_bed_flag",
]


def _safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except Exception:
        return 0


def compute_risk_score(row: pd.Series) -> int:
    """Compute a simple additive risk score from flags and numeric features.
    
    Original algorithm - maintains consistency with previous results.
    Enhanced with critical risk factors (pregnancy, missing school, etc.)
    """
    score = 0
    
    # CRITICAL RISK FACTORS (extreme weights - automatically push to HIGH priority)
    # Pregnancy is the highest risk factor - should push to HIGH immediately
    pregnancy_flag = _safe_int(row.get("pregnancy_flag", 0))
    if pregnancy_flag:
        score += 10  # Automatic HIGH priority for pregnancy
    
    # Missing school frequently is a major academic crisis indicator
    missing_school_flag = _safe_int(row.get("missing_school_flag", 0))
    if missing_school_flag:
        score += 3  # Major academic crisis
    
    # Withdrawn/emotional distress indicates severe mental health issues
    withdrawn_flag = _safe_int(row.get("withdrawn_flag", 0))
    if withdrawn_flag:
        score += 2  # Mental health risk
    
    # Elderly caregiver unable to provide consistent care
    # Also check if living with grandmother/grandfather (elderly caregiver indicator)
    elderly_caregiver_flag = _safe_int(row.get("elderly_caregiver_flag", 0))
    
    # Check text for elderly caregiver mentions
    text_content = str(row.get("text_content", "")).lower() if "text_content" in row else ""
    if not text_content:
        text_content = str(row.get("raw_text", "")).lower() if "raw_text" in row else ""
    
    elderly_in_text = any(phrase in text_content for phrase in [
        "grandmother", "grandfather", "grandparent", "elderly", "old caregiver",
        "lives with grandmother", "lives with grandfather", "cared for by grandmother"
    ]) if text_content else False
    
    if elderly_caregiver_flag or elderly_in_text:
        score += 2  # Caregiver instability
    
    # Economic hardship / fees / hunger (adjusted weights)
    score += 2 * _safe_int(row.get("rent_arrears_flag"))
    
    # Hunger scoring with severity consideration
    hunger_severity = _safe_int(row.get("hunger_severity", 0))
    intermittent_hunger = _safe_int(row.get("intermittent_hunger_flag", 0))
    hunger_flag = _safe_int(row.get("hunger_flag", 0))
    
    if hunger_severity == 2:
        # Constant/severe hunger - full weight
        score += 2
    elif hunger_severity == 1 or (hunger_flag == 1 and intermittent_hunger == 1):
        # Intermittent hunger - reduced weight (less critical)
        score += 1
    elif hunger_flag == 1:
        # General hunger flag (severity unknown) - default to intermittent weight
        score += 1
    # else: no hunger, no points
    
    score += 2 * _safe_int(row.get("no_school_fees_flag"))
    
    # Family instability (adjusted weights - consider mother presence)
    father_absent = _safe_int(row.get("father_absent_flag", 0))
    single_parent = _safe_int(row.get("single_parent_flag", 0))
    mother_present = _safe_int(row.get("mother_present", 0))
    
    # Father absent: only count if mother is also absent (reduced weight)
    if father_absent == 1 and mother_present == 0:
        score += 1  # Both parents absent = more critical
    elif father_absent == 1 and mother_present == 1:
        score += 0.5  # Father absent but mother present = less critical (round down to 0)
    
    # Single parent: only if both flags present (already counted father_absent)
    if single_parent == 1 and father_absent == 0:
        score += 1  # Single parent but not due to father absence

    # Housing proxies (original weights)
    score += 1 * _safe_int(row.get("no_electric_flag"))
    score += 1 * _safe_int(row.get("shared_bed_flag"))

    # Meals per day (original scoring)
    meals = row.get("meals_per_day", np.nan)
    if pd.notna(meals):
        try:
            meals_val = float(meals)
            if meals_val <= 1:
                score += 2
            elif meals_val == 2:
                score += 1
        except Exception:
            pass

    # Academic score (adjusted scoring - consider positive indicators)
    # If missing school is flagged, treat exam score as 0 (crisis)
    attends_regularly = _safe_int(row.get("attends_regularly", 0))
    decent_academic = _safe_int(row.get("decent_academic", 0))
    has_support = _safe_int(row.get("has_support", 0))
    
    last_exam = row.get("last_exam_score", np.nan)
    if missing_school_flag and pd.isna(last_exam):
        score += 2  # Missing school + no exam score = academic crisis
    elif pd.notna(last_exam):
        try:
            last_exam_val = float(last_exam)
            if last_exam_val < 200:
                score += 2
            elif last_exam_val < 250:
                score += 1
            # If attends regularly and has decent academic indicators, reduce penalty
            elif attends_regularly == 1 and decent_academic == 1:
                # Decent score (240-250) + attends regularly = no penalty
                pass
        except Exception:
            pass
    
    # POSITIVE INDICATORS: Reduce score for manageable cases
    # BUT: Don't reduce too much if case has multiple needs or elderly caregiver
    # A case with 3+ needs or elderly caregiver should still be at least MEDIUM priority
    positive_count = attends_regularly + decent_academic + has_support + mother_present
    
    # Count total needs (approximate - check if multiple risk factors present)
    multiple_risk_factors = (
        _safe_int(row.get("hunger_flag", 0)) +
        _safe_int(row.get("no_school_fees_flag", 0)) +
        _safe_int(row.get("rent_arrears_flag", 0)) +
        _safe_int(row.get("works_unstable_flag", 0)) +
        _safe_int(row.get("father_absent_flag", 0))
    )
    has_elderly_caregiver = elderly_caregiver_flag or elderly_in_text
    
    # Only reduce score if case doesn't have multiple needs AND doesn't have elderly caregiver
    # Cases with 3+ needs or elderly caregiver are inherently more complex and need intervention
    if (multiple_risk_factors >= 3 or has_elderly_caregiver):
        # Don't reduce score - case has multiple needs or elderly caregiver = needs intervention
        pass
    elif positive_count >= 2:
        # Multiple positive indicators suggest case is manageable - reduce score by 1
        score -= 1
    elif positive_count == 1:
        # Some positive indicator - reduce score by 0.5 (round down to 0)
        score -= 0.5

    return max(0, int(score))  # Ensure score doesn't go negative


def score_to_label(score: int, high_threshold: int = 5, med_threshold: int = 3) -> str:
    if score >= high_threshold:
        return "high"
    if score >= med_threshold:
        return "medium"
    return "low"


def apply_heuristics(
    df: pd.DataFrame,
    high_threshold: int = 7,
    med_threshold: int = 4,
) -> pd.DataFrame:
    """Return a copy of df with heuristic columns added.

    Adds columns: heuristic_score, weak_label, weak_confidence.
    weak_confidence is a simple proxy using LF agreement among core flags.
    """
    out = df.copy()
    out["heuristic_score"] = out.apply(compute_risk_score, axis=1)
    out["weak_label"] = out["heuristic_score"].apply(
        lambda s: score_to_label(s, high_threshold=high_threshold, med_threshold=med_threshold)
    )

    # Simple confidence proxy: proportion of positive core flags among non-missing
    available = out[CORE_FLAG_COLUMNS].notna().sum(axis=1).replace(0, np.nan)
    positives = out[CORE_FLAG_COLUMNS].fillna(0).sum(axis=1)
    conf = (positives / available).fillna(0.0)
    out["weak_confidence"] = conf.clip(0, 1)
    return out


def summarize_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts per weak_label for quick reporting."""
    if "weak_label" not in df.columns:
        df = apply_heuristics(df)
    return df["weak_label"].value_counts(dropna=False).rename_axis("tier").reset_index(name="count")


