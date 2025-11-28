#!/usr/bin/env python3
"""
Test script to verify the upload system is working correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from complete_upload_processor import CompleteUploadProcessor

def test_upload_system():
    """Test the complete upload system with the case from the image."""
    
    print("🧪 TESTING UPLOAD SYSTEM")
    print("=" * 50)
    
    # Initialize processor
    processor = CompleteUploadProcessor()
    
    # Load ML models
    models = {
        'needs': joblib.load('models/realistic_random_forest_needs_model.pkl'),
        'priority': joblib.load('models/realistic_random_forest_priority_model.pkl'),
        'dropout': joblib.load('models/realistic_random_forest_dropout_model.pkl')
    }
    
    # The case from the image (14-year-old girl)
    test_text = """
The pupil is a 14-year-old girl in Class 7 living in an informal settlement with her mother and two younger siblings.
The father left the family two years ago and has not been supporting them since.
The mother sells vegetables by the roadside but sometimes goes for days without earning anything.
The pupil said they are often sent home for school fees and sometimes go without meals.
Their single-room house was recently locked by the landlord due to rent arrears, and they spent two nights at a neighbor's house.
The pupil reports frequent fights between her mother and the landlord when rent is delayed.
Teachers say she looks withdrawn and has missed several days of school this term.
She mentioned feeling "too tired to read" and "worried about what will happen to my family."
"""
    
    print(f"📝 TEST CASE:")
    print(f"  Age: 14-year-old girl")
    print(f"  Class: 7")
    print(f"  Issues: Father absent, rent arrears, hunger, school fees, housing locked")
    print()
    
    # Process the case
    print("🔄 PROCESSING CASE...")
    features = processor.process_upload(test_text, 'test_case')
    
    print(f"\n🚩 FLAG DETECTION:")
    flag_features = {k: v for k, v in features.items() if k.endswith('_flag')}
    detected_flags = []
    for flag, value in flag_features.items():
        if value == 1:
            detected_flags.append(flag)
            print(f"  ✅ {flag}")
        else:
            print(f"  ❌ {flag}")
    
    print(f"\n🎯 HEURISTIC ASSESSMENT:")
    from heuristics import compute_risk_score, score_to_label
    heuristic_score = compute_risk_score(pd.Series(features))
    heuristic_priority = score_to_label(heuristic_score)
    heuristic_dropout = 1 if heuristic_score >= 7 else 0
    
    print(f"  Heuristic Score: {heuristic_score}")
    print(f"  Priority: {heuristic_priority.upper()}")
    print(f"  Dropout Risk: {'HIGH' if heuristic_dropout else 'LOW'}")
    
    print(f"\n📋 NEEDS ASSESSMENT:")
    predictions = processor.predict_risk_profile(features, models)
    needs = predictions['needs']
    detected_needs = []
    for need, value in needs.items():
        if value == 1:
            detected_needs.append(need)
            print(f"  ✅ {need}")
        else:
            print(f"  ❌ {need}")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Priority: {predictions['priority'].upper()}")
    print(f"  Dropout Risk: {'HIGH' if predictions['dropout_risk'] else 'LOW'}")
    print(f"  Total Needs: {sum(needs.values())}/7")
    print(f"  Method: {predictions['method']}")
    
    print(f"\n✅ EXPECTED RESULTS:")
    print(f"  Priority: HIGH (should be HIGH)")
    print(f"  Dropout Risk: HIGH (should be HIGH)")
    print(f"  Needs: 4-5 needs (should be 4-5)")
    print(f"  Flags: 9+ flags (should be 9+)")
    
    # Verify results
    success = True
    if predictions['priority'].lower() != 'high':
        print(f"❌ FAIL: Priority should be HIGH, got {predictions['priority'].upper()}")
        success = False
    else:
        print(f"✅ PASS: Priority is HIGH")
    
    if not predictions['dropout_risk']:
        print(f"❌ FAIL: Dropout risk should be HIGH, got LOW")
        success = False
    else:
        print(f"✅ PASS: Dropout risk is HIGH")
    
    if sum(needs.values()) < 4:
        print(f"❌ FAIL: Should have 4+ needs, got {sum(needs.values())}")
        success = False
    else:
        print(f"✅ PASS: Has {sum(needs.values())} needs")
    
    if len(detected_flags) < 8:
        print(f"❌ FAIL: Should have 8+ flags, got {len(detected_flags)}")
        success = False
    else:
        print(f"✅ PASS: Has {len(detected_flags)} flags")
    
    print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    return success

if __name__ == "__main__":
    test_upload_system()



