#!/usr/bin/env python3
"""
recommendations.py

Recommendation engine that maps predicted needs to specific actionable interventions.
"""
from typing import Dict, List, Tuple
import pandas as pd

# Action recommendations mapped to needs
NEED_ACTIONS = {
    'need_food': [
        "Provide food vouchers or meal subsidies",
        "Connect to school feeding program",
        "Link to local food bank or community kitchen",
        "Assess for government food assistance programs"
    ],
    'need_school_fees': [
        "Apply for school fee waiver or scholarship",
        "Connect to education sponsorship programs", 
        "Arrange payment plan with school administration",
        "Seek community education fund support"
    ],
    'need_housing': [
        "Assess housing conditions and safety",
        "Connect to housing improvement programs",
        "Provide temporary accommodation if needed",
        "Link to housing assistance organizations"
    ],
    'need_economic': [
        "Connect to microfinance or small loan programs",
        "Provide job training or skills development",
        "Link to income-generating activities",
        "Connect to emergency financial assistance"
    ],
    'need_family_support': [
        "Provide family counseling services",
        "Connect to single parent support groups",
        "Arrange parenting skills training",
        "Link to community family support programs"
    ],
    'need_health': [
        "Schedule medical assessment and treatment",
        "Connect to healthcare services and insurance",
        "Provide health education and prevention",
        "Link to specialized medical care if needed"
    ],
    'need_counseling': [
        "Provide individual counseling sessions",
        "Connect to peer support groups",
        "Arrange academic counseling and mentoring",
        "Provide stress management and coping skills training"
    ]
}

# Priority-based urgency levels
PRIORITY_URGENCY = {
    'high': 'URGENT - Schedule within 1 week',
    'medium': 'MODERATE - Schedule within 2-4 weeks', 
    'low': 'ROUTINE - Schedule within 1-2 months'
}

# Dropout risk specific actions
DROPOUT_ACTIONS = [
    "IMMEDIATE: Schedule emergency case review",
    "Develop intensive support plan with multiple interventions",
    "Assign dedicated case worker for close monitoring",
    "Coordinate with school administration for academic support",
    "Consider alternative education pathways if appropriate"
]

def generate_personalized_recommendations(needs_pred: Dict[str, int], priority: str, 
                                         dropout_risk: int, flags_detected: Dict[str, int] = None,
                                         confidence_scores: Dict[str, float] = None,
                                         case_uid: str = None) -> Dict[str, any]:
    """
    Generate personalized, actionable recommendations based on detected needs and flags.
    Creates interventions that can be directly linked to case management.
    
    Args:
        needs_pred: Dictionary of need predictions (0/1 for each need type)
        priority: Priority level ('high', 'medium', 'low')
        dropout_risk: Dropout risk flag (0/1)
        flags_detected: Dictionary of detected flags (for personalized interventions)
        confidence_scores: Optional confidence scores for predictions
        case_uid: Case ID for creating actionable links
        
    Returns:
        Dictionary with personalized recommendations, urgency, and action items
    """
    recommendations = {
        'priority_level': priority,
        'urgency': PRIORITY_URGENCY.get(priority, 'ROUTINE'),
        'dropout_risk': bool(dropout_risk),
        'immediate_actions': [],
        'personalized_interventions': [],
        'action_items': [],  # New: actionable items with case management links
        'follow_up_timeline': {},
        'case_notes': [],
        'case_uid': case_uid
    }
    
    # Handle dropout risk first (highest priority)
    if dropout_risk:
        recommendations['immediate_actions'].extend(DROPOUT_ACTIONS)
        recommendations['action_items'].append({
            'title': 'Emergency Case Review',
            'description': 'Schedule immediate case review for high dropout risk',
            'category': 'Urgent',
            'action': 'schedule_review',
            'case_uid': case_uid
        })
        recommendations['case_notes'].append("HIGH DROPOUT RISK - Requires immediate intervention")
    
    # Generate personalized recommendations based on flags and needs
    active_needs = [need for need, value in needs_pred.items() if value == 1]
    
    # Flag-specific personalized interventions
    flag_interventions = {
        'pregnancy_flag': {
            'title': 'Pregnancy Support Program',
            'interventions': [
                'Medical care and prenatal services',
                'Counseling for teen pregnancy',
                'Academic support to prevent dropout',
                'Family planning education'
            ],
            'category': 'Health & Counseling'
        },
        'missing_school_flag': {
            'title': 'School Attendance Intervention',
            'interventions': [
                'Investigate root causes of absenteeism',
                'Connect with school administration',
                'Provide transportation support if needed',
                'Address underlying health or family issues'
            ],
            'category': 'Education & Health'
        },
        'hunger_flag': {
            'title': 'Food Security Program',
            'interventions': [
                'School feeding program enrollment',
                'Food vouchers or meal subsidies',
                'Community food bank connection',
                'Nutrition education for family'
            ],
            'category': 'Food Security'
        },
        'no_school_fees_flag': {
            'title': 'Education Financial Support',
            'interventions': [
                'School fee waiver application',
                'Education sponsorship programs',
                'Payment plan negotiation',
                'Community education fund'
            ],
            'category': 'Education Finance'
        },
        'iron_sheets_flag': {
            'title': 'Housing Improvement',
            'interventions': [
                'Housing condition assessment',
                'Housing improvement grants',
                'Temporary accommodation if unsafe',
                'Connection to housing NGOs'
            ],
            'category': 'Housing'
        },
        'father_absent_flag': {
            'title': 'Family Support Services',
            'interventions': [
                'Single parent support group',
                'Parenting skills training',
                'Family counseling',
                'Community support network'
            ],
            'category': 'Family Support'
        },
        'economic_stress_indicators': {
            'title': 'Economic Empowerment',
            'interventions': [
                'Income-generating activities',
                'Skills training for parents',
                'Microfinance programs',
                'Emergency financial assistance'
            ],
            'category': 'Economic Support'
        }
    }
    
    # Create personalized interventions based on detected flags
    if flags_detected:
        for flag, detected in flags_detected.items():
            if detected and flag in flag_interventions:
                intervention_info = flag_interventions[flag]
                recommendations['personalized_interventions'].append({
                    'category': intervention_info['category'],
                    'title': intervention_info['title'],
                    'actions': intervention_info['interventions'],
                    'triggered_by': flag,
                    'action_item': {
                        'title': f"Initiate {intervention_info['title']}",
                        'description': f"Set up {intervention_info['title'].lower()} for this case",
                        'category': intervention_info['category'],
                        'action': 'add_intervention',
                        'case_uid': case_uid,
                        'intervention_type': intervention_info['title']
                    }
                })
    
    # Generate recommendations for each identified need
    for need in active_needs:
        if need in NEED_ACTIONS:
            actions = NEED_ACTIONS[need]
            confidence_note = ""
            if confidence_scores and need in confidence_scores:
                confidence_note = f" (Confidence: {confidence_scores[need]:.2f})"
            
            recommendations['personalized_interventions'].append({
                'need_category': need.replace('need_', '').replace('_', ' ').title(),
                'actions': actions,
                'confidence': confidence_scores.get(need, 0.0) if confidence_scores else 0.0,
                'note': f"Identified {need.replace('need_', '').replace('_', ' ')} need{confidence_note}",
                'action_item': {
                    'title': f"Address {need.replace('need_', '').replace('_', ' ').title()} Need",
                    'description': f"Coordinate interventions for {need.replace('need_', '').replace('_', ' ')}",
                    'category': need.replace('need_', '').replace('_', ' ').title(),
                    'action': 'add_intervention',
                    'case_uid': case_uid,
                    'intervention_type': need
                }
            })
            
            # Add to action items
            recommendations['action_items'].append({
                'title': f"Address {need.replace('need_', '').replace('_', ' ').title()} Need",
                'description': f"Coordinate interventions for {need.replace('need_', '').replace('_', ' ')}",
                'category': need.replace('need_', '').replace('_', ' ').title(),
                'action': 'add_intervention',
                'case_uid': case_uid
            })
    
    # Set follow-up timeline based on priority and needs
    if priority == 'high' or dropout_risk:
        recommendations['follow_up_timeline'] = {
            'initial_contact': '1-3 days',
            'first_intervention': '1 week',
            'progress_review': '2 weeks',
            'case_review': '1 month'
        }
    elif priority == 'medium':
        recommendations['follow_up_timeline'] = {
            'initial_contact': '1 week',
            'first_intervention': '2-3 weeks',
            'progress_review': '1 month',
            'case_review': '2 months'
        }
    else:
        recommendations['follow_up_timeline'] = {
            'initial_contact': '2-4 weeks',
            'first_intervention': '1-2 months',
            'progress_review': '3 months',
            'case_review': '6 months'
        }
    
    # Add general case notes
    if len(active_needs) >= 4:
        recommendations['case_notes'].append("Multiple needs identified - consider comprehensive support plan")
        recommendations['action_items'].append({
            'title': 'Develop Comprehensive Support Plan',
            'description': 'Create coordinated intervention plan for multiple needs',
            'category': 'Case Planning',
            'action': 'create_plan',
            'case_uid': case_uid
        })
    elif len(active_needs) == 0:
        recommendations['case_notes'].append("No immediate needs identified - routine monitoring recommended")
    
    # Specific combinations that require special attention
    if needs_pred.get('need_food', 0) and needs_pred.get('need_school_fees', 0):
        recommendations['case_notes'].append("Food insecurity + school fees - high risk combination")
        recommendations['action_items'].append({
            'title': 'Coordinated Food & Education Support',
            'description': 'Address food insecurity and school fees together',
            'category': 'Urgent',
            'action': 'add_intervention',
            'case_uid': case_uid
        })
    
    if needs_pred.get('need_family_support', 0) and needs_pred.get('need_counseling', 0):
        recommendations['case_notes'].append("Family issues + counseling needs - consider family therapy")
        recommendations['action_items'].append({
            'title': 'Family Therapy Referral',
            'description': 'Coordinate family counseling and support services',
            'category': 'Family Support',
            'action': 'add_intervention',
            'case_uid': case_uid
        })
    
    return recommendations


def generate_recommendations(needs_pred: Dict[str, int], priority: str, 
                           dropout_risk: int, confidence_scores: Dict[str, float] = None) -> Dict[str, any]:
    """
    Generate actionable recommendations based on predicted needs and risk levels.
    
    Args:
        needs_pred: Dictionary of need predictions (0/1 for each need type)
        priority: Priority level ('high', 'medium', 'low')
        dropout_risk: Dropout risk flag (0/1)
        confidence_scores: Optional confidence scores for predictions
        
    Returns:
        Dictionary with recommendations, urgency, and next steps
    """
    recommendations = {
        'priority_level': priority,
        'urgency': PRIORITY_URGENCY.get(priority, 'ROUTINE'),
        'dropout_risk': bool(dropout_risk),
        'immediate_actions': [],
        'recommended_interventions': [],
        'follow_up_timeline': {},
        'case_notes': []
    }
    
    # Handle dropout risk first (highest priority)
    if dropout_risk:
        recommendations['immediate_actions'].extend(DROPOUT_ACTIONS)
        recommendations['case_notes'].append("HIGH DROPOUT RISK - Requires immediate intervention")
    
    # Generate recommendations for each identified need
    active_needs = [need for need, value in needs_pred.items() if value == 1]
    
    for need in active_needs:
        if need in NEED_ACTIONS:
            actions = NEED_ACTIONS[need]
            
            # Add confidence information if available
            confidence_note = ""
            if confidence_scores and need in confidence_scores:
                confidence_note = f" (Confidence: {confidence_scores[need]:.2f})"
            
            recommendations['recommended_interventions'].append({
                'need_category': need.replace('need_', '').replace('_', ' ').title(),
                'actions': actions,
                'confidence': confidence_scores.get(need, 0.0) if confidence_scores else 0.0,
                'note': f"Identified {need.replace('need_', '').replace('_', ' ')} need{confidence_note}"
            })
    
    # Set follow-up timeline based on priority and needs
    if priority == 'high' or dropout_risk:
        recommendations['follow_up_timeline'] = {
            'initial_contact': '1-3 days',
            'first_intervention': '1 week',
            'progress_review': '2 weeks',
            'case_review': '1 month'
        }
    elif priority == 'medium':
        recommendations['follow_up_timeline'] = {
            'initial_contact': '1 week',
            'first_intervention': '2-3 weeks',
            'progress_review': '1 month',
            'case_review': '2 months'
        }
    else:
        recommendations['follow_up_timeline'] = {
            'initial_contact': '2-4 weeks',
            'first_intervention': '1-2 months',
            'progress_review': '3 months',
            'case_review': '6 months'
        }
    
    # Add general case notes
    if len(active_needs) >= 4:
        recommendations['case_notes'].append("Multiple needs identified - consider comprehensive support plan")
    elif len(active_needs) == 0:
        recommendations['case_notes'].append("No immediate needs identified - routine monitoring recommended")
    
    # Specific combinations that require special attention
    if needs_pred.get('need_food', 0) and needs_pred.get('need_school_fees', 0):
        recommendations['case_notes'].append("Food insecurity + school fees - high risk combination")
    
    if needs_pred.get('need_family_support', 0) and needs_pred.get('need_counseling', 0):
        recommendations['case_notes'].append("Family issues + counseling needs - consider family therapy")
    
    return recommendations

def format_recommendations_for_display(recommendations: Dict[str, any]) -> str:
    """Format recommendations for display in GUI or reports."""
    output = []
    
    # Header
    output.append(f"RISK ASSESSMENT & RECOMMENDATIONS")
    output.append(f"Priority Level: {recommendations['priority_level'].upper()}")
    output.append(f"Urgency: {recommendations['urgency']}")
    
    if recommendations['dropout_risk']:
        output.append("⚠️  DROPOUT RISK IDENTIFIED")
    
    output.append("")
    
    # Immediate actions
    if recommendations['immediate_actions']:
        output.append("IMMEDIATE ACTIONS REQUIRED:")
        for action in recommendations['immediate_actions']:
            output.append(f"• {action}")
        output.append("")
    
    # Interventions by category
    if recommendations['recommended_interventions']:
        output.append("RECOMMENDED INTERVENTIONS:")
        for intervention in recommendations['recommended_interventions']:
            output.append(f"\n{intervention['need_category']} (Confidence: {intervention['confidence']:.1%}):")
            for action in intervention['actions']:
                output.append(f"  • {action}")
    
    # Timeline
    output.append(f"\nFOLLOW-UP TIMELINE:")
    for milestone, timeframe in recommendations['follow_up_timeline'].items():
        output.append(f"• {milestone.replace('_', ' ').title()}: {timeframe}")
    
    # Case notes
    if recommendations['case_notes']:
        output.append(f"\nCASE NOTES:")
        for note in recommendations['case_notes']:
            output.append(f"• {note}")
    
    return "\n".join(output)

def batch_generate_recommendations(df: pd.DataFrame, needs_cols: List[str], 
                                 priority_col: str = 'priority_pred', 
                                 dropout_col: str = 'dropout_pred') -> pd.DataFrame:
    """Generate recommendations for a batch of profiles."""
    recommendations_list = []
    
    for idx, row in df.iterrows():
        needs_pred = {col: int(row.get(col, 0)) for col in needs_cols}
        priority = row.get(priority_col, 'low')
        dropout_risk = int(row.get(dropout_col, 0))
        
        recs = generate_recommendations(needs_pred, priority, dropout_risk)
        
        # Flatten for DataFrame
        rec_summary = {
            'uid': row.get('uid', idx),
            'priority_level': recs['priority_level'],
            'urgency': recs['urgency'],
            'dropout_risk': recs['dropout_risk'],
            'num_interventions': len(recs['recommended_interventions']),
            'initial_contact_timeline': recs['follow_up_timeline']['initial_contact'],
            'case_notes': '; '.join(recs['case_notes']),
            'full_recommendations': format_recommendations_for_display(recs)
        }
        
        recommendations_list.append(rec_summary)
    
    return pd.DataFrame(recommendations_list)
