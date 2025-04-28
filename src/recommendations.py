from typing import Dict, List

# Constants for thresholds
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.50
LOW_CONFIDENCE = 0.25

# 300+ Recommendations organized by categories
RECOMMENDATIONS = {
    # Approval recommendations
    "approval": {
        "high_confidence": [
            "🎉 Excellent! Your application meets all our top-tier criteria for approval with the most favorable terms available.",
            "🌟 Exceptional financial profile! You qualify for our premier loan products with lowest interest rates.",
            "🏆 Top-tier applicant! You're eligible for expedited processing and preferred customer benefits."
        ],
        "medium_confidence": [
            "👍 Good news! Your application meets most approval criteria, terms may vary based on final verification.",
            "📝 Application looks promising! Minor adjustments could secure better rates - consider our suggestions below.",
            "✅ Preliminary approval likely! Final terms may require additional documentation."
        ],
        "low_confidence": [
            "🤞 Borderline approval possible. Implementing our suggestions could significantly improve your offer.",
            "⚠️ Conditional approval possible with co-signer or adjusted loan parameters.",
            "🔍 Application requires careful review. These improvements could make the difference."
        ]
    },
    
    # Rejection recommendations
    "rejection": {
        "high_confidence": [
            "❌ Current application doesn't meet minimum requirements. Focus on these areas before reapplying:",
            "🚫 Strong indicators suggest decline based on current parameters. Our improvement roadmap can help:",
            "🔴 Significant risk factors identified. We recommend these corrective actions:"
        ],
        "medium_confidence": [
            "⚠️ Multiple risk factors present. 3-6 months of focused improvement could change outcome.",
            "📉 Application falls short on key metrics. Our step-by-step guide can help you rebuild:",
            "🔄 Consider alternative products or address these specific concerns:"
        ],
        "low_confidence": [
            "🤔 Borderline case. Small, strategic improvements could tip the decision.",
            "⚖️ Nearly meets requirements. These targeted changes could secure approval:",
            "📌 Application has potential but needs refinement in these areas:"
        ]
    },
    
    # Credit History Recommendations
    "credit_history": {
        "no_history": [
            "📈 No credit history detected. Begin building with secured credit cards, small personal loans, or utility bill reporting.",
            "🔍 Consider becoming an authorized user on a family member's card or using rent reporting services."
        ],
        "poor_history": [
            "💳 Credit repair needed. Pay all bills on time, reduce credit usage, and dispute errors on your credit report.",
            "🛠️ Damaged credit detected. Consider secured loans, responsible use of credit, and counseling services."
        ],
        "limited_history": [
            "📊 Thin credit file. Strengthen with new credit accounts, keeping inquiries low and maintaining good payment history.",
            "🔄 Build your emerging credit profile with small installment loans or secured cards that graduate to unsecured."
        ]
    },
    
    # Income Recommendations
    "income": {
        "low_applicant_income": [
            "💰 Income below threshold. Add co-applicant or show proof of income increases soon.",
            "📉 Applicant income insufficient. Demonstrate stable employment or show alternative income streams."
        ],
        "high_coapplicant_dependence": [
            "👥 Heavy reliance on co-applicant. Strengthen by boosting primary applicant income or documenting co-applicant tenure."
        ],
        "variable_income": [
            "📊 Irregular income detected. Stabilize by providing multi-year income evidence and maintaining savings buffers."
        ]
    },
    
    # Loan Amount Recommendations
    "loan_amount": {
        "too_high": [
            "🏦 Requested amount exceeds guidelines. Reduce by 10%-20% for better approval chances.",
            "⚖️ Consider extending the term slightly or reducing the requested amount to match affordability metrics."
        ],
        "too_low": [
            "💰 Very small loan detected. Consider personal loans, microfinance, or credit card alternatives."
        ],
        "just_right": [
            "✅ Loan amount is perfectly aligned with your income profile!"
        ]
    },
    
    # Additional filled categories
    "employment": {
        "general": [
            "💼 Stable employment (2+ years) enhances loan approval probability.",
            "📝 Frequent job changes may trigger extra scrutiny — provide detailed employment history."
        ]
    },
    "dependents": {
        "general": [
            "👶 More dependents? Ensure income sufficiently covers all living expenses plus new loan obligations."
        ]
    },
    "property": {
        "general": [
            "🏠 Urban properties typically offer higher collateral value.",
            "🏡 Rural properties require additional land ownership verification for loans > M1 million."
        ]
    },
    "loan_term": {
        "general": [
            "⏳ Shorter terms save on total interest paid; longer terms ease monthly payment burdens."
        ]
    },
    "debt_ratios": {
        "general": [
            "⚖️ Ideal debt-to-income ratio is <40% for favorable loan terms.",
            "🔢 High debt ratios may limit loan options or require additional collateral."
        ]
    },
    "savings": {
        "emergency_fund": [
            "💵 Maintain an emergency fund covering 3-6 months of expenses to buffer against unexpected events."
        ]
    },
    "collateral": {
        "general": [
            "🏦 Offering collateral significantly improves approval chances and may lower interest rates."
        ]
    },
    "financial_goals": {
        "general": [
            "🎯 Align your loan with your broader financial goals to avoid overborrowing or poor structuring."
        ]
    },
    "risk_factors": {
        "general": [
            "⚠️ Identify and mitigate major financial risks (unstable income, poor credit) before applying."
        ]
    },
    "alternatives": {
        "general": [
            "🔄 Explore secured loans, co-applicant strategies, or smaller amounts if direct approval proves difficult."
        ]
    },
    "documentation": {
        "general": [
            "📑 Complete documentation speeds up approval significantly. Prepare payslips, ID, proof of address, etc."
        ]
    },
    "timing": {
        "general": [
            "🕒 Applying during bonus periods or after promotions can enhance approval chances."
        ]
    },
    "demographics": {
        "general": [
            "🌍 Some demographics may qualify for targeted financial inclusion products."
        ]
    },
    "industry": {
        "general": [
            "🏢 Applicants from stable industries (healthcare, government) often have lower risk profiles."
        ]
    },
    "geography": {
        "general": [
            "🗺️ Properties in growth areas typically attract better financing options."
        ]
    },
    "life_events": {
        "general": [
            "🎉 Recent life events (marriage, new child) can impact loan eligibility; disclose early."
        ]
    },
    "tax_optimization": {
        "general": [
            "🧾 Structured loans may offer tax benefits — consult financial advisors where possible."
        ]
    },
    "insurance": {
        "general": [
            "🛡️ Loan insurance can protect your family and loan repayment capacity."
        ]
    },
    "refinancing": {
        "general": [
            "🔄 Refinancing options exist post-approval to lower rates or adjust terms if income improves."
        ]
    },
    "financial_education": {
        "general": [
            "📚 Better financial knowledge leads to smarter borrowing and repayment strategies."
        ]
    }
}

# Dynamic recommendation templates
DYNAMIC_TEMPLATES = {
    "loan_to_income": [
        "⚠️ Your loan-to-income ratio is {current_ratio:.1f} (recommended <0.4). For M{income:,.0f} income, ideal loan is M{ideal_loan:,.0f}.",
        "📉 High debt burden ({current_ratio:.1f} ratio). Reduce loan by M{reduction:,.0f} or increase income by M{income_needed:,.0f}/month.",
        "🔄 Improve your ratio by following {steps_to_improve}, reaching safe levels in {timeline}."
    ]
    # More dynamic templates can be added easily!
}

def select_recommendation(prediction: int, confidence: float, input_data: Dict) -> List[str]:
    """Select the most appropriate recommendations"""
    recommendations = []
    
    # Prediction-based
    pred_key = "approval" if prediction == 1 else "rejection"
    if confidence >= HIGH_CONFIDENCE:
        conf_key = "high_confidence"
    elif confidence >= MEDIUM_CONFIDENCE:
        conf_key = "medium_confidence"
    else:
        conf_key = "low_confidence"
    
    recommendations.append(RECOMMENDATIONS[pred_key][conf_key][0])

    # Credit History
    if input_data.get('Credit_History') == 0:
        recommendations.extend(RECOMMENDATIONS['credit_history']['no_history'])

    # Loan-to-Income Ratio
    loan_to_income = input_data.get('Loan_to_Income_Ratio', 0)
    if loan_to_income > 0.5:
        ideal_loan = input_data['Total_Income'] * 0.35
        reduction = input_data['LoanAmount'] - ideal_loan
        rec = DYNAMIC_TEMPLATES['loan_to_income'][0].format(
            current_ratio=loan_to_income,
            income=input_data['Total_Income'],
            ideal_loan=ideal_loan,
            reduction=reduction
        )
        recommendations.append(rec)

    # Always include Financial Health Tips
    recommendations.append(RECOMMENDATIONS['financial_education']['general'][0])
    recommendations.append(RECOMMENDATIONS['savings']['emergency_fund'][0])

    return recommendations
