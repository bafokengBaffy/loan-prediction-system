# input_validation.py
from typing import Dict, List, Optional

def validate_gender(input_data: Dict) -> Optional[str]:
    gender = input_data.get('Gender')
    dependents = input_data.get('Dependents', '0')
    income = input_data.get('ApplicantIncome', 0)

    if not gender:
        return "Gender is required"
    if gender not in ['Male', 'Female']:
        return "Invalid gender specified"
    
    if gender == 'Female':
        if input_data.get('Married') == 'No':
            return "Note: Unmarried female applicants may qualify for special programs"
        elif income > 1_000_000:
            return "High-income female applicants require independent income verification"
        elif dependents == '3+':
            return "Female applicants with 3+ dependents require additional support assessment"
    
    elif gender == 'Male':
        if dependents == '3+':
            return "Note: Male applicants with 3+ dependents have different income requirements"
        elif income < 10_000:
            return "Low-income male applicants may not meet standard affordability criteria"
    
    return None

def validate_income(input_data: Dict) -> Optional[str]:
    ai = input_data.get('ApplicantIncome', 0)
    ci = input_data.get('CoapplicantIncome', 0)
    total = ai + ci

    if ai <= 0:
        return "Applicant income cannot be zero"
    if ai < 10_000:
        return "Applicant income below absolute minimum (M10,000)"
    if 10_000 <= ai < 25_000:
        if ci > ai * 2:
            return "Coapplicant income unusually high compared to applicant"
    elif 25_000 <= ai < 50_000:
        if total < 40_000:
            return "Combined income too low for mid-tier products"
    elif 50_000 <= ai < 200_000:
        if (ai / total) < 0.4:
            return "Primary applicant should contribute ≥40% of total income"
    elif 200_000 <= ai <= 2_000_000:
        if total > 5_000_000:
            return "Combined income exceeds M5,000,000 - jumbo loan category"
    elif ai > 2_000_000:
        return "High Net-worth Individual (HNI) classification - special processing required"
    
    if ci > 500_000 and ai < 50_000:
        return "Unusual coapplicant situation - needs strong explanation"

    return None

def validate_loan_amount(input_data: Dict) -> Optional[str]:
    loan = input_data.get('LoanAmount', 0)
    income = input_data.get('ApplicantIncome', 0) + input_data.get('CoapplicantIncome', 0)
    credit = input_data.get('Credit_History')

    if loan < 50_000:
        return "Loan amount below minimum for mortgage products (M50,000)"
    elif 50_000 <= loan < 100_000:
        return "Small loans under M100,000 - consider MSME or personal loan products"
    elif 100_000 <= loan <= 500_000:
        if loan > income * 5:
            return "Loan amount exceeds typical underwriting limits (5x income)"
    elif 500_000 < loan <= 1_000_000:
        if credit == 'No History':
            return "Large loan requests with no credit history not allowed"
    elif 1_000_000 < loan <= 3_000_000:
        if income < 500_000:
            return "Large loans require strong income profile"
    elif 3_000_000 < loan <= 5_000_000:
        return "Loan above M3,000,000 requires upfront property appraisal"
    elif loan > 5_000_000:
        return "Jumbo loan requires special approval process"

    if loan > income * 10:
        return "Loan amount exceeds underwriting limits (10x income)"
    
    return None

def validate_loan_term(input_data: Dict) -> Optional[str]:
    term = input_data.get('Loan_Amount_Term', 0)
    age = input_data.get('ApplicantAge', 0)

    if term < 12:
        return "Minimum loan term is 12 months"
    elif 12 <= term <= 60:
        if term % 6 != 0:
            return "Short-term loan: term must be multiple of 6 months for best rates"
    elif 61 <= term <= 180:
        if term % 6 != 0:
            return "Medium-term loan: term must align to 6-month multiples"
    elif 181 <= term <= 360:
        if age > 55:
            return "Applicants over 55 need extra review for long-term loans"
    elif 361 <= term <= 480:
        if (age + term // 12) > 75:
            return "Applicant age plus loan term exceeds 75 years"
    else:
        return "Maximum allowed loan term is 40 years (480 months)"
    
    return None

def validate_credit_history(input_data: Dict) -> Optional[str]:
    credit = input_data.get('Credit_History')
    education = input_data.get('Education')
    loan = input_data.get('LoanAmount', 0)
    married = input_data.get('Married')
    dti = input_data.get('DebtToIncome', 0)

    if not credit:
        return "Credit history must be provided"
    if credit not in ['Good', 'No History']:
        return "Invalid credit history selection"

    if credit == 'No History':
        if loan > 100_000:
            return "Large loans require established credit history"
        if married == 'No':
            return "Unmarried applicants with no credit history are higher risk"
        if education == 'Graduate':
            return "Graduates with no credit history may qualify for first-time borrower programs"
    
    if credit == 'Good':
        if dti > 0.60:
            return "Good credit but debt-to-income ratio >60% - manual verification needed"
    
    return None

def validate_additional(input_data: Dict) -> Optional[str]:
    area = input_data.get('Property_Area')
    loan = input_data.get('LoanAmount', 0)
    education = input_data.get('Education')
    self_employed = input_data.get('Self_Employed')
    dependents = input_data.get('Dependents', '0')
    income = input_data.get('ApplicantIncome', 0)
    age = input_data.get('ApplicantAge', 0)
    married = input_data.get('Married')
    co_income = input_data.get('CoapplicantIncome', 0)
    business_proof = input_data.get('BusinessProofUploaded', False)

    if area == 'Rural':
        if loan > 1_000_000:
            return "Rural properties over M1,000,000 require land ownership proof"
        if income < 50_000:
            return "Rural applicants need minimum M50,000 income"

    if education == 'Not Graduate':
        if loan > 500_000:
            return "Non-graduates applying for high loan amounts require strong justification"
    
    if self_employed == 'Yes':
        if not business_proof:
            return "Self-employed applicants must provide business income proofs"
    
    if dependents == '3+' and income < 50_000:
        return "Applicants with many dependents and low income are high-risk"
    
    if age < 21:
        return "Applicant must be at least 21 years old"
    elif age > 65:
        return "Applicant age exceeds maximum limit (65 years)"
    
    if married == 'Yes' and co_income == 0:
        return "Married applicants should consider including coapplicant income for better eligibility"

    return None

def validate_all_inputs(input_data: Dict) -> List[str]:
    """Run all validation checks"""
    validators = [
        validate_gender,
        validate_income,
        validate_loan_amount,
        validate_loan_term,
        validate_credit_history,
        validate_additional,
    ]
    
    messages = []
    for validator in validators:
        result = validator(input_data)
        if result:
            messages.append(result)
    
    return messages
