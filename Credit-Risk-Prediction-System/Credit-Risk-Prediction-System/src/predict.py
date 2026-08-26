import joblib
import pandas as pd

MODEL_PATH = "models/best_credit_risk_model.joblib"
model = joblib.load(MODEL_PATH)

print("Enter Customer Details")
customer = {
    "Age": int(input("Age: ")),
    "Income": float(input("Annual Income: ")),
    "Loan_Amount": float(input("Loan Amount: ")),
    "Loan_Duration": int(input("Loan Duration (months): ")),
    "Credit_History": float(input("Credit History (0-1): ")),
    "Employment_Status": input("Employment Status (Employed/Self-employed/Unemployed): "),
    "Housing": input("Housing (Own/Rent/Mortgage): "),
    "Existing_Loans": int(input("Existing Loans: ")),
    "Savings": float(input("Savings: ")),
    "Dependents": int(input("Dependents: "))
}

prediction = model.predict(pd.DataFrame([customer]))[0]
print("\nCredit Risk Prediction:", "GOOD RISK" if prediction == 1 else "BAD RISK")
