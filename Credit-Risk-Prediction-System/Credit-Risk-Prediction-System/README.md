# Credit Risk Prediction System

A Python Machine Learning project that predicts customer credit risk using customer and financial information.

## Features
- Data preprocessing and cleaning
- Exploratory data analysis
- Categorical feature encoding
- Feature scaling
- Logistic Regression, Decision Tree, and Random Forest classification
- Model performance comparison
- Best-model selection
- New customer credit-risk prediction

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

## Project Structure
```text
Credit-Risk-Prediction-System/
├── data/
│   └── credit_risk.csv
├── models/
│   └── .gitkeep
├── src/
│   ├── train_model.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
```

The training script compares the classification models and saves the best model in `models/`.

## Input Features
Age, Income, Loan Amount, Loan Duration, Credit History, Employment Status, Housing, Existing Loans, Savings, and Dependents.

## Output
The system predicts **Good Risk** or **Bad Risk**.
