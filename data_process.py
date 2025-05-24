import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns to retain for modeling (selecting key features from LendingClub dataset)
KEEP_COLUMNS = [
    'loan_amnt', 'term', 'int_rate', 'installment',
    'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'annual_inc', 'purpose', 'dti', 'open_acc',
    'total_acc', 'earliest_cr_line', 'loan_status'
]

# Mapping loan status into a binary target variable
# 0 = Non-default (Fully Paid), 1 = Default (Charged Off or Default)
TARGET_MAPPING = {
    'Fully Paid': 0,
    'Charged Off': 1,
    'Default': 1
}

def load_data(filepath):
    """
    Load raw LendingClub data from CSV file.
    Parameters:
        filepath (str): Path to the loan data CSV file.
    Returns:
        pd.DataFrame: Raw DataFrame loaded from file.
    """
    df = pd.read_csv(filepath, low_memory=False)
    return df

def convert_emp_length(emp):
    """
    Convert employment length from string to numeric values.
    For example:
        '10+ years' -> 10
        '< 1 year'  -> 0.5
        '3 years'   -> 3
    """
    if pd.isnull(emp):
        return np.nan
    if emp == '10+ years':
        return 10
    if emp == '< 1 year':
        return 0.5
    try:
        return float(emp.strip().split()[0])
    except:
        return np.nan

def engineer_features(df):
    """
    Generate new financial features to enhance model performance.
    Features include:
        - credit_utilization: Ratio of loan amount to income
        - log_income: Log-transformed annual income
        - monthly_rate: Monthly interest rate
        - emp_length_num: Numeric version of employment length
    """
    df['credit_utilization'] = df['loan_amnt'] / df['annual_inc']
    df['log_income'] = np.log1p(df['annual_inc'])
    df['monthly_rate'] = df['int_rate'] / 12.0
    df['emp_length_num'] = df['emp_length'].apply(convert_emp_length)
    return df

def encode_categoricals(df):
    """
    Encode all categorical string columns using Label Encoding.
    This is a simplified approach and suitable for tree-based models like XGBoost.
    """
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def clean_and_engineer(df):
    """
    Perform the full preprocessing pipeline:
        1. Retain only selected columns
        2. Filter target labels and map to binary
        3. Drop columns/rows with excessive missing data
        4. Create new features
        5. Encode categorical variables
    Parameters:
        df (pd.DataFrame): Raw input DataFrame.
    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame.
    """
    df = df.copy()

    # 1. Keep only relevant columns
    df = df[[col for col in KEEP_COLUMNS if col in df.columns]]

    # 2. Convert loan status to binary label
    df = df[df['loan_status'].isin(TARGET_MAPPING)]
    df['loan_status'] = df['loan_status'].map(TARGET_MAPPING)

    # 3. Drop columns with too many missing values and then drop incomplete rows
    df.dropna(thresh=0.8 * len(df), axis=1, inplace=True)
    df.dropna(inplace=True)

    # 4. Feature engineering
    df = engineer_features(df)

    # 5. Encode categorical variables
    df = encode_categoricals(df)

    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def split_features(df):
    """
    Split the dataset into training and testing sets.
    Parameters:
        df (pd.DataFrame): Cleaned dataset including target variable.
    Returns:
        X_train, X_test, y_train, y_test: Feature and label splits.
    """
    y = df['loan_status']
    X = df.drop(columns=['loan_status'])
    return train_test_split(X, y, test_size=0.2, random_state=42)



if __name__ == "__main__":
    import joblib
    import os

    # Step 1: Load and preprocess
    print("Loading data...")
    df = load_data("data/accepted_2007_to_2018Q4.csv")
    df = clean_and_engineer(df)

    # Step 2: Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_features(df)

    # Step 3: Scale
    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 4: Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    # Step 5: Confirm shape
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")