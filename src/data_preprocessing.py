import pandas as pd
import numpy as np

def preprocess_data(df, is_training=True):
    """
    Performs all preprocessing steps on the raw UNSW-NB15 data.

    Args:
        df (pd.DataFrame): The raw input dataframe.
        is_training (bool): If True, expects 'label' column and returns X and y.
                            If False, returns only X.

    Returns:
        (pd.DataFrame, pd.Series) or pd.DataFrame: Preprocessed X and optionally y.
    """
    print("Starting data preprocessing...")

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Check for privileged port
    if 'dsport' in df.columns:
        print("Creating 'is_privileged_port' feature...")
        dsport_numeric = pd.to_numeric(df['dsport'], errors='coerce')
        df['is_privileged_port'] = (dsport_numeric < 1024).astype(int)
    
    if 'dur' in df.columns:
        print("Applying log transformation to 'dur' feature...")
        df['dur'] = np.log1p(df['dur'])

    # Replace '-' with 'none' in 'service' column  
    if 'service' in df.columns:
        print("Replacing '-' in 'service' column with 'none'...")
        df['service'] = df['service'].replace('-', 'none')

    # One-hot encoding
    print("One-hot encoding categorical features...")
    categorical_cols = ['proto', 'service', 'state']
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_categorical_cols)

    # Separate labels and finalize feature set 
    y = None
    if is_training:
        if 'label' not in df.columns:
            raise ValueError("Training mode requires a 'label' column in the dataframe.")
        y = df['label'].copy()

    # Drop all columns that should not be used for training
    cols_to_drop = ['srcip', 'dstip', 'sport', 'dsport', 'attack_cat', 'label']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    # Clean Data
    print("Coercing all features to numeric and handling missing values...")
    # Convert all col in X to numeric values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    if is_training:
        # Combine X and y to safely drop rows with NaNs in either
        X_y = pd.concat([X, y], axis=1)
        X_y.dropna(inplace=True)
        X = X_y.drop(columns=['label'])
        y = X_y['label'].astype(int)
        print(f"Preprocessing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
        return X, y
    else:
        # Don't  drop rows for prediction, just fill with 0.
        X.fillna(0, inplace=True)
        print(f"Preprocessing complete. Shape of X: {X.shape}")
        return X
