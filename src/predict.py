import pandas as pd
from joblib import load
import os
import sys

# Import our preprocessing function from the other file
from data_preprocessing import preprocess_data

# Define paths to saved artifacts
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'training_columns.joblib')

def predict_anomalies(input_df):
    """
    Loads a pre-trained model and associated artifacts to make predictions on new data.

    Args:
        input_df (pd.DataFrame): A DataFrame with new, raw network data matching the
                                 original format.

    Returns:
        pd.Series: A pandas Series containing the predictions (-1 for anomaly, 1 for normal).
    """
    # Load Artifacts 
    print("Loading model, scaler, and training columns...")
    try:
        model = load(MODEL_PATH)
        scaler = load(SCALER_PATH)
        training_columns = load(COLUMNS_PATH)
    except FileNotFoundError:
        print("Error: Model artifacts not found. Please run train.py first to create them.")
        sys.exit(1)
    print("Artifacts loaded successfully.")

    # Preprocess new data 
    X_new = preprocess_data(input_df, is_training=False)

    # Align Columns 
    # Make sure new data has the exact same columns as the training data
    print("Aligning columns of new data with training data...")
    X_new_aligned = X_new.reindex(columns=training_columns, fill_value=0)
    
    # Scale and Predict
    print("Scaling new data...")
    X_new_scaled = scaler.transform(X_new_aligned)
    
    print("Making predictions...")
    predictions = model.predict(X_new_scaled)
    
    print("Prediction complete.")
    return pd.Series(predictions, index=X_new_aligned.index)

if __name__ == '__main__':
    # This block demonstrates how to use the predict_anomalies function.
    # Load a small sample of the original data to simulate 'new' incoming data.
    print("\n--- Running Prediction Example ---")
    
    try:
        # Load 10 rows of sample data from one of the original CSVs
        sample_df = pd.read_csv(
            'Datasets/UNSW-NB15/UNSW-NB15_1.csv', 
            header=None, 
            nrows=10
        )
        # Manually assign the original column names for the sample
        column_names = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
            'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sinpkt',
            'dinpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
        sample_df.columns = column_names
        
        # Get predictions for the sample data
        anomaly_predictions = predict_anomalies(sample_df)

        # Display the predictions alongside some original data
        results_df = sample_df.copy()
        results_df['prediction'] = anomaly_predictions.map({-1: 'Anomaly', 1: 'Normal'})
        print("\n--- Prediction Results on Sample Data ---")
        print(results_df[['srcip', 'dstip', 'proto', 'service', 'prediction']])
        
    except FileNotFoundError:
        print("\nCould not run example: Dataset file 'Datasets/UNSW-NB15/UNSW-NB15_1.csv' not found.")
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
