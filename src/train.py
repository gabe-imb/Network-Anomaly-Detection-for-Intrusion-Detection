import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

# Import our preprocessing function from the other file
from data_preprocessing import preprocess_data

DATA_DIR = 'Datasets/UNSW-NB15/'
MODEL_DIR = 'models'

def load_data(data_dir):
    """Loads and concatenates the UNSW-NB15 dataset from multiple CSV files."""
    print("Loading data...")
    try:
        csv_files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.csv') and 'features' not in f]
        if not csv_files:
            raise FileNotFoundError(f"No CSV data files found in directory: {data_dir}")
    except FileNotFoundError:
        print(f"Error: The data directory '{data_dir}' was not found.")
        exit()

    # Define colmun names
    column_names = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
        'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
        'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sinpkt',
        'dinpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
    ]

    df_list = [pd.read_csv(file, header=None, names=column_names, low_memory=False) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Data loaded successfully. Total records: {len(df)}")
    return df

def main():
    """Main training script to preprocess data, train a model, and save artifacts."""
    # Load and preprocess data 
    raw_df = load_data(DATA_DIR)
    X, y = preprocess_data(raw_df, is_training=True)

    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the artifacts needed for prediction
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    columns_path = os.path.join(MODEL_DIR, 'training_columns.joblib')
    dump(X.columns, columns_path)
    print(f"Training columns saved to {columns_path}")

    # Train Isolation Forest Model
    # Best parameters found during experimentation
    print("Training Isolation Forest model...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=0.05, 
        random_state=42, 
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(X_scaled)

    # Save trained model
    model_path = os.path.join(MODEL_DIR, 'anomaly_model.joblib')
    dump(iso_forest, model_path)
    print(f"Model saved to {model_path}")

    # Evaluate final model
    print("\n--- Final Model Evaluation ---")
    # Convert model's predictions: -1 for anomaly -> 1 (Attack), 1 for normal -> 0 (Normal)
    predicted_labels = [1 if p == -1 else 0 for p in predictions]
    
    print("\nClassification Report:")
    print(classification_report(y, predicted_labels, target_names=['Normal', 'Attack'], digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predicted_labels))

if __name__ == '__main__':
    main()
