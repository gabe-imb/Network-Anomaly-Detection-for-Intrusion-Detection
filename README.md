# Unsupervised Anomaly Detection for Network Intrusion

This project uses an unsupervised machine learning model (`IsolationForest`) to identify anomalous network traffic from the UNSW-NB15 dataset.

The original analysis and model prototyping was performed in a Jupyter Notebook, which has now been refactored into a structured Python application.

## Project Structure

- `notebooks/`: Contains the original Jupyter Notebook for exploration and analysis.
- `src/`: Contains the Python source code.
  - `data_preprocessing.py`: A module for all data cleaning, feature engineering, and preparation.
  - `train.py`: The main script to train the model and save the final artifacts.
  - `predict.py`: A script to load the saved model and make predictions on new data.
- `models/`: This directory is created by `train.py` and will store the saved model (`anomaly_model.joblib`), scaler (`scaler.joblib`), and column list (`training_columns.joblib`).
- `Datasets/`: This directory is where the raw data should be placed (it is ignored by Git).

## Setup Instructions

**1. Clone the Repository:**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

**2. Create a Virtual Environment (Recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download the Dataset:**
- Download the UNSW-NB15 dataset: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Place the four CSV files (`UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, etc.) inside the following directory structure: `Datasets/UNSW-NB15/`. You will need to create these folders.

## How to Run

**1. To Train the Model:**
This command will run the entire pipeline: loading and preprocessing the data, training the model, saving the artifacts to the `models/` directory, and printing a final evaluation report.
```bash
python3 src/train.py
```

**2. To Make Predictions:**
This command will load the artifacts saved during training and run a demonstration on 10 sample rows of data to show how to predict on new, unseen data.
```bash
python3 src/predict.py
```

## Model Performance

The final `IsolationForest` model achieved the following performance on the "Attack" class:
- **Precision:** 0.37
- **Recall:** 0.73
- **F1-Score:** 0.50

This indicates that this mode balanced and is effective at identifying a majority of attacks while maintaining a reasonable rate of false alarms.