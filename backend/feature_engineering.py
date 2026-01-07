import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

def process_data(input_file="synthetic_traffic_data.csv", output_dir="backend"):
    print("Starting feature engineering...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found. Please run generate_data.py first.")

    df = pd.read_csv(input_file)
    
    # Separate Features and Label
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode Labels
    # Normal -> 0, Attacks -> 1 (Binary Classification for RL Reward)
    # Or Multi-class. For this IDS, let's strictly do Binary Anomaly Detection for the Reward System
    # 0 = Normal, 1 = Attack
    y_binary = y.apply(lambda x: 0 if x == "Normal" else 1)
    
    # Save label encoder for attack types to map back later if needed
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for realtime inference
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    print("Feature scaling complete. Scaler saved.")
    
    # Split data (for offline training if needed, though RL usually iterates)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)
    
    # Save processed arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    print("Feature engineering completed. Arrays saved.")

if __name__ == "__main__":
    process_data()
