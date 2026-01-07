import pandas as pd
import numpy as np
import random
import time

def generate_synthetic_data(num_samples=15000):
    print("Generating synthetic network traffic data with expanded attack types...")
    
    data = []
    
    # Expanded Attack types
    possible_attacks = ["DDoS", "PortScan", "Botnet", "Malware", "WebAttack"]
    
    for _ in range(num_samples):
        # 70% Normal, 30% Attack
        is_attack = random.random() < 0.3
        label = "Normal"
        
        if is_attack:
            label = random.choice(possible_attacks)
        
        # --- Feature Distributions for Different Classes ---
        
        if label == "Normal":
            duration = random.randint(0, 500)
            src_bytes = random.randint(100, 5000)
            dst_bytes = random.randint(100, 10000)
            count = random.randint(1, 10)
            srv_count = random.randint(1, 10)
            serror_rate = random.uniform(0.0, 0.05)
            dst_host_count = random.randint(1, 60)
            dst_host_srv_count = random.randint(1, 60)
            same_srv_rate = random.uniform(0.9, 1.0)
            diff_srv_rate = random.uniform(0.0, 0.1)
            dst_host_same_srv_rate = random.uniform(0.9, 1.0)
            dst_host_diff_srv_rate = random.uniform(0.0, 0.1)
        
        elif label == "DDoS":
            # High volume, fast, same target
            duration = random.randint(0, 5)
            src_bytes = random.randint(0, 50)
            dst_bytes = 0
            count = random.randint(200, 511)
            srv_count = random.randint(200, 511)
            serror_rate = random.uniform(0.0, 0.1)
            dst_host_count = 255
            dst_host_srv_count = 255
            same_srv_rate = 1.0
            diff_srv_rate = 0.0
            dst_host_same_srv_rate = 1.0
            dst_host_diff_srv_rate = 0.0
            
        elif label == "PortScan":
            # Probing many ports, often failing
            duration = 0
            src_bytes = 0
            dst_bytes = 0
            count = random.randint(1, 5)
            srv_count = random.randint(1, 2)
            serror_rate = random.uniform(0.8, 1.0) # Connection refused/reset
            dst_host_count = 255
            dst_host_srv_count = random.randint(1, 10) # Few valid services found
            same_srv_rate = random.uniform(0.0, 0.1)
            diff_srv_rate = random.uniform(0.8, 1.0)
            dst_host_same_srv_rate = random.uniform(0.0, 0.1)
            dst_host_diff_srv_rate = random.uniform(0.8, 1.0)

        elif label == "Botnet":
            # C&C Heartbeats, periodic
            duration = random.randint(500, 2000)
            src_bytes = random.randint(40, 100)
            dst_bytes = random.randint(40, 100)
            count = random.randint(5, 20)
            srv_count = random.randint(5, 20)
            serror_rate = 0.0
            dst_host_count = random.randint(10, 100)
            dst_host_srv_count = random.randint(10, 100)
            same_srv_rate = random.uniform(0.8, 1.0)
            diff_srv_rate = 0.0
            dst_host_same_srv_rate = random.uniform(0.8, 1.0)
            dst_host_diff_srv_rate = 0.0
            
        elif label == "Malware":
            # Represents Malware Download/Exfiltration (High Data Transfer)
            duration = random.randint(1000, 5000)
            src_bytes = random.randint(5000, 20000) # Exfiltration
            dst_bytes = random.randint(5000, 20000) # Download
            count = random.randint(1, 5)
            srv_count = random.randint(1, 5)
            serror_rate = 0.0
            dst_host_count = random.randint(20, 100)
            dst_host_srv_count = random.randint(20, 100)
            same_srv_rate = 1.0
            diff_srv_rate = 0.0
            dst_host_same_srv_rate = 1.0
            dst_host_diff_srv_rate = 0.0
            
        elif label == "WebAttack":
            # SQL Injection / XSS (Short bursts, weird payload sizes, rapid succession if fuzzing)
            duration = random.randint(0, 100)
            src_bytes = random.randint(500, 1500) # HTTP Requests
            dst_bytes = random.randint(500, 1500) # HTTP Responses
            count = random.randint(20, 50) # Fuzzing/Scanner
            srv_count = random.randint(20, 50)
            serror_rate = random.uniform(0.0, 0.2) # Some 4xx/5xx errors
            dst_host_count = random.randint(1, 20)
            dst_host_srv_count = random.randint(1, 20)
            same_srv_rate = 1.0
            diff_srv_rate = 0.0
            dst_host_same_srv_rate = 1.0
            dst_host_diff_srv_rate = 0.0

        row = {
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": count,
            "srv_count": srv_count,
            "serror_rate": serror_rate,
            "dst_host_count": dst_host_count,
            "dst_host_srv_count": dst_host_srv_count,
            "same_srv_rate": same_srv_rate,
            "diff_srv_rate": diff_srv_rate,
            "dst_host_same_srv_rate": dst_host_same_srv_rate,
            "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
            "label": label
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv("synthetic_traffic_data.csv", index=False)
    print(f"Generated {num_samples} samples saved to 'synthetic_traffic_data.csv'")

if __name__ == "__main__":
    generate_synthetic_data()
