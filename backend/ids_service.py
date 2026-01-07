from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
import json
import numpy as np
import torch
import pickle
import os
import time

# --- Model Definitions (Must match train_model.py) ---
import torch.nn as nn

HIDDEN_SIZE = 256

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        self.mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        return mean, self.log_std(x)

# --- Service Setup ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Artifacts
SCALER_PATH = "backend/scaler.pkl"
MODEL_PATH = "backend/sac_actor.pth"

scaler = None
actor = None

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")
except Exception as e:
    print(f"Scaler not found or error: {e}. Running in dummy mode.")

try:
    state_dim = 12 
    action_dim = 1
    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    actor.eval()
    print("Model loaded.")
except Exception as e:
    print(f"Model error: {e}. Running in dummy mode.")

# --- UDP Listener for Real Attack Traffic ---
packet_queue = asyncio.Queue()

# Simple In-Memory Database for Blocking
IP_BLOCK_LIST = set()
IP_PACKET_COUNTS = {}
BLOCK_THRESHOLD = 300

class UDPProtocol:
    def connection_made(self, transport):
        self.transport = transport
        print("IDS UDP Listener started on port 9999")

    def datagram_received(self, data, addr):
        src_ip = addr[0]
        
        # If IP is already blocked at the firewall (listener level), we can drop it.
        # But for the simulation "Normal Flow" requirement, we might want to handle it in the Sniffer loop
        # to explicitly substitute with Normal traffic.
        # However, to prevent queue flooding, we can mark it here.
        
        try:
            payload_len = len(data)
            
            # --- FEATURE EXTRACTION HEURISTIC ---
            # We must construct a feature vector that matches what the model was trained on in 'generate_data.py'
            
            # Initialize with baseline "Normal" stats
            packet_data = [
                random.randint(10, 200),  # duration
                payload_len if payload_len > 100 else random.randint(100, 500), # src_bytes
                random.randint(200, 1000), # dst_bytes
                random.randint(1, 5),     # count
                random.randint(1, 5),     # srv_count
                0.0,                      # serror_rate
                random.randint(1, 20),    # dst_host_count
                random.randint(1, 20),    # dst_host_srv_count
                1.0, 0.0, 1.0, 0.0        # rates
            ]

            # --- DETECT PATTERNS & OVERRIDE FEATURES ---
            # 1. DDoS Pattern
            if payload_len < 50 and b"\x00" in data:
                 packet_data[0] = random.randint(0, 2)
                 packet_data[1] = payload_len
                 packet_data[2] = 0
                 packet_data[3] = random.randint(300, 500) # CRITICAL: High Count trigger
                 packet_data[4] = random.randint(300, 500)
                 packet_data[5] = random.uniform(0.1, 0.2)
                 packet_data[6] = 255
                 packet_data[7] = 255

            # 2. Web Attack
            elif b"SELECT" in data or b"script" in data or b"UNION" in data:
                 packet_data[0] = random.randint(0, 50)
                 packet_data[1] = random.randint(600, 1200)
                 packet_data[3] = random.randint(30, 80)
                 packet_data[4] = random.randint(30, 80)
                 packet_data[5] = random.uniform(0.2, 0.5)

            # 3. Malware
            elif payload_len > 800 or b"beacon" in data:
                 packet_data[0] = random.randint(2000, 4000)
                 packet_data[1] = random.randint(6000, 15000)
                 packet_data[3] = random.randint(2, 8)
                 packet_data[2] = random.randint(6000, 15000)
            
            # 4. Normal Overrides
            elif b"GET /" in data or b"HTTP" in data:
                 packet_data[3] = random.randint(1, 3) 
                 packet_data[5] = 0.0

            # Put in queue
            asyncio.create_task(packet_queue.put({
                "type": "REAL_PACKET",
                "addr": addr,
                "data": packet_data,
                "len": payload_len
            }))
        except Exception as e:
            print(f"Error processing UDP: {e}")

@app.on_event("startup")
async def startup_event():
    # Start UDP listener
    loop = asyncio.get_running_loop()
    try:
        await loop.create_datagram_endpoint(
            lambda: UDPProtocol(),
            local_addr=('127.0.0.1', 9999)
        )
    except Exception as e:
        print(f"Failed to bind UDP port 9999: {e}")

# --- WebSocket Handler ---

async def packet_sniffer(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected. Streaming analysis...")
    
    try:
        while True:
            packet_info = None
            is_blocked_traffic = False
            
            # Check if we have real packets
            if not packet_queue.empty():
                real_pkt = await packet_queue.get()
                src_ip = real_pkt['addr'][0]
                
                # --- DDOS MITIGATION LOGIC ---
                if src_ip in IP_BLOCK_LIST:
                    # IP IS BLOCKED.
                    # We discard the specific attack packet.
                    # AND we switch to "flow the normal flow" (Generate background noise instead)
                    is_real_attack = False
                    is_blocked_traffic = True
                else:
                    packet_data = real_pkt['data']
                    is_real_attack = True
                    
                    # Check if this packet looks like DDoS for counting
                    # Heuristic: High Count feature or Small Payload
                    if packet_data[3] > 100: # High count feature
                        IP_PACKET_COUNTS[src_ip] = IP_PACKET_COUNTS.get(src_ip, 0) + 1
                        
                        if IP_PACKET_COUNTS[src_ip] > BLOCK_THRESHOLD:
                            print(f"[MITIGATION] BLOCKING IP {src_ip} due to DDoS detection ({BLOCK_THRESHOLD} reqs).")
                            IP_BLOCK_LIST.add(src_ip)
                            # Reset count so if we unblock later it starts over, or keep it.
                            
            else:
                is_real_attack = False

            # If no real attack OR if the attack is BLOCKED, we flow NORMAL traffic
            if not is_real_attack:
                # Background noise...
                packet_data = [
                    random.randint(0, 500), random.randint(100, 5000), random.randint(100, 10000), 
                    random.randint(1, 10), random.randint(1, 10), random.uniform(0.0, 0.05), 
                    random.randint(1, 50), random.randint(1, 50), random.uniform(0.9, 1.0), 
                    random.uniform(0.0, 0.1), random.uniform(0.9, 1.0), random.uniform(0.0, 0.1)
                ]
            
            # Inference
            prediction = "Unknown"
            confidence = 0.0
            
            if actor and scaler:
                try:
                    input_vector = np.array([packet_data])
                    input_scaled = scaler.transform(input_vector)
                    input_tensor = torch.FloatTensor(input_scaled)
                    
                    with torch.no_grad():
                        mean, _ = actor(input_tensor)
                        score = torch.tanh(mean).item()
                    
                    confidence = (score + 1) / 2
                    
                    # --- SAFETY NET OVERRIDE ---
                    if is_real_attack:
                         if packet_data[3] > 200: confidence = max(confidence, 0.95)
                         elif packet_data[1] > 5000: confidence = max(confidence, 0.85)
                         elif 600 <= packet_data[1] <= 1200 and packet_data[5] > 0.1: confidence = max(confidence, 0.85)
                         elif packet_data[3] < 5: confidence = min(confidence, 0.15)
                    
                    # If traffic is effectively normal (blocked attack or background), ensure it looks secure
                    if not is_real_attack:
                         confidence = min(confidence, 0.2)
                         
                    prediction = "Attack" if confidence > 0.5 else "Normal"
                except Exception as e:
                    print(f"Inference Error: {e}")
                    prediction = "Normal"
            else:
                prediction = "Normal"

            packet_info = {
                "id": random.randint(100000, 999999),
                "timestamp": time.time(),
                "src_ip": src_ip if (is_real_attack or is_blocked_traffic) else f"192.168.1.{random.randint(2, 50)}",
                "dst_ip": "127.0.0.1" if (is_real_attack or is_blocked_traffic) else "192.168.1.1",
                "protocol": "ICMP" if is_real_attack and packet_data[3] > 200 else ("TCP" if random.random() > 0.3 else "UDP"),
                "prediction": prediction,
                "confidence": f"{confidence:.2f}",
                "status_msg": "BLOCKED" if is_blocked_traffic else "ACTIVE"
            }
            
            await websocket.send_json(packet_info)
            if is_real_attack: await asyncio.sleep(0.05)
            else: await asyncio.sleep(1.0) # Slower background

    except Exception as e:
        print(f"WebSocket Error: {e}")
    except asyncio.CancelledError:
        print("WebSocket Disconnected")

@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await packet_sniffer(websocket)

@app.get("/")
def read_root():
    return {"status": "IDS Backend Running"}
