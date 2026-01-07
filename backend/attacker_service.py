from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import socket
import random
import time
import asyncio
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IDS_HOST = "127.0.0.1"
IDS_PORT = 9999

def send_udp_packet(data_bytes):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(data_bytes, (IDS_HOST, IDS_PORT))
    except Exception as e:
        print(f"UDP Send Error: {e}")

# --- Traffic Logic ---

def send_normal_traffic(duration=10):
    print(f"Sending Normal Traffic for {duration}s")
    end_time = time.time() + duration
    while time.time() < end_time:
        # Simulate normal browsing: random small payloads, moderate interval
        # HTTP GET like data (not attacks)
        payloads = [
            b"GET /index.html HTTP/1.1",
            b"POST /submit_form HTTP/1.1", 
            b"Data content for transfer...",
            random._urandom(random.randint(50, 200)) # Random data packets
        ]
        send_udp_packet(random.choice(payloads))
        time.sleep(random.uniform(0.1, 0.5)) # Human-like interval

def attack_ddos(duration=5):
    print(f"Launching DDoS for {duration}s")
    end_time = time.time() + duration
    while time.time() < end_time:
        # DDoS packet simulation: Small, Empty, Fast
        send_udp_packet(b"\x00" * 32)
        time.sleep(0.001)

def attack_malware():
    print("Launching Malware Infection Sim")
    # Simulate download of payload + heartbeat
    for _ in range(5):
        # Large packet (Download)
        send_udp_packet(random._urandom(1024))
        time.sleep(0.2)
        # Small packet (C&C)
        send_udp_packet(b"beacon")
        time.sleep(0.1)

def attack_web_exploit(type="sql"):
    print(f"Launching {type} Attack")
    # Simulate HTTP Requests (ASCII text mostly)
    payloads = []
    if type == "sql":
        payloads = [
            b"GET /login?user=' OR '1'='1 HTTP/1.1",
            b"POST /search HTTP/1.1\r\nContent-Length: 50\r\nq=admin'--",
            b"UNION SELECT 1,2,3--"
        ]
    elif type == "xss":
        payloads = [
            b"<script>alert(1)</script>",
            b"GET /?search=<img src=x onerror=alert(1)>",
            b"javascript:alert('XSS')"
        ]
    
    # Rapid fire these payloads like a scanner
    for _ in range(20):
        p = random.choice(payloads)
        send_udp_packet(p)
        time.sleep(0.05)

# --- API Endpoints ---

@app.post("/traffic/normal")
async def trigger_normal():
    threading.Thread(target=send_normal_traffic).start()
    return {"status": "Generating Normal Network Traffic..."}

@app.post("/attack/ddos")
async def trigger_ddos():
    threading.Thread(target=attack_ddos).start()
    return {"status": "DDoS Attack Launched"}

@app.post("/attack/malware")
async def trigger_malware():
    threading.Thread(target=attack_malware).start()
    return {"status": "Malware Infection Simulation Launched"}

@app.post("/attack/sqli")
async def trigger_sqli():
    threading.Thread(target=attack_web_exploit, args=("sql",)).start()
    return {"status": "SQL Injection Attack Launched"}

@app.post("/attack/xss")
async def trigger_xss():
    threading.Thread(target=attack_web_exploit, args=("xss",)).start()
    return {"status": "XSS Attack Launched"}

@app.get("/")
def root():
    return {"status": "Attacker Console Backend Ready"}
