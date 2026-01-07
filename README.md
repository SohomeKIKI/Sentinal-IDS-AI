🛡️ Sentinel AI — Intrusion Detection System (IDS)
Next-Generation AI-Powered Network Security using Soft Actor-Critic Reinforcement Learning
<img width="204" height="258" alt="image" src="https://github.com/user-attachments/assets/0e4e46a8-1aaf-4d96-9810-aa2699b2c4ea" />

📖 Overview

Sentinel AI is an advanced AI-powered Intrusion Detection System (IDS) designed to detect and mitigate cyber attacks in real time.
Unlike traditional rule-based IDS, Sentinel AI leverages Deep Reinforcement Learning (Soft Actor-Critic) to continuously learn network behavior and adapt to evolving attack patterns.

The platform includes:

🛡️ Blue Team Dashboard for defense monitoring

⚔️ Red Team Console for controlled attack simulation

📊 Live Cyberpunk-styled Visualization for threat analysis

This project was developed as a Major Project by Sohom Mandal.

✨ Key Features

🧠 AI-Driven Detection
Classifies packets as Normal or Attack using a trained SAC neural network.

⚡ Real-Time Analysis
Live packet inspection with instant feedback via WebSockets.

⚔️ Red Team Console
Simulate real-world attacks:

UDP Flood (DDoS)

SQL Injection (SQLi)

Cross-Site Scripting (XSS)

Malware Command & Control (C2)

Normal Traffic (baseline behavior)

🛡️ Active Mitigation Engine
Automatically blocks malicious IPs after sustained DDoS activity and restores network stability.

🖥️ Split-View Mode
Side-by-side Attack Console + Defense Dashboard for evaluation.

📊 Live Visualization
Interactive real-time charts using Recharts displaying AI confidence levels and threat trends.

🛠️ Technology Stack
Backend

FastAPI — High-performance IDS API

PyTorch — Deep RL & SAC model

Scikit-Learn — Feature scaling & preprocessing

NumPy / Pandas — Data processing

AsyncIO — Concurrent packet handling & streaming

Frontend

React + Vite — Lightning-fast UI

Recharts — Live data visualization

Lucide React — Icons

TailwindCSS — Cyberpunk UI styling

🚀 Installation & Setup
Prerequisites

Python 3.8+

Node.js 16+

⚡ Automatic Setup (Windows)
START_SYSTEM.bat


This installs all dependencies, trains the model, and launches the entire system.

🧩 Manual Setup

Open three terminals:

🧠 Terminal 1 — IDS Backend
cd "Major Project"
pip install -r requirements.txt
python -m uvicorn backend.ids_service:app --reload --port 8000

⚔️ Terminal 2 — Attacker Simulator
cd "Major Project"
python -m uvicorn backend.attacker_service:app --reload --port 8001

🖥️ Terminal 3 — Frontend Dashboard
cd "Major Project/frontend"
npm install
npm run dev

🕹️ How to Use

Launch system:

START_SYSTEM.bat


Open dashboard:
http://localhost:5173 or http://localhost:5174

Enable Split View from the top-right corner.

Simulate traffic:

NORMAL TRAFFIC → Graph remains secure (Blue/Green)

SQL INJECTION → Graph spikes Red → Intrusion Detected

UDP FLOOD (DDoS) → After ~15s → Mitigation Active: IP Blocked

📂 Project Structure
Major Project/
├── backend/
│   ├── ids_service.py
│   ├── attacker_service.py
│   ├── train_model.py
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── sac_actor.pth
│   └── scaler.pkl
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── AttackConsole.jsx
│   │   └── ...
├── START_SYSTEM.bat
├── PUSH_TO_GITHUB.bat
└── requirements.txt

🤖 Model Architecture — Soft Actor-Critic (SAC)

State Space: 12 network features

Action Space: Continuous → mapped to binary classification

Reward Function:

+1 → Correct classification

-1 → Incorrect classification

Learning: Off-policy deep reinforcement learning

👨‍💻 Author

Sohom Mandal
Major Project — Cybersecurity & Artificial Intelligence
