# 🛡️ Sentinel AI - Intrusion Detection System (IDS)
**A Next-Generation AI-Powered Network Security System utilizing Soft Actor-Critic (SAC) Reinforcement Learning.**
![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Backend-FastAPI%20%7C%20PyTorch-blue)
![Frontend](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Recharts-cyan)
![AI Model](https://img.shields.io/badge/Model-Soft%20Actor--Critic%20(SAC)-orange)
---
## 📖 Overview
**Sentinel AI** is a fully functional Intrusion Detection System (IDS) designed to detect and mitigate network attacks in real-time. Unlike traditional rule-based IDS, Sentinel AI uses **Deep Reinforcement Learning (SAC Algorithm)** to learn traffic patterns and identify anomalies with high precision.
The system includes a **Cyberpunk-styled Dashboard** for the Blue Team (Defenders) and a dedicated **Red Team Console** for the Attacker to simulate threats like DDoS, SQL Injection, and Malware.
---
## ✨ Key Features
*   **🧠 AI-Driven Detection**: Uses a trained **Soft Actor-Critic (SAC)** neural network to classify network packets as "Normal" or "Attack".
*   **⚡ Real-Time Analysis**: Processes live UDP packets and visualizes threat levels instantly via WebSockets.
*   **⚔️ Red Team Console**: Built-in Attack Simulator to manually launch:
    *   **UDP Flood (DDoS)**
    *   **SQL Injection (SQLi)**
    *   **Cross-Site Scripting (XSS)**
    *   **Malware C2 Traffic**
    *   **Normal User Traffic** (for baseline testing)
*   **🛡️ Active Mitigation**: Automatically **BLOCKS** IP addresses after detecting sustained DDoS attacks (300+ malicious packets) and restores normal traffic flow.
*   **🖥️ Split View Mode**: Evaluate the system effectively with a dual-pane view showing the **Attack Console** and **IDS Dashboard** side-by-side.
*   **📊 Live Visualization**: Interactive Area Charts powered by Recharts showing AI Confidence scores over time.
---
## 🛠️ Technology Stack
### **Backend (Python)**
*   **FastAPI**: High-performance API for the IDS engine and Attacker service.
*   **PyTorch**: Framework for the SAC Reinforcement Learning model.
*   **Scikit-Learn**: For feature scaling (StandardScaler) and data preprocessing.
*   **Pandas/Numpy**: Data generation and manipulation.
*   **AsyncIO**: For handling asynchronous packet sniffing and WebSocket streaming.
### **Frontend (React)**
*   **Vite**: Next-generation frontend tooling.
*   **Recharts**: For real-time data visualization graphs.
*   **Lucide React**: Beautiful icons.
*   **TailwindCSS**: For rapid, futuristic styling.
---
## 🚀 Installation & Setup
### **Prerequisites**
*   **Python 3.8+**
*   **Node.js 16+**
### **1. Automatic Setup (Windows)**
Simply run the included batch script to install dependencies, train the model, and launch everything:
```bash
START_SYSTEM.bat
2. Manual Setup
If you prefer running components individually, open 3 Terminal Windows:

Terminal 1: IDS Backend (The Brain)

bash
cd "Major Project"
pip install -r requirements.txt
python -m uvicorn backend.ids_service:app --reload --port 8000
Terminal 2: Attacker Service (The Simulator)

bash
cd "Major Project"
python -m uvicorn backend.attacker_service:app --reload --port 8001
Terminal 3: Frontend ( The Dashboard)

bash
cd "Major Project/frontend"
npm install
npm run dev
🕹️ How to Use
Launch the System: Run 
START_SYSTEM.bat
.
Open the Dashboard: Go to http://localhost:5173 (or 5174).
Enter Split View: Click the "SPLIT VIEW" button in the top right.
Left Side: Your Defense Dashboard (Blue Team).
Right Side: Your Attack Console (Red Team).
Simulate Attacks:
Click "NORMAL TRAFFIC": Watch the graph stay Blue/Green (Secure).
Click "SQL INJECTION": Watch the graph spike Red and status change to "INTRUSION DETECTED".
Click "UDP FLOOD (DDoS)": Watch the system detect the high volume. After ~15 seconds, the status will change to "MITIGATION ACTIVE: IP BLOCKED".
📂 Project Structure
Major Project/
├── backend/
│   ├── ids_service.py        # Main IDS Engine (FastAPI + AI Inference)
│   ├── attacker_service.py   # Attack Simulation API
│   ├── train_model.py        # SAC Training Script
│   ├── generate_data.py      # Synthetic Dataset Generator
│   ├── feature_engineering.py# Data Preprocessing & Scaling
│   ├── sac_actor.pth         # Trained Model Weights
│   └── scaler.pkl            # Saved Scaler
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main Routing & Dashboard Logic
│   │   ├── AttackConsole.jsx # Red Team UI
│   │   └── ...
├── START_SYSTEM.bat          # One-click Launcher
├── PUSH_TO_GITHUB.bat        # Git Automation Script
└── requirements.txt          # Python Dependencies
🤖 Model Details (SAC)
The system uses Soft Actor-Critic, an off-policy reinforcement learning algorithm.

State Space: 12 features (Duration, Src/Dst Bytes, Count, Error Rate, etc.)
Action Space: Continuous (mapped to Binary Classification: Normal vs Attack).
Reward System: +1 for correct classification, -1 for incorrect.
Developed by Sohom Mandal for Major Project.


make it perfect for read me for github
