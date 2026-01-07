import sys
import time
import random
import socket
import threading

def attack_ddos(target_ip, target_port, duration=10):
    print(f"[Attack] Starting DDoS on {target_ip}:{target_port} for {duration}s...")
    timeout = time.time() + duration
    sent = 0
    while time.time() < timeout:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            bytes_to_send = random._urandom(1024)
            s.sendto(bytes_to_send, (target_ip, target_port))
            sent += 1
            if sent % 1000 == 0:
                print(f"[Attack] Sent {sent} packets...")
        except Exception as e:
            pass
    print(f"[Attack] DDoS Finished. Sent {sent} packets.")

def attack_port_scan(target_ip):
    print(f"[Attack] Starting Port Scan on {target_ip}...")
    open_ports = []
    # Scan random subset of ports
    ports = random.sample(range(1, 1024), 50)
    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            result = s.connect_ex((target_ip, port))
            if result == 0:
                open_ports.append(port)
            s.close()
        except:
            pass
    print(f"[Attack] Port Scan Finished. Open ports found: {open_ports}")

def attack_botnet_sim(target_ip):
    print(f"[Attack] Simulating Botnet Traffic to {target_ip}...")
    # Simulate meaningful periodic traffic
    for i in range(10):
        print(f"[Attack] Botnet beacon {i+1}/10 sent.")
        time.sleep(0.5)
    print(f"[Attack] Botnet Simulation Finished.")

def main():
    target_ip = "127.0.0.1" # Localhost
    target_port = 9999
    
    while True:
        print("\n--- Attack Simulator Menu ---")
        print("1. DDoS Attack")
        print("2. Port Scan")
        print("3. Botnet Simulation")
        print("4. Exit")
        
        choice = input("Select attack (1-4): ")
        
        if choice == '1':
            attack_ddos(target_ip, target_port)
        elif choice == '2':
            attack_port_scan(target_ip)
        elif choice == '3':
            attack_botnet_sim(target_ip)
        elif choice == '4':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
