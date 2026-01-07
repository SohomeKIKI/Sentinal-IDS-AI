import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from collections import deque

# --- Hyperparameters ---
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
HIDDEN_SIZE = 256
LR = 3e-4
BATCH_SIZE = 256
MEMORY_SIZE = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Networks ---

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        self.mean = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Linear(HIDDEN_SIZE, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (correction for tanh)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, HIDDEN_SIZE)
        self.l5 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l6 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = nn.functional.relu(self.l1(sa))
        q1 = nn.functional.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = nn.functional.relu(self.l4(sa))
        q2 = nn.functional.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = nn.functional.relu(self.l1(sa))
        q1 = nn.functional.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]
    
    def predict(self, state):
        # Wrapper for inference to return classification (Attack vs Normal)
        # Action is continuous between -1 and 1. 
        # We map: > 0 => Attack (1), <= 0 => Normal (0)
        action = self.select_action(state, evaluate=True)
        # In this simplistic mapping: Action tells "Confidence of Attack"
        confidence = (action[0] + 1) / 2 # Normalize to 0-1
        is_attack = confidence > 0.5
        return is_attack, confidence

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - ALPHA * next_log_prob
            target_Q = reward + (1 - done) * GAMMA * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_prob_new = self.actor.sample(state)
        Q1_new, Q2_new = self.critic(state, action_new)
        Q_new = torch.min(Q1_new, Q2_new)
        actor_loss = (ALPHA * log_prob_new - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# --- Training Loop ---
def train_sac_model():
    print("Loading Data...")
    try:
        X_train = np.load("backend/X_train.npy")
        y_train = np.load("backend/y_train.npy")
    except FileNotFoundError:
        print("Data not found, running processing scripts...")
        import feature_engineering # Feature Engineering will rely on generate_data being run manually or implicitly
        # Assuming generate_data and feature_engineering are in the same directory context or already run.
        # But let's assume successful load for now as per prior steps.
        X_train = np.load("backend/X_train.npy")
        y_train = np.load("backend/y_train.npy")

    state_dim = X_train.shape[1]
    action_dim = 1 # Continuous output 

    agent = SACAgent(state_dim, action_dim)
    
    print("Starting Training (Simulated Environment)...")
    episodes = 50 # Increase for better accuracy
    
    for episode in range(episodes):
        state_idx = np.random.randint(0, len(X_train))
        state = X_train[state_idx]
        total_reward = 0
        
        # Simulate a "session" of network traffic classification
        for step in range(200): # Max steps per episode
            action = agent.select_action(state)
            
            # Environment Step (Simulated)
            # Interpret Action: > 0 => Predict Attack (1), <= 0 => Predict Normal (0)
            prediction = 1 if action[0] > 0 else 0
            actual = y_train[state_idx]
            
            # Reward Function
            if prediction == actual:
                reward = 1.0 # Correct classification
            else:
                reward = -1.0 # Incorrect
                
            agent.replay_buffer.push(state, action, reward, state, 0) # Next state is approximated as same or ignored for this static classification task simplification
            agent.update()
            
            total_reward += reward
            
            # Move to next sample for next step
            state_idx = (state_idx + 1) % len(X_train)
            state = X_train[state_idx]
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    # Save Model
    torch.save(agent.actor.state_dict(), "backend/sac_actor.pth")
    torch.save(agent.critic.state_dict(), "backend/sac_critic.pth")
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train_sac_model()
