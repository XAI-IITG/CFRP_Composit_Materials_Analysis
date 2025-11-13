import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RULPredictionEnvironment:
    """
    Proper environment for RUL prediction using temporal sequences
    Treats RUL prediction as a sequential decision problem
    """
    
    def __init__(self, sequences, targets, sequence_length=10):
        """
        Args:
            sequences: (n_samples, seq_len, n_features) - temporal sequences
            targets: (n_samples,) - RUL targets
            sequence_length: length of sequences to use
        """
        self.sequences = sequences
        self.targets = targets
        self.sequence_length = sequence_length
        self.n_samples = len(sequences)
        
        # Normalize targets for stable learning
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
        self.normalized_targets = (targets - self.target_mean) / (self.target_std + 1e-8)
        
        # Current episode state
        self.current_sample = 0
        self.current_timestep = 0
        self.episode_length = self.sequence_length
    
    def reset(self):
        """Reset environment to start new episode"""
        self.current_sample = np.random.randint(0, self.n_samples)
        self.current_timestep = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state - partial sequence up to current timestep"""
        # Create state with padding for incomplete sequences
        state = np.zeros((self.sequence_length, self.sequences.shape[-1]))
        
        # Fill available timesteps
        available_steps = min(self.current_timestep + 1, self.sequence_length)
        state[:available_steps] = self.sequences[self.current_sample][:available_steps]
        
        return state
    
    def step(self, action):
        """
        Execute action (RUL prediction) and return reward
        
        Args:
            action: Predicted RUL value
            
        Returns:
            next_state, reward, done, info
        """
        # Current actual RUL
        actual_rul = self.normalized_targets[self.current_sample]
        predicted_rul = action
        
        # Calculate error
        error = abs(predicted_rul - actual_rul)
        
        # Reward function: Encourage accurate predictions
        # Positive reward for good predictions, negative for bad ones
        if error < 0.05:  # Very accurate (within 5% of normalized range)
            reward = 20.0 - error * 200
        elif error < 0.1:  # Good accuracy (within 10%)
            reward = 10.0 - error * 100
        elif error < 0.2:  # Moderate accuracy (within 20%)
            reward = 5.0 - error * 25
        else:  # Poor prediction
            reward = -error * 20
        
        # Add bonus for conservative predictions (better to overestimate RUL)
        if predicted_rul > actual_rul:
            reward += 1.0  # Small bonus for conservative predictions
        
        # Progress in sequence
        self.current_timestep += 1
        done = self.current_timestep >= self.episode_length
        
        # Get next state
        if done:
            next_state = np.zeros((self.sequence_length, self.sequences.shape[-1]))
        else:
            next_state = self._get_state()
        
        info = {
            'error': error,
            'actual_rul': actual_rul,
            'predicted_rul': predicted_rul,
            'timestep': self.current_timestep
        }
        
        return next_state, reward, done, info

class RULActorCritic(nn.Module):
    """
    Actor-Critic network for continuous RUL prediction
    Uses LSTM to process temporal sequences
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(RULActorCritic, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM backbone for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Actor network (policy) - predicts RUL
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1] for normalized RUL
        )
        
        # Actor std (for exploration)
        self.actor_std = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive std
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len, input_dim)
            hidden: LSTM hidden state
            
        Returns:
            mean, std, value, hidden
        """
        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last timestep output
        features = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Actor outputs
        mean = self.actor_mean(features).squeeze(-1)  # (batch_size,)
        std = self.actor_std(features).squeeze(-1) + 1e-6  # Ensure positive
        
        # Critic output
        value = self.critic(features).squeeze(-1)  # (batch_size,)
        
        return mean, std, value, hidden
    
    def act(self, state, hidden=None, deterministic=False):
        """
        Select action based on current policy
        
        Args:
            state: Current state
            hidden: LSTM hidden state
            deterministic: If True, return mean action
            
        Returns:
            action, log_prob, value, hidden
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            mean, std, value, hidden = self.forward(state_tensor, hidden)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            # Sample from normal distribution
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Clamp action to valid range [0, 1]
            action = torch.clamp(action, 0, 1)
        
        return action.cpu().numpy()[0], log_prob, value.cpu().numpy()[0], hidden

class PPORULAgent:
    """
    PPO agent specifically designed for RUL prediction
    """
    
    def __init__(self, input_dim, hidden_dim=128, lr=3e-4, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        
        # Networks
        self.actor_critic = RULActorCritic(input_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_values = []
        
    def select_action(self, state, hidden=None, training=True):
        """Select action using current policy"""
        action, log_prob, value, hidden = self.actor_critic.act(
            state, hidden, deterministic=not training
        )
        
        # Store for training
        if training:
            self.states.append(state.copy())
            self.actions.append(action)
            self.values.append(value)
            if log_prob is not None:
                self.log_probs.append(log_prob.cpu().numpy())
        
        return action, hidden
    
    def store_transition(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self, next_value=0):
        """Compute advantages using GAE"""
        advantages = []
        returns = []
        
        gae = 0
        gamma = 0.99
        lam = 0.95
        
        values = self.values + [next_value]
        
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0}
        
        # Compute advantages
        advantages, returns = self.compute_advantages()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of training
        total_losses = []
        policy_losses = []
        value_losses = []
        
        for epoch in range(4):
            # Forward pass
            mean, std, values = self.actor_critic(states)[:3]
            
            # Current policy log probabilities
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        # Clear memory
        self.clear_memory()
        
        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for action noise in DDPG"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.BoolTensor(done).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

class DDPGActor(nn.Module):
    """DDPG Actor Network with LSTM for temporal processing"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(DDPGActor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM backbone for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Actor layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single continuous action (RUL prediction)
        
        # Weight initialization
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        """
        Forward pass
        Args:
            state: (batch_size, seq_len, input_dim)
        Returns:
            action: (batch_size,) - RUL prediction in [0, 1]
        """
        # LSTM processing
        lstm_out, _ = self.lstm(state)
        
        # Use last timestep output
        features = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Actor forward pass
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        action = torch.sigmoid(self.fc3(x))  # Output in [0, 1] range
        
        return action.squeeze(-1)

class DDPGCritic(nn.Module):
    """DDPG Critic Network with LSTM for temporal processing"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(DDPGCritic, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM backbone for state processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Critic layers (state + action)
        self.fc1 = nn.Linear(hidden_dim + 1, 64)  # +1 for action
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Q-value output
        
        # Weight initialization
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """
        Forward pass
        Args:
            state: (batch_size, seq_len, input_dim)
            action: (batch_size,) - RUL predictions
        Returns:
            q_value: (batch_size,) - Q-values
        """
        # LSTM processing of state
        lstm_out, _ = self.lstm(state)
        
        # Use last timestep output
        state_features = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Concatenate state features with action
        if action.dim() == 1:
            action = action.unsqueeze(1)  # (batch_size, 1)
        
        x = torch.cat([state_features, action], dim=1)  # (batch_size, hidden_dim + 1)
        
        # Critic forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value.squeeze(-1)

class RULPredictionEnvironmentDDPG:
    """
    Environment for RUL prediction using temporal sequences (Same logic as PPO environment)
    """
    
    def __init__(self, sequences, targets, sequence_length=10):
        """
        Args:
            sequences: (n_samples, seq_len, n_features) - temporal sequences
            targets: (n_samples,) - RUL targets
            sequence_length: length of sequences to use
        """
        self.sequences = sequences
        self.targets = targets
        self.sequence_length = sequence_length
        self.n_samples = len(sequences)
        
        # Normalize targets for stable learning
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
        self.normalized_targets = (targets - self.target_mean) / (self.target_std + 1e-8)
        
        # Current episode state
        self.current_sample = 0
        self.current_timestep = 0
        self.episode_length = self.sequence_length
    
    def reset(self):
        """Reset environment to start new episode"""
        self.current_sample = np.random.randint(0, self.n_samples)
        self.current_timestep = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state - partial sequence up to current timestep"""
        # Create state with padding for incomplete sequences
        state = np.zeros((self.sequence_length, self.sequences.shape[-1]))
        
        # Fill available timesteps
        available_steps = min(self.current_timestep + 1, self.sequence_length)
        state[:available_steps] = self.sequences[self.current_sample][:available_steps]
        
        return state
    
    def step(self, action):
        """
        Execute action (RUL prediction) and return reward
        
        Args:
            action: Predicted RUL value
            
        Returns:
            next_state, reward, done, info
        """
        # Current actual RUL
        actual_rul = self.normalized_targets[self.current_sample]
        predicted_rul = action
        
        # Calculate error
        error = abs(predicted_rul - actual_rul)
        
        # # Reward function: Same as PPO
        # if error < 0.05:  # Very accurate (within 5% of normalized range)
        #     reward = 20.0 - error * 200
        # elif error < 0.1:  # Good accuracy (within 10%)
        #     reward = 10.0 - error * 100
        # elif error < 0.2:  # Moderate accuracy (within 20%)
        #     reward = 5.0 - error * 25
        # else:  # Poor prediction
        #     reward = -error * 20
        
        # # Add bonus for conservative predictions (better to overestimate RUL)
        # if predicted_rul > actual_rul:
        #     reward += 1.0  # Small bonus for conservative predictions

        # NEW (more balanced):
        if error < 0.05:  # Very accurate
            reward = 10.0
        elif error < 0.1:  # Good
            reward = 5.0 - error * 50
        elif error < 0.2:  # Acceptable
            reward = 2.0 - error * 10
        else:  # Poor
            reward = -error * 5  # ✅ Less harsh penalty

        # Keep the conservative prediction bonus
        if predicted_rul > actual_rul:
            reward += 0.5  # ✅ Reduced from 1.0
                
        # Progress in sequence
        self.current_timestep += 1
        done = self.current_timestep >= self.episode_length
        
        # Get next state
        if done:
            next_state = np.zeros((self.sequence_length, self.sequences.shape[-1]))
        else:
            next_state = self._get_state()
        
        info = {
            'error': error,
            'actual_rul': actual_rul,
            'predicted_rul': predicted_rul,
            'timestep': self.current_timestep
        }
        
        return next_state, reward, done, info

class DDPGAgent:
    """DDPG Agent for RUL Prediction"""
    
    # def __init__(self, input_dim, lr_actor=1e-4, lr_critic=1e-3, 
    #              gamma=0.99, tau=0.001, buffer_size=100000, batch_size=64):
    def __init__(self, input_dim, lr_actor=3e-5, lr_critic=1e-3, 
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64):
        
        self.input_dim = input_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Initialize networks
        self.actor = DDPGActor(input_dim).to(device)
        self.actor_target = DDPGActor(input_dim).to(device)
        self.critic = DDPGCritic(input_dim).to(device)
        self.critic_target = DDPGCritic(input_dim).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize noise
        # self.noise = OrnsteinUhlenbeckNoise(size=1, mu=0.0, theta=0.15, sigma=0.2)
        self.noise = OrnsteinUhlenbeckNoise(size=1, mu=0.5, theta=0.15, sigma=0.3)  # ✅ Center noise at 0.5, increase exploration
        
        # Copy weights to target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
    
    def hard_update(self, target, source):
        """Hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def select_action(self, state, noise_mu=0.5, noise_sigma=0.3, training=True):
        """Select action with optional noise for exploration"""
        self.actor.eval()
        self.noise = OrnsteinUhlenbeckNoise(size=1, mu=noise_mu, theta=0.15, sigma=noise_sigma)  # ✅ Center noise at 0.5, increase exploration

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if training:
            # Add exploration noise
            noise = self.noise.sample()[0]
            action += noise
            action = np.clip(action, 0, 1)  # Ensure action stays in [0, 1]
        
        return action
    
    def update(self):
        """Update actor and critic networks"""
        
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * next_q_values * ~dones)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

class RULEnvironment:
    """Environment for RUL prediction using DRL"""
    
    def __init__(self, sequences, targets, max_steps=20):
        self.sequences = sequences
        self.targets = targets
        self.max_steps = max_steps
        self.current_step = 0
        self.current_sequence_idx = 0
        self.n_sequences = len(sequences)
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_sequence_idx = np.random.randint(0, self.n_sequences)
        return self._get_state()
    
    def _get_state(self):
        """Get current state (feature sequence)"""
        if self.current_sequence_idx < self.n_sequences:
            return self.sequences[self.current_sequence_idx].flatten()
        else:
            return np.zeros(self.sequences[0].flatten().shape)
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Action is the predicted RUL
        predicted_rul = action
        actual_rul = self.targets[self.current_sequence_idx]
        
        # Calculate reward (negative error)
        error = abs(predicted_rul - actual_rul)
        reward = -error / max(actual_rul, 1e-6)  # Normalize by actual RUL
        
        # Move to next sequence
        self.current_step += 1
        self.current_sequence_idx += 1
        
        done = (self.current_step >= self.max_steps or 
                self.current_sequence_idx >= self.n_sequences)
        
        if not done:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.sequences[0].flatten().shape)
        
        return next_state, reward, done

class DQNAgent(nn.Module):
    """Deep Q-Network Agent for RUL prediction"""
    
    def __init__(self, state_dim, hidden_dim=256):
        super(DQNAgent, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),  # Output RUL prediction
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, x):
        return self.network(x).squeeze(-1)

class DRLRULPredictor:
    """DRL-based RUL predictor"""
    
    def __init__(self, state_dim, lr=0.001, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Networks
        self.q_network = DQNAgent(state_dim).to(device)
        self.target_network = DQNAgent(state_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Random action (random RUL prediction)
            return np.random.random()
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
