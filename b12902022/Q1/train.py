import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import copy

# --- Environment Setup ---
def make_env():
    """Creates the Pendulum-v1 environment."""
    # Create Pendulum-v1 environment
    # The state is 3-dimensional: [cos(theta), sin(theta), theta_dot]
    # The action is 1-dimensional: [torque], ranging from -2.0 to 2.0
    env = gym.make("Pendulum-v1", render_mode=None) # Use None for faster training, 'rgb_array' for rendering
    return env

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        """Initialize a ReplayBuffer object.
        Params
        ======
            capacity (int): maximum size of buffer
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Add a new experience to memory."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# --- Ornstein-Uhlenbeck Noise ---
# Often used for exploration in DDPG, helps with temporally correlated noise
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

# --- Actor Network ---
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        """Initialize parameters and build model.
        Params
        ======
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            max_action (float): Highest possible action value
            hidden_dim (int): Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        """Build an actor network that maps states -> actions."""
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        # Use tanh to output actions between -1 and 1, then scale
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

# --- Critic Network ---
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """Initialize parameters and build model.
        Params
        ======
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            hidden_dim (int): Number of nodes in hidden layers
        """
        super(Critic, self).__init__()
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture (for TD3, but can be helpful for stability in DDPG too)
        # If you want pure DDPG, you can remove Q2
        self.layer_4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_6 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        """Build a critic network that maps (state, action) pairs -> Q-values."""
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)

        # If using twin Q networks (like TD3)
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2 # Return both Q values

    def Q1(self, state, action):
        """ Returns Q-value from the first network"""
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)
        return q1


# --- DDPG Agent ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, device,
                 buffer_capacity=1000000, batch_size=256, gamma=0.99,
                 tau=0.005, lr_actor=3e-4, lr_critic=3e-4, weight_decay=0,
                 noise_seed=42):
        """Initialize the DDPG agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize Actor network and Target Actor network
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize Critic network and Target Critic network
        # Using two Q-networks and targets like in TD3 often improves stability
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Initialize Noise process
        self.noise = OUNoise(action_dim, seed=noise_seed)
        # Alternatively, use Gaussian noise:
        # self.noise_stddev = 0.1 # Adjust standard deviation as needed

        self.learn_step_counter = 0 # For updating target networks

    def select_action(self, state, add_noise=True):
        """Selects an action from the actor network, optionally adding noise."""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.actor.eval() # Set actor to evaluation mode
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train() # Set actor back to training mode

        if add_noise:
            # OU Noise
            noise = self.noise.sample()
            # Gaussian Noise (alternative)
            # noise = np.random.normal(0, self.noise_stddev * self.max_action, size=self.action_dim)
            action = (action + noise)

        # Clip action to the valid range
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """Updates actor and critic networks."""
        if len(self.replay_buffer) < self.batch_size:
            return # Don't update if buffer doesn't have enough samples

        self.learn_step_counter += 1

        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch elements to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.FloatTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)

        # --- Update Critic ---

        # Get next actions from target actor network
        next_actions = self.actor_target(next_states)

        # Compute the target Q value using target critic network(s)
        target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2) # Use minimum of two Q-targets (TD3 trick)
        # If using single Q critic (pure DDPG):
        # target_Q = self.critic_target.Q1(next_states, next_actions)

        target_Q = rewards + (self.gamma * target_Q * (1 - dones)).detach() # detach to stop gradients

        # Get current Q estimates from critic network(s)
        current_Q1, current_Q2 = self.critic(states, actions)
        # If using single Q critic (pure DDPG):
        # current_Q1 = self.critic.Q1(states, actions)

        # Compute critic loss (MSE loss)
        # If using twin Q networks:
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # If using single Q critic:
        # critic_loss = F.mse_loss(current_Q1, target_Q)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) # Optional: Gradient Clipping
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Compute actor loss (negative Q-value for the actions taken by the actor)
        # We want to maximize Q(s, actor(s)), which is equivalent to minimizing -Q(s, actor(s))
        actor_actions = self.actor(states)
        q_values_actor = self.critic.Q1(states, actor_actions) # Use Q1 from the critic
        actor_loss = -q_values_actor.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) # Optional: Gradient Clipping
        self.actor_optimizer.step()

        # --- Update Target Networks ---
        # Soft update target networks
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        """Saves the actor and critic models."""
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")
        print(f"Models saved to {filename}_*.pth")

    def load(self, filename):
        """Loads the actor and critic models."""
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
        self.actor_target = copy.deepcopy(self.actor)
        print(f"Models loaded from {filename}_*.pth")


# --- Training Loop ---
def train():
    # Hyperparameters
    env_name = "Pendulum-v1"
    seed = 0
    max_episodes = 200 # Number of training episodes
    max_timesteps = 200 # Max steps per episode for Pendulum-v1
    start_timesteps = 1000 # Number of steps for random actions before training starts
    expl_noise = 0.1 # Std of Gaussian exploration noise (used if not using OU noise)
    batch_size = 256
    gamma = 0.99 # Discount factor
    tau = 0.005 # Target network update rate
    lr_actor = 3e-4
    lr_critic = 3e-4
    buffer_capacity = 100000
    save_models = True # Save models after training
    model_save_path = "./ddpg_pendulum"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = make_env()

    # Set seeds
    # env.seed(seed) # Deprecated in newer Gymnasium
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}")

    # Initialize DDPG agent
    agent = DDPGAgent(state_dim, action_dim, max_action, device,
                      buffer_capacity=buffer_capacity, batch_size=batch_size,
                      gamma=gamma, tau=tau, lr_actor=lr_actor, lr_critic=lr_critic,
                      noise_seed=seed)

    # Training loop
    total_timesteps = 0
    episode_rewards = []

    for episode in range(max_episodes):
        state, info = env.reset(seed=seed + episode) # Use different seed per episode for reset
        episode_reward = 0
        agent.noise.reset() # Reset noise process for each episode if using OU Noise

        for t in range(max_timesteps):
            total_timesteps += 1

            # Select action: random for initial steps, then policy + noise
            if total_timesteps < start_timesteps:
                action = env.action_space.sample() # Sample random action
            else:
                # Use OU Noise from agent.select_action
                action = agent.select_action(state, add_noise=True)
                # If using Gaussian noise instead of OU:
                # action = agent.select_action(state, add_noise=False) # Get deterministic action
                # noise = np.random.normal(0, max_action * expl_noise, size=action_dim)
                # action = np.clip(action + noise, -max_action, max_action)


            # Perform action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # Episode ends if terminated or truncated

            # Store transition in replay buffer
            # Pendulum reward is often negative, consider normalizing or adjusting if needed
            agent.store_transition(state, action, reward, next_state, float(done)) # Store done as float (0.0 or 1.0)

            # Move to the next state
            state = next_state
            episode_reward += reward

            # Update agent networks after initial random steps
            if total_timesteps >= start_timesteps:
                agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) # Moving average of last 100 episodes

        print(f"Episode: {episode+1}, Timesteps: {t+1}, Total Timesteps: {total_timesteps}, Reward: {episode_reward:.2f}, Avg Reward (Last 100): {avg_reward:.2f}")

    # Save the final models
    if save_models:
        agent.save(model_save_path)

    env.close()

    # You can add plotting code here to visualize episode_rewards

# --- Main Execution ---
if __name__ == "__main__":
    train()


