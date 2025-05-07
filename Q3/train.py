import os
import random
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

try:
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)
    sys.path.append(parent_dir)
    from dmc import make_dmc_env
except ImportError:
    print(
        "Warning: Could not import make_dmc_env. Ensure dmc.py is accessible in the parent directory."
    )

    # Define a dummy function if import fails, to avoid crashing later
    def make_dmc_env(*args, **kwargs):
        raise NotImplementedError(
            "make_dmc_env is not available. Please ensure dmc.py is in the parent directory of this script."
        )


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "humanoid-walk"  # Replace with your specific humanoid environment


def make_env():
    env = make_dmc_env(
        ENV_NAME, np.random.randint(0, 1000000), flatten=True, use_pixels=False
    )
    return env


# Hyperparameters (these will likely need extensive tuning)
RANDOM_SEED = 42
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
LR_ICM = 1e-3

GAMMA = 0.99  # Discount factor for extrinsic rewards
TAU = 0.005  # Target network update rate
ALPHA_INITIAL = 0.2  # Initial value for temperature parameter alpha
TARGET_ENTROPY = -torch.prod(
    torch.Tensor(make_env().action_space.shape).to(DEVICE)
).item()  # Heuristic

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
HIDDEN_DIM = 256

# ICM Specific Hyperparameters
ICM_FEATURE_DIM = 128
ICM_ETA = 0.1  # Weight of intrinsic reward vs extrinsic reward
ICM_BETA = 0.2  # Weight of forward loss vs inverse loss in ICM
ICM_FORWARD_LOSS_WEIGHT = 10.0  # Scaling factor for forward model loss


# --- Helper Functions ---
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Potentially make PyTorch deterministic (can slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(states).to(DEVICE),
            torch.FloatTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(next_states).to(DEVICE),
            torch.FloatTensor(dones).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# --- SAC Networks ---
class Actor(nn.Module):
    """
    Policy Network: maps state to action distribution (mean and std_dev)
    """

    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        # Constrain log_std to avoid numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state, reparameterize=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if reparameterize:
            # Reparameterization trick: z = mu + sigma * epsilon
            # This allows gradients to flow back through the sampling process
            x_t = normal.rsample()
        else:
            x_t = normal.sample()

        # Apply Tanh squashing to bound actions between -1 and 1
        # (or -max_action and max_action if scaled)
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        # Calculate log probability of the squashed action
        # This is derived from the change of variables formula
        # log_prob = normal.log_prob(x_t) - torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        # Simpler form:
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)  # Correction for Tanh squashing
        log_prob = log_prob.sum(1, keepdim=True)

        return (
            action,
            log_prob,
            mean,
        )  # Return mean for potential exploration strategies


class Critic(nn.Module):
    """
    Q-Value Network: maps (state, action) pair to Q-value
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture (for Clipped Double Q-Learning)
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        return q1


# --- ICM Module ---
class ICMEncoder(nn.Module):
    """
    Encodes states into a feature representation for ICM.
    """

    def __init__(self, state_dim, feature_dim, hidden_dim=HIDDEN_DIM):
        super(ICMEncoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        feature = self.fc3(x)  # No activation on the output feature
        return feature


class ICMForwardModel(nn.Module):
    """
    Predicts the feature representation of the next state given
    the feature representation of the current state and the action.
    """

    def __init__(self, feature_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(ICMForwardModel, self).__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.apply(init_weights)

    def forward(self, state_feature, action):
        x = torch.cat([state_feature, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predicted_next_state_feature = self.fc3(x)
        return predicted_next_state_feature


class ICMInverseModel(nn.Module):
    """
    Predicts the action taken given the feature representations
    of the current and next states.
    """

    def __init__(self, feature_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(ICMInverseModel, self).__init__()
        self.fc1 = nn.Linear(
            feature_dim * 2, hidden_dim
        )  # Current and next state features
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, state_feature, next_state_feature):
        x = torch.cat([state_feature, next_state_feature], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predicted_action = self.fc3(x)  # Output is continuous action
        return predicted_action


class ICMModule:
    def __init__(
        self,
        state_dim,
        action_dim,
        feature_dim,
        lr=LR_ICM,
        beta=ICM_BETA,
        forward_loss_weight=ICM_FORWARD_LOSS_WEIGHT,
    ):
        self.encoder = ICMEncoder(state_dim, feature_dim).to(DEVICE)
        self.forward_model = ICMForwardModel(feature_dim, action_dim).to(DEVICE)
        self.inverse_model = ICMInverseModel(feature_dim, action_dim).to(DEVICE)

        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_forward = optim.Adam(self.forward_model.parameters(), lr=lr)
        self.optimizer_inverse = optim.Adam(self.inverse_model.parameters(), lr=lr)

        self.beta = beta
        self.forward_loss_weight = forward_loss_weight
        self.action_dim = action_dim  # For inverse model loss calculation

    def get_intrinsic_reward(self, state, action, next_state):
        """
        Calculates the intrinsic reward based on the prediction error of the forward model.
        This is done WITHOUT updating the ICM models here.
        """
        with torch.no_grad():  # No gradients needed for reward calculation
            state_feat = self.encoder(state)
            next_state_feat = self.encoder(next_state)
            predicted_next_state_feat = self.forward_model(state_feat, action)
            # Intrinsic reward is proportional to the squared error
            intrinsic_reward = (
                (next_state_feat - predicted_next_state_feat)
                .pow(2)
                .sum(dim=1, keepdim=True)
            )
        return intrinsic_reward * ICM_ETA  # Scale by eta

    def update(self, state, action, next_state):
        """
        Updates the ICM models (Encoder, Forward, Inverse).
        """
        state_feat = self.encoder(state)
        next_state_feat_actual = self.encoder(
            next_state
        )  # Target for forward model, detached
        next_state_feat_pred = self.forward_model(
            state_feat.detach(), action
        )  # state_feat detached for forward model training

        # Forward Loss: L_F = || phi(s_t+1) - f_F(phi(s_t), a_t) ||^2
        # Predicts the embedding of the next state.
        forward_loss = (
            F.mse_loss(next_state_feat_pred, next_state_feat_actual.detach())
            * self.forward_loss_weight
        )

        # Inverse Loss: L_I = CrossEntropy(a_pred, a_t) for discrete, or MSE for continuous
        # Predicts the action taken to transition from s_t to s_t+1.
        # Here, state_feat and next_state_feat_actual are NOT detached as encoder is also trained via inverse model.
        action_pred = self.inverse_model(state_feat, next_state_feat_actual)
        inverse_loss = F.mse_loss(action_pred, action)  # Assuming continuous actions

        # Total ICM Loss
        icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        # Optimize Encoder (gradients from both forward and inverse loss)
        self.optimizer_encoder.zero_grad()
        # Gradients from forward loss (through next_state_feat_actual)
        # Gradients from inverse loss (through state_feat and next_state_feat_actual)
        icm_loss.backward()  # This will populate gradients for encoder, forward, and inverse models
        self.optimizer_encoder.step()

        # Optimize Forward Model (gradients only from forward_loss part)
        # We need to re-calculate forward_loss with detached state_feat for forward model specific update
        # Or, more simply, since encoder is already updated, we can just update forward and inverse models
        # based on the already computed icm_loss gradients.

        # The backward pass on icm_loss already computes gradients for all three.
        # We just need to step the optimizers.
        self.optimizer_forward.step()
        self.optimizer_inverse.step()

        return forward_loss.item(), inverse_loss.item()


# --- SAC Agent ---
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        lr_alpha=LR_ALPHA,
        gamma=GAMMA,
        tau=TAU,
        alpha_initial=ALPHA_INITIAL,
        target_entropy=TARGET_ENTROPY,
    ):

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.target_entropy = target_entropy

        # Actor Network
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic Networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Target Critic Networks (for stability)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for (
            param
        ) in (
            self.critic_target.parameters()
        ):  # Target networks are not trained directly
            param.requires_grad = False

        # Alpha (Temperature Parameter for Entropy Regularization)
        self.log_alpha = torch.tensor(
            np.log(alpha_initial),
            dtype=torch.float32,
            requires_grad=True,
            device=DEVICE,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp()  # Keep alpha positive

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.icm = ICMModule(state_dim, action_dim, ICM_FEATURE_DIM)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        if evaluate:  # During evaluation, use mean action for deterministic policy
            _, _, action_mean = self.actor.sample(state_tensor, reparameterize=False)
            action = torch.tanh(action_mean) * self.max_action
        else:  # During training, sample from the policy distribution
            action, _, _ = self.actor.sample(state_tensor, reparameterize=False)
        return action.cpu().data.numpy().flatten()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None, None, None, None, None  # Not enough samples yet

        states, actions, extrinsic_rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )

        # --- Calculate Intrinsic Reward and update ICM ---
        # Note: ICM uses raw states, not normalized states if you normalize for SAC.
        # Here, assuming states are consistent.
        intrinsic_rewards = self.icm.get_intrinsic_reward(states, actions, next_states)
        f_loss, i_loss = self.icm.update(states, actions, next_states)

        # Combine rewards: r_total = r_extrinsic + eta * r_intrinsic
        # The intrinsic_rewards from get_intrinsic_reward already has eta factored in.
        total_rewards = (
            extrinsic_rewards + intrinsic_rewards.detach()
        )  # Detach intrinsic rewards for SAC update

        # --- Critic Update ---
        with torch.no_grad():
            # Get next actions and log probabilities from the policy
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            # Get target Q-values from target critic networks
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            # Bellman equation for Q-target: Q_target = r + gamma * (1-done) * (Q_next - alpha * log_prob_next)
            q_target = total_rewards + self.gamma * (1 - dones) * (
                target_q - self.alpha * next_log_pi
            )

        # Current Q-values from critic networks
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (MSE loss)
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(
            current_q2, q_target
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update (Delayed, typically less frequent than critic) ---
        # Freeze critic networks to avoid computing gradients for them during actor update
        for params in self.critic.parameters():
            params.requires_grad = False

        # Get actions and log probabilities from the policy for current states
        pi_actions, log_pi, _ = self.actor.sample(states)
        # Q-values for these actions from the critic network
        q1_pi, q2_pi = self.critic(states, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss: J_pi = E[alpha * log_prob - Q_val]
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()  # Use detached alpha

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # --- Alpha (Temperature) Update ---
        # We want to keep policy entropy close to target_entropy
        # Alpha loss: J_alpha = E[-alpha * (log_prob + target_entropy)]
        alpha_loss = -(
            self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = (
            self.log_alpha.exp().detach()
        )  # Update alpha value, detach for next actor loss

        # --- Soft Update Target Networks ---
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return (
            actor_loss.item(),
            critic_loss.item(),
            alpha_loss.item(),
            self.alpha.item(),
            f_loss,
            i_loss,
        )

    def save_models(self, filename_prefix="sac_icm_humanoid"):
        torch.save(self.actor.state_dict(), f"{filename_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename_prefix}_critic.pth")
        torch.save(self.icm.encoder.state_dict(), f"{filename_prefix}_icm_encoder.pth")
        torch.save(
            self.icm.forward_model.state_dict(), f"{filename_prefix}_icm_forward.pth"
        )
        torch.save(
            self.icm.inverse_model.state_dict(), f"{filename_prefix}_icm_inverse.pth"
        )
        print(f"Models saved with prefix: {filename_prefix}")

    def load_models(self, filename_prefix="sac_icm_humanoid"):
        self.actor.load_state_dict(
            torch.load(f"{filename_prefix}_actor.pth", map_location=DEVICE)
        )
        self.critic.load_state_dict(
            torch.load(f"{filename_prefix}_critic.pth", map_location=DEVICE)
        )
        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )  # Ensure target is same as loaded critic
        self.icm.encoder.load_state_dict(
            torch.load(f"{filename_prefix}_icm_encoder.pth", map_location=DEVICE)
        )
        self.icm.forward_model.load_state_dict(
            torch.load(f"{filename_prefix}_icm_forward.pth", map_location=DEVICE)
        )
        self.icm.inverse_model.load_state_dict(
            torch.load(f"{filename_prefix}_icm_inverse.pth", map_location=DEVICE)
        )
        print(f"Models loaded from prefix: {filename_prefix}")


# --- Main Training Loop ---
def main():
    set_seeds(RANDOM_SEED)

    # Initialize environment
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # Assuming symmetric action space

    print(f"Environment: {ENV_NAME}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Target Entropy: {TARGET_ENTROPY}")
    print(f"Device: {DEVICE}")

    agent = SACAgent(state_dim, action_dim, max_action, target_entropy=TARGET_ENTROPY)

    max_episodes = 2000
    max_timesteps_per_episode = 1000  # Humanoid envs often terminate early
    log_interval = 10  # Log every N episodes
    save_interval = 100  # Save models every N episodes
    warmup_steps = (
        10000  # Number of steps with random actions to populate buffer initially
    )

    total_steps = 0
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state, info = env.reset(
            seed=RANDOM_SEED + episode
        )  # Ensure different seed per episode for reset
        episode_extrinsic_reward = 0
        episode_intrinsic_reward = 0  # For tracking

        for t in range(max_timesteps_per_episode):
            total_steps += 1

            if total_steps < warmup_steps:
                action = env.action_space.sample()  # Random action during warmup
            else:
                action = agent.select_action(state)

            next_state, extrinsic_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Store experience in replay buffer ---
            # ICM needs raw state, action, next_state to calculate intrinsic reward
            # The intrinsic reward itself is calculated during the agent.update() step
            # or can be calculated here if needed for immediate use (but usually calculated from batch)
            agent.replay_buffer.add(state, action, extrinsic_reward, next_state, done)

            # --- Train agent ---
            if total_steps > warmup_steps:  # Start training after warmup
                # The update function handles intrinsic reward calculation and ICM update internally
                update_results = agent.update(BATCH_SIZE)
                if update_results:
                    (
                        actor_loss,
                        critic_loss,
                        alpha_loss,
                        current_alpha,
                        fwd_loss,
                        inv_loss,
                    ) = update_results
                    # You can log these losses if needed

            state = next_state
            episode_extrinsic_reward += extrinsic_reward

            if done:
                break

        episode_rewards.append(episode_extrinsic_reward)
        avg_reward = np.mean(
            episode_rewards[-100:]
        )  # Moving average of last 100 episodes

        if episode % log_interval == 0:
            print(
                f"Episode: {episode}, Timesteps: {t+1}, Total Steps: {total_steps}, Extrinsic Reward: {episode_extrinsic_reward:.2f}, Avg Ext. Reward: {avg_reward:.2f}"
            )
            if total_steps > warmup_steps and update_results:
                print(
                    f"  Losses -> Actor: {actor_loss:.3f}, Critic: {critic_loss:.3f}, Alpha: {alpha_loss:.3f} (Val: {current_alpha:.3f})"
                )
                print(
                    f"  ICM Losses -> Forward: {fwd_loss:.3f}, Inverse: {inv_loss:.3f}"
                )

        if episode % save_interval == 0 and total_steps > warmup_steps:
            agent.save_models(f"humanoid_sac_icm_ep{episode}")

    env.close()
    print("Training finished.")
    # Example of loading models (optional)
    # agent.load_models("humanoid_sac_icm_epYOUR_EPISODE_NUMBER")


if __name__ == "__main__":
    main()
