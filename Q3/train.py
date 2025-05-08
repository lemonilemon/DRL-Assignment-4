import argparse  # Added for command-line arguments
import os
import pickle  # Added for saving/loading replay buffer
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
CHECKPOINTS_DIR = "checkpoints"  # Directory for saving all checkpoints and models
DEFAULT_CHECKPOINT_PATH = os.path.join(
    CHECKPOINTS_DIR, "sac_icm_humanoid_interrupt_checkpoint.pth"
)


def make_env_for_config():
    # Helper to create a temporary env just for config like action space
    try:
        env = make_dmc_env(
            ENV_NAME, np.random.randint(0, 1000000), flatten=True, use_pixels=False
        )
        return env
    except NotImplementedError:
        print(
            "Cannot create DMC env for config, TARGET_ENTROPY might be incorrect if action space is unknown."
        )

        class MockActionSpace:
            def __init__(self, shape):
                self.shape = shape

        return type("MockEnv", (), {"action_space": MockActionSpace((6,))})()


# Hyperparameters
RANDOM_SEED = 42
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
LR_ICM = 3e-4

GAMMA = 0.99
TAU = 0.005
ALPHA_INITIAL = 0.2

temp_env_for_config = make_env_for_config()
TARGET_ENTROPY = -torch.prod(
    torch.Tensor(temp_env_for_config.action_space.shape).to(DEVICE)
).item()
del temp_env_for_config


BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
HIDDEN_DIM = 512

ICM_FEATURE_DIM = 128
ICM_ETA = 0.1
ICM_BETA = 0.2
ICM_FORWARD_LOSS_WEIGHT = 10.0


# --- Helper Functions ---
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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

    def get_buffer_data(self):
        return list(self.buffer)

    def load_buffer_data(self, data):
        self.buffer = deque(data, maxlen=self.buffer.maxlen)


# --- SAC Networks (Actor, Critic) ---
class Actor(nn.Module):
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
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state, reparameterize=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if reparameterize:
            x_t = normal.rsample()
        else:
            x_t = normal.sample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
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
    def __init__(self, state_dim, feature_dim, hidden_dim=HIDDEN_DIM):
        super(ICMEncoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feature_dim)
        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ICMForwardModel(nn.Module):
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
        return self.fc3(x)


class ICMInverseModel(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(ICMInverseModel, self).__init__()
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, state_feature, next_state_feature):
        x = torch.cat([state_feature, next_state_feature], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
        self.action_dim = action_dim

    def get_intrinsic_reward(self, state, action, next_state):
        with torch.no_grad():
            state_feat = self.encoder(state)
            next_state_feat = self.encoder(next_state)
            predicted_next_state_feat = self.forward_model(state_feat, action)
            intrinsic_reward = (
                (next_state_feat - predicted_next_state_feat)
                .pow(2)
                .sum(dim=1, keepdim=True)
            )
        return intrinsic_reward * ICM_ETA

    def update(self, state, action, next_state):
        state_feat = self.encoder(state)
        next_state_feat_actual = self.encoder(next_state)
        next_state_feat_pred = self.forward_model(state_feat.detach(), action)
        forward_loss = (
            F.mse_loss(next_state_feat_pred, next_state_feat_actual.detach())
            * self.forward_loss_weight
        )
        action_pred = self.inverse_model(state_feat, next_state_feat_actual)
        inverse_loss = F.mse_loss(action_pred, action)
        icm_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        self.optimizer_encoder.zero_grad()
        self.optimizer_forward.zero_grad()
        self.optimizer_inverse.zero_grad()

        icm_loss.backward()

        self.optimizer_encoder.step()
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

        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.log_alpha = torch.tensor(
            np.log(alpha_initial),
            dtype=torch.float32,
            requires_grad=True,
            device=DEVICE,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha = self.log_alpha.exp()

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.icm = ICMModule(state_dim, action_dim, ICM_FEATURE_DIM, lr=LR_ICM)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        if evaluate:
            _, _, action_mean = self.actor.sample(state_tensor, reparameterize=False)
            action = torch.tanh(action_mean) * self.max_action
        else:
            action, _, _ = self.actor.sample(state_tensor, reparameterize=False)
        return action.cpu().data.numpy().flatten()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None, None, None, None, None

        states, actions, extrinsic_rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )

        intrinsic_rewards = self.icm.get_intrinsic_reward(states, actions, next_states)
        f_loss, i_loss = self.icm.update(states, actions, next_states)
        total_rewards = extrinsic_rewards + intrinsic_rewards.detach()

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            q_target = total_rewards + self.gamma * (1 - dones) * (
                target_q - self.alpha * next_log_pi
            )

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(
            current_q2, q_target
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for params in self.critic.parameters():
            params.requires_grad = False
        pi_actions, log_pi, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for params in self.critic.parameters():
            params.requires_grad = True

        alpha_loss = -(
            self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

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
        """Saves only model weights (actor, critic, ICM) to the CHECKPOINTS_DIR."""
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)  # Ensure directory exists

        # Construct full path for saving models inside CHECKPOINTS_DIR
        actor_path = os.path.join(CHECKPOINTS_DIR, f"{filename_prefix}_actor.pth")
        critic_path = os.path.join(CHECKPOINTS_DIR, f"{filename_prefix}_critic.pth")
        icm_encoder_path = os.path.join(
            CHECKPOINTS_DIR, f"{filename_prefix}_icm_encoder.pth"
        )
        icm_forward_path = os.path.join(
            CHECKPOINTS_DIR, f"{filename_prefix}_icm_forward.pth"
        )
        icm_inverse_path = os.path.join(
            CHECKPOINTS_DIR, f"{filename_prefix}_icm_inverse.pth"
        )

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.icm.encoder.state_dict(), icm_encoder_path)
        torch.save(self.icm.forward_model.state_dict(), icm_forward_path)
        torch.save(self.icm.inverse_model.state_dict(), icm_inverse_path)
        print(
            f"Basic models saved to directory: {CHECKPOINTS_DIR} with prefix: {filename_prefix}"
        )

    def load_models(self, filename_prefix="sac_icm_humanoid"):
        """Loads only model weights from the CHECKPOINTS_DIR."""
        actor_path = os.path.join(CHECKPOINTS_DIR, f"{filename_prefix}_actor.pth")
        critic_path = os.path.join(CHECKPOINTS_DIR, f"{filename_prefix}_critic.pth")
        icm_encoder_path = os.path.join(
            CHECKPOINTS_DIR, f"{filename_prefix}_icm_encoder.pth"
        )
        icm_forward_path = os.path.join(
            CHECKPOINTS_DIR, f"{filename_prefix}_icm_forward.pth"
        )
        icm_inverse_path = os.path.join(
            CHECKPOINTS_DIR, f"{filename_prefix}_icm_inverse.pth"
        )

        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
            self.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
            self.critic_target.load_state_dict(
                self.critic.state_dict()
            )  # Sync target critic
            self.icm.encoder.load_state_dict(
                torch.load(icm_encoder_path, map_location=DEVICE)
            )
            self.icm.forward_model.load_state_dict(
                torch.load(icm_forward_path, map_location=DEVICE)
            )
            self.icm.inverse_model.load_state_dict(
                torch.load(icm_inverse_path, map_location=DEVICE)
            )
            print(
                f"Basic models loaded from directory: {CHECKPOINTS_DIR} with prefix: {filename_prefix}"
            )
        except FileNotFoundError as e:
            print(
                f"Error loading basic models: {e}. Ensure files exist in {CHECKPOINTS_DIR}."
            )
            # Depending on desired behavior, you might want to re-raise or handle differently
            raise

    def save_checkpoint(self, filepath, episode, total_steps, episode_rewards_list):
        """Saves the complete agent state for resuming training."""
        checkpoint_dir = os.path.dirname(filepath)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "total_steps": total_steps,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "icm_encoder_state_dict": self.icm.encoder.state_dict(),
            "icm_forward_model_state_dict": self.icm.forward_model.state_dict(),
            "icm_inverse_model_state_dict": self.icm.inverse_model.state_dict(),
            "icm_optimizer_encoder_state_dict": self.icm.optimizer_encoder.state_dict(),
            "icm_optimizer_forward_state_dict": self.icm.optimizer_forward.state_dict(),
            "icm_optimizer_inverse_state_dict": self.icm.optimizer_inverse.state_dict(),
            "replay_buffer_data": self.replay_buffer.get_buffer_data(),
            "episode_rewards_list": episode_rewards_list,
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Loads agent state from a checkpoint to resume training."""
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return None

        checkpoint = torch.load(filepath, map_location=DEVICE)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        self.log_alpha = torch.tensor(
            checkpoint["log_alpha"],
            dtype=torch.float32,
            requires_grad=True,
            device=DEVICE,
        )
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        self.alpha = self.log_alpha.exp().detach()

        self.icm.encoder.load_state_dict(checkpoint["icm_encoder_state_dict"])
        self.icm.forward_model.load_state_dict(
            checkpoint["icm_forward_model_state_dict"]
        )
        self.icm.inverse_model.load_state_dict(
            checkpoint["icm_inverse_model_state_dict"]
        )
        self.icm.optimizer_encoder.load_state_dict(
            checkpoint["icm_optimizer_encoder_state_dict"]
        )
        self.icm.optimizer_forward.load_state_dict(
            checkpoint["icm_optimizer_forward_state_dict"]
        )
        self.icm.optimizer_inverse.load_state_dict(
            checkpoint["icm_optimizer_inverse_state_dict"]
        )

        self.replay_buffer.load_buffer_data(checkpoint["replay_buffer_data"])

        random.setstate(checkpoint["random_rng_state"])
        np.random.set_state(checkpoint["np_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "torch_cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])

        print(f"Checkpoint loaded from {filepath}")
        return (
            checkpoint["episode"],
            checkpoint["total_steps"],
            checkpoint["episode_rewards_list"],
        )


# --- Main Training Loop ---
def main(args):
    set_seeds(RANDOM_SEED)

    def make_training_env():
        return make_dmc_env(
            ENV_NAME, np.random.randint(0, 1000000), flatten=True, use_pixels=False
        )

    env = make_training_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Environment: {ENV_NAME}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Target Entropy: {TARGET_ENTROPY}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoints will be saved in: ./{CHECKPOINTS_DIR}/")
    print(f"Default interruption checkpoint path: {DEFAULT_CHECKPOINT_PATH}")

    agent = SACAgent(state_dim, action_dim, max_action, target_entropy=TARGET_ENTROPY)

    max_episodes = 2000
    max_timesteps_per_episode = 1000
    log_interval = 10
    save_model_interval = 100
    warmup_steps = 10000

    start_episode = 1
    total_steps = 0
    episode_rewards = []

    if args.resume_checkpoint:
        # Ensure the resume_checkpoint path is correctly pointing into CHECKPOINTS_DIR if it's a relative path
        resume_path = args.resume_checkpoint
        if not os.path.isabs(resume_path) and not resume_path.startswith(
            CHECKPOINTS_DIR + os.sep
        ):
            resume_path = os.path.join(CHECKPOINTS_DIR, resume_path)
            print(
                f"Assuming relative resume checkpoint path is inside '{CHECKPOINTS_DIR}'. Full path: {resume_path}"
            )

        print(f"Attempting to resume from checkpoint: {resume_path}")
        load_result = agent.load_checkpoint(
            resume_path
        )  # Use potentially modified resume_path
        if load_result:
            loaded_episode, total_steps, episode_rewards = load_result
            start_episode = loaded_episode + 1
            print(f"Resuming from Episode {start_episode}, Total Steps: {total_steps}")
        else:
            print(f"Could not load checkpoint. Starting training from scratch.")

    try:
        for episode in range(start_episode, max_episodes + 1):
            current_episode_seed = RANDOM_SEED + episode
            state, info = env.reset(seed=current_episode_seed)
            episode_extrinsic_reward = 0

            for t in range(max_timesteps_per_episode):
                total_steps += 1

                if total_steps < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, extrinsic_reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated

                agent.replay_buffer.add(
                    state, action, extrinsic_reward, next_state, done
                )

                update_results = None
                if (
                    total_steps > warmup_steps
                    and len(agent.replay_buffer) >= BATCH_SIZE
                ):
                    update_results = agent.update(BATCH_SIZE)

                state = next_state
                episode_extrinsic_reward += extrinsic_reward

                if done:
                    break

            if episode >= start_episode:
                while len(episode_rewards) < episode - 1:
                    episode_rewards.append(0)
                if len(episode_rewards) == episode - 1:
                    episode_rewards.append(episode_extrinsic_reward)
                elif len(episode_rewards) > episode - 1:
                    episode_rewards[episode - 1] = episode_extrinsic_reward

            avg_reward_window = min(len(episode_rewards), 100)
            if avg_reward_window > 0:
                avg_reward = np.mean(episode_rewards[-avg_reward_window:])
            else:
                avg_reward = 0.0

            if episode % log_interval == 0:
                print(
                    f"Episode: {episode}, Timesteps: {t+1}, Total Steps: {total_steps}, Extrinsic Reward: {episode_extrinsic_reward:.2f}, Avg Ext. Reward ({avg_reward_window} ep): {avg_reward:.2f}"
                )
                if total_steps > warmup_steps and update_results:
                    (
                        actor_loss,
                        critic_loss,
                        alpha_loss,
                        current_alpha,
                        fwd_loss,
                        inv_loss,
                    ) = update_results
                    print(
                        f"  Losses -> Actor: {actor_loss:.3f}, Critic: {critic_loss:.3f}, Alpha: {alpha_loss:.3f} (Val: {current_alpha:.3f})"
                    )
                    print(
                        f"  ICM Losses -> Forward: {fwd_loss:.3f}, Inverse: {inv_loss:.3f}"
                    )

            if episode % save_model_interval == 0 and total_steps > warmup_steps:
                # save_models now saves to CHECKPOINTS_DIR
                agent.save_models(f"humanoid_sac_icm_ep{episode}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        current_episode_for_save = (
            episode
            if "episode" in locals() and episode >= start_episode
            else start_episode - 1
        )
        if current_episode_for_save < 0:
            current_episode_for_save = 0

        agent.save_checkpoint(
            DEFAULT_CHECKPOINT_PATH,
            current_episode_for_save,
            total_steps,
            episode_rewards,
        )
        print(f"Current state saved to {DEFAULT_CHECKPOINT_PATH}. Exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Attempting to save current state before exiting...")
        current_episode_for_save = (
            episode
            if "episode" in locals() and episode >= start_episode
            else start_episode - 1
        )
        if current_episode_for_save < 0:
            current_episode_for_save = 0

        agent.save_checkpoint(
            DEFAULT_CHECKPOINT_PATH,
            current_episode_for_save,
            total_steps,
            episode_rewards,
        )
        print(f"Current state saved to {DEFAULT_CHECKPOINT_PATH}. Exiting.")
        raise
    finally:
        env.close()
        print("Training finished or terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC with ICM agent for humanoid control."
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=f"Path to a checkpoint file to resume training from. If relative, assumed to be in '{CHECKPOINTS_DIR}'. E.g., 'my_checkpoint.pth' or '{os.path.join(CHECKPOINTS_DIR, 'my_checkpoint.pth')}'",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
