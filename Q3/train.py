import argparse
import os
import pickle
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
from tqdm import tqdm

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)
    sys.path.append(parent_dir)
    from dmc import make_dmc_env
except ImportError:
    print(
        "Warning: Could not import make_dmc_env. Ensure dmc.py is accessible in the parent directory."
    )

    def make_dmc_env(*args, **kwargs):
        raise NotImplementedError(
            "make_dmc_env is not available. Please ensure dmc.py is in the parent directory of this script."
        )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS_DIR = "checkpoints"
MODELS_DIR = "models"


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
        if m.bias is not None:
            m.bias.data.fill_(0.01)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(
        self, state_dim, action_dim, max_size, device, reward_norm_clipping=5.0
    ):
        self.max_size = int(max_size)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros(
            (self.max_size, state_dim), dtype=torch.float32, device=device
        )
        self.action = torch.zeros(
            (self.max_size, action_dim), dtype=torch.float32, device=device
        )
        self.next_state = torch.zeros(
            (self.max_size, state_dim), dtype=torch.float32, device=device
        )
        self.reward = torch.zeros(
            (self.max_size, 1), dtype=torch.float32, device=device
        )
        self.done = torch.zeros((self.max_size, 1), dtype=torch.float32, device=device)

        # For reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0
        self.reward_count = 0
        self.reward_norm_clipping = reward_norm_clipping

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.action[self.ptr] = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        )
        self.next_state[self.ptr] = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device
        )
        self.done[self.ptr] = torch.as_tensor(
            done, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Update reward statistics for normalization
        self.reward_count += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward**2

        if self.reward_count > 1:
            self.reward_mean = self.reward_sum / self.reward_count
            variance = (self.reward_sq_sum / self.reward_count) - self.reward_mean**2
            self.reward_std = np.sqrt(max(variance, 1e-6))
        else:
            self.reward_mean = reward
            self.reward_std = 1.0

        normalized_reward = (reward - self.reward_mean) / self.reward_std
        normalized_reward = np.clip(
            normalized_reward, -self.reward_norm_clipping, self.reward_norm_clipping
        )
        self.reward[self.ptr] = torch.as_tensor(
            normalized_reward, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
        )

    def __len__(self):
        return self.size

    # Methods for checkpointing replay buffer data
    def get_buffer_data_for_checkpoint(self):
        return {
            "state": self.state[: self.size].cpu().numpy(),
            "action": self.action[: self.size].cpu().numpy(),
            "next_state": self.next_state[: self.size].cpu().numpy(),
            "reward": self.reward[: self.size].cpu().numpy(),
            "done": self.done[: self.size].cpu().numpy(),
            "ptr": self.ptr,
            "size": self.size,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_sum": self.reward_sum,
            "reward_sq_sum": self.reward_sq_sum,
            "reward_count": self.reward_count,
        }

    def load_buffer_data_from_checkpoint(self, data):
        self.ptr = data["ptr"]
        self.size = data["size"]

        load_size = min(self.size, self.max_size)

        self.state[:load_size] = torch.as_tensor(
            data["state"][:load_size], dtype=torch.float32, device=self.device
        )
        self.action[:load_size] = torch.as_tensor(
            data["action"][:load_size], dtype=torch.float32, device=self.device
        )
        self.next_state[:load_size] = torch.as_tensor(
            data["next_state"][:load_size], dtype=torch.float32, device=self.device
        )
        self.reward[:load_size] = torch.as_tensor(
            data["reward"][:load_size], dtype=torch.float32, device=self.device
        )
        self.done[:load_size] = torch.as_tensor(
            data["done"][:load_size], dtype=torch.float32, device=self.device
        )

        self.reward_mean = data["reward_mean"]
        self.reward_std = data["reward_std"]
        self.reward_sum = data["reward_sum"]
        self.reward_sq_sum = data["reward_sq_sum"]
        self.reward_count = data["reward_count"]
        print(f"Replay buffer loaded with {self.size} samples.")


# --- SAC Networks (Actor, Critic) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = Normal(mu, std)
        z = normal.rsample()  # Reparameterization trick
        action_tanh = torch.tanh(z)
        action = action_tanh * self.max_action

        # Calculate log_prob correctly for TanhNormal distribution
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.max_action * (1 - action_tanh.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


# --- ICM Module ---
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, hidden_dim):
        super(ICM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_weights)
        self.feature_dim = feature_dim

    def encode(self, state):
        return self.encoder(state)

    def get_intrinsic_reward(self, state_feat, action, next_state_feat_actual):
        predicted_next_state_feat = self.forward_model(
            torch.cat([state_feat, action], dim=1)
        )
        intrinsic_reward = 0.5 * F.mse_loss(
            predicted_next_state_feat, next_state_feat_actual, reduction="none"
        ).mean(dim=1, keepdim=True)
        return intrinsic_reward

    def calculate_losses(self, state_feat, action, next_state_feat_actual):
        # Forward model loss
        predicted_next_state_feat = self.forward_model(
            torch.cat([state_feat.detach(), action], dim=1)
        )
        forward_loss = F.mse_loss(
            predicted_next_state_feat, next_state_feat_actual.detach()
        )

        # Inverse model loss
        predicted_action = self.inverse_model(
            torch.cat([state_feat, next_state_feat_actual], dim=1)
        )
        inverse_loss = F.mse_loss(predicted_action, action)

        return forward_loss, inverse_loss


# --- SAC Agent ---
class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        args,
    ):
        self.args = args
        self.device = DEVICE

        self.gamma = args.gamma
        self.tau = args.tau
        self.max_action = max_action

        if args.target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(args.target_entropy)

        # Actor
        self.actor = Actor(state_dim, action_dim, args.hidden_dim, max_action).to(
            self.device
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        # Critic
        self.critic_1 = Critic(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), lr=args.lr_critic
        )
        self.critic_target_1 = Critic(state_dim, action_dim, args.hidden_dim).to(
            self.device
        )
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), lr=args.lr_critic
        )
        self.critic_target_2 = Critic(state_dim, action_dim, args.hidden_dim).to(
            self.device
        )
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        for param in self.critic_target_1.parameters():
            param.requires_grad = False
        for param in self.critic_target_2.parameters():
            param.requires_grad = False

        # Alpha (Entropy temperature)
        self.log_alpha = torch.tensor(
            np.log(args.alpha_initial),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr_alpha)
        self.alpha = self.log_alpha.exp().detach()

        # ICM
        self.icm = ICM(
            state_dim, action_dim, args.icm_feature_dim, args.icm_hidden_dim
        ).to(self.device)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=args.lr_icm)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            state_dim,
            action_dim,
            args.buffer_size,
            self.device,
            args.reward_norm_clipping,
        )

    def select_action(self, state, evaluate=False):
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        if evaluate:
            mu, _ = self.actor.forward(state_tensor)
            action = torch.tanh(mu) * self.max_action
        else:
            action, _ = self.actor.sample(state_tensor)
        return action.cpu().data.numpy().flatten()

    def update(self):
        if len(self.replay_buffer) < self.args.batch_size:
            return None, None, None, None, None, None

        states, actions, extrinsic_rewards, next_states, dones = (
            self.replay_buffer.sample(self.args.batch_size)
        )

        # ICM: Get intrinsic reward and calculate ICM losses
        with torch.no_grad():
            state_feat_icm = self.icm.encode(states)
            next_state_feat_icm = self.icm.encode(next_states)
            intrinsic_rewards = self.icm.get_intrinsic_reward(
                state_feat_icm, actions, next_state_feat_icm
            )

        total_rewards = (
            extrinsic_rewards
            + self.args.intrinsic_reward_weight * intrinsic_rewards.detach()
        )

        # ICM update
        state_feat_for_loss = self.icm.encode(states)
        next_state_feat_for_loss = self.icm.encode(next_states)

        icm_forward_loss, icm_inverse_loss = self.icm.calculate_losses(
            state_feat_for_loss, actions, next_state_feat_for_loss
        )
        icm_loss = (
            self.args.icm_forward_loss_weight * icm_forward_loss
            + self.args.icm_inverse_loss_weight * icm_inverse_loss
        )

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        if self.args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.icm.parameters(), self.args.grad_clip_norm
            )
        self.icm_optimizer.step()

        # Critic update
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            target_q1_next = self.critic_target_1(next_states, next_actions)
            target_q2_next = self.critic_target_2(next_states, next_actions)
            target_q_next = (
                torch.min(target_q1_next, target_q2_next) - self.alpha * next_log_pi
            )
            q_target = total_rewards + self.gamma * (1 - dones) * target_q_next

        current_q1 = self.critic_1(states, actions)
        critic_1_loss = F.mse_loss(current_q1, q_target)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        if self.args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic_1.parameters(), self.args.grad_clip_norm
            )
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(states, actions)
        critic_2_loss = F.mse_loss(current_q2, q_target)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        if self.args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic_2.parameters(), self.args.grad_clip_norm
            )
        self.critic_2_optimizer.step()

        critic_loss_val = (critic_1_loss + critic_2_loss).item() / 2

        # Actor update
        for params in self.critic_1.parameters():
            params.requires_grad = False
        for params in self.critic_2.parameters():
            params.requires_grad = False

        pi_actions, log_pi = self.actor.sample(states)
        q1_pi = self.critic_1(states, pi_actions)
        q2_pi = self.critic_2(states, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.args.grad_clip_norm
            )
        self.actor_optimizer.step()

        for params in self.critic_1.parameters():
            params.requires_grad = True
        for params in self.critic_2.parameters():
            params.requires_grad = True

        # Alpha update
        alpha_loss = -(
            self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # Soft update target networks
        for target_param, param in zip(
            self.critic_target_1.parameters(), self.critic_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, param in zip(
            self.critic_target_2.parameters(), self.critic_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        return (
            actor_loss.item(),
            critic_loss_val,
            alpha_loss.item(),
            self.alpha.item(),
            icm_forward_loss.item(),
            icm_inverse_loss.item(),
        )

    def save_checkpoint(self, filepath, episode, total_steps, episode_rewards_list):
        checkpoint_dir = os.path.dirname(filepath)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            "episode": episode,
            "total_steps": total_steps,
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "icm_state_dict": self.icm.state_dict(),
            "icm_optimizer_state_dict": self.icm_optimizer.state_dict(),
            "replay_buffer_data": self.replay_buffer.get_buffer_data_for_checkpoint(),
            "episode_rewards_list": episode_rewards_list,
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "args": self.args,
        }
        if torch.cuda.is_available():
            checkpoint["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            print(f"Checkpoint file not found: {filepath}")
            return None

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_1_optimizer.load_state_dict(
            checkpoint["critic_1_optimizer_state_dict"]
        )
        self.critic_2_optimizer.load_state_dict(
            checkpoint["critic_2_optimizer_state_dict"]
        )

        self.log_alpha = torch.tensor(
            checkpoint["log_alpha"],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.lr_alpha)
        if "alpha_optimizer_state_dict" in checkpoint:
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
        self.alpha = self.log_alpha.exp().detach()

        self.icm.load_state_dict(checkpoint["icm_state_dict"])
        self.icm_optimizer.load_state_dict(checkpoint["icm_optimizer_state_dict"])

        if "replay_buffer_data" in checkpoint:
            self.replay_buffer.load_buffer_data_from_checkpoint(
                checkpoint["replay_buffer_data"]
            )

        # random.setstate(checkpoint["random_rng_state"])
        # np.random.set_state(checkpoint["np_rng_state"])
        # torch.set_rng_state(checkpoint["torch_rng_state"])
        # if torch.cuda.is_available() and "torch_cuda_rng_state" in checkpoint:
        #     torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])

        print(f"Checkpoint loaded from {filepath}")
        return (
            checkpoint["episode"],
            checkpoint["total_steps"],
            checkpoint["episode_rewards_list"],
            checkpoint.get("args", self.args),
        )

    def save_actor_model(self, filename="actor.pth"):
        """Saves only the actor model weights for evaluation."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, filename)
        torch.save(self.actor.state_dict(), model_path)
        print(f"Actor model saved to {model_path}")


# --- Evaluation Function ---
def evaluate_agent(eval_env, agent_actor, max_action, num_episodes=10, seed_offset=100):
    """Evaluates the agent's actor."""
    avg_reward = 0.0
    agent_actor.eval()

    for i in range(num_episodes):
        state, _ = eval_env.reset(seed=args.seed + seed_offset + i)
        episode_reward = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            state_tensor = torch.as_tensor(
                state.reshape(1, -1), dtype=torch.float32, device=DEVICE
            )
            with torch.no_grad():
                mu, _ = agent_actor.forward(state_tensor)
                action_env = torch.tanh(mu) * max_action
            action_env = action_env.cpu().data.numpy().flatten()

            next_state, reward, terminated, truncated, _ = eval_env.step(action_env)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= num_episodes

    agent_actor.train()
    return avg_reward


# --- Main Training Loop ---
def main(args):
    set_seeds(args.seed)

    env = make_dmc_env(args.env_name, seed=args.seed, flatten=True, use_pixels=False)
    eval_env = make_dmc_env(
        args.env_name, seed=args.seed + 1, flatten=True, use_pixels=False
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.target_entropy is None:
        args.target_entropy = -float(action_dim)
    else:
        args.target_entropy = float(args.target_entropy)

    print(f"Environment: {args.env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Target Entropy: {args.target_entropy}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoints will be saved in: ./{CHECKPOINTS_DIR}/")
    default_interrupt_checkpoint_path = os.path.join(
        CHECKPOINTS_DIR, f"{args.env_name.replace('-', '_')}_sac_icm_interrupt.pth"
    )
    print(f"Default interruption checkpoint path: {default_interrupt_checkpoint_path}")

    agent = SAC(state_dim, action_dim, max_action, args)

    start_episode = 1
    total_steps = 0
    episode_rewards_list = []

    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint
        if not os.path.isabs(resume_path) and not resume_path.startswith(
            CHECKPOINTS_DIR + os.sep
        ):
            resume_path = os.path.join(CHECKPOINTS_DIR, resume_path)

        print(f"Attempting to resume from checkpoint: {resume_path}")
        load_result = agent.load_checkpoint(resume_path)
        if load_result:
            loaded_episode, total_steps, episode_rewards_list, loaded_args = load_result
            start_episode = loaded_episode + 1
            print(f"Resuming from Episode {start_episode}, Total Steps: {total_steps}")
        else:
            print(f"Could not load checkpoint. Starting training from scratch.")

    recent_episode_rewards = deque(maxlen=args.avg_reward_window)

    try:
        for episode in tqdm(
            range(start_episode, args.max_episodes + 1), desc="Training Progress"
        ):
            current_episode_seed = args.seed + episode
            state, info = env.reset(seed=current_episode_seed)
            episode_extrinsic_reward = 0

            for t in range(args.max_timesteps_per_episode):
                total_steps += 1

                if total_steps < args.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, evaluate=False)

                next_state, extrinsic_reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated

                agent.replay_buffer.add(
                    state, action, extrinsic_reward, next_state, float(done)
                )

                if (
                    total_steps > args.warmup_steps
                    and len(agent.replay_buffer) >= args.batch_size
                ):
                    for _ in range(args.num_updates_per_step):
                        update_results = agent.update()

                state = next_state
                episode_extrinsic_reward += extrinsic_reward

                if done:
                    break

            while len(episode_rewards_list) < episode - 1:
                episode_rewards_list.append(0)
            if len(episode_rewards_list) == episode - 1:
                episode_rewards_list.append(episode_extrinsic_reward)
            elif len(episode_rewards_list) > episode - 1:
                episode_rewards_list[episode - 1] = episode_extrinsic_reward
            else:
                episode_rewards_list.append(episode_extrinsic_reward)

            recent_episode_rewards.append(episode_extrinsic_reward)
            avg_reward = (
                np.mean(recent_episode_rewards) if recent_episode_rewards else 0.0
            )

            if episode % args.log_interval == 0:
                log_msg = (
                    f"Ep: {episode}, Steps: {t+1}, Total Steps: {total_steps}, "
                    f"Ext Reward: {episode_extrinsic_reward:.2f}, Avg Ext Reward ({args.avg_reward_window} ep): {avg_reward:.2f}"
                )
                if (
                    total_steps > args.warmup_steps
                    and "update_results" in locals()
                    and update_results
                ):
                    (
                        actor_loss,
                        critic_loss,
                        alpha_loss,
                        current_alpha,
                        fwd_loss,
                        inv_loss,
                    ) = update_results
                    log_msg += f"\n  Losses -> Actor: {actor_loss:.3f}, Critic: {critic_loss:.3f}, Alpha: {alpha_loss:.3f} (Val: {current_alpha:.3f})"
                    log_msg += (
                        f"\n  ICM Losses -> Fwd: {fwd_loss:.3f}, Inv: {inv_loss:.3f}"
                    )
                tqdm.write(log_msg)

            if episode % args.eval_interval == 0 and total_steps > args.warmup_steps:
                eval_avg_reward = evaluate_agent(
                    eval_env,
                    agent.actor,
                    max_action,
                    num_episodes=args.eval_episodes,
                    seed_offset=episode,
                )
                tqdm.write(
                    f"Evaluation after Episode {episode}: Avg Reward over {args.eval_episodes} episodes: {eval_avg_reward:.2f}"
                )
                agent.save_actor_model(
                    filename=f"{args.env_name.replace('-', '_')}_actor_ep{episode}.pth"
                )

            if (
                episode % args.save_checkpoint_interval == 0
                and total_steps > args.warmup_steps
            ):
                chkpt_path = os.path.join(
                    CHECKPOINTS_DIR,
                    f"{args.env_name.replace('-', '_')}_sac_icm_ep{episode}.pth",
                )
                agent.save_checkpoint(
                    chkpt_path, episode, total_steps, episode_rewards_list
                )

        agent.save_actor_model(
            filename=f"{args.env_name.replace('-', '_')}_actor_final.pth"
        )
        final_chkpt_path = os.path.join(
            CHECKPOINTS_DIR, f"{args.env_name.replace('-', '_')}_sac_icm_final.pth"
        )
        agent.save_checkpoint(
            final_chkpt_path, args.max_episodes, total_steps, episode_rewards_list
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        print("Attempting to save current state before exiting...")
    finally:
        current_episode_for_save = (
            episode
            if "episode" in locals() and episode >= start_episode
            else start_episode - 1
        )
        if current_episode_for_save < 0:
            current_episode_for_save = 0

        agent.save_checkpoint(
            default_interrupt_checkpoint_path,
            current_episode_for_save,
            total_steps,
            episode_rewards_list,
        )
        print(f"Current state saved to {default_interrupt_checkpoint_path}. Exiting.")
        env.close()
        eval_env.close()
        print("Training finished or terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAC with ICM agent for DeepMind Control Suite."
    )
    # Environment
    parser.add_argument(
        "--env_name", type=str, default="humanoid-walk", help="DMC environment name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # SAC Hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Soft update coefficient"
    )
    parser.add_argument(
        "--lr_actor", type=float, default=3e-4, help="Actor learning rate"
    )
    parser.add_argument(
        "--lr_critic", type=float, default=3e-4, help="Critic learning rate"
    )
    parser.add_argument(
        "--lr_alpha", type=float, default=3e-4, help="Alpha learning rate"
    )
    parser.add_argument(
        "--alpha_initial", type=float, default=0.2, help="Initial value for alpha"
    )
    parser.add_argument(
        "--target_entropy", type=float, default=None, help="Target entropy"
    )

    # Network Hyperparameters
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden layer dimension"
    )

    # ICM Hyperparameters
    parser.add_argument("--lr_icm", type=float, default=3e-4, help="ICM learning rate")
    parser.add_argument(
        "--icm_feature_dim", type=int, default=128, help="ICM feature dimension"
    )
    parser.add_argument(
        "--icm_hidden_dim", type=int, default=256, help="ICM hidden dimension"
    )
    parser.add_argument(
        "--intrinsic_reward_weight",
        type=float,
        default=0.01,
        help="Intrinsic reward weight",
    )
    parser.add_argument(
        "--icm_forward_loss_weight",
        type=float,
        default=0.2,
        help="ICM forward loss weight",
    )
    parser.add_argument(
        "--icm_inverse_loss_weight",
        type=float,
        default=0.8,
        help="ICM inverse loss weight",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--buffer_size", type=int, default=int(1e6), help="Replay buffer size"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--max_episodes", type=int, default=5000, help="Maximum training episodes"
    )
    parser.add_argument(
        "--max_timesteps_per_episode",
        type=int,
        default=1000,
        help="Maximum timesteps per episode",
    )
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument(
        "--num_updates_per_step",
        type=int,
        default=1,
        help="Agent updates per environment step",
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm"
    )
    parser.add_argument(
        "--reward_norm_clipping",
        type=float,
        default=5.0,
        help="Normalized reward clipping",
    )

    # Logging and Saving
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument(
        "--eval_interval", type=int, default=100, help="Evaluation interval"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=100,
        help="Checkpoint saving interval",
    )
    parser.add_argument(
        "--avg_reward_window", type=int, default=100, help="Average reward window size"
    )

    # Resuming Training
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    args = parser.parse_args()

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    main(args)
