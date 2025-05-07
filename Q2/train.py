import os
import sys

import gymnasium as gym  # Assuming you might use Gymnasium features, though DMC is the primary focus
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Use Normal distribution for continuous actions
from torch.distributions import Normal

# Assuming dmc.py is in the parent directory relative to the script's location
# If not, adjust the path accordingly
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

except NameError:  # __file__ is not defined in some environments (e.g. Jupyter)
    print(
        "Warning: __file__ is not defined. Assuming dmc.py is in a discoverable path or parent directory."
    )
    # Try a relative import path that might work in some notebook setups
    try:
        from ..dmc import make_dmc_env  # If the script is run as part of a package
    except (ImportError, ValueError):
        # Define a dummy function if import fails, to avoid crashing later
        def make_dmc_env(*args, **kwargs):
            raise NotImplementedError(
                "make_dmc_env is not available. Please ensure dmc.py is accessible."
            )


# --- Hyperparameters ---
# Environment
ENV_NAME = "cartpole-balance"  # Using DMC environment
# Training
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
GAMMA = 0.99  # Discount factor for future rewards
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
PPO_EPSILON = 0.2  # Epsilon for clipping in PPO objective
PPO_EPOCHS = 10  # Number of epochs to update policy per PPO iteration
PPO_BATCH_SIZE = 64  # Batch size for PPO updates
ENTROPY_COEF = 0.01  # Coefficient for entropy bonus (encourages exploration)
MAX_TIMESTEPS_PER_EPISODE = 1000  # Default for many DMC tasks
UPDATE_TIMESTEPS = 2048  # Number of environment steps to collect before updating policy
TOTAL_TRAINING_TIMESTEPS = 1000000
TEST_EPISODES = 5  # Number of episodes to test the trained agent
LOG_STD_INIT = 0.0  # Initial log standard deviation for action distribution

# Early Stopping
EARLY_STOP_THRESHOLD = 970.0  # Average reward threshold for early stopping
EARLY_STOP_WINDOW = 100  # Number of episodes for the moving average reward

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Rollout Buffer ---
class RolloutBuffer:
    """
    Stores trajectories collected by the agent interacting with the environment.
    Adapted for continuous actions.
    """

    def __init__(
        self, buffer_size, observation_shape, action_shape, gae_lambda, gamma, device
    ):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape  # Should be a tuple like (action_dim,)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device

        # Initialize buffers
        self.observations = np.zeros(
            (self.buffer_size,) + self.observation_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.buffer_size,) + self.action_shape, dtype=np.float32
        )
        self.log_probs = np.zeros(
            self.buffer_size, dtype=np.float32
        )  # Log prob is scalar per step
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.bool_)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = (
            0  # This seems to be an unused variable based on the provided code.
        )

    def add(self, obs, action, log_prob, reward, value, done):
        """Add one transition to the buffer."""
        if self.ptr >= self.buffer_size:
            # Simple rollover if buffer overflows (can also raise error)
            print("Warning: Buffer overflow. Overwriting oldest data.")
            self.ptr = 0  # Reset pointer to overwrite
        #   raise ValueError("Buffer overflow.")

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action  # Action is already a numpy array
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_advantages_and_returns(self, last_value, last_done):
        """Computes advantages and returns using GAE."""
        # Ensure ptr is within bounds for slicing, especially if buffer wasn't full
        current_buffer_size = self.ptr

        effective_last_value = last_value * (1.0 - last_done)
        # Append the bootstrapped value for the last state to the values array for GAE calculation
        values_for_gae = np.append(
            self.values[:current_buffer_size], effective_last_value
        )
        gae = 0
        # Iterate backwards through the collected transitions
        for t in reversed(range(current_buffer_size)):
            next_value = values_for_gae[t + 1]
            next_non_terminal = 1.0 - self.dones[t]  # 1 if not done, 0 if done
            # Calculate the TD error (delta)
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            # Calculate GAE: delta + gamma * lambda * not_done * previous_gae
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Calculate returns by adding advantages to values
        self.returns[:current_buffer_size] = (
            self.advantages[:current_buffer_size] + self.values[:current_buffer_size]
        )

        # Normalize advantages only over the filled part of the buffer
        if current_buffer_size > 1:  # Avoid division by zero or NaN for single sample
            adv_mean = np.mean(self.advantages[:current_buffer_size])
            adv_std = (
                np.std(self.advantages[:current_buffer_size]) + 1e-8
            )  # Epsilon for stability
            self.advantages[:current_buffer_size] = (
                self.advantages[:current_buffer_size] - adv_mean
            ) / adv_std
        elif current_buffer_size == 1:
            self.advantages[0] = (
                0.0  # Or handle as per specific requirements, e.g. no normalization
            )

    def get_batch(self, batch_size):
        """Yields minibatches of experiences from the filled part of the buffer."""
        num_samples = self.ptr  # Only use samples up to the current pointer
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # Shuffle indices for random batch sampling

        # Iterate through the shuffled indices to create batches
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield (
                torch.tensor(self.observations[batch_indices], dtype=torch.float32).to(
                    self.device
                ),
                torch.tensor(self.actions[batch_indices], dtype=torch.float32).to(
                    self.device
                ),
                torch.tensor(self.log_probs[batch_indices], dtype=torch.float32).to(
                    self.device
                ),
                torch.tensor(self.advantages[batch_indices], dtype=torch.float32).to(
                    self.device
                ),
                torch.tensor(self.returns[batch_indices], dtype=torch.float32).to(
                    self.device
                ),
                torch.tensor(self.values[batch_indices], dtype=torch.float32).to(
                    self.device
                ),
            )

    def clear(self):
        """Resets the buffer pointer and path start index."""
        self.ptr = 0
        self.path_start_idx = 0


# --- Actor-Critic Network for Continuous Actions ---
class ActorCriticContinuous(nn.Module):
    """
    PPO Actor-Critic network for continuous action spaces.
    Outputs mean of Gaussian distribution from actor.
    Uses a learnable log standard deviation parameter.
    """

    def __init__(self, state_dim, action_dim, log_std_init):
        super(ActorCriticContinuous, self).__init__()
        self.action_dim = action_dim

        # Actor network (outputs mean of the action distribution)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # Critic network (outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Learnable log standard deviation for the action distribution
        # Initialized with log_std_init and has one parameter per action dimension
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim, device=DEVICE) * log_std_init
        )

    def forward(self, state):
        # This method is not typically called directly in PPO.
        # Use act() for action selection during rollout and evaluate() during training.
        raise NotImplementedError(
            "Use act() or evaluate() methods for ActorCriticContinuous."
        )

    def act(self, state):
        """
        Given a state, returns a sampled action, its log probability, and the state value.
        Used during trajectory collection (rollout phase).
        """
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.actor_log_std)  # Standard deviation from log_std

        # Create a Normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()  # Sample an action from the distribution
        action_log_prob = dist.log_prob(action).sum(
            dim=-1
        )  # Sum log probabilities for multi-dimensional actions
        state_value = self.critic(state)  # Get state value from critic

        return action, action_log_prob, state_value

    def evaluate(self, state, action):
        """
        Given a state and an action, returns the action's log probability under
        the current policy, the state value (from critic), and the distribution's entropy.
        Used during PPO update phase.
        """
        action_mean = self.actor_mean(state)
        # Ensure std matches shape of mean, especially for batch processing
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)  # Create a Normal distribution

        # Calculate log probability of the given actions
        action_log_probs = dist.log_prob(action).sum(
            dim=-1
        )  # Sum for multi-dimensional actions
        # Calculate entropy of the distribution
        dist_entropy = dist.entropy().sum(dim=-1)  # Sum for multi-dimensional actions
        # Get state values from critic and remove the last dimension (squeeze)
        state_values = self.critic(state).squeeze(-1)

        return action_log_probs, state_values, dist_entropy


# --- PPO Agent for Continuous Actions ---
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        lr_actor,
        lr_critic,
        gamma,
        gae_lambda,
        ppo_epsilon,
        ppo_epochs,
        ppo_batch_size,
        entropy_coef,
        log_std_init,
        device,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.entropy_coef = entropy_coef
        self.device = device
        # Store action bounds as tensors on the correct device
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # Initialize policy network (ActorCritic)
        self.policy = ActorCriticContinuous(state_dim, action_dim, log_std_init).to(
            self.device
        )

        # Define optimizer for actor parameters (mean network parameters + log_std parameter)
        actor_params = list(self.policy.actor_mean.parameters()) + [
            self.policy.actor_log_std
        ]
        self.optimizer_actor = optim.Adam(actor_params, lr=lr_actor)

        # Define optimizer for critic parameters
        self.optimizer_critic = optim.Adam(
            self.policy.critic.parameters(), lr=lr_critic
        )

        # Loss function for critic (Mean Squared Error)
        self.mse_loss = nn.MSELoss()

    def get_action_and_value(self, state):
        """
        Selects an action from the policy, gets its log probability, and the state value from the critic.
        Clips the action to the environment's bounds.
        Used during rollout.
        """
        with torch.no_grad():  # No gradient calculation needed for action selection
            # Convert state to tensor and add batch dimension if it's a single state
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if (
                state_tensor.ndim == 1
            ):  # If input is a single state (e.g. shape [state_dim])
                state_tensor = state_tensor.unsqueeze(
                    0
                )  # Add batch dimension: shape [1, state_dim]

            # Get action, log_prob, and value from the policy network
            action_tensor, log_prob_tensor, value_tensor = self.policy.act(state_tensor)

            # Clip the sampled action to the environment's action space bounds
            action_clipped = torch.clamp(
                action_tensor, self.action_low, self.action_high
            )

            # Prepare outputs: convert to NumPy for environment interaction, keep as scalars/items for single state
            if state_tensor.shape[0] == 1:  # If input was a single state
                action_to_env = (
                    action_clipped.squeeze(0).cpu().numpy()
                )  # Remove batch dim, move to CPU, convert to NumPy
                log_prob_item = log_prob_tensor.item()  # Scalar log probability
                value_item = value_tensor.item()  # Scalar state value
                return action_to_env, log_prob_item, value_item
            else:  # If input was a batch of states (not typical for this function during rollout)
                action_to_env = action_clipped.cpu().numpy()
                log_prob_item = log_prob_tensor.cpu().numpy()
                value_item = (
                    value_tensor.squeeze(-1).cpu().numpy()
                )  # Squeeze value if it has an extra dim
                return action_to_env, log_prob_item, value_item

    def update(self, rollout_buffer):
        """Updates the policy and value networks using PPO."""
        # Iterate for PPO_EPOCHS (multiple updates on the same batch of data)
        for _ in range(self.ppo_epochs):
            # Get batches from the rollout buffer
            for (
                batch_obs,
                batch_actions,
                batch_old_log_probs,
                batch_advantages,
                batch_returns,
                batch_old_values,  # This is collected but not directly used in the simpler critic loss here
            ) in rollout_buffer.get_batch(self.ppo_batch_size):

                # Evaluate actions taken previously (batch_actions) using the current policy
                # This gives new log_probs, new state values, and entropy of the current policy's action distribution
                new_log_probs, new_values, entropy = self.policy.evaluate(
                    batch_obs, batch_actions
                )

                # Calculate the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate surrogate loss (clipped PPO objective)
                # Term 1: ratio * advantage
                surr1 = ratios * batch_advantages
                # Term 2: clipped_ratio * advantage
                surr2 = (
                    torch.clamp(ratios, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
                    * batch_advantages
                )
                # Actor loss is the negative minimum of the two surrogate objectives (we want to maximize it)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate critic loss (MSE between new values predicted by critic and actual returns)
                # 0.5 coefficient is common but can be omitted or adjusted
                critic_loss = 0.5 * self.mse_loss(new_values, batch_returns)

                # Calculate entropy loss (to encourage exploration by maximizing entropy)
                # Negative sign because we want to maximize entropy, so we minimize -entropy_coef * entropy
                entropy_loss = -self.entropy_coef * entropy.mean()

                # Total loss for the actor (PPO objective + entropy bonus)
                total_actor_loss = actor_loss + entropy_loss

                # --- Update Actor ---
                self.optimizer_actor.zero_grad()  # Clear old gradients
                total_actor_loss.backward()  # Compute gradients
                # Clip gradients for actor parameters to prevent large updates
                nn.utils.clip_grad_norm_(
                    self.policy.actor_mean.parameters(), max_norm=0.5
                )
                nn.utils.clip_grad_norm_(
                    [
                        self.policy.actor_log_std
                    ],  # log_std is a Parameter, treat as a list
                    max_norm=0.5,
                )
                self.optimizer_actor.step()  # Apply updates

                # --- Update Critic ---
                self.optimizer_critic.zero_grad()  # Clear old gradients
                critic_loss.backward()  # Compute gradients
                # Clip gradients for critic parameters
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()  # Apply updates

    def save_model(self, filepath="ppo_actor_critic.pth"):
        """Saves the policy network's state_dict and optimizers' state_dicts."""
        try:
            # Create a checkpoint dictionary
            checkpoint = {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
                "optimizer_critic_state_dict": self.optimizer_critic.state_dict(),
                # Optionally, save other training metadata like hyperparameters or current progress
                "env_name": ENV_NAME,
                # 'total_timesteps': current_total_timesteps, # would need to pass this in
            }
            # Save the checkpoint to the specified file
            torch.save(checkpoint, filepath)
            print(f"Model and optimizers saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath="ppo_actor_critic.pth", load_optimizers=True):
        """Loads the policy network's state_dict and optionally optimizers' state_dicts."""
        try:
            # Load the checkpoint from the file, mapping to the current device
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load the policy state dict
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.policy.to(self.device)  # Ensure the model is on the correct device

            # Optionally load optimizer states
            if load_optimizers:
                if (
                    "optimizer_actor_state_dict" in checkpoint
                    and "optimizer_critic_state_dict" in checkpoint
                ):
                    self.optimizer_actor.load_state_dict(
                        checkpoint["optimizer_actor_state_dict"]
                    )
                    self.optimizer_critic.load_state_dict(
                        checkpoint["optimizer_critic_state_dict"]
                    )
                    print("Optimizers loaded from checkpoint.")
                else:
                    print(
                        "Optimizer states not found in checkpoint. Initializing new optimizers if training continues."
                    )

            # Optionally, load and print other metadata
            env_name_loaded = checkpoint.get("env_name", "Unknown")
            # total_timesteps_loaded = checkpoint.get('total_timesteps', 0) # If you save this
            print(f"Loaded model previously trained on {env_name_loaded}.")

            print(f"Model loaded from {filepath}")
            # Set the policy to evaluation mode by default after loading.
            # If continuing training, this should be set back to train() mode.
            self.policy.eval()
            # return total_timesteps_loaded # If you need to resume training progress
        except FileNotFoundError:
            print(
                f"Error: Model file not found at {filepath}. Starting with a new model."
            )
        except Exception as e:
            print(f"Error loading model: {e}. Starting with a new model.")
            # return 0 # If you need to resume training progress


# --- Main Training Loop ---
if __name__ == "__main__":
    # --- Environment Initialization ---
    # Set a seed for reproducibility
    seed = np.random.randint(0, 1000000)  # Or set a fixed seed: e.g., seed = 42
    print(f"Using seed: {seed}")

    # Initialize training and testing environments
    # flatten=True and use_pixels=False gives state vector observations
    try:
        env = make_dmc_env(ENV_NAME, seed, flatten=True, use_pixels=False)
        test_env = make_dmc_env(
            ENV_NAME,
            seed + 1,  # Use a different seed for the test environment
            flatten=True,
            use_pixels=False,
        )
    except NotImplementedError as e:
        print(f"Could not initialize DMC environment: {e}")
        print("Exiting script. Please ensure dmc.py is correctly set up.")
        sys.exit(1)  # Exit if environment cannot be created
    except Exception as e:
        print(f"An unexpected error occurred during environment creation: {e}")
        sys.exit(1)

    # Get state and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_shape = env.action_space.shape  # Tuple, e.g., (action_dim,)
    action_low = env.action_space.low  # Min action value(s)
    action_high = env.action_space.high  # Max action value(s)
    print(
        f"State Dim: {state_dim}, Action Dim: {action_dim}, "
        f"Action Shape: {action_shape}, Action Low: {action_low}, Action High: {action_high}"
    )

    # --- Agent and Buffer Initialization ---
    agent = PPOAgent(
        state_dim,
        action_dim,
        action_low,
        action_high,
        LEARNING_RATE_ACTOR,
        LEARNING_RATE_CRITIC,
        GAMMA,
        GAE_LAMBDA,
        PPO_EPSILON,
        PPO_EPOCHS,
        PPO_BATCH_SIZE,
        ENTROPY_COEF,
        LOG_STD_INIT,
        DEVICE,
    )

    buffer = RolloutBuffer(
        UPDATE_TIMESTEPS,
        env.observation_space.shape,  # Pass the full observation shape tuple
        action_shape,  # Pass the action shape tuple
        GAE_LAMBDA,
        GAMMA,
        DEVICE,
    )

    # --- Model Loading (Optional) ---
    MODEL_FILENAME = (
        f"ppo_agent_{ENV_NAME.replace('-', '_')}_actor_critic.pth"  # Sanitize filename
    )
    LOAD_PREVIOUS_MODEL = False  # Set to True to attempt loading a saved model

    # Initialize training progress variables
    current_total_timesteps = 0
    episode_num = 0

    if LOAD_PREVIOUS_MODEL:
        print(f"Attempting to load model from: {MODEL_FILENAME}")
        agent.load_model(
            MODEL_FILENAME, load_optimizers=True
        )  # Load optimizers if continuing training
        # If you were saving/loading current_total_timesteps, you'd update it here:
        # current_total_timesteps = agent.load_model(...)
        # The load_model sets policy to eval(), so set it to train() if continuing training
        # This will be done at the start of the training loop anyway.

    # --- Logging ---
    episode_rewards = []  # List to store rewards for each episode

    print(f"Starting training for {TOTAL_TRAINING_TIMESTEPS} timesteps...")

    # Reset environment for the first episode
    state, _ = env.reset()
    current_episode_reward = 0
    episode_len = 0

    training_achieved_goal = False  # Flag for early stopping

    # --- Main Training Loop ---
    while current_total_timesteps < TOTAL_TRAINING_TIMESTEPS:
        agent.policy.train()  # Set model to training mode at the start of each update cycle

        # --- Rollout Phase: Collect trajectories ---
        for t_rollout in range(UPDATE_TIMESTEPS):
            # Get action, log probability, and value from the agent
            action, log_prob, value = agent.get_action_and_value(state)

            # Ensure action is a NumPy array for the environment step
            action_np = np.array(action, dtype=np.float32)

            # Take a step in the environment
            next_state, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated  # Combine termination flags

            # Add transition to the rollout buffer
            buffer.add(state, action_np, log_prob, reward, value, done)

            # Update current state and episode statistics
            state = next_state
            current_episode_reward += reward
            episode_len += 1
            current_total_timesteps += 1

            # Check if the episode has ended (due to termination or truncation)
            # DMC often uses internal time limits, check info or use MAX_TIMESTEPS_PER_EPISODE
            is_truncated_by_env_limit = (
                "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
            )
            # actual_done_for_reset determines if the environment should be reset
            actual_done_for_reset = terminated or is_truncated_by_env_limit

            if actual_done_for_reset or episode_len >= MAX_TIMESTEPS_PER_EPISODE:
                episode_rewards.append(current_episode_reward)
                # Calculate average reward over the last 'EARLY_STOP_WINDOW' episodes (or fewer if not enough data yet)
                avg_reward_last_N = (
                    np.mean(episode_rewards[-EARLY_STOP_WINDOW:])
                    if episode_rewards
                    else 0.0
                )
                print(
                    f"Ep: {episode_num + 1}, Total Steps: {current_total_timesteps}, "
                    f"Reward: {current_episode_reward:.2f}, Ep Len: {episode_len}, "
                    f"Avg Rwd ({EARLY_STOP_WINDOW}): {avg_reward_last_N:.2f}"
                )

                # --- EARLY STOPPING CHECK ---
                if (
                    len(episode_rewards) >= EARLY_STOP_WINDOW
                    and avg_reward_last_N > EARLY_STOP_THRESHOLD
                ):
                    print(
                        f"\nEarly stopping: Average reward over last {EARLY_STOP_WINDOW} episodes ({avg_reward_last_N:.2f}) "
                        f"exceeded threshold ({EARLY_STOP_THRESHOLD})."
                    )
                    training_achieved_goal = True  # Set flag to stop training
                    break  # Break from the inner rollout loop

                # Reset environment for the next episode
                state, _ = env.reset()
                current_episode_reward = 0
                episode_len = 0
                episode_num += 1

            # Exit rollout phase if total training timesteps reached
            if current_total_timesteps >= TOTAL_TRAINING_TIMESTEPS:
                break

        # If early stopping condition met or total timesteps reached, break main training loop
        if (
            training_achieved_goal
            or current_total_timesteps >= TOTAL_TRAINING_TIMESTEPS
        ):
            break

        # --- Update Phase: Train the agent ---
        # Calculate the value of the last state reached in the rollout for GAE
        # 'done' flag from the *last actual step collected* in the buffer
        last_step_in_buffer_idx = buffer.ptr - 1 if buffer.ptr > 0 else 0
        last_step_done_in_buffer = (
            buffer.dones[last_step_in_buffer_idx] if buffer.ptr > 0 else True
        )

        if last_step_done_in_buffer:  # If the last collected step was terminal
            last_val_for_gae = 0.0  # Value of terminal state is 0
        else:
            # 'state' here is the state *after* the last collected step (s_T+1)
            # We need its value for bootstrapping GAE if the episode didn't end
            _, _, last_val_for_gae = agent.get_action_and_value(state)

        # Compute advantages and returns for the collected trajectories
        buffer.compute_advantages_and_returns(
            last_val_for_gae, last_step_done_in_buffer
        )

        # Update the agent's policy and critic networks
        agent.update(buffer)

        # Clear the rollout buffer for the next set of trajectories
        buffer.clear()

    if training_achieved_goal:
        print("Training stopped early due to achieving reward threshold.")
    else:
        print("Training finished after reaching total timesteps.")

    # --- Save the Final Model ---
    agent.save_model(MODEL_FILENAME)

    # --- Logging Final Rewards ---
    if episode_rewards:
        final_avg_reward = (
            np.mean(episode_rewards[-EARLY_STOP_WINDOW:])
            if len(episode_rewards) >= EARLY_STOP_WINDOW
            else np.mean(episode_rewards)
        )
        print(
            f"Training completed. Final average reward (last {min(len(episode_rewards), EARLY_STOP_WINDOW)} episodes): {final_avg_reward:.2f}"
        )
        try:
            log_filename = f"ppo_rewards_{ENV_NAME.replace('-', '_')}.txt"
            with open(log_filename, "w") as f:
                for r_idx, r_val in enumerate(episode_rewards):
                    f.write(f"Episode {r_idx+1}: {r_val}\n")
            print(f"Episode rewards logged to {log_filename}")
        except Exception as e:
            print(f"Could not write rewards log: {e}")
    else:
        print("No episodes completed during training to log rewards.")

    # --- Test the Trained Agent (using the agent currently in memory) ---
    print("\nTesting trained agent (from memory)...")
    agent.policy.eval()  # Set policy to evaluation mode for testing
    total_test_rewards = []

    for i in range(TEST_EPISODES):
        state, _ = test_env.reset()
        episode_reward = 0
        # terminated = False # Not needed here as 'actual_test_done' covers it
        # truncated = False # Not needed here
        test_step = 0
        while True:
            # For testing, one might use the mean of the action distribution for deterministic behavior.
            # The current get_action_and_value samples. For deterministic:
            # with torch.no_grad():
            #     state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            #     action_mean = agent.policy.actor_mean(state_tensor)
            #     action = torch.clamp(action_mean, agent.action_low, agent.action_high).squeeze(0).cpu().numpy()
            action, _, _ = agent.get_action_and_value(state)  # Uses sampling
            action_np = np.array(action, dtype=np.float32)

            state, reward, terminated, truncated, info = test_env.step(action_np)
            episode_reward += reward
            test_step += 1

            is_test_truncated_by_limit = (
                "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
            )
            actual_test_done = terminated or is_test_truncated_by_limit

            if actual_test_done or test_step >= MAX_TIMESTEPS_PER_EPISODE:
                break
        total_test_rewards.append(episode_reward)
        print(
            f"Test Episode {i + 1}: Reward = {episode_reward:.2f}, Steps = {test_step}"
        )

    if total_test_rewards:
        print(
            f"Average Test Reward (from memory) over {TEST_EPISODES} episodes: "
            f"{np.mean(total_test_rewards):.2f} +/- {np.std(total_test_rewards):.2f}"
        )
    else:
        print("No test episodes were run (from memory).")

    # --- Example: Load and Test the Saved Model Separately ---
    print("\n--- Example: Loading and Testing the Saved Model from File ---")
    # Create a new agent instance (or re-use, but for a clean test, new is better)
    # Ensure parameters match the saved model's architecture
    loaded_agent = PPOAgent(
        state_dim,
        action_dim,
        action_low,
        action_high,
        LEARNING_RATE_ACTOR,  # These LR are for optimizer re-init, not strictly needed if not training further
        LEARNING_RATE_CRITIC,
        GAMMA,
        GAE_LAMBDA,
        PPO_EPSILON,
        PPO_EPOCHS,
        PPO_BATCH_SIZE,
        ENTROPY_COEF,
        LOG_STD_INIT,
        DEVICE,
    )
    # Load the model (optimizers not strictly needed for testing only)
    loaded_agent.load_model(MODEL_FILENAME, load_optimizers=False)
    # The load_model method already sets the policy to eval() mode.

    print("\nTesting agent loaded from file...")
    total_test_rewards_loaded = []
    # It's good practice to re-initialize or use a dedicated test environment
    # to avoid state leakage if the same test_env object was used before.
    # Using a different seed for this test environment
    test_env_for_loaded = make_dmc_env(
        ENV_NAME, seed + 2, flatten=True, use_pixels=False
    )

    for i in range(TEST_EPISODES):
        state, _ = test_env_for_loaded.reset()
        episode_reward = 0
        # terminated = False # Not needed
        # truncated = False # Not needed
        test_step = 0
        while True:
            action, _, _ = loaded_agent.get_action_and_value(state)  # Use loaded_agent
            action_np = np.array(action, dtype=np.float32)
            state, reward, terminated, truncated, info = test_env_for_loaded.step(
                action_np
            )
            episode_reward += reward
            test_step += 1
            is_loaded_test_truncated = (
                "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
            )
            actual_loaded_test_done = terminated or is_loaded_test_truncated
            if actual_loaded_test_done or test_step >= MAX_TIMESTEPS_PER_EPISODE:
                break
        total_test_rewards_loaded.append(episode_reward)
        print(
            f"Test Episode {i + 1} (Loaded Model): Reward = {episode_reward:.2f}, Steps = {test_step}"
        )

    if total_test_rewards_loaded:
        print(
            f"Average Test Reward (Loaded Model) over {TEST_EPISODES} episodes: "
            f"{np.mean(total_test_rewards_loaded):.2f} +/- {np.std(total_test_rewards_loaded):.2f}"
        )
    else:
        print("No test episodes were run with the loaded model.")

    # --- Cleanup ---
    env.close()
    test_env.close()
    if "test_env_for_loaded" in locals():  # Check if it was created
        test_env_for_loaded.close()
    print("\nScript finished.")
