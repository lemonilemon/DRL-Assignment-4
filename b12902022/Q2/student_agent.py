import os
import sys

import gymnasium  # For the action_space definition
import numpy as np
import torch

# --- Attempt to import necessary components from train.py ---
# This assumes train.py (the script with PPOAgent, ActorCriticContinuous, etc.)
# is in the same directory as this eval_agent.py script, or is accessible via PYTHONPATH.
try:
    # Import other hyperparameters required by PPOAgent.__init__
    # These are mostly for training, but the constructor needs them.
    from train import DEVICE  # Device (cpu/cuda)
    from train import ENTROPY_COEF  # Core agent and network classes
    from train import ENV_NAME  # Environment name to ensure consistency
    from train import LOG_STD_INIT  # Hyperparameter for ActorCriticContinuous
    from train import (
        GAE_LAMBDA,
        GAMMA,
        LEARNING_RATE_ACTOR,
        LEARNING_RATE_CRITIC,
        PPO_BATCH_SIZE,
        PPO_EPOCHS,
        PPO_EPSILON,
        ActorCriticContinuous,
        PPOAgent,
    )
except ImportError as e:
    print(f"Error importing from train.py: {e}")
    print(
        "Please ensure train.py (containing PPOAgent, ActorCriticContinuous, and hyperparameters) "
        "is in the same directory as this script or accessible in PYTHONPATH."
    )
    sys.exit(1)

# --- Define Environment-Specific Parameters ---
# These parameters *must* match the environment the agent was trained on.
# They are derived from ENV_NAME imported from train.py.
if ENV_NAME == "cartpole-balance":
    # CORRECTED: The observation space from eval.py and the saved model indicate 5 dimensions.
    EVAL_STATE_DIM = 5
    EVAL_ACTION_DIM = 1  # Cartpole has a single continuous action
    EVAL_ACTION_LOW = np.array([-1.0], dtype=np.float32)  # Action bounds
    EVAL_ACTION_HIGH = np.array([1.0], dtype=np.float32)
    # DMC environments usually use float32 for actions and observations
    EVAL_DTYPE = np.float32
elif ENV_NAME == "another-env-example":  # Example for extension
    # EVAL_STATE_DIM = ... # Define for other envs
    # EVAL_ACTION_DIM = ...
    # EVAL_ACTION_LOW = np.array([...], dtype=np.float32)
    # EVAL_ACTION_HIGH = np.array([...], dtype=np.float32)
    # EVAL_DTYPE = np.float32
    print(
        f"Warning: Using placeholder parameters for '{ENV_NAME}'. Please configure them in eval_agent.py."
    )
    # Defaulting to cartpole-balance values to prevent immediate crash, but this should be fixed.
    EVAL_STATE_DIM = 5
    EVAL_ACTION_DIM = 1
    EVAL_ACTION_LOW = np.array([-1.0], dtype=np.float32)
    EVAL_ACTION_HIGH = np.array([1.0], dtype=np.float32)
    EVAL_DTYPE = np.float32
else:
    raise ValueError(
        f"Environment parameters for '{ENV_NAME}' are not defined in eval_agent.py. "
        "Please add them or ensure ENV_NAME in train.py is one of the configured environments."
    )


class Agent(object):
    """
    Evaluation Agent that loads a pre-trained PPO model from train.py
    and uses it to act in the environment.
    """

    def __init__(self):
        """
        Initializes the agent, loads the PPO model, and sets up the action space.
        The __init__ signature should not be modified as per the user's request.
        """
        # Define the action space based on the environment parameters.
        # This matches the format requested by the user.
        self.action_space = gymnasium.spaces.Box(
            low=EVAL_ACTION_LOW,
            high=EVAL_ACTION_HIGH,
            shape=(EVAL_ACTION_DIM,),
            dtype=EVAL_DTYPE,  # Use float32 for consistency with DMC/PyTorch
        )

        # Instantiate the PPOAgent from train.py.
        # The state_dim now correctly uses the resolved EVAL_STATE_DIM.
        self.ppo_agent = PPOAgent(
            state_dim=EVAL_STATE_DIM,  # Corrected
            action_dim=EVAL_ACTION_DIM,
            action_low=EVAL_ACTION_LOW,
            action_high=EVAL_ACTION_HIGH,
            lr_actor=LEARNING_RATE_ACTOR,
            lr_critic=LEARNING_RATE_CRITIC,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ppo_epsilon=PPO_EPSILON,
            ppo_epochs=PPO_EPOCHS,
            ppo_batch_size=PPO_BATCH_SIZE,
            entropy_coef=ENTROPY_COEF,
            log_std_init=LOG_STD_INIT,
            device=DEVICE,
        )

        # Define the model filename. This must match how it's saved in train.py.
        model_filename = f"ppo_agent_{ENV_NAME.replace('-', '_')}_actor_critic.pth"

        # Check if the model file exists before attempting to load
        if not os.path.exists(model_filename):
            print(f"CRITICAL ERROR: Model file not found at '{model_filename}'.")
            print(
                "Please ensure the agent has been trained using train.py and the model file is present "
                "in the same directory as this script, or provide the correct path."
            )
            raise FileNotFoundError(
                f"Model file '{model_filename}' not found. Cannot initialize evaluation agent."
            )

        # Load the trained model weights.
        print(f"Loading model from: {model_filename}")
        self.ppo_agent.load_model(model_filename, load_optimizers=False)
        # The load_model method in train.py should handle setting the policy to eval mode.

        print(
            f"Evaluation agent initialized successfully. Loaded model for '{ENV_NAME}'."
        )

    def act(self, observation):
        """
        Takes an observation from the environment and returns an action
        determined by the loaded PPO policy.
        The 'observation' input signature should not be modified.
        """
        # Ensure the observation is a NumPy array with the correct dtype.
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=EVAL_DTYPE)
        elif observation.dtype != EVAL_DTYPE:
            observation = observation.astype(EVAL_DTYPE)

        # Validate observation shape if possible (optional, but good for debugging)
        if observation.shape[0] != EVAL_STATE_DIM:
            print(
                f"Warning: Observation shape mismatch. Expected {EVAL_STATE_DIM}, got {observation.shape[0]}."
            )
            # Depending on strictness, you might raise an error or try to reshape/pad.
            # For now, we'll let PPOAgent handle it, but this is a potential issue.

        action, _log_prob, _value = self.ppo_agent.get_action_and_value(observation)

        return action


# --- Example Usage (for testing the Agent class directly) ---
if __name__ == "__main__":
    print("--- Testing Evaluation Agent Standalone ---")

    print(f"Target environment for this agent: {ENV_NAME}")
    print(f"Expected state dimension: {EVAL_STATE_DIM}")
    print(f"Expected action dimension: {EVAL_ACTION_DIM}")

    dummy_observation = np.random.rand(EVAL_STATE_DIM).astype(EVAL_DTYPE)
    print(
        f"Dummy observation created with shape: {dummy_observation.shape}, dtype: {dummy_observation.dtype}"
    )

    try:
        eval_agent = Agent()
        action_output = eval_agent.act(dummy_observation)

        print(f"\nAgent produced action: {action_output}")
        print(f"Action type: {type(action_output)}")
        print(
            f"Action shape: {action_output.shape if hasattr(action_output, 'shape') else 'N/A'}"
        )
        print(
            f"Action dtype: {action_output.dtype if hasattr(action_output, 'dtype') else 'N/A'}"
        )

        if eval_agent.action_space.contains(action_output):
            print("Action is within the agent's defined action space.")
        else:
            print("WARNING: Action is OUTSIDE the agent's defined action space.")
            print(
                f"  Action space details: Low={eval_agent.action_space.low}, High={eval_agent.action_space.high}"
            )
            print(f"  Action produced: {action_output}")

    except FileNotFoundError as e:
        print(f"\nERROR during example usage: Could not find the model file.")
        print(f"  Details: {e}")
        print(
            "  Please ensure 'train.py' has been run and has saved a model file named "
            f"'ppo_agent_{ENV_NAME.replace('-', '_')}_actor_critic.pth' in the same directory."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during example usage: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Standalone Test Finished ---")
