import os

import gymnasium as gym
import numpy as np
import torch

# Attempt to import Actor and DEVICE from the training script (assumed to be sac_train.py)
# and make_dmc_env from dmc.py
try:
    from dmc import make_dmc_env

    from train import DEVICE, Actor
except ImportError as e:
    print(f"Error importing necessary components: {e}")
    print(
        "Please ensure 'sac_train.py' (with Actor, DEVICE) and 'dmc.py' (with make_dmc_env) are accessible."
    )
    raise

# --- Configuration for the Inference Agent ---
# These are hardcoded due to the __init__ signature constraint.
# For a more flexible agent, these would ideally be parameters.

# Default environment name: This should match the environment the model was trained on.
# The training script's default env_name was "humanoid-stand".
# Let's use "humanoid-walk" as another common example, or you can change it to "humanoid-stand".
DEFAULT_ENV_NAME = "humanoid-walk"  # Or "humanoid-stand", "cheetah-run", etc.

# Default hidden dimension for the actor network (must match the trained model's architecture)
DEFAULT_ACTOR_HIDDEN_DIM = 256

# Construct the default model filename based on the environment name,
# matching the saving convention in train.py (e.g., "models/humanoid_walk_actor_final.pth")
DEFAULT_MODEL_FILENAME = f"{DEFAULT_ENV_NAME.replace('-', '_')}_actor_final.pth"
DEFAULT_MODEL_PATH = os.path.join("models", DEFAULT_MODEL_FILENAME)


class Agent(object):
    """
    Agent that loads a pre-trained SAC actor model for inference.
    Adheres to the specified __init__ and act function signatures.
    """

    def __init__(self):
        # --- Adhering to the constraint: Do not modify the input of this function. ---
        # Model path, environment name, and actor's hidden dimension are set
        # using the default global variables defined above.

        self.device = DEVICE
        self.env_name = DEFAULT_ENV_NAME
        self.model_path = DEFAULT_MODEL_PATH
        self.actor_hidden_dim = DEFAULT_ACTOR_HIDDEN_DIM

        # Create a temporary environment to get observation and action space details.
        # The seed for this temporary env doesn't critically affect space definitions for inference.
        # 'use_pixels=False' because the pre-trained SAC actor expects state vectors.
        # 'flatten=True' to match the expected observation format for state-based agents.
        try:
            temp_env = make_dmc_env(
                self.env_name, seed=0, flatten=True, use_pixels=False
            )
        except Exception as e:
            print(f"Error creating temporary environment '{self.env_name}': {e}")
            print("Please ensure the environment name is correct and dmc.py is set up.")
            raise

        state_dim = temp_env.observation_space.shape[0]
        self.action_space = (
            temp_env.action_space
        )  # Use the actual action space from the environment
        action_dim = self.action_space.shape[0]

        # max_action is needed to initialize the Actor and scale its output
        max_action = float(self.action_space.high[0])

        temp_env.close()

        # Initialize actor network (architecture must match the saved model)
        self.actor = Actor(state_dim, action_dim, self.actor_hidden_dim, max_action).to(
            self.device
        )

        # Load the trained model weights
        if os.path.exists(self.model_path):
            print(f"Loading trained model from: {self.model_path}")
            self.actor.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            # This behavior aligns with the original template's "Agent that acts randomly"
            # if no model is found, though here it would be an untrained policy.
            # For a truly random agent if model not found, one might fall back to action_space.sample().
            # However, the prompt is to create an agent that *uses the imported structures*.
            print(f"WARNING: Model file not found at '{self.model_path}'.")
            print(
                "The agent will act using an uninitialized (randomly weighted) policy."
            )
            print("Ensure the model path is correct and the model file exists.")

        self.actor.eval()  # Set the actor to evaluation mode (e.g., disables dropout)

    def act(self, observation: np.ndarray) -> np.ndarray:
        # --- Adhering to the constraint: Do not modify the input of this function. ---

        # Ensure observation is a NumPy array and then convert to a PyTorch tensor
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)

        # The SAC actor expects a batch dimension, so add one if it's a flat observation
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():  # Disable gradient calculations for inference
            # The Actor's forward method returns mu, std.
            # For deterministic action during inference, we typically use the mean (mu).
            mu, _ = self.actor.forward(obs_tensor)

            # Apply tanh squashing and scale by max_action, consistent with how actions
            # are generated and used during SAC training.
            action_tensor = torch.tanh(mu) * self.actor.max_action

        # Convert action tensor to a flattened NumPy array
        action_np = action_tensor.cpu().numpy().flatten()

        return action_np
