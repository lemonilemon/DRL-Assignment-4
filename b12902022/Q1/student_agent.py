import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # Required for the Actor's forward pass

# --- Actor Network Definition ---
# This class definition must match the Actor class used during training.
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        """Initialize parameters and build model.
        Params
        ======
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            max_action (float): Highest possible action value (magnitude)
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
        # Use tanh to output actions between -1 and 1, then scale by max_action
        x = torch.tanh(self.layer_3(x)) * self.max_action
        return x

# --- Inference Agent ---
# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that uses a pre-trained DDPG Actor model to select actions."""
    def __init__(self):
        """
        Initializes the agent by loading the pre-trained DDPG Actor model.
        Ensure that the actor model file (e.g., 'ddpg_pendulum_actor.pth')
        is present at the specified path.
        """
        # --- Parameters for Pendulum-v1 ---
        # State dimension: [cos(theta), sin(theta), theta_dot]
        self.state_dim = 3
        # Action dimension: [torque]
        self.action_dim = 1
        # Max action value (torque magnitude for Pendulum-v1)
        self.max_action = 2.0
        # Hidden dimension for the actor network (must match training)
        self.hidden_dim = 256

        # --- Device Configuration ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference Agent using device: {self.device}")

        # --- Initialize Actor Network ---
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, self.hidden_dim).to(self.device)

        # --- Load Pre-trained Actor Model Weights ---
        # IMPORTANT: Replace "ddpg_pendulum_actor.pth" with the actual path to your trained actor model file.
        # This file should have been saved during training, typically as "<filename_prefix>_actor.pth".
        actor_model_path = "ddpg_pendulum_actor.pth" # MODIFY THIS PATH IF NEEDED

        try:
            # Load the state dictionary. map_location ensures model loads correctly even if trained on a different device.
            self.actor.load_state_dict(torch.load(actor_model_path, map_location=self.device))
            print(f"Successfully loaded actor model from: {actor_model_path}")
        except FileNotFoundError:
            print(f"ERROR: Actor model file not found at '{actor_model_path}'.")
            print("Please ensure the path is correct and the model file exists.")
            print("The agent will use a randomly initialized actor network.")
        except Exception as e:
            print(f"ERROR loading actor model: {e}")
            print("The agent will use a randomly initialized actor network.")

        # Set the actor network to evaluation mode
        # This disables layers like dropout or batch normalization if they were used (not in this Actor, but good practice).
        self.actor.eval()

        # --- Action Space (as per the provided template) ---
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        # While the DDPG actor inherently learns the action scale,
        # this defines the expected action space format.
        self.action_space = gym.spaces.Box(-self.max_action, self.max_action, (self.action_dim,), dtype=np.float32)


    def act(self, observation):
        """
        Selects an action based on the given observation using the loaded Actor model.

        Args:
            observation (np.ndarray): The current observation (state) from the environment.

        Returns:
            np.ndarray: The action selected by the agent.
        """
        # Convert the observation (NumPy array) to a PyTorch tensor
        # Reshape to (1, state_dim) as the network expects a batch of states.
        state_tensor = torch.FloatTensor(observation.reshape(1, -1)).to(self.device)

        # Perform inference with no gradient calculation
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)

        # Convert the action tensor back to a NumPy array and flatten it
        action_numpy = action_tensor.cpu().data.numpy().flatten()

        # The actor's output is already scaled by max_action due to the tanh activation and scaling.
        # No explicit clipping is needed here if the actor is well-trained and matches the environment's action bounds.
        return action_numpy

# --- Example Usage (Optional: for testing the Agent class) ---
if __name__ == '__main__':
    # This is an example of how you might use the Agent.
    # You'll need to have 'ddpg_pendulum_actor.pth' (or your model file) in the same directory
    # or provide the correct path.

    # Create a dummy environment for testing (not used for actual model logic here)
    # env = gym.make("Pendulum-v1", render_mode='human') # Uncomment to render
    env = gym.make("Pendulum-v1")
    observation, info = env.reset()

    # Initialize the agent
    # This will attempt to load the model specified in Agent.__init__
    agent = Agent()

    print("\nTesting Agent with a sample observation...")
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial observation: {observation}")

    # Get an action from the agent
    action = agent.act(observation)
    print(f"Action selected by agent: {action}")
    print(f"Action shape: {action.shape}")

    # Example of stepping in the environment with the agent's action
    # next_observation, reward, terminated, truncated, info = env.step(action)
    # print(f"Next observation: {next_observation}")
    # print(f"Reward: {reward}")

    # Remember to close the environment if you create one
    env.close()

    print("\n--- Agent Test Complete ---")
    print("If no 'ERROR' messages appeared above regarding model loading,")
    print("and you see an action printed, the Agent class is likely set up correctly.")
    print("Ensure 'ddpg_pendulum_actor.pth' (or your specified model file) exists and contains valid weights.")
