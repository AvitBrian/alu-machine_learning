from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from resource_allocation_env import ResourceAllocationEnv
from keras import __version__
import tensorflow as tf

tf.keras.__version__ = __version__


def build_model(states, actions):
    """
    Builds a neural network model for the DQN agent.

    Parameters:
    - states: The shape of the observation space.
    - actions: The number of actions in the action space.

    Returns:
    - A compiled Keras model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    """
    Builds a DQN agent with the specified model and parameters.

    Parameters:
    - model: The neural network model to use for the agent.
    - actions: The number of actions in the action space.

    Returns:
    - A DQNAgent instance.
    """
    policy = EpsGreedyQPolicy(eps=1.0)
    memory = SequentialMemory(limit=2000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn

if __name__ == "__main__":
    # Initialize the environment
    env = ResourceAllocationEnv(grid_size=5, resources=50)
    states = env.observation_space.shape
    actions = env.action_space.n

    # Build and summarize the model
    model = build_model(states, actions)
    model.summary()

    # Build the DQN agent
    dqn = build_agent(model, actions)

    # Compile the agent
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Train the agent
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # Save the trained model
    dqn.save_weights('Policies/resource_allocation_dqn_weights.h5', overwrite=True)
    print("Training completed and model saved.")
