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
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
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

    # Load the pre-trained weights (if available)
    try:
        dqn.load_weights('Policies/resource_allocation_dqn_weights.h5')
    except Exception as e:
        print(f"Error loading weights: {e}")

    # Test the agent
    dqn.test(env, nb_episodes=10, visualize=True, verbose=1)
