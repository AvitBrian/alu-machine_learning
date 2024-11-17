import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from gym.wrappers import AtariPreprocessing, FrameStack
import cv2

class AtariProcessor(Processor):
    def process_observation(self, observation):
        # Convert to grayscale and resize
        processed = cv2.resize(observation, (84, 84))
        return np.expand_dims(processed, axis=-1).astype('float32') / 255.

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

def build_model(input_shape, nb_actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape),
        Conv2D(64, (4, 4), strides=2, activation='relu'),
        Conv2D(64, (3, 3), strides=1, activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(nb_actions, activation='linear')
    ])
    return model

def main():
    # Create environment
    env = gym.make('Breakout-v0', render_mode='human')
    env = AtariPreprocessing(env, 
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=True
    )
    env = FrameStack(env, 4)
    nb_actions = env.action_space.n

    # Build model with same architecture as training
    input_shape = (84, 84, 4)
    model = build_model(input_shape, nb_actions)

    # Create agent with GreedyQPolicy for exploitation
    memory = SequentialMemory(limit=50000, window_length=4)
    policy = GreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=5000,
        enable_double_dqn=True
    )
    dqn.compile(optimizer='adam')

    # Load trained weights
    try:
        dqn.load_weights('policy.h5')
        print("Weights loaded successfully")
    except:
        print("Error loading weights. Make sure 'policy.h5' exists")
        return

    # Test for 5 episodes
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()

if __name__ == "__main__":
    main() 