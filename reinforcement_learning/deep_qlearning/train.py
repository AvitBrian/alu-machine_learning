#!/usr/bin/env python3
"""Deep Q-Learning implementation for Atari's Breakout game"""

import gym
import numpy as np
from PIL import Image
from tensorflow.keras import layers, Model, optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor
from rl.callbacks import Callback
import time


class AtariProcessor(Processor):
    """Processor for Atari game frames"""
    
    def __init__(self, target_size=(84, 84)):
        super(AtariProcessor, self).__init__()
        self.target_size = target_size

    def process_observation(self, observation):
        """Process raw game frames for neural network input"""
        assert observation.ndim == 3, "Expected 3D array (height, width, channel)"
        
        image = Image.fromarray(observation)
        image = image.resize(self.target_size, Image.Resampling.LANCZOS).convert("L")
        processed_observation = np.array(image)
        
        assert processed_observation.shape == self.target_size
        return processed_observation.astype("uint8")

    def process_state_batch(self, batch):
        """Normalize state batch to [0, 1] range"""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards to [-1, 1] range"""
        return np.clip(reward, -1.0, 1.0)


def create_CNN_model(number_actions, frames=4, input_shape=(84, 84)):
    """Create CNN model for Deep Q-Learning following DeepMind's architecture"""
    
    conv_layers = [
        {'filters': 32, 'kernel_size': 8, 'strides': 4, 'name': 'conv1'},
        {'filters': 64, 'kernel_size': 4, 'strides': 2, 'name': 'conv2'},
        {'filters': 64, 'kernel_size': 3, 'strides': 1, 'name': 'conv3'}
    ]
    
    dense_layers = [
        {'units': 512, 'name': 'dense1'},
        {'units': number_actions, 'activation': 'linear', 'name': 'output'}
    ]

    inputs = layers.Input(shape=(frames,) + input_shape, name='input')
    x = layers.Permute((2, 3, 1), name='permute')(inputs)

    for conv in conv_layers:
        x = layers.Conv2D(
            filters=conv['filters'],
            kernel_size=conv['kernel_size'],
            strides=conv['strides'],
            activation='relu',
            name=conv['name']
        )(x)

    x = layers.Flatten(name='flatten')(x)

    for dense in dense_layers:
        x = layers.Dense(
            units=dense['units'],
            activation=dense.get('activation', 'relu'),
            name=dense['name']
        )(x)

    return Model(inputs=inputs, outputs=x), frames


def setup_dqn_agent(model, nb_actions, processor):
    """Configure DQN agent with specified hyperparameters"""
    memory = SequentialMemory(limit=1000000, window_length=4)
    
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )
    
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )
    
    dqn.compile(optimizers.Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


class EarlyStoppingCallback(Callback):
    def __init__(self, reward_threshold, window_size=100):
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.rewards = []
        
    def on_episode_end(self, episode, logs):
        self.rewards.append(logs['episode_reward'])
        
        if len(self.rewards) >= self.window_size:
            mean_reward = np.mean(self.rewards[-self.window_size:])
            if mean_reward >= self.reward_threshold:
                print(f"\nEarly stopping triggered! Mean reward: {mean_reward:.2f}")
                self.model.stop_training = True


class TrainingMetricsCallback(Callback):
    """Simple callback to track training metrics"""
    def __init__(self):
        self.rewards = []
        self.start_time = time.time()
        
    def on_episode_end(self, episode, logs={}):
        self.rewards.append(logs.get('episode_reward', 0))
        
    def on_train_end(self, logs={}):
        """Display final training statistics"""
        print("\nTraining Summary:")
        print("=" * 30)
        print(f"Total Episodes: {len(self.rewards)}")
        print(f"Training Duration: {(time.time() - self.start_time) / 60:.2f} minutes")
        print(f"Best Reward: {max(self.rewards):.2f}")
        print(f"Average Reward: {np.mean(self.rewards):.2f}")
        print(f"Final 100 Episodes Average: {np.mean(self.rewards[-100:]):.2f}")

def train():
    """Train the DQN agent on Breakout"""
    env = gym.make("Breakout-v0")
    env.reset()
    
    nb_actions = env.action_space.n
    model, frames = create_CNN_model(nb_actions)
    model.summary()
    
    processor = AtariProcessor()
    dqn = setup_dqn_agent(model, nb_actions, processor)
    
    early_stopping = EarlyStoppingCallback(
        reward_threshold=50.0,
        window_size=100
    )

    metrics_callback = TrainingMetricsCallback()
    
    dqn.fit(
        env, 
        nb_steps=1000000, 
        log_interval=10000, 
        visualize=False, 
        verbose=2,
        callbacks=[early_stopping, metrics_callback]
    )
    
    dqn.save_weights("policy.h5", overwrite=True)


if __name__ == "__main__":
    train()