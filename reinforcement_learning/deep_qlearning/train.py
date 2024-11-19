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
        assert observation.ndim == 3
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
    """Configure DQN agent with optimized hyperparameters"""
    memory = SequentialMemory(limit=1000000, window_length=4)
    
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.02,
        nb_steps=75000
    )
    
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=25000, 
        gamma=0.99,
        target_model_update=5000, 
        train_interval=4,
        delta_clip=1.0,
        batch_size=64
    )
    
    dqn.compile(
        optimizer=optimizers.Adam(learning_rate=0.00025),
        metrics=['mae']
    )
    return dqn


class EarlyStoppingCallback(Callback):
    def __init__(self, reward_threshold=50.0, window_size=100, patience=20):
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.patience = patience
        self.rewards = []
        self.last_mean = -float('inf')
        self.no_improvement_count = 0
        
    def on_episode_end(self, episode, logs):
        reward = logs['episode_reward']
        self.rewards.append(reward)
        
        if len(self.rewards) >= self.window_size:
            mean_reward = np.mean(self.rewards[-self.window_size:])
            
            if mean_reward <= self.last_mean - 0.5: 
                self.no_improvement_count += 1
            elif mean_reward > self.last_mean:
                self.no_improvement_count = 0
                self.last_mean = mean_reward
            
            if self.no_improvement_count >= self.patience:
                print(f"\nStopping training - No improvement for {self.patience} episodes")
                print(f"Best mean reward: {self.last_mean:.2f}")
                self.model.stop_training = True


class TrainingMetricsCallback(Callback):
    def __init__(self, min_episodes=10):
        self.rewards = []
        self.start_time = time.time()
        self.best_mean_reward = -float('inf')
        self.best_episode_reward = -float('inf')
        self.min_episodes = min_episodes
        
    def on_episode_end(self, episode, logs={}):
        reward = logs.get('episode_reward', 0)
        self.rewards.append(reward)
        
        if len(self.rewards) >= self.min_episodes:
            mean_reward = np.mean(self.rewards[-self.min_episodes:])
            
            if mean_reward > self.best_mean_reward + 0.5:
                self.best_mean_reward = mean_reward
                print(f"\nNew best average reward: {mean_reward:.2f}")
                self.model.save_weights(f"policy_avg_{mean_reward:.1f}.h5")
            
            if reward > self.best_episode_reward + 1:
                self.best_episode_reward = reward
                print(f"\nNew best episode: {reward:.2f}")
                self.model.save_weights(f"policy_best_{reward:.1f}.h5")


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

    metrics_callback = TrainingMetricsCallback(min_episodes=10)
    
    try:
        dqn.fit(
            env, 
            nb_steps=250000,
            log_interval=1000,
            visualize=False, 
            verbose=1,
            callbacks=[early_stopping, metrics_callback]
        )
    except Exception as e:
        print(f"Training interrupted: {e}")
        dqn.save_weights("policy_final.h5", overwrite=True)
    
    dqn.save_weights("policy.h5", overwrite=True)


if __name__ == "__main__":
    train()