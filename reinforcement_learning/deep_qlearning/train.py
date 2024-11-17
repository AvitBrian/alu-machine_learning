import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from gym.wrappers import AtariPreprocessing, FrameStack
import cv2

class AtariProcessor(Processor):
    def process_observation(self, observation):
        """Process observation for DQN input"""
        # Convert LazyFrames to numpy array properly
        observation = np.asarray(observation)
        
        # Debug print
        print(f"Original observation shape: {observation.shape}")
        
        # If observation is LazyFrames, it needs special handling
        if len(observation.shape) == 1:
            # Convert list of frames to proper numpy array
            frames = [np.asarray(frame) for frame in observation]
            observation = np.stack(frames, axis=-1)
            
        # Add batch dimension if needed
        if len(observation.shape) == 3:
            observation = np.expand_dims(observation, axis=0)
            
        # Debug print
        print(f"Processed observation shape: {observation.shape}")
        
        return observation.astype('float32') / 255.

    def process_state_batch(self, batch):
        """Process batch of states"""
        return batch.astype('float32')

    def process_reward(self, reward):
        """Clip rewards"""
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
    # Create base environment with frame-skip disabled
    env = gym.make(
        'ALE/Breakout-v5',
        render_mode=None,
        frameskip=1,  # Disable frame-skip in base environment
        repeat_action_probability=0.0,
        full_action_space=False
    )
    
    # Add preprocessing wrapper
    env = AtariPreprocessing(
        env,
        frame_skip=4,  # Handle frame-skip in the preprocessing wrapper
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=True,
        screen_size=84
    )
    
    # Stack frames
    env = FrameStack(env, 4)
    
    nb_actions = env.action_space.n
    input_shape = (84, 84, 4)  # Shape matches preprocessed and stacked frames

    # Build model
    model = build_model(input_shape, nb_actions)
    print(model.summary())

    # Configure agent
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=5000,
        target_model_update=1e-2,
        policy=policy,
        processor=processor,
        batch_size=32,
        train_interval=4,
        delta_clip=1.0,
        enable_double_dqn=True
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train
    try:
        dqn.fit(env, 
                nb_steps=50000,
                visualize=False,
                verbose=1)
        
        print("Training completed successfully")
        dqn.save_weights('policy.h5', overwrite=True)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        dqn.save_weights('policy.h5', overwrite=True)
        
    finally:
        env.close()

if __name__ == "__main__":
    main()