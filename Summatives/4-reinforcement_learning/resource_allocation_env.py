import time
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

class ResourceAllocationEnv(gym.Env):
    """
    This class represents the environment for the resource allocation problem.
    """
    def __init__(self, grid_size=5, resources=50, threshold=1):
        """
        Initialize the environment.
        """
        super(ResourceAllocationEnv, self).__init__()
        self.grid_size = grid_size
        self.resources = resources
        self.threshold = threshold 
        
        # 4 movement actions + 1 redistribution action
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=self.resources,
            shape=(self.grid_size, self.grid_size),
            dtype=np.int32,
        )

        self.state = None
        self.agent_pos = None
        self.steps = 0
        self.max_steps = 500

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # Start with a grid of zeros
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Create clusters of high-resource and low-resource areas
        high_resource_schools = random.sample(
            [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)], 
            k=max(1, self.grid_size // 2)
        )
        remaining_resources = self.resources

        for school in high_resource_schools:
            if remaining_resources <= 0:
                break
            allocation = random.randint(1, min(remaining_resources, self.resources // 4))
            self.state[school] = allocation
            remaining_resources -= allocation

        # Distribute remaining resources randomly across the grid
        while remaining_resources > 0:
            i, j = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            allocation = random.randint(1, remaining_resources)
            self.state[i, j] += allocation
            remaining_resources -= allocation

        self.agent_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.steps = 0
        return self.state


    def step(self, action):
        """
        Take a step in the environment.
        """
        self.steps += 1
        x, y = self.agent_pos
        reward = 0

        if action == 0 and x > 0:
            # Move left
            self.agent_pos[0] -= 1
            reward += self._redistribute_resources()
        elif action == 1 and x < self.grid_size - 1:
            # Move right
            self.agent_pos[0] += 1
            reward += self._redistribute_resources()
        elif action == 2 and y > 0:
            # Move up
            self.agent_pos[1] -= 1
            reward += self._redistribute_resources()
        elif action == 3 and y < self.grid_size - 1:
            # Move down
            self.agent_pos[1] += 1
            reward += self._redistribute_resources()
        elif action == 4:
            # Redistribute resources in the current school
            reward += self._redistribute_resources()

        done = self._check_equitable_distribution()
        return self.state, reward, done, {}

    def _redistribute_resources(self):
        """
        Redistribute resources between the current school and its neighbors.
        """
        x, y = self.agent_pos
        neighbors = self._get_neighbors(x, y)
        schools = [(x, y)] + neighbors

        resources = sum(self.state[cx, cy] for cx, cy in schools)

        # Distribute resources evenly and return a reward
        avg_resources = resources // len(schools)
        remainder = resources % len(schools)

        for i, (cx, cy) in enumerate(schools):
            self.state[cx, cy] = avg_resources + (1 if i < remainder else 0)

        return 1 

    def _get_neighbors(self, x, y):
        """
        Get the coordinates of the neighboring schools (up, down, left, right).
        """
        neighbors = []
        # Up
        if x > 0: neighbors.append((x - 1, y))
        # Down
        if x < self.grid_size - 1: neighbors.append((x + 1, y))
        # Left
        if y > 0: neighbors.append((x, y - 1))
        # Right
        if y < self.grid_size - 1: neighbors.append((x, y + 1))
        return neighbors

    def _check_equitable_distribution(self):
        """
        Checks if the grid has been balanced.
        """
        max_resources = np.max(self.state)
        min_resources = np.min(self.state)
        return max_resources - min_resources <= self.threshold

    def render(self, mode="human", display_type='terminal', show_agent=True):
        """
        Render the environment.
        """
        grid_with_agent = self.state.copy()
        x, y = self.agent_pos
        if show_agent:
            grid_with_agent[x, y] = 999

        if display_type == 'terminal':
            print("\033c", end="")
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if grid_with_agent[row, col] == 999 and show_agent:
                        print("🤖 ", end="")
                    else:
                        print(f"{grid_with_agent[row, col]:2} ", end="")
                print("")
            time.sleep(0.2)

        if display_type == 'matplotlib':
            if not hasattr(self, 'fig') or not hasattr(self, 'ax'):
                self.fig, self.ax = plt.subplots()
                plt.ion()

            self.ax.clear()
            self.ax.imshow(self.state, cmap='coolwarm', interpolation='nearest')
            if show_agent:
                robot_emoji = "\U0001F916"
                self.ax.text(self.agent_pos[1], self.agent_pos[0], robot_emoji, ha='center', va='center',fontsize=12)

            self.ax.set_title("Resource Allocation")
            plt.draw()
            plt.pause(0.1)

    def seed(self, seed=None):
        """
        Seed the random number generator.
        """
        self.np_random, seed = np.random.default_rng(seed), seed
        random.seed(seed)
        return [seed]
