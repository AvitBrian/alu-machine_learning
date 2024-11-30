import time
import random
from resource_allocation_env import ResourceAllocationEnv

if __name__ == "__main__":
    # Ask the user for the number of resources
    resources = int(input("Enter the number of available resources: "))
    env = ResourceAllocationEnv(resources=resources)

    # Reset the environment
    obs = env.reset()
    
    display_type = input("\nEnter display type (terminal/matplotlib) or (q) to quit: ").strip().lower()

    #Running state 
    running = True
    # Choose display type
    while running:
        if display_type == 'terminal' or display_type == 'matplotlib':
            print("\nStarting the simulation...\n")
            time.sleep(1)

            for step in range(10000):
                print(f"\nStep {step + 1}:")

                # Randomly select an action
                action = random.randint(0, env.action_space.n - 1)
                obs, reward, done, _ = env.step(action)
                
                if done:
                    print("\nSimulation complete.")
                    running = False
                    break

                # Display the state
                env.render(mode="human", display_type=display_type)

                # Display additional information
                print(f"Action taken: {action}")
                print(f"Reward: {reward}")
                print(f"Current State: \n {obs}")

                # Add a short delay for readability
                time.sleep(0.5)
        elif display_type == 'q':
            print('Quitting...')
            break
    
        else:
            print("You can only view the simulation in terminal or plot.")
            display_type = input("\nEnter display type (terminal/matplotlib) or (q) to quit: ").strip().lower()
