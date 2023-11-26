import gym
import readchar
import numpy as np

# # MACROS
Push_Left = 0
No_Push = 1
Push_Right = 2

# Key mapping
arrow_keys = {
    '\x1b[D': Push_Left,
    '\x1b[B': No_Push,
    '\x1b[C': Push_Right}

env = gym.make('MountainCar-v0')#, render_mode="human")

trajectories = []
episode_step = 0

for episode in range(1):  # n_trajectories : 20
    trajectory = []
    step = 0

    env.reset()
    print("episode_step", episode_step)

    while True:
        env.render()
        print("step", step)

        key = readchar.readkey()
        if key not in arrow_keys.keys():
            break

        action = arrow_keys[key]
        state, reward, done, _, _ = env.step(action)

        if state[0] >= env.env.goal_position and step > 129:  # trajectory_length : 130
            break

        trajectory.append((state[0], state[1], action))
        step += 1
    print(trajectory)
