"""
==========================
8.1 Using KAgents
==========================
Here we will show how to instanciate and use any of the Kernel Reinforcement Learning algorithms. 
Most of the code is similar to what you would find with standard RL libraries. Specificities are commented and explained below.
"""
# Importing necessary modules
import codpy.KQLearning as KQLearning
import gymnasium as gym
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make("CartPole-v1", render_mode="rgb_array")

frames = []

# The agent requires action and observation space dimensions
agent = KQLearning.KQLearning(env.action_space.n, env.observation_space.shape[0], gamma=0.99)
steps = 0
games = 0

while games < 15:
    print("start game")
    # Store the game history for training
    states, actions, next_states, rewards, dones = [], [], [], [], []
    state, _ = env.reset()
    steps = 0
    while steps < 1000:
        action = agent(state)
        next_state, reward, done, _, _ = env.step(action)

        frame = env.render()
        frames.append(frame)
        
        # The agent expects the game to be passed in reverse order
        (
            states.insert(0, state),
            actions.insert(0, action),
            next_states.insert(0, next_state),
            rewards.insert(0, reward),
            dones.insert(0, done),
        )
        steps += 1
        state = next_state
        if done:
            break
    # You train your agent once at the end of every episode
    agent.train((states, actions, next_states, rewards, dones))
    games += 1
    print(f"Game {games}")
env.close()

# imageio.mimsave("cartpole.gif", frames, fps=30)

from IPython.display import HTML, Video, display

# HTML('<img src="cartpole.gif" style="max-width: 100%; height: auto;">')

fig = plt.figure()
plt.axis("off")
im = plt.imshow(frames[0])

def update(frame):
    im.set_array(frame)
    return im,

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames,  # Pass the list of frames
    interval=50,    # Delay between frames (ms)
    blit=True       # Optimize rendering
)
ani.save("cartpole.mp4", writer="ffmpeg", fps=30, dpi=100)
display(Video("cartpole.mp4", embed=True))
# plt.show()