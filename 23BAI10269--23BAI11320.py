# -*- coding: utf-8 -*-
"""
# Name:Sanskriti Singh(23BAI10269), Divya Shukla(23BAI10269)
# Project: DQN on Atari Breakout
# Description: Deep Q-Network implementation using Gymnasium Atari environment

Original file is located at
    https://colab.research.google.com/drive/1Y2R57VuHE5rKoGhg9MWQiSGAEANrcEhY

"""

"""##Register Atari Environment"""

import gymnasium as gym
import torch.nn as nn
import ale_py

gym.register_envs(ale_py)

"""##Test Environment"""

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
obs, _ = env.reset()
print(obs.shape)

"""##Preprocessing Wrapper"""

import cv2
import numpy as np
from collections import deque

class AtariPreprocessing:
    def __init__(self, env):
        self.env = env
        self.frames = deque(maxlen=4)

        # forward attributes
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
      obs, info = self.env.reset(**kwargs)
      frame = self.process(obs)
      self.frames.clear()
      for _ in range(4):
        self.frames.append(frame)

      return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.process(obs)
        self.frames.append(frame)

        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

    def process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        return frame

"""##Wrap Environment"""

env = gym.make("ALE/Breakout-v5")
env = AtariPreprocessing(env)

state = env.reset()
print(state) # test

"""##DQN Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

"""##Replay Buffers"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)

"""##Train"""

import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("ALE/Breakout-v5")
env = AtariPreprocessing(env)

action_size = env.action_space.n

q_net = DQN(action_size).to(DEVICE)
target_net = DQN(action_size).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
buffer = ReplayBuffer(100000)

GAMMA = 0.99
BATCH_SIZE = 32
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 1e-6

UPDATE_TARGET_EVERY = 1000
STEPS = 0

state, _ = env.reset()
env.step(1)
episode_rewards = []
losses = []

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    for t in range(10000):
        STEPS += 1

        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = q_net(s).argmax().item()

        next_state, reward, done, truncated, _ = env.step(action)
        reward = np.sign(reward)


        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) > BATCH_SIZE:
            s, a, r, s_next, d = buffer.sample(BATCH_SIZE)

            s = torch.tensor(s, dtype=torch.float32).to(DEVICE)
            a = torch.tensor(a).to(DEVICE)
            r = torch.tensor(r, dtype=torch.float32).to(DEVICE)
            s_next = torch.tensor(s_next, dtype=torch.float32).to(DEVICE)
            d = torch.tensor(d, dtype=torch.float32).to(DEVICE)  # FIX

            q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q = target_net(s_next).max(1)[0]
                target = r + GAMMA * next_q * (1 - d)

            loss = (q_values - target).pow(2).mean()
            loss1 = nn.MSELoss()(q_values, target.detach())
            losses.append(loss1.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if STEPS % UPDATE_TARGET_EVERY == 0:
            target_net.load_state_dict(q_net.state_dict())

        EPSILON = max(EPSILON_MIN, EPSILON - EPSILON_DECAY)

        if done or truncated:
            break
    episode_rewards.append(total_reward)
    print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {EPSILON:.3f}")

import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward vs Episodes")
plt.show()

plt.plot(losses)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

import matplotlib.pyplot as plt

# Create new environment for rendering
test_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
test_env = AtariPreprocessing(test_env)

state, _ = test_env.reset()

# FIRE action to start game
test_env.step(1)

for _ in range(300):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action = q_net(s).argmax().item()

    state, reward, done, truncated, _ = test_env.step(action)

    frame = test_env.env.render()

    plt.imshow(frame)
    plt.axis('off')
    plt.pause(0.01)

    if done or truncated:
        break

test_env.env.close()

pip install imageio imageio-ffmpeg

import imageio

video_filename = "breakout_dqn.mp4"
writer = imageio.get_writer(video_filename, fps=30)

test_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
test_env = AtariPreprocessing(test_env)

state, _ = test_env.reset()
test_env.env.step(1)  # FIRE to start

for _ in range(500):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action = q_net(s).argmax().item()

    state, reward, done, truncated, _ = test_env.step(action)

    frame = test_env.env.render()
    writer.append_data(frame)

    if done or truncated:
        break

writer.close()
test_env.env.close()

print("Video saved as:", video_filename)