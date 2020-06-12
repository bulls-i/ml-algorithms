# Reinforcement learning (Frozen lake)
# Q-learning algorithm

import gym
import numpy as np 
import random 
import time 

# SFFF
# FHFH
# FFFH
# HFFG

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

env = gym.make('FrozenLake-v0')

a_size = env.action_space.n 
s_size = env.observation_space.n

q_table = np.zeros((s_size, a_size))
num_episodes = 10000
max_steps = 100
lr = 0.1
dr = 0.99
er = 1
max_er = 1
min_er = 0.01
e_decay = 0.001

rewards = []
for episode in range(num_episodes):
    state = env.reset()

    done = False
    cur_reward = 0

    for step in range(max_steps):
        e_threshold = random.uniform(0,1)
        if e_threshold > er:
            action = np.argmax(q_table[state, :])
        else: 
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)

        q_table[state, action] = (1 - lr) * q_table[state, action] + \
            lr * (reward + dr * np.max(q_table[new_state, :]))
        
        state = new_state
        cur_reward += reward

        if done == True:
            break 

    er = min_er + (max_er - min_er) * np.exp(-e_decay * episode)
    rewards.append(cur_reward)

r = np.split(np.array(rewards), num_episodes / 1000)
count = 1000

for r1 in r:
    print(count, ':', sum(r1 / 1000))
    count += 1000

# print(q_table)
