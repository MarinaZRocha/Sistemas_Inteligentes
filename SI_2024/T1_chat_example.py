import numpy as np
import gym
import matplotlib.pyplot as plt

# Create the environment
env = gym.make("FrozenLake-v1", is_slippery=True)

def run_q_learning(env, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-Learning update rule
            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])
            
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
    
    return q_table, rewards

def run_sarsa(env, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Epsilon-greedy action selection for SARSA
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # Epsilon-greedy action selection for the next state
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            
            # SARSA update rule
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])
            
            state = next_state
            action = next_action
            total_reward += reward
        
        rewards.append(total_reward)
    
    return q_table, rewards

# Parameters
episodes = 1000
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Run Q-learning and SARSA
q_table_qlearning, rewards_qlearning = run_q_learning(env, episodes, alpha, gamma, epsilon)
q_table_sarsa, rewards_sarsa = run_sarsa(env, episodes, alpha, gamma, epsilon)

# Plot the comparison
plt.plot(rewards_qlearning, label="Q-learning")
plt.plot(rewards_sarsa, label="SARSA")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Q-learning vs SARSA")
plt.legend()
plt.show()
