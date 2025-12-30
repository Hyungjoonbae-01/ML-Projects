import gymnasium as gym
import random
import numpy as np

random.seed(1234)

# Setting up the Taxi environment
streets = gym.make("Taxi-v3", render_mode="ansi").env
# Setting a specific initial state for consistency
initial_state = streets.unwrapped.encode(
    4, 3, 4, 3
)  # (taxi row, taxi column, passenger index, destination index)
streets.unwrapped.s = initial_state
streets.unwrapped.lastaction = None

# Initializing Q-table, in which rows represent states and columns represent actions
# For example , q_table[initial_state] gives the Q-values for all actions in the initial state
q_table = np.zeros([streets.observation_space.n, streets.action_space.n])


# Hyperparameters
learning_rate = 0.1
discount_factor = 0.6
exploration = 0.1
epochs = 10000


# Training the agent
for taxi_run in range(epochs):
    state, _ = streets.reset()  # Setting the state to the predefined initial state
    done = False

    while not done:
        random_value = random.uniform(0, 1)
        if random_value < exploration:
            action = streets.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = streets.step(action)
        done = terminated or truncated

        prev_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * prev_q + learning_rate * (
            reward + discount_factor * next_max_q
        )
        q_table[state, action] = new_q
        state = next_state
