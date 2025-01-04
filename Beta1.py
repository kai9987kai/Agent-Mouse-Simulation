import pyautogui
import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
from PIL import ImageGrab
from collections import deque

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Simulation settings
settings = {
    "agent": {"max_speed": 5, "energy": 100, "energy_decay": 0.1, "energy_regeneration": 0.01},
    "food": {"count": 5, "size": 20, "color": "green"},
    "spawn_rate": 2,  # How often new food spawns (in seconds)
    "reward": {"click": 10, "right_click": 20, "scroll": 5, "change": 50, "task_completion": 100},
    "memory_size": 10000,  # Experience replay buffer size
    "batch_size": 32,  # Mini-batch size for training
    "gamma": 0.95,  # Discount factor
    "epsilon": 1.0,  # Exploration rate
    "epsilon_min": 0.01,  # Minimum exploration rate
    "epsilon_decay": 0.995,  # Decay rate for exploration
}

# Initialize entities
foods = []
tasks_completed = 0

# Visualization setup
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, screen_width)
ax.set_ylim(0, screen_height)
ax.set_title("Agent Mouse Simulation")

class Agent:
    def __init__(self):
        self.energy = settings["agent"]["energy"]
        self.brain = self.create_brain()
        self.target_brain = self.create_brain()  # Target network
        self.target_brain.set_weights(self.brain.get_weights())
        self.memory = deque(maxlen=settings["memory_size"])  # Experience replay buffer
        self.epsilon = settings["epsilon"]

    def create_brain(self):
        # Neural network with convolutional layers for screen capture input
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100, 100, 3)),  # Explicit Input layer
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(6, activation="linear"),  # 6 actions: move left, right, up, down, left-click, right-click
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy strategy for exploration vs. exploitation
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 6)  # Random action
        q_values = self.brain.predict(state[np.newaxis, ...], verbose=0)[0]
        return np.argmax(q_values)  # Predicted action

    def replay(self):
        # Train the agent using experiences from the replay buffer
        if len(self.memory) < settings["batch_size"]:
            return
        minibatch = random.sample(self.memory, settings["batch_size"])
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Predict Q-values for current and next states
        current_q = self.brain.predict(states, verbose=0)
        next_q = self.target_brain.predict(next_states, verbose=0)
        target_q = current_q.copy()

        # Update Q-values using Bellman equation
        for i in range(settings["batch_size"]):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + settings["gamma"] * np.max(next_q[i])

        # Train the neural network
        self.brain.fit(states, target_q, verbose=0)

        # Decay exploration rate
        if self.epsilon > settings["epsilon_min"]:
            self.epsilon *= settings["epsilon_decay"]

    def take_screenshot(self):
        # Capture a 100x100 region around the mouse cursor
        x, y = pyautogui.position()
        screenshot = ImageGrab.grab(bbox=(x-50, y-50, x+50, y+50))
        screenshot = np.array(screenshot) / 255.0  # Normalize pixel values
        return screenshot

    def calculate_mse(self, before, after):
        # Calculate Mean Squared Error (MSE) between two screenshots
        return np.mean((before - after) ** 2)

    def move(self):
        # Take a screenshot before the action
        before_screenshot = self.take_screenshot()

        # Predict the next action
        state = self.take_screenshot()
        action = self.act(state)

        # Execute action
        x, y = pyautogui.position()
        reward = 0
        if action == 0:  # Move right
            x += settings["agent"]["max_speed"]
        elif action == 1:  # Move left
            x -= settings["agent"]["max_speed"]
        elif action == 2:  # Move down
            y += settings["agent"]["max_speed"]
        elif action == 3:  # Move up
            y -= settings["agent"]["max_speed"]
        elif action == 4:  # Left-click
            pyautogui.click()
            reward += settings["reward"]["click"]
        elif action == 5:  # Right-click
            pyautogui.rightClick()
            reward += settings["reward"]["right_click"]

        # Ensure the mouse stays within the screen bounds
        x = max(0, min(screen_width, x))
        y = max(0, min(screen_height, y))
        pyautogui.moveTo(x, y)

        # Take a screenshot after the action
        after_screenshot = self.take_screenshot()

        # Calculate reward based on screen change (using MSE)
        change = self.calculate_mse(before_screenshot, after_screenshot)
        reward += settings["reward"]["change"] * change  # More change = higher reward

        # Energy management
        self.energy -= settings["agent"]["energy_decay"]
        self.energy += settings["agent"]["energy_regeneration"]
        self.energy = min(self.energy, settings["agent"]["energy"])
        done = self.energy <= 0
        if done:
            print("Agent ran out of energy!")

        # Check for food
        for food in foods:
            distance = np.hypot(x - food.x, y - food.y)
            if distance < settings["food"]["size"]:
                self.energy = min(self.energy + 20, settings["agent"]["energy"])
                foods.remove(food)
                global tasks_completed
                tasks_completed += 1
                reward += settings["reward"]["task_completion"]

        # Store experience in replay buffer
        next_state = self.take_screenshot()
        self.remember(state, action, reward, next_state, done)

        # Train the agent using experience replay
        self.replay()

        return not done, reward

class Food:
    def __init__(self):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["food"]["size"]
        self.color = settings["food"]["color"]

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

# Initialize simulation
def initialize_simulation():
    global foods, tasks_completed
    foods = [Food() for _ in range(settings["food"]["count"])]
    tasks_completed = 0

# Main simulation loop
def run_simulation():
    initialize_simulation()
    agent = Agent()
    last_spawn_time = time.time()
    total_reward = 0

    while agent.energy > 0:
        ax.clear()
        ax.set_xlim(0, screen_width)
        ax.set_ylim(0, screen_height)
        ax.set_title(f"Agent Mouse Simulation | Tasks: {tasks_completed} | Energy: {agent.energy:.2f} | Reward: {total_reward}")

        # Update and draw entities
        for food in foods:
            food.draw()

        # Move the agent
        alive, reward = agent.move()
        total_reward += reward
        if not alive:
            break

        # Spawn new food periodically
        if time.time() - last_spawn_time > settings["spawn_rate"]:
            foods.append(Food())
            last_spawn_time = time.time()

        # Refresh the plot
        plt.pause(0.01)

    print("Simulation ended. Final tasks completed:", tasks_completed)
    print("Total reward:", total_reward)

# Run the simulation
if __name__ == "__main__":
    run_simulation()
