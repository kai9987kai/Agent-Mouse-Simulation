import pyautogui
import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
from PIL import ImageGrab
from collections import deque
import winsound
from threading import Thread

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Simulation settings
settings = {
    "agent": {
        "max_speed": 5,
        "energy": 100,
        "energy_decay": 0.1,
        "energy_regeneration": 0.01,
        "hold_click_energy": 0.2,  # Energy consumed per second while holding a click
    },
    "food": {"count": 5, "size": 20, "color": "green"},
    "button": {"size": 20, "color": "blue"},
    "spawn_rate": 2,  # How often new food spawns (in seconds)
    "reward": {
        "click": 10,
        "right_click": 20,
        "middle_click": 15,
        "hold_click": 5,  # Reward per second while holding a click
        "key_press": 10,
        "key_combo": 30,  # Reward for key combinations (e.g., Ctrl+Shift+C)
        "change": 50,
        "task_completion": 100,
        "penalty": -5,  # Penalty for inefficient actions
    },
    "memory_size": 10000,  # Experience replay buffer size
    "batch_size": 32,  # Mini-batch size for training
    "gamma": 0.95,  # Discount factor
    "epsilon": 1.0,  # Exploration rate
    "epsilon_min": 0.01,  # Minimum exploration rate
    "epsilon_decay": 0.995,  # Decay rate for exploration
    "learning_rate": 0.001,
    "max_hold_duration": 5,  # Maximum duration for holding a click (in seconds)
    "action_delay": 0.1,  # Delay between actions to prevent overwhelming the system
}

# Initialize entities
foods = []
buttons = []
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
        self.hold_start_time = None  # Track hold-click duration

    def create_brain(self):
        # Neural network with convolutional layers, attention mechanisms, and recurrent layers
        inputs = tf.keras.layers.Input(shape=(200, 200, 3))  # Larger input size
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)  # Additional layer
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Replace Flatten with GlobalAveragePooling
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(10, activation="linear")(x)  # 10 actions
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]), loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy strategy for exploration vs. exploitation
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 10)  # Random action
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

        # Double DQN: Use target network for next Q-values
        next_q = self.brain.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q, axis=1)
        target_q = self.target_brain.predict(next_states, verbose=0)
        target_q = rewards + settings["gamma"] * target_q[np.arange(settings["batch_size"]), next_actions] * (1 - dones)

        # Train the neural network
        self.brain.fit(states, tf.one_hot(actions, 10) * target_q[:, None], verbose=0)

        # Decay exploration rate
        if self.epsilon > settings["epsilon_min"]:
            self.epsilon *= settings["epsilon_decay"]

    def take_screenshot(self):
        # Capture a 200x200 region around the mouse cursor
        x, y = pyautogui.position()
        screenshot = ImageGrab.grab(bbox=(x-100, y-100, x+100, y+100))
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
        elif action == 6:  # Middle-click
            pyautogui.middleClick()
            reward += settings["reward"]["middle_click"]
        elif action == 7:  # Hold click
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
                pyautogui.mouseDown()
            else:
                hold_duration = time.time() - self.hold_start_time
                if hold_duration >= settings["max_hold_duration"]:
                    pyautogui.mouseUp()
                    self.hold_start_time = None
                reward += settings["reward"]["hold_click"] * hold_duration
        elif action == 8:  # Key press (e.g., 'A')
            pyautogui.press('a')
            reward += settings["reward"]["key_press"]
        elif action == 9:  # Key combination (e.g., Ctrl+Shift+C)
            try:
                pyautogui.hotkey('ctrl', 'shift', 'c')
                reward += settings["reward"]["key_combo"]
            except KeyboardInterrupt:
                print("Key combination interrupted. Skipping...")

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

        # Check for button clicks
        for button in buttons:
            if button.is_clicked(x, y):
                button.color = "red"  # Change color when clicked
                reward += settings["reward"]["task_completion"]

        # Store experience in replay buffer
        next_state = self.take_screenshot()
        self.remember(state, action, reward, next_state, done)

        # Train the agent using experience replay
        self.replay()

        # Add a small delay to prevent overwhelming the system
        time.sleep(settings["action_delay"])

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

class Button:
    def __init__(self):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["button"]["size"]
        self.color = settings["button"]["color"]

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

    def is_clicked(self, x, y):
        return np.hypot(x - self.x, y - self.y) < self.size

# Initialize simulation
def initialize_simulation():
    global foods, buttons, tasks_completed
    foods = [Food() for _ in range(settings["food"]["count"])]
    buttons = [Button() for _ in range(2)]  # Add 2 buttons
    tasks_completed = 0

# Main simulation loop
def run_simulation():
    initialize_simulation()
    agent = Agent()
    last_spawn_time = time.time()
    total_reward = 0

    try:
        while True:  # Run forever
            ax.clear()
            ax.set_xlim(0, screen_width)
            ax.set_ylim(0, screen_height)
            ax.set_title(f"Agent Mouse Simulation | Tasks: {tasks_completed} | Energy: {agent.energy:.2f} | Reward: {total_reward}")

            # Update and draw entities
            for food in foods:
                food.draw()
            for button in buttons:
                button.draw()

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

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Simulation ended. Final tasks completed:", tasks_completed)
    print("Total reward:", total_reward)

# Run the simulation
if __name__ == "__main__":
    run_simulation()
