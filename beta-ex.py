#!/usr/bin/env python3
"""
advanced_rl_mouse_simulation.py

An advanced RL simulation for a "mouse agent" with improved mouse movements, drag-and-drop,
enhanced annotations, and innovative learning mechanisms.
"""

import random
import time
import numpy as np
import pyautogui
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageGrab
from collections import deque
from typing import List, Tuple
import threading

# IMPORTANT: for interactive mode in some IDEs
matplotlib.use('TkAgg')

###############################################################################
#                           GLOBAL CONFIGURATION                               #
###############################################################################
# Fix seeds for reproducibility (optional)
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Simulation parameters
settings = {
    "agent": {
        "max_speed": 5,                 # pixels per move
        "energy": 100,
        "energy_decay": 0.1,
        "energy_regeneration": 0.02,
        "energy_surge_interval": 30.0,  # seconds between surges
        "energy_surge_amount": 20.0,    # energy gained during a surge
    },
    "food": {
        "count": 5,
        "size": 20,
        "color": "green",
    },
    "special_food": {
        "count": 1,
        "size": 25,
        "color": "gold",
    },
    "rest_zone": {
        "size": 30,
        "color": "blue",
    },
    "danger_zone": {
        "count": 2,
        "size": 30,
        "color": "red",
    },
    "goal_zone": {
        "size": 40,
        "color": "purple",
    },
    "dynamic_obstacle": {
        "count": 2,
        "size": 30,
        "color": "black",
        "speed": 2,  # pixels per move
    },
    "spawn_rate": 2,          # new food spawn interval in seconds
    "reward": {
        "exploration": 1,     # reward per pixel traveled
        "task_completion": 100,
        "special_food": 50,
        "danger_penalty": -50,
        "goal_reward": 200,   # reward for reaching the goal
    },
    "memory_size": 20000,
    "batch_size": 64,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "learning_rate": 0.0005,
    "action_indicator_time": 1.0,  # seconds to display a click/keypress indicator
}

###############################################################################
#                           MATPLOTLIB SETUP                                   #
###############################################################################
# Smaller default window size
plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))  # Reduced window size
ax.set_xlim(0, screen_width)
ax.set_ylim(0, screen_height)
ax.set_title("Advanced RL Mouse Simulation")

###############################################################################
#                           NOISY LAYER                                       #
###############################################################################
class NoisyDense(tf.keras.layers.Layer):
    """
    A Dense layer with parameter noise for exploration (Fortunato et al.).
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.w_mu = self.add_weight(
            name='w_mu',
            shape=(in_dim, self.units),
            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
            trainable=True
        )
        self.w_sigma = self.add_weight(
            name='w_sigma',
            shape=(in_dim, self.units),
            initializer=tf.constant_initializer(0.017),
            trainable=True
        )
        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(self.units,),
            initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
            trainable=True
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(self.units,),
            initializer=tf.constant_initializer(0.017),
            trainable=True
        )

    def call(self, inputs, training=None):
        if training:
            eps_in = tf.random.normal((tf.shape(inputs)[-1],))
            eps_out = tf.random.normal((self.units,))
            w_epsilon = tf.tensordot(eps_in, eps_out, axes=0)
            b_epsilon = eps_out
            w = self.w_mu + self.w_sigma * w_epsilon
            b = self.b_mu + self.b_sigma * b_epsilon
        else:
            w = self.w_mu
            b = self.b_mu
        return tf.matmul(inputs, w) + b

###############################################################################
#                           DUELING DQN MODEL                                  #
###############################################################################
def build_dueling_dqn(input_shape, num_actions):
    """
    Constructs a Dueling DQN with Noisy Layers.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Convolutional feature extractor
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Value stream
    v = NoisyDense(128, name="value_fc1")(x)
    v = tf.keras.layers.ReLU()(v)
    v = NoisyDense(1, name="value")(v)

    # Advantage stream
    a = NoisyDense(128, name="adv_fc1")(x)
    a = tf.keras.layers.ReLU()(a)
    a = NoisyDense(num_actions, name="adv")(a)

    # Combine: Q = V + (A - mean(A))
    a_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(a)
    q_values = tf.keras.layers.Add()([v, tf.keras.layers.Subtract()([a, a_mean])])

    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss='mse'
    )
    return model

###############################################################################
#                           ENVIRONMENT ENTITIES                               #
###############################################################################
class Food:
    def __init__(self, special=False):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["special_food"]["size"] if special else settings["food"]["size"]
        self.color = settings["special_food"]["color"] if special else settings["food"]["color"]

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

class DangerZone:
    def __init__(self):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["danger_zone"]["size"]
        self.color = settings["danger_zone"]["color"]

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

    def is_in_zone(self, agent_x, agent_y):
        return np.hypot(agent_x - self.x, agent_y - self.y) < self.size

class RestZone:
    def __init__(self):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["rest_zone"]["size"]
        self.color = settings["rest_zone"]["color"]

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

    def is_in_zone(self, agent_x, agent_y):
        return np.hypot(agent_x - self.x, agent_y - self.y) < self.size

class GoalZone:
    def __init__(self):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["goal_zone"]["size"]
        self.color = settings["goal_zone"]["color"]

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

    def is_in_zone(self, agent_x, agent_y):
        return np.hypot(agent_x - self.x, agent_y - self.y) < self.size

class DynamicObstacle:
    def __init__(self):
        self.x = random.uniform(0, screen_width)
        self.y = random.uniform(0, screen_height)
        self.size = settings["dynamic_obstacle"]["size"]
        self.color = settings["dynamic_obstacle"]["color"]
        self.speed = settings["dynamic_obstacle"]["speed"]
        self.direction = np.random.uniform(-1, 1, 2)  # Random direction vector

    def draw(self):
        circle = plt.Circle((self.x, self.y), self.size, color=self.color)
        ax.add_patch(circle)

    def move(self):
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed

        # Bounce off screen edges
        if self.x < 0 or self.x > screen_width:
            self.direction[0] *= -1
        if self.y < 0 or self.y > screen_height:
            self.direction[1] *= -1

    def is_in_zone(self, agent_x, agent_y):
        return np.hypot(agent_x - self.x, agent_y - self.y) < self.size

###############################################################################
#                      ACTION INDICATOR FOR VISUAL FEEDBACK                   #
###############################################################################
class ActionIndicator:
    """
    A small label or marker to visualize a mouse click or keypress near the agent.
    """
    def __init__(self, x, y, text="", color="yellow", duration=1.0):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.duration = duration
        self.start_time = time.time()

    def is_expired(self):
        return (time.time() - self.start_time) >= self.duration

    def draw(self):
        # Draw text at the specified location
        ax.text(self.x, self.y, self.text, color=self.color, fontsize=9, ha='center')

###############################################################################
#                            AGENT CLASS                                       #
###############################################################################
class Agent:
    def __init__(self, num_actions=12):
        """
        Actions:
         0: Move right
         1: Move left
         2: Move down
         3: Move up
         4: Left-click
         5: Right-click
         6: Middle-click
         7: Press 'a'
         8: Press 'b'
         9: Press 'esc'
         10: Press 'ctrl+shift+c'
         11: Press 'enter'
        """
        self.num_actions = num_actions
        self.energy = settings["agent"]["energy"]
        self.memory = deque(maxlen=settings["memory_size"])
        self.epsilon = settings["epsilon"]
        # Start near center of screen
        self.position = np.array([screen_width * 0.5, screen_height * 0.5])

        # Create Dueling DQN models
        self.model = build_dueling_dqn((84, 84, 3), num_actions)
        self.target_model = build_dueling_dqn((84, 84, 3), num_actions)
        self.target_model.set_weights(self.model.get_weights())

        # For periodic energy surges
        self.last_surge_time = time.time()

    def take_screenshot(self):
        """
        Capture an 84x84 region around the agent's position.
        """
        x, y = self.position
        half_side = 42
        left = int(max(0, x - half_side))
        top = int(max(0, y - half_side))
        right = int(min(screen_width, x + half_side))
        bottom = int(min(screen_height, y + half_side))

        bbox = (left, top, right, bottom)
        screenshot = ImageGrab.grab(bbox=bbox)
        screenshot = screenshot.resize((84, 84))
        screenshot = np.array(screenshot, dtype=np.float32) / 255.0
        return screenshot

    def act(self, state):
        """
        Epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Sample from memory and train the network (Double DQN style).
        """
        if len(self.memory) < settings["batch_size"]:
            return

        batch = random.sample(self.memory, settings["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Next Q from local model
        next_q_local = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_local, axis=1)

        # Next Q from target
        q_target_vals = self.target_model.predict(next_states, verbose=0)

        # Current Q
        target_f = self.model.predict(states, verbose=0)

        for i in range(settings["batch_size"]):
            a = next_actions[i]
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + settings["gamma"] * q_target_vals[i][a]
            target_f[i][actions[i]] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > settings["epsilon_min"]:
            self.epsilon *= settings["epsilon_decay"]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def handle_energy_surge(self):
        """
        Periodic automatic energy boost.
        """
        now = time.time()
        if (now - self.last_surge_time) >= settings["agent"]["energy_surge_interval"]:
            self.energy = min(self.energy + settings["agent"]["energy_surge_amount"], settings["agent"]["energy"])
            print(f"[ENERGY SURGE] Agent energy now {self.energy:.2f}")
            self.last_surge_time = now

    def step(self, foods, special_foods, rest_zones, danger_zones, goal_zone, dynamic_obstacles, indicators):
        """
        One iteration: decide action, move, compute reward, etc.
        """
        # Possibly boost energy
        self.handle_energy_surge()

        state = self.take_screenshot()
        action = self.act(state)
        old_pos = self.position.copy()

        # Execute action
        if action == 0:   # Move right
            self.position[0] += settings["agent"]["max_speed"]
        elif action == 1: # Move left
            self.position[0] -= settings["agent"]["max_speed"]
        elif action == 2: # Move down
            self.position[1] += settings["agent"]["max_speed"]
        elif action == 3: # Move up
            self.position[1] -= settings["agent"]["max_speed"]
        elif action == 4: # Left-click
            threading.Thread(target=pyautogui.click).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="L-Click", color="green",
                                              duration=settings["action_indicator_time"]))
        elif action == 5: # Right-click
            threading.Thread(target=pyautogui.rightClick).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="R-Click", color="orange",
                                              duration=settings["action_indicator_time"]))
        elif action == 6: # Middle-click
            threading.Thread(target=pyautogui.middleClick).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="M-Click", color="purple",
                                              duration=settings["action_indicator_time"]))
        elif action == 7: # Press 'a'
            threading.Thread(target=pyautogui.press, args=('a',)).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="Key:a", color="gray",
                                              duration=settings["action_indicator_time"]))
        elif action == 8: # Press 'b'
            threading.Thread(target=pyautogui.press, args=('b',)).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="Key:b", color="gray",
                                              duration=settings["action_indicator_time"]))
        elif action == 9: # Press 'esc'
            threading.Thread(target=pyautogui.press, args=('esc',)).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="Key:esc", color="gray",
                                              duration=settings["action_indicator_time"]))
        elif action == 10: # ctrl+shift+c
            try:
                threading.Thread(target=pyautogui.hotkey, args=('ctrl', 'shift', 'c')).start()
                indicators.append(ActionIndicator(self.position[0], self.position[1],
                                                  text="Ctrl+Shift+C", color="gray",
                                                  duration=settings["action_indicator_time"]))
            except Exception:
                pass
        elif action == 11: # Press 'enter'
            threading.Thread(target=pyautogui.press, args=('enter',)).start()
            indicators.append(ActionIndicator(self.position[0], self.position[1],
                                              text="Key:enter", color="gray",
                                              duration=settings["action_indicator_time"]))

        # Clamp agent to screen
        self.position[0] = np.clip(self.position[0], 0, screen_width)
        self.position[1] = np.clip(self.position[1], 0, screen_height)

        # Energy usage
        self.energy -= settings["agent"]["energy_decay"]
        if self.energy < 0:
            self.energy = 0

        # Reward shaping
        reward = 0.0

        # Movement-based exploration
        dist_moved = np.hypot(self.position[0] - old_pos[0], self.position[1] - old_pos[1])
        reward += settings["reward"]["exploration"] * dist_moved

        # Collision with normal food
        for fd in foods[:]:
            if np.hypot(fd.x - self.position[0], fd.y - self.position[1]) < fd.size:
                reward += settings["reward"]["task_completion"]
                self.energy = min(self.energy + 20, settings["agent"]["energy"])
                foods.remove(fd)

        # Collision with special food
        for sfd in special_foods[:]:
            if np.hypot(sfd.x - self.position[0], sfd.y - self.position[1]) < sfd.size:
                reward += settings["reward"]["special_food"]
                self.energy = min(self.energy + 30, settings["agent"]["energy"])
                special_foods.remove(sfd)

        # Check rest zones
        for rz in rest_zones:
            if rz.is_in_zone(self.position[0], self.position[1]):
                self.energy = min(self.energy + 0.5, settings["agent"]["energy"])

        # Check danger zones
        for dz in danger_zones:
            if dz.is_in_zone(self.position[0], self.position[1]):
                reward += settings["reward"]["danger_penalty"]

        # Check dynamic obstacles
        for obstacle in dynamic_obstacles:
            if obstacle.is_in_zone(self.position[0], self.position[1]):
                reward += settings["reward"]["danger_penalty"]

        # Check goal zone
        if goal_zone.is_in_zone(self.position[0], self.position[1]):
            reward += settings["reward"]["goal_reward"]
            print("Goal reached! Simulation ended.")
            return False, reward

        next_state = self.take_screenshot()
        done = (self.energy <= 0)

        self.remember(state, action, reward, next_state, done)
        self.replay()

        return not done, reward

###############################################################################
#                           MAIN SIMULATION LOOP                               #
###############################################################################
def run_simulation():
    # Create an agent
    agent = Agent(num_actions=12)

    # Create environment entities
    foods = [Food() for _ in range(settings["food"]["count"])]
    special_foods = [Food(special=True) for _ in range(settings["special_food"]["count"])]
    rest_zones = [RestZone()]  # Add more if desired
    danger_zones = [DangerZone() for _ in range(settings["danger_zone"]["count"])]
    goal_zone = GoalZone()
    dynamic_obstacles = [DynamicObstacle() for _ in range(settings["dynamic_obstacle"]["count"])]
    
    total_reward = 0.0
    last_spawn_time = time.time()
    update_target_counter = 0

    # Indicators for visual feedback (clicks, keys)
    indicators = []

    try:
        while True:
            ax.clear()
            ax.set_xlim(0, screen_width)
            ax.set_ylim(0, screen_height)

            # Plot title = status
            ax.set_title(
                f"Energy: {agent.energy:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Reward: {total_reward:.2f}"
            )

            # Draw environment
            for f in foods:
                f.draw()
            for sf in special_foods:
                sf.draw()
            for rz in rest_zones:
                rz.draw()
            for dz in danger_zones:
                dz.draw()
            for obstacle in dynamic_obstacles:
                obstacle.draw()
                obstacle.move()  # Move dynamic obstacles
            goal_zone.draw()

            # Draw agent
            energy_ratio = agent.energy / settings["agent"]["energy"]
            # Color from red (low) to green (high)
            agent_color = (1 - energy_ratio, energy_ratio, 0)
            ax.add_patch(plt.Circle((agent.position[0], agent.position[1]), 10, color=agent_color))

            # Update indicators
            for ind in indicators[:]:
                if ind.is_expired():
                    indicators.remove(ind)
                else:
                    ind.draw()

            # Step the agent
            alive, step_reward = agent.step(foods, special_foods, rest_zones, danger_zones, goal_zone, dynamic_obstacles, indicators)
            total_reward += step_reward
            if not alive:
                print("Agent has run out of energy or ended episode.")
                break

            # Spawn new food periodically
            if (time.time() - last_spawn_time) > settings["spawn_rate"]:
                foods.append(Food())
                last_spawn_time = time.time()

            # Update target network periodically
            update_target_counter += 1
            if update_target_counter % 100 == 0:
                agent.update_target_model()

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    print("Simulation ended.")
    print(f"Total reward accumulated: {total_reward:.2f}")

###############################################################################
#                           SCRIPT ENTRY POINT                                 #
###############################################################################
if __name__ == "__main__":
    run_simulation()
