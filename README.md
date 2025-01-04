Agent Mouse Simulation
This project simulates an AI-controlled mouse agent that interacts with the screen using pyautogui. The agent uses a convolutional neural network (CNN) to make decisions based on screen captures, and it learns through Q-learning. The simulation includes features like energy management, dynamic food spawning, reward systems, and real-time visualization.

Key Features:
Screen Capture: The agent captures a 100x100 region around the mouse cursor to make decisions.

Neural Network: A CNN processes the screenshot and predicts actions (move, click, etc.).

Q-Learning: The agent uses Q-values to decide actions and updates its neural network based on rewards.

Reward System: The agent is rewarded for clicking, right-clicking, and causing screen changes.

Energy Management: The agent loses energy over time and must click on food targets to replenish it.

Dynamic Environment: Food targets spawn randomly on the screen.

Real-Time Visualization: The simulation is visualized using matplotlib.

Use Cases:
AI-controlled automation.

Reinforcement learning experiments.

Interactive simulations.

Python Setup Guide
1. Prerequisites
Python 3.8 or higher.

pip (Python package manager).

2. Set Up a Virtual Environment (Optional but Recommended)
bash
Copy
# Create a virtual environment
python -m venv myenv

# Activate the virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
3. Install Required Packages
Install the required Python packages using pip:

bash
Copy
pip install tensorflow pyautogui numpy matplotlib pillow
4. Run the Simulation
Run the simulation script:

bash
Copy
python agent_mouse_full.py
5. Customize the Simulation
You can customize the simulation by modifying the settings dictionary in the script:

python
Copy
settings = {
    "agent": {"max_speed": 5, "energy": 100, "energy_decay": 0.1},
    "food": {"count": 5, "size": 20, "color": "green"},
    "spawn_rate": 2,  # How often new food spawns (in seconds)
    "reward": {"click": 10, "right_click": 20, "scroll": 5, "change": 50},
}
6. Troubleshooting
TensorFlow Installation Issues:

Ensure you have Python 3.8–3.11 (TensorFlow does not support Python 3.12+ yet).

If TensorFlow fails to install, try the CPU-only version:

bash
Copy
pip install tensorflow-cpu
PyAutoGUI Permissions:

On macOS, grant accessibility permissions to Python in System Preferences > Security & Privacy > Accessibility.

Matplotlib Backend Issues:

If the visualization window does not appear, install a different backend for matplotlib:

bash
Copy
pip install pyqt5
Then set the backend in your script:

python
Copy
import matplotlib
matplotlib.use('Qt5Agg')
7. Contributing
Contributions are welcome! If you'd like to contribute:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeature).

Open a pull request.

8. License
This project is licensed under the MIT License. See the LICENSE file for details.

9. Contact
For questions or feedback, open an issue on GitHub or contact kai9987kai@gmail.com

This README.md provides a clear description of the project and a step-by-step guide for setting it up. Let me know if you need further assistance! 🚀

