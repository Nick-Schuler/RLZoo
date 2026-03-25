import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO, A2C
from wandb.integration.sb3 import WandbCallback


# --- 1. Environment Definition ---
class Maze5x5(gym.Env):
    def __init__(self, is_deep=False):
        super().__init__()
        self.size = 5
        self.is_deep = is_deep
        # Deep RL needs a Box (vector) or it will crash; Tabular needs Discrete
        if is_deep:
            self.observation_space = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Discrete(self.size * self.size)

        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0  # Start at (0,0)
        return self._get_obs(), {}

    def _get_obs(self):
        if self.is_deep:
            return np.array(divmod(self.state, self.size), dtype=np.float32)
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.size)
        if action == 0 and row > 0:
            row -= 1  # Up
        elif action == 1 and col < self.size - 1:
            col += 1  # Right
        elif action == 2 and row < self.size - 1:
            row += 1  # Down
        elif action == 3 and col > 0:
            col -= 1  # Left

        self.state = row * self.size + col
        done = (self.state == 24)  # Goal at bottom-right
        reward = 10.0 if done else -0.1
        return self._get_obs(), reward, done, False, {}


# --- 2. Tabular Algorithms (Q-Learning & SARSA) ---
def run_tabular(algo_name):
    env = Maze5x5(is_deep=False)
    wandb.init(project="maze-5x5-benchmark", name=algo_name)

    q_table = np.zeros((25, 4))
    alpha, gamma, epsilon = 0.1, 0.95, 0.1

    for ep in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Action selection for SARSA (on-policy)
        action = env.action_space.sample() if np.random.random() < epsilon else np.argmax(q_table[state])

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = env.action_space.sample() if np.random.random() < epsilon else np.argmax(q_table[next_state])

            # Update Rules
            if algo_name == "Q-Learning":
                q_table[state, action] += alpha * (
                            reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            else:  # SARSA
                q_table[state, action] += alpha * (
                            reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            state, action = next_state, next_action
            total_reward += reward

        wandb.log({"reward": total_reward, "episode": ep})
    wandb.finish()


# --- 3. Deep RL Algorithms (PPO & A2C) ---
def run_deep(algo_class, algo_name):
    env = Maze5x5(is_deep=True)
    run = wandb.init(project="maze-5x5-benchmark", name=algo_name, sync_tensorboard=True)

    model = algo_class("MlpPolicy", env, verbose=0, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=10000, callback=WandbCallback())
    run.finish()


# --- 4. Main Execution ---
if __name__ == "__main__":
    # Test Tabular
    run_tabular("Q-Learning")
    run_tabular("SARSA")

    # Test Deep RL
    run_deep(PPO, "PPO")
    run_deep(A2C, "A2C")
