import numpy as np
import random
from typing import Dict, Tuple, List


class GridWorld:
   

    def __init__(self, width: int = 5, height: int = 5):
        self.width = width
        self.height = height

        # Special states and their transitions
        self.special_states = {'A': (0, 1), 'B': (0, 3)}
        self.next_states = {'A': (4, 1), 'B': (2, 3)}
        self.special_rewards = {'A': 10, 'B': 5}

        # Define actions: north, south, east, west
        self.actions = ['north', 'south', 'east', 'west']
        self.action_effects = {
            'north': (-1, 0),
            'south': (1, 0),
            'east': (0, 1),
            'west': (0, -1)
        }

        # Initialize Q-table with zeros
        self.q_table = np.zeros((height, width, len(self.actions)))

        # For visualization
        self.arrow_map = {
            'north': '↑',
            'south': '↓',
            'east': '→',
            'west': '←'
        }

    def is_special_state(self, state: Tuple[int, int]) -> bool:
        """Check if the current state is a special state (A or B)"""
        return state in self.special_states.values()

    def get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float]:
        
        i, j = state

        # Check for special states A and B
        if self.is_special_state(state):
            if state == self.special_states['A']:
                return self.next_states['A'], self.special_rewards['A']
            elif state == self.special_states['B']:
                return self.next_states['B'], self.special_rewards['B']

        # Apply the action
        di, dj = self.action_effects[action]
        new_i, new_j = i + di, j + dj

        # Check if out of bounds
        if new_i < 0 or new_i >= self.height or new_j < 0 or new_j >= self.width:
            return (i, j), -1  # Stay in same position, reward -1

        return (new_i, new_j), 0  # Normal move, reward 0

    def q_learning(self, gamma: float = 0.9, epsilon: float = 0.1,
                   alpha: float = 0.2, episodes: int = 5000, steps: int = 5000):
        
        for episode in range(episodes):
            # Start from a random state
            state = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))

            for step in range(steps):
                # Choose action using epsilon-greedy policy
                if random.random() < epsilon:
                    action = random.choice(self.actions)
                else:
                    action_idx = np.argmax(self.q_table[state[0], state[1]])
                    action = self.actions[action_idx]

                # Take action and observe next state and reward
                next_state, reward = self.get_next_state(state, action)

                # Update Q-table
                current_q = self.q_table[state[0], state[1], self.actions.index(action)]
                max_next_q = np.max(self.q_table[next_state[0], next_state[1]])

                new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
                self.q_table[state[0], state[1], self.actions.index(action)] = new_q

                state = next_state

                # Early termination if in terminal state (though GridWorld is continuing)
                if self.is_special_state(state):
                    break

    def get_optimal_value_function(self) -> np.ndarray:
        """Extract the optimal value function from the Q-table"""
        return np.max(self.q_table, axis=2)

    def get_optimal_policy(self) -> np.ndarray:
        """Extract the optimal policy from the Q-table"""
        policy = np.empty((self.height, self.width), dtype=object)
        for i in range(self.height):
            for j in range(self.width):
                best_action_idx = np.argmax(self.q_table[i, j])
                policy[i, j] = self.actions[best_action_idx]
        return policy

    def print_results(self):
        """Print the optimal value function and policy"""
        print("Optimal Value Function:")
        value_function = self.get_optimal_value_function()
        for row in value_function:
            print("  ".join(f"{val:5.2f}" for val in row))

        print("\nOptimal Policy:")
        policy = self.get_optimal_policy()
        for row in policy:
            print("  ".join(f"{action:6}" for action in row))

        print("\nOptimal Policy (arrows):")
        for row in policy:
            print("  ".join(self.arrow_map[action] for action in row))


def main():
    print("Initializing Gridworld...")
    grid = GridWorld()
    print(f"Grid size: {grid.height}x{grid.width}")
    print(f"Special_states: {grid.special_states}")
    print(f"Next_states: {grid.next_states}")
    print(f"Special_rewards: {grid.special_rewards}")

    print("\nStarting Q-learning with parameters:")
    print(f"γ (discount factor): 0.9")
    print(f"ε (exploration rate): 0.1")
    print(f"α (learning rate): 0.2")
    print(f"Episodes: 5000")
    print(f"Steps per episode: 5000\n")

    # Run Q-learning
    grid.q_learning()

    print("Evaluating optimal value function and policy...")
    grid.print_results()

    # Compare with the expected results from the assignment
    expected_values = np.array([
        [22.0, 24.4, 22.0, 19.4, 17.5],
        [19.8, 22.0, 19.8, 17.8, 16.0],
        [17.8, 19.8, 17.8, 16.0, 14.4],
        [16.0, 17.8, 16.0, 14.4, 13.0],
        [14.4, 16.0, 14.4, 13.0, 11.7]
    ])

    print("\nComparing with expected optimal values:")
    optimal_values = grid.get_optimal_value_function()
    diff = np.abs(optimal_values - expected_values)
    print("\nDifference between computed and expected values:")
    for row in diff:
        print("  ".join(f"{val:5.2f}" for val in row))

    print("\nMean absolute difference:", np.mean(diff))
    print("Maximum difference:", np.max(diff))


if __name__ == "__main__":
    main()