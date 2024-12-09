import numpy as np


class ForestFire:
    def __init__(self, size=20, density=0.6):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.initialize_forest(density)

    def initialize_forest(self, density):
        # Initialize trees randomly (0: Empty, 1: Tree)
        self.grid = np.random.choice(
            [0, 1], size=(self.size, self.size), p=[1 - density, density]
        )

        # Create 1-3 initial fire points near center
        center = self.size // 2
        num_fires = np.random.randint(1, 4)
        offset = int(self.size * 0.2)

        for _ in range(num_fires):
            fire_x = center + np.random.randint(-offset, offset + 1)
            fire_y = center + np.random.randint(-offset, offset + 1)
            fire_x = max(0, min(fire_x, self.size - 1))
            fire_y = max(0, min(fire_y, self.size - 1))
            self.grid[fire_x, fire_y] = 2  # 2: Burning

    def step(self):
        new_grid = self.grid.copy()
        fire_exists = False

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 2:  # Burning cell
                    new_grid[i, j] = 0  # Burns out
                elif self.grid[i, j] == 1:  # Tree cell
                    # Check four adjacent neighbors
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < self.size
                            and 0 <= nj < self.size
                            and self.grid[ni, nj] == 2
                        ):  # If neighbor is burning
                            if np.random.random() < 0.6:  # 60% chance to catch fire
                                new_grid[i, j] = 2
                                fire_exists = True
                                break

        self.grid = new_grid
        return fire_exists

    def run_simulation(self, max_steps=150):
        initial_state = self.grid.copy()

        steps = 0
        while steps < max_steps:
            if not self.step():  # Stop if no more fire
                break
            steps += 1

        # Calculate fraction of trees that burned
        initial_trees = np.sum(initial_state == 1)
        if initial_trees == 0:
            return initial_state, 0.0

        final_empty = np.sum(self.grid == 0)
        initial_empty = np.sum(initial_state == 0)
        trees_burned = final_empty - initial_empty
        burn_fraction = trees_burned / initial_trees

        return initial_state, burn_fraction


def generate_dataset(num_samples=1000, size=20, density_range=(0.3, 0.7)):
    initial_states = []
    burn_fractions = []

    # Create density range
    densities = np.concatenate(
        [
            np.random.uniform(0.3, 0.5, num_samples // 2),  # Lower range
            np.random.uniform(0.5, 0.9, num_samples // 2),  # Higher range
        ]
    )
    np.random.shuffle(densities)

    for i, density in enumerate(densities):
        forest = ForestFire(size=size, density=density)
        initial, fraction = forest.run_simulation()

        initial_states.append(initial)
        burn_fractions.append(fraction)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} samples")

    return np.array(initial_states), np.array(burn_fractions)
