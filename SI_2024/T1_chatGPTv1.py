import pygame
import numpy as np
import random

# Environment setup using Pygame
class GridEnv:
    def __init__(self, grid_size, start, target, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.target = target
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        new_pos = list(self.pos)

        if action == 'up':
            new_pos[0] = max(0, self.pos[0] - 1)
        elif action == 'down':
            new_pos[0] = min(self.grid_size - 1, self.pos[0] + 1)
        elif action == 'left':
            new_pos[1] = max(0, self.pos[1] - 1)
        elif action == 'right':
            new_pos[1] = min(self.grid_size - 1, self.pos[1] + 1)

        if tuple(new_pos) in self.obstacles:
            new_pos = self.pos

        reward = -1  # penalty for each move
        done = False

        if tuple(new_pos) == self.target:
            reward = 10  # reward for reaching the target
            done = True

        self.pos = tuple(new_pos)
        return self.pos, reward, done

    def possible_actions(self):
        return ['up', 'down', 'left', 'right']

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.possible_actions())
        else:
            return self.get_best_action(state)

    def get_best_action(self, state):
        q_values = self.q_table.get(state, {a: 0 for a in self.env.possible_actions()})
        return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        q_values = self.q_table.get(state, {a: 0 for a in self.env.possible_actions()})
        next_best_action = self.get_best_action(next_state)
        q_values[action] = q_values[action] + self.alpha * (
            reward + self.gamma * self.q_table.get(next_state, {a: 0 for a in self.env.possible_actions()})[next_best_action] - q_values[action]
        )
        self.q_table[state] = q_values

    def train(self, episodes):
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

# Pygame setup
def draw_grid(screen, env, grid_size, cell_size):
    for x in range(0, grid_size * cell_size, cell_size):
        for y in range(0, grid_size * cell_size, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            grid_pos = (y // cell_size, x // cell_size)
            if grid_pos == env.start:
                pygame.draw.rect(screen, (0, 0, 255), rect)  # Pacman start position
            elif grid_pos == env.target:
                pygame.draw.rect(screen, (0, 255, 0), rect)  # Target position
            elif grid_pos in env.obstacles:
                pygame.draw.rect(screen, (255, 0, 0), rect)  # Obstacles
            elif grid_pos == env.pos:
                pygame.draw.rect(screen, (255, 255, 0), rect)  # Pacman current position

def main():
    grid_size = 5
    cell_size = 100
    start = (0, 0)
    target = (4, 4)
    obstacles = [(1, 1), (2, 2), (3, 3)]

    env = GridEnv(grid_size, start, target, obstacles)
    agent = QLearningAgent(env)

    pygame.init()
    screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
    pygame.display.set_caption("Pacman Q-learning")
    clock = pygame.time.Clock()

    episodes = 1000
    for episode in range(episodes):
        env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            state = env.pos
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)

            screen.fill((0, 0, 0))
            draw_grid(screen, env, grid_size, cell_size)
            pygame.display.flip()
            clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
