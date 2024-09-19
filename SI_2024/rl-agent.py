import pygame
import numpy as np
import numpy.ma as ma
import random

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 7
CELL_SIZE = 80
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
MANUAL_MODE_FPS = 30
AUTO_MODE_FPS = 120

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# RL parameters
LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Set to True for SARSA, False for Q-learning
USE_SARSA = False

class GridWorld:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.negative = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.agent = self.start
        # Initialize Q-values
        self.q_values = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        # Mask for visited state-action pairs (True for unvisited)
        self.unvisited_mask = np.ones((GRID_SIZE, GRID_SIZE, 4), dtype=bool)

    def reset(self):
        self.agent = self.start
        return self.agent

    def step(self, intended_action):
        # 80% chance to take the intended action, 20% chance for a random action
        if random.random() < 0.8:
            action = intended_action
        else:
            action = random.randint(0, 3)

        if action == 0:  # Up
            new_pos = (max(0, self.agent[0] - 1), self.agent[1])
        elif action == 1:  # Right
            new_pos = (self.agent[0], min(GRID_SIZE - 1, self.agent[1] + 1))
        elif action == 2:  # Down
            new_pos = (min(GRID_SIZE - 1, self.agent[0] + 1), self.agent[1])
        else:  # Left
            new_pos = (self.agent[0], max(0, self.agent[1] - 1))

        self.agent = new_pos

        if self.agent == self.goal:
            return self.agent, 1, True
        elif self.agent == self.negative:
            return self.agent, -1, True
        else:
            return self.agent, 0, False

    def get_max_q(self, state):
        masked_q_values = ma.masked_array(self.q_values[state[0], state[1]], mask=self.unvisited_mask[state[0], state[1]])
        if masked_q_values.count() == 0:  # All actions are unvisited
            return 0
        return ma.max(masked_q_values)

    def get_best_action(self, state):
        masked_q_values = ma.masked_array(self.q_values[state[0], state[1]], mask=self.unvisited_mask[state[0], state[1]])
        if masked_q_values.count() == 0:  # All actions are unvisited
            return np.random.randint(4)  # Choose randomly if all actions are unvisited
        return ma.argmax(masked_q_values)

    def update_q_value(self, state, action, reward, next_state, next_action=None):
        self.unvisited_mask[state[0], state[1], action] = False
        current_q = self.q_values[state[0], state[1], action]
        
        if USE_SARSA:
            next_q = self.q_values[next_state[0], next_state[1], next_action] if next_action is not None else 0
        else:
            next_q = self.get_max_q(next_state)
        
        self.q_values[state[0], state[1], action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - current_q)

def select_action(env, state, epsilon):
    '''
    Epsilon Greedy Policy
    '''
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return env.get_best_action(state)

def get_color_from_value(value, is_unvisited):
    if is_unvisited:
        return (200, 200, 200)  # Light gray for unvisited
    
    # Normalize value to be between -1 and 1
    normalized_value = max(min(value, 1), -1)
    
    if normalized_value >= 0:
        # Positive values: from white (0) to blue (1)
        blue = 255
        red = green = int(255 * (1 - normalized_value))
    else:
        # Negative values: from white (0) to red (-1)
        red = 255
        blue = green = int(255 * (1 + normalized_value))
    
    return (red, green, blue)

def draw_grid(screen, env):
    screen.fill(WHITE)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Draw colored triangles for each action
            for action in range(4):
                q_value = env.q_values[i, j, action]
                is_unvisited = env.unvisited_mask[i, j, action]
                color = get_color_from_value(q_value, is_unvisited)
                draw_triangle(screen, i, j, action, color)

    # Draw grid lines
    GRID_COLOR = BLACK  # Changed to black for better visibility
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRID_COLOR, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))
        pygame.draw.line(screen, GRID_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))

    # Draw X inside each cell
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (j * CELL_SIZE, i * CELL_SIZE), 
                             ((j + 1) * CELL_SIZE, (i + 1) * CELL_SIZE))
            pygame.draw.line(screen, GRID_COLOR, ((j + 1) * CELL_SIZE, i * CELL_SIZE), 
                             (j * CELL_SIZE, (i + 1) * CELL_SIZE))

    # Draw start, goal, and negative states
    pygame.draw.rect(screen, GREEN, (env.start[1] * CELL_SIZE, env.start[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, BLUE, (env.goal[1] * CELL_SIZE, env.goal[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, (env.negative[1] * CELL_SIZE, env.negative[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw agent
    pygame.draw.circle(screen, BLACK, (env.agent[1] * CELL_SIZE + CELL_SIZE // 2, env.agent[0] * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

def draw_triangle(screen, i, j, action, color):
    x, y = j * CELL_SIZE, i * CELL_SIZE
    if action == 0:  # Up
        pygame.draw.polygon(screen, color, [(x, y), (x + CELL_SIZE, y), (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])
    elif action == 1:  # Right
        pygame.draw.polygon(screen, color, [(x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])
    elif action == 2:  # Down
        pygame.draw.polygon(screen, color, [(x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])
    else:  # Left
        pygame.draw.polygon(screen, color, [(x, y), (x, y + CELL_SIZE), (x + CELL_SIZE // 2, y + CELL_SIZE // 2)])

def main():
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Grid World RL " + ("SARSA" if USE_SARSA else "Q-Learning"))
    clock = pygame.time.Clock()

    env = GridWorld()
    state = env.reset()
    action = select_action(env, state, EPSILON)  # Select initial action
    auto_mode = False

    prev_state, prev_action = None, None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                elif not auto_mode:
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_LEFT:
                        action = 3

        if auto_mode:
            action = select_action(env, state, EPSILON)

        if action is not None:
            next_state, reward, done = env.step(action)

            if prev_state is not None and prev_action is not None:
                env.update_q_value(prev_state, prev_action, reward, state, action)

            prev_state, prev_action = state, action
            state = next_state

            if done:
                # Final update for the terminal state
                env.update_q_value(prev_state, prev_action, reward, state, None)
                
                state = env.reset()
                action = select_action(env, state, EPSILON)
                prev_state, prev_action = None, None

            if not auto_mode:
                action = None  # Reset action for manual mode

        draw_grid(screen, env)
        pygame.display.flip()
        clock.tick(AUTO_MODE_FPS if auto_mode else MANUAL_MODE_FPS)

    pygame.quit()

if __name__ == "__main__":
    main()