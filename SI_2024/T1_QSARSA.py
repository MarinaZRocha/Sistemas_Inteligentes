import pygame
import numpy as np
import numpy.ma as ma
import random
import matplotlib.pyplot as plt

class DogEnv:
    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.treat = (grid_size - 1, grid_size - 1)
        self.sock = (grid_size // 2, grid_size // 2)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((grid_size, grid_size, 4))
        # Mask for visited state-action pairs (True for unvisited)
        self.unvisited_mask = np.ones((grid_size, grid_size, 4), dtype=bool)
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, user_action):
        # 80% chance to take the intended action, 20% chance for a random action
        if random.random() < 0.8:
            action = user_action
        else:
            action = random.randint(0, 3)

        new_pos = self.pos
        done = False
        # Correctly map actions based on indices
        if action == 0:  # Up
            new_pos = (max(0, self.pos[0] - 1), self.pos[1])
        elif action == 1:  # Down
            new_pos = (min(self.grid_size - 1, self.pos[0] + 1), self.pos[1])
        elif action == 2:  # Left
            new_pos = (self.pos[0], max(0, self.pos[1] - 1))
        elif action == 3:  # Right
            new_pos = (self.pos[0], min(self.grid_size - 1, self.pos[1] + 1))

        self.pos = new_pos

        if self.pos == self.treat:
            reward = 100  # Large reward for getting the treat
            done = True
            self.reset()  # Reset position to start after getting the treat
        elif self.pos == self.sock:
            reward = -10  # Larger penalty for hitting a sock
            done = True
            self.reset()  # Reset position to start after hitting a sock
        else:
            reward = -1  # Small penalty for each move to encourage efficiency

        return self.pos, reward, done

    def get_max_q(self, state):
        masked_q_values = ma.masked_array(self.q_table[state[0], state[1]], mask=self.unvisited_mask[state[0], state[1]])
        if masked_q_values.count() == 0:  # All actions are unvisited
            return 0
        return ma.max(masked_q_values)

    def get_best_action(self, state):
        masked_q_values = ma.masked_array(self.q_table[state[0], state[1]], mask=self.unvisited_mask[state[0], state[1]])
        if masked_q_values.count() == 0:  # All actions are unvisited
            return np.random.randint(4)  # Choose randomly if all actions are unvisited
        return ma.argmax(masked_q_values)

    def update_q_table(self, state, action, reward, next_state, alpha, gamma, mode, next_action=None):
        self.unvisited_mask[state[0], state[1], action] = False
        current_q = self.q_table[state[0], state[1], action]
        if mode == "SARSA":
            if next_action is not None:
                next_q = self.q_table[next_state[0], next_state[1], next_action]  
            else:
                next_q = 0
        else:
            next_q = self.get_max_q(next_state)

        self.q_table[state[0], state[1], action] += alpha * (reward + gamma * next_q - current_q)

def select_action(env, state, epsilon):
    '''Epsilon Greedy Policy'''
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return env.get_best_action(state)


def train_and_visualize(env, episodes=100):
    pygame.init()
    grid_size = env.grid_size
    cell_size = 100
    screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
    clock = pygame.time.Clock()
    rewards = []
    mode = "Q"

    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)

    images = {
        'dog': pygame.transform.scale(pygame.image.load('dog.png'), (cell_size, cell_size)),
        'treat': pygame.transform.scale(pygame.image.load('treat.png'), (cell_size, cell_size)),
        'sock': pygame.transform.scale(pygame.image.load('sock.png'), (cell_size, cell_size))
    }

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        action = select_action(env, state, env.epsilon)

        while not done:
            next_state, reward, done = env.step(action)
            next_action = select_action(env, next_state, env.epsilon) if mode == "SARSA" else None
            env.update_q_table(state, action, reward, next_state, env.alpha, env.gamma, mode)
            state = next_state
            action = next_action if mode == "SARSA" else select_action(env, state, env.epsilon)
            total_reward += reward

            screen.fill((255, 255, 255))  # White background
            draw_grid(screen, env, grid_size, cell_size, images)
            # Render episode number
            episode_text = font.render(f'Episode: {episode + 1}/{episodes}', True, (0, 0, 0))
            screen.blit(episode_text, (10, 470))  # Position the text at the bottom left corner
            pygame.display.flip()
            clock.tick(30)

        rewards.append(total_reward)
        if episode % 10 == 0:
            env.epsilon *= 0.95  # Decay epsilon

    pygame.quit()
    plot_rewards(rewards, mode)

def plot_rewards(rewards, mode):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title(f'Learning results - {mode}')
    plt.show()

def get_color_from_value(value, is_unvisited):
    if is_unvisited:
        return (200, 200, 200)  # Light gray for unvisited
    normalized_value = max(min(value, 1), -1)
    if normalized_value >= 0:
        green = 255
        red = blue = int(255 * (1 - normalized_value))
    else:
        red = 255
        blue = green = int(255 * (1 + normalized_value))
    return (red, green, blue)

def draw_grid(screen, env, grid_size, cell_size, images):
    for i in range(grid_size):
        for j in range(grid_size):
            for action in range(4):
                q_table = env.q_table[i, j, action]
                is_unvisited = env.unvisited_mask[i, j, action]
                color = get_color_from_value(q_table, is_unvisited)
                draw_triangle(screen, i, j, action, color, cell_size)

    for i in range(grid_size + 1):
        black = (0, 0, 0)
        pygame.draw.line(screen, black, (0, i * cell_size), (grid_size * cell_size, i * cell_size))
        pygame.draw.line(screen, black, (i * cell_size, 0), (i * cell_size, grid_size * cell_size))

    for i in range(grid_size):
        for j in range(grid_size):
            pygame.draw.line(screen, black, (j * cell_size, i * cell_size), 
                             ((j + 1) * cell_size, (i + 1) * cell_size))
            pygame.draw.line(screen, black, ((j + 1) * cell_size, i * cell_size), 
                             (j * cell_size, (i + 1) * cell_size))
    screen.blit(images['treat'], pygame.Rect(env.treat[1] * cell_size, env.treat[0] * cell_size, cell_size, cell_size))
    screen.blit(images['sock'], pygame.Rect(env.sock[1] * cell_size, env.sock[0] * cell_size, cell_size, cell_size))
    screen.blit(images['dog'], pygame.Rect(env.pos[1] * cell_size, env.pos[0] * cell_size, cell_size, cell_size))

def draw_triangle(screen, i, j, action, color, cell_size):
    x, y = j * cell_size, i * cell_size
    if action == 0:  # Up
        pygame.draw.polygon(screen, color, [(x, y), (x + cell_size, y), (x + cell_size // 2, y + cell_size // 2)])
    elif action == 1:  # Right
        pygame.draw.polygon(screen, color, [(x + cell_size, y), (x + cell_size, y + cell_size), (x + cell_size // 2, y + cell_size // 2)])
    elif action == 2:  # Down
        pygame.draw.polygon(screen, color, [(x, y + cell_size), (x + cell_size, y + cell_size), (x + cell_size // 2, y + cell_size // 2)])
    else:  # Left
        pygame.draw.polygon(screen, color, [(x, y), (x, y + cell_size), (x + cell_size // 2, y + cell_size // 2)])

def main():
    grid_size = 5
    alpha = 0.4
    gamma = 0.9
    epsilon = 0.1

    env = DogEnv(grid_size, alpha, gamma, epsilon)
    train_and_visualize(env)

if __name__ == "__main__":
    main()
