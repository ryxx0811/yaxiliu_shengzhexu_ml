import os
import pickle
import random
import numpy as np
from collections import deque
from random import shuffle
import tensorflow as tf
from tensorflow.python.keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 0.995
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5
METHODS = 'SARSA'
FILENAME = "self_agent"
TRANSITION_HISTORY_SIZE = 1000
BATCH_SIZE = 500
FEATURE_NUM = 8
NUM_EPISODES = 10
input_shape = (FEATURE_NUM,)
num_actions = len(ACTIONS)
epsilon = 0.1
gamma = 0.9
learning_rate = 0.01
max_step = 400


class SARSA:
    def __init__(self, input_shape, num_actions, epsilon, gamma, learning_rate):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.q_network = self.build_q_network()
        self.optimizer = RMSprop(learning_rate=self.learning_rate)
        self.transition_memory = deque(maxlen=TRANSITION_HISTORY_SIZE)
        self.loss_history = []

    def build_q_network(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(10, activation='relu')) # input_shape=self.input_shape
        model.add(layers.Dense(self.num_actions, activation=None))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.q_network.predict(state)
            return np.argmax(q_values)

    def store_transition(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.transition_memory.append(transition)

    def train(self):
        if len(self.transition_memory) < BATCH_SIZE:
            return
        batch = np.random.choice(self.transition_memory, BATCH_SIZE, replace=False)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        q_values = self.q_network.predict(np.vstack(state_batch))
        next_q_values = self.q_network.predict(np.vstack(next_state_batch))
        for i in range(BATCH_SIZE):
            if done_batch[i]:
                q_values[i, action_batch[i]] = reward_batch[i]
            else:
                q_values[i, action_batch[i]] = reward_batch[i] + self.gamma * next_q_values[i, action_batch[i]]
        self.q_network.fit(np.vstack(state_batch), q_values, verbose=0)

        loss = self.q_network.train_on_batch(np.vstack(state_batch), q_values)
        self.loss_history.append(loss)

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon -= 1e-6

    def update(self):
        self.train()
        self.update_epsilon()

    def plot_loss_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(float(len(self.loss_history)), self.loss_history, label='Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss History')
        plt.show()


sarsa_agent = SARSA(input_shape, num_actions, epsilon, gamma, learning_rate)


def setup(self):
    self.actions = ACTIONS
    self.methods = METHODS
    # set the agent model as untrained
    self.trained_model = False
    if self.train or not os.path.isfile(f"{FILENAME}.pt"):
        self.logger.info("Setting up model from scratch.")
        # Initializing model
        self.model = SARSA(num_actions=len(self.actions))
        self.trained_model = False
    else:
        self.logger.info("Loading model from saved state.")
        with open(f"{FILENAME}", "rb") as file:
            self.model = pickle.load(file)
        self.trained_model = True


def act(self, game_state: dict) -> str:
    state = state_to_features(game_state)
    action = self.sarsa_agent.select_action(state)
    return ACTIONS[action]


def state_to_features(game_state: dict) -> np.array:
    # The current state of the game world is passed in the dictionary game_state. It has the following entries:
    _, _, bombs_left, (x, y) = game_state['self']  # self:(str,int,bool,(int,int))
    field = game_state['field']  # field:np.array(width,height)
    bombs = game_state['bombs']  # bombs:[(int,int),int]
    explosion_map = game_state['explosion_map']  # explosion_map:np.array(width,height)
    coins = game_state['coins']  # coins:[(x,y)]
    others = game_state['others']  # others:[(str,int,bool, (int,int))]

    distance_to_crates = distance_crates(field,x,y)
    distance_to_coin = distance_coins(coins,x,y)
    distance_to_opponents = distance_others(others,x,y)
    features = np.concatenate((distance_to_crates, distance_to_coin, distance_to_opponents), axis=None)

    return features[np.newaxis, :]


def distance_crates(field,x,y):
    # calculate the distance from the agents to the crates
    #(x,y) is the agent coordinate
    crate_positions = np.argwhere(field==1)
    if len(crate_positions ) == 0:
        return np.zeros_like(field)

    distances = np.linalg.norm(crate_positions-np.array([x,y]), axis=1)
    min_distances = np.min(distances)
    return min_distances


def distance_coins(coins,x,y):

    if not coins:
        return np.zeros(0)

    coin_positions=np.array(coins)
    distances = np.linalg.norm(coin_positions-np.array([x,y]),axis=1)
    min_distances = np.min(distances)
    return min_distances


def distance_others(others,x,y):

    if not others:
        return np.zeros(0)

    opponent_positions = np.array([pos for _, _, _, pos in others])
    distances = np.linalg.norm(opponent_positions - np.array([x, y]), axis=1)
    min_distances = np.min(distances)
    return min_distances
#def next_position(x: int, y: int, action: str) -> (int, int):
    # action:up.down,left,right ;
    # 0 | 1 | 2 |
    # 1 |
    # 2 |
    # return next position
    #if action == 'UP':
    #    y = y - 1
    #elif action == 'DOWN':
    #    y = y + 1
    #elif action == 'LEFT':
    #    x = x - 1
    #elif action == 'RIGHT':
    #    x = x + 1
    #else:
    #    raise ValueError("incorrect action")
    #return x, y


#def type_of_field(x: int, y: int, field: np.array, types: str) -> bool:
    # the entries in field are 1 for crates,-1 for stone walls and 0 for free tiles
    #if types == 'crates':
    #    return field[x, y] == 1
    #elif types == 'stone_walls':
    #    return field[x, y] == -1
    #elif types == 'free_tiles':
    #    return field[x, y] == 0
    #else:
    #    raise ValueError("incorrect type of field")


#def explosion_coordinate(field, bombs, explosion_map):
    # Once a bomb is dropped, it will detonate after four steps and create an explosion that extends three
    # tiles up, down, left and right. The explosion destroys crates and agents, but will stop at stone
    # walls and does not reach around corners. The explosion remains
    # dangerous for one more round before it vanishes in smoke.
    #actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    #explode_coordinate = set()

    #def exp_area(x, y):
    #    for a in actions:
    #        n_x, n_y = next_position(x, y, a)
    #        while (0 <= n_x < len(field) and 0 <= n_y < len(field[0]) and not type_of_field(n_x, n_y, field, 'stone_walls')
    #               and abs(n_x - x) <= 3 and abs(n_y - y) <= 3):
    #            explode_coordinate.add((n_x, n_y))
    #            n_x, n_y = next_position(n_x, n_y, a)

    #if bombs:
    #    for (nx, ny), _ in bombs:
    #        explode_coordinate.add((nx, ny))
    #        exp_area(nx, ny)

    #for i in range(np.shape(explosion_map)[0]):
    #    for j in range(np.shape(explosion_map)[1]):
    #        if explosion_map[i, j] == 1:
    #            explode_coordinate.add((i, j))

    #return np.array(list(explode_coordinate))


#def find_object(begin_cor, available_way, object_cor):
    # we are using the Manhattan distance to all objects and find the minimum distance
    #queue = deque()
    #queue.append(begin_cor)

    # create a dictionary to store the parent of each visited tile
    #parent_dict = {begin_cor: None}
    #best_direction = None
    #best_distance = float('inf')

    #if len(object_cor) == 0:
    #    return None

    #while queue:
    #    current = queue.popleft()

    #    distances = [abs(ox - current[0]) + abs(oy - current[1]) for ox, oy in object_cor]
    #    min_dist = min(distances)

    #    if min_dist < best_distance:
    #        best_distance = min_dist
    #        best_direction = parent_dict[current]
    #    if min_dist == 0:
    #        break
        # get the neighboring positions
    #    neighbors = [(current[0]+dx, current[1]+dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
    #    random.shuffle(neighbors)

    #    for neighbor in neighbors:
    #        x, y = neighbor
    #        if (
    #             0 <= x < available_way[0] and
    #             0 <= y < available_way[1] and
    #             available_way[x][y] and
    #             neighbor not in parent_dict
    #        ):
    #            queue.append(neighbor)
    #            parent_dict[neighbor] = current

    #return best_direction


#def if_in_explosion_area(explode_coordinate, x: int, y: int) -> int:
#    return int((x, y) in explode_coordinate)


#def find_crates(field:np.array, x:int,y:int,explode_coordinate, available_way, bombs: list, bombs_left:bool) ->(int, bool):
    # find the nearest crate and return the action to towards it.
    # first get the locations of crates that are not in the explore_coordinate
    #crate_positions = np.argwhere(field == 1) & ~np.in1d(tuple(map(tuple, np.argwhere(field == 1))), explode_coordinate)
    #direction = find_object((x, y), available_way, crate_positions)
    #if direction is None:
    #    return WAIT
    #direction_x, direction_y = direction
    # calculate the direction towards the nearest crate
    #dx, dy = direction_x - x, direction_y - y
    #if dx == 0:
    #    if dy < 0:
    #        return UP
    #    elif dy > 0:
    #        return DOWN
    #elif dy == 0:
    #    if dx < 0:
    #        return LEFT
    #    elif dx > 0:
    #        return RIGHT
    # if the agent reached the crate position, check if it can place a bomb
    #if dx == 0 and dy == 0:
    #    reach_crate = True
    #    if bombs_left:
    #        return BOMB
    #    else:
    #        return WAIT
    #return WAIT  # no specific direction found


#def find_coins(field:np.array, x:int,y:int,explode_coordinate, available_way, coins: list,bombs:list, bombs_left:bool) -> int:
    # find the coins that are not in the explosion area
    #reachable_coins = [coin for coin in coins if coin not in explode_coordinate]
    #direction = find_object((x, y), available_way, reachable_coins)
    #direction_x, direction_y = direction
    # calculate the direction towards the nearest crate
    #dx, dy = direction_x - x, direction_y - y
    #if dx == 0:
    #    if dy < 0:
    #        return UP
    #    elif dy > 0:
    #        return DOWN
    #elif dy == 0:
    #    if dx < 0:
    #        return LEFT
    #    elif dx > 0:
    #        return RIGHT
    #else:
        # if not in a explosion area and no reachable coins,bombing obstacles or wait
        #if any(field[x+dx][y+dy] == 1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]):
        #    return BOMB
        #else:
            #return WAIT


#def find_opponent(field: np.array, x: int, y: int, explode_coordinate, available_way, others: list, bombs: list, bombs_left: bool) -> (int,bool):
    #opponent_positions = [(xy[0], xy[1]) for _, _, b, xy in others if if_in_explosion_area(explode_coordinate, xy[0], xy[1]) == 0]
    #nearest_opponent = find_object((x, y), available_way, opponent_positions)
    # assign the value of True to this two variables are same
    #directions = [(0, -1, UP), (0, 1, DOWN), (-1, 0, LEFT), (1, 0, RIGHT)]
    #for dx, dy, a in directions:
    #    if nearest_opponent == (x+dx, y+dy):
    #        return a

    #if any(field[x+dx][y+dy] == 1 for dx, dy, _ in directions) :
    #    if bombs_left:
    #        return WAIT


sarsa_agent.plot_loss_history()