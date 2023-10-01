import math
import os
import pickle
from queue import Queue
from random import shuffle

from torch import optim

import settings as s
import numpy as np
from collections import namedtuple, deque
import torch.nn as nn
import torch
import torch.nn.functional
import multiprocessing





ACTIONS=['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5

def setup(self):
    from .train import dqnnet
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model=dqnnet()
        checkpoint = torch.load('my_saved_model.pt')
        self.model.q_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(self.model.q_net.parameters(), lr=checkpoint['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("experience.pickle","rb") as file:
            self.model.transitions=pickle.load(file)

def act(self,game_state)-> str:
    feature = state_to_features(game_state)

    explosion_map = game_state['explosion_map']
    field = game_state['field']
    step = game_state['step']
    (x, y) = game_state['self'][3]
    hasbomb = game_state['self'][2]

    if np.random.uniform() < 0.99995**self.model.counter:

        p = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

        _action = np.random.choice(ACTIONS, p=p)

    else:

        q = self.model.action(feature).flatten()

        _action = ACTIONS[torch.argmax(q).item()]
    return _action


def state_to_features(game_state: dict) -> np.array:

    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    # For example, you could construct several channels of equal shape, ...
    channels = []

    field = game_state['field']
    bombs = game_state['bombs']
    bombs_map=np.zeros((s.ROWS,s.COLS))
    for bombs_coor,bombs_t in bombs:
        bombs_map[bombs_coor]=bombs_t
    coins_map=np.zeros((s.ROWS,s.COLS))
    coins = game_state['coins']
    for coin in coins:
        coins_map[coin]=1
    self=game_state['self']
    self_map=np.zeros((s.ROWS,s.COLS))
    _,_,hasbomb,coor=self
    self_map[coor]=1 if hasbomb else -1
    others_map=np.zeros((s.ROWS,s.COLS))
    others = game_state['others']
    for _,_,hasbomb,coor in others:
        others_map[coor]= 1 if hasbomb else -1
    explosion_map=game_state['explosion_map']

    feature = np.concatenate((field[np.newaxis, ...],
                                         explosion_map[np.newaxis, ...],
                                         bombs_map[np.newaxis, ...],
                                         coins_map[np.newaxis, ...],
                                         self_map[np.newaxis, ...],
                                         others_map[np.newaxis, ...]
                                         ), axis=0)
    return feature













def explosion_area(bombs,field,explosion_map):
    """
    Return the coordinates of all positions where there will be explosion according the bombs.
    """
    dangerous_area=[]
    for (bx, by), _ in bombs:
        dangerous_area.append((bx,by))
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for step in range(1, s.BOMB_POWER+1):
                new_x, new_y = bx + dx * step, by + dy * step

                # Check if the new position is within the game board boundaries
                if 0 <= new_x < s.COLS and 0 <= new_y < s.COLS:
                # If the new position is a wall, stop the blast in that direction
                    if field[new_x][new_y] ==-1:
                        break
                    dangerous_area.append((new_x, new_y))
                else:
                    break  # Stop if we reach the edge of the game board
    for x in range(s.COLS):
        for y in  range(s.ROWS):
            if explosion_map[x][y]!=0:
                dangerous_area.append((x,y))
    return dangerous_area


def safe_area_coordinates(field,explosion_map):
    safe_coordinates = []

    for x in range(s.COLS):
        for y in range(s.ROWS):
            if explosion_map[x][y] == 0 and field[x][y]==0:
                safe_coordinates.append((x, y))
    return safe_coordinates

def get_crate_coordinates(field):
    crate_coordinates = []

    for y in range(s.ROWS):
        for x in range(s.COLS):
            if field[x][y] == 1:
                crate_coordinates.append((x, y))

    return crate_coordinates
def is_in_explosion_area(position,bombs,field,explosion_map):
    """
    Given a position (x,y), check if it is in explosion area
    """
    if position in explosion_area(bombs,field,explosion_map):
        return True
    return False

def is_in_dangerous_area(self_position,explosion_map):
    (x,y)=self_position
    if explosion_map[x][y]==1:
        return True
    return False

def dis_path(field, self_coor, position):
    """
    Given a position(x,y), check if it can be reached in 4 steps.If yes, return distance, path and True.
    Otherwise, return 1000,None and False
    """
    visited = set()
    queue = deque([(self_coor, 0, [])])

    while queue:
        (x, y), distance, path = queue.popleft()
        visited.add((x, y))

        if (x, y) == position:

            return distance, path, True

        for dx, dy, direction in [(1, 0, 'RIGHT'), (-1, 0, 'LEFT'), (0, 1, 'DOWN'), (0, -1, 'UP')]:
            new_x, new_y = x + dx, y + dy
            if distance+1<=4:
                if (
                    0 <= new_x < len(field) and
                    0 <= new_y < len(field[0]) and
                    field[new_x][new_y] == 0 and
                    (new_x, new_y) not in visited
                ):
                    new_path = path + [direction]
                    queue.append(((new_x, new_y), distance + 1, new_path))

    return 1000, None, False

def to_nearest_safe_place(field,self_position,safe_places):
    """
    to find the nearest safe place
    Args:
        field: field
        self_position: current position of agent
        safe_places: list of positions where explosion_map[position]=0 and field[position]=0

    Returns: return the direction to nearest safe place including UP, RIGHT,DOWN,LEFT,WAIT

    """
    distances = []
    actions = []
    for safe_place in safe_places:
        dis,path,reachable=dis_path(field,self_position,safe_place)
        if reachable:
            distances.append(dis)
            if safe_place!=self_position:
                actions.append(path)
            else:
                actions.append(['WAIT'])
    if distances and len(actions)<=4:
        return actions[np.argmin(distances)][0]
    else:
        return 'WAIT'


def to_nearest_coin(field,self_position,coins):
    """

    Args:
        field: field
        self_position: current position of agent
        coins: coins not in explosion area


    Returns: if there are at least one coins ,return the direction to nearest coins
             if no coins can be reached return the direction to nearest crate

    """
    distances=[]
    actions=[]
    if not coins:
        return 'None'
    if coins:
        for coin in coins:
            distance,action,reachable=dis_path(field,self_position,coin)
            if reachable:
                distances.append(distance)
                actions.append(action)
    if distances:
        return actions[np.argmin(distances)][0]
    else:
        return 'None'


def dis_path_crates(field, self_coor, position):
    """
    Given a position(x,y), check if it can be reached in 5 steps.If yes, return distance, path and True.
    Otherwise, return 1000,None and False
    """
    x,y=self_coor
    visited = set()
    queue = deque([(self_coor, 0, [])])
    while queue:
        (x, y), distance, path = queue.popleft()
        visited.add((x, y))
        if (x, y) == position:
            return distance, path, True
        for dx, dy, direction in [(1, 0, 'RIGHT'), (-1, 0, 'LEFT'), (0, 1, 'DOWN'), (0, -1, 'UP')]:
            new_x, new_y = x + dx, y + dy
            if distance + 1 <= 4:
                if (
                    0 <= new_x < len(field) and
                    0 <= new_y < len(field[0]) and
                    field[new_x][new_y] !=-1 and
                    (new_x, new_y) not in visited
                ):
                    new_path = path + [direction]
                    queue.append(((new_x, new_y), distance + 1, new_path))

    return 1000, None, False



def to_nearest_crate(field,self_position,self_hasbomb,crates):
    """

    Args:
        field: field
        self_position: current position of agent
        self_hasbomb: if the agent has bomb
        crates: crates not in explosion area
        coins: coins not in explosion area
        coins_direction: direction to nearest coin
        others_direction: direction to nearest other

    Returns: return the direction to nearest crate

    """
    distances=[]
    paths=[]
    if not crates:
        return 'None'
    for crate in crates:
        distance,path,reachable=dis_path_crates(field,self_position,crate)
        if distance==1 and self_hasbomb:
            return 'BOMB'
        if distance==1 and not self_hasbomb:
            return 'WAIT'
        if reachable:
            distances.append(distance)
            paths.append(path)

    if distances:
        return paths[np.argmin(distances)][0]
    else:
        return 'None'


def to_nearest_others(field,self_position,self_hasbomb,others):
    """

    Args:
        field:
        self_position:
        self_hasbomb:


    Returns:

    """
    distances=[]
    paths=[]
    if not others:
        return 'None'
    if others:
        for _,_,has_bomb,other in others:

            if not has_bomb:
                distance, path, reachable = dis_path(field, self_position,other)
                if distance==1 and self_hasbomb:
                    return 'BOMB'
                if distance==1 and not self_hasbomb:
                    return 'WAIT'
                if reachable:
                    distances.append(distance)
                    paths.append(path)
    if distances:
        return paths[np.argmin(distances)][0]
    return 'None'

