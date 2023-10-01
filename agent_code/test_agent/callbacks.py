import os
import pickle

from torch import optim

import settings as s
import numpy as np
from collections import namedtuple, deque
import torch.nn as nn
import torch
import torch.nn.functional
import random


ACTIONS=['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5

def setup(self):
    from .train import dqnnet
    if self.train and not os.path.isfile("test-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model=dqnnet()
    else:
        self.logger.info("Loading model from saved state.")

        with open('test-model.pt','rb') as file:
            self.model = torch.load(file,map_location='cpu')

        #with open("experience.pickle","rb") as file:
            #self.model.transitions=pickle.load(file)


def act(self,game_state)-> str:

    feature=state_to_features(game_state)

    #print(feature)
    #print(len(feature))
    round=game_state['round']
    explosion_map=game_state['explosion_map']

    field = game_state['field']
    step=game_state['step']
    (x, y) = game_state['self'][3]
    explosion_map[x][y] = 3
    hasbomb = game_state['self'][2]
    ##########filter
    if np.random.uniform() <=0.99995**round:
        p=[.2,.2,.2,.2,.1,.1]
        if field[x][y-1]==-1:
            p[0]=0
        if field[x+1][y]==-1:
            p[1]=0
        if field[x][y+1]==-1:
            p[2]=0
        if field[x-1][y]==-1:
            p[3]=0

        if explosion_map[x][y]!=0:
            p[4]=0

        total = sum(p)
        normalized_p = [x / total for x in p]
        _action=np.random.choice(ACTIONS,p=normalized_p)

    else:
    #filter
        q=self.model.action(feature).flatten()
        _action=ACTIONS[torch.argmax(q).item()]
    self.model.counter+=1
    return _action
def state_to_features(game_state: dict) -> np.array:
    round = game_state['round']
    step=game_state['step']
    field=game_state['field']
    explosion_map = game_state['explosion_map']
    bombs=game_state['bombs']
    others=game_state['others']
    coins=game_state['coins']
    Self=game_state['self']

    explosion = explosion_area(bombs, field, explosion_map)
    safe_places = safe_area_coordinates(field, explosion_map)
    danger_coins = list(set(coins).intersection(set(explosion)))
    safe_coins = [x for x in coins if x not in danger_coins]
    crates = get_crate_coordinates(field)
    danger_crates = list(set(crates).intersection(set(explosion)))
    safe_crates = [x for x in crates if x not in danger_crates]
    danger_others = []
    for other in others:
        if not is_in_dangerous_area(other[3], bombs, field, explosion_map):
            danger_others.append(other)
    safe_others = [x for x in others if x not in danger_others]

    danger_feature=int(is_in_dangerous_area(Self[3],bombs,field,explosion_map))
    escape_feature=ACTIONS.index(to_nearest_safe_place(Self[3],bombs,field,safe_places))
    coins_feature=ACTIONS.index(to_nearest_coin(Self[3],safe_coins,field))
    crates_feature=ACTIONS.index(to_nearest_crate(Self[3],safe_crates,bombs,field,Self[2]))
    others_feature=ACTIONS.index(to_nearest_others(Self[3],safe_others,field,explosion_map))
    feature=[danger_feature,escape_feature,coins_feature,crates_feature,others_feature]

    feature=np.array(feature)
    return feature

# def state_to_features(game_state: dict) -> np.array:
#
#     """
#     *This is not a required function, but an idea to structure your code.*
#
#     Converts the game state to the input of your model, i.e.
#     a feature vector.
#
#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.
#
#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """
#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None
#     # For example, you could construct several channels of equal shape, ...
#     channels = []
#
#     round=game_state['round']
#     step=game_state['step']
#     field=game_state['field'].flatten()
#     explosion_map = game_state['explosion_map'].flatten()
#     bombs=game_state['bombs']
#     others=game_state['others']
#     coins=game_state['coins']
#     Self=game_state['self']
#
#
#     self_feature=np.array([-1]*4)
#     _,score,hasbomb,(x,y)=Self
#     self_feature[0]=score
#     self_feature[1]=hasbomb
#     self_feature[2]=x
#     self_feature[3]=y
#
#     bombs_feature=np.array([-1]*12)
#     if bombs:
#         for i in range(len(bombs)):
#             (x,y),t=bombs[i]
#             bombs_feature[3*i]=x
#             bombs_feature[3*i+1]=y
#             bombs_feature[3*i+2] = t
#
#     coins_feature=np.array([-1]*18)
#     if coins:
#         for i in range(len(coins)):
#             (x,y)=coins[i]
#             coins_feature[2*i]=x
#             coins_feature[2*i+1]=y
#
#     others_feature=np.array([-1]*12)
#     if others:
#         for i in range(len(others)):
#             _,score,hasbomb,(x,y)=others[i]
#             others_feature[4*i]=score
#             others_feature[4 * i+1] =hasbomb
#             others_feature[4 * i+2] =x
#             others_feature[4 * i+3] =y
#
#     feature=np.concatenate((round,
#                                    step,
#                                    field,
#                                    explosion_map,
#                                    self_feature,
#                                    bombs_feature,
#                                    coins_feature,
#                                    others_feature),axis=None)
#     # concatenate them as a feature tensor (they must have the same shape), ...
#
#     # and return them as a vector
#     return feature
def to_dangerous_area(position,explosion_map):
    x,y=position
    if explosion_map[x][y]==1:
        return ['WAIT']
    if explosion_map[x][y+1]==1:
        return ['RIGHT']
    if explosion_map[x][y-1]==1:
        return ['UP']
    if explosion_map[x+1][y]==1:
        return ['RIGHT']
    if explosion_map[x-1][y]==1:
        return ['LEFT']
    else:
        return []

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]




def dangerous_area(bombs,field,explosion_map):
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
            if explosion_map[x][y]==1:
                dangerous_area.append((x,y))
    return dangerous_area

def bomb_area(bomb_position,field):
    dangerous_area = []
    (bx, by) = bomb_position
    dangerous_area.append((bx, by))
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for step in range(1, s.BOMB_POWER + 1):
            new_x, new_y = bx + dx * step, by + dy * step
            # Check if the new position is within the game board boundaries
            if 0 <= new_x < s.COLS and 0 <= new_y < s.COLS:
                # If the new position is a wall, stop the blast in that direction
                if field[new_x][new_y] == 0:
                    break
                dangerous_area.append((new_x, new_y))
            else:
                break  # Stop if we reach the edge of the game board
    return  dangerous_area
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
def has_safe_place_around(self_position,position, field,bombs,explosion_map):
    x, y = position
    adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    for adj_x, adj_y in adjacent_positions:
        if 0 <= adj_x < s.COLS and 0 <= adj_y < s.ROWS:
            if field[adj_x][adj_y] == 0 and not is_in_dangerous_area((adj_x,adj_y,),bombs,field,explosion_map)  and np.sum(np.abs(np.subtract((adj_x,adj_y),self_position)))<=3:
                return True
    return False

def has_crate_around(position, field,bombs,explosion_map):
    x, y = position
    adjacent_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    for adj_x, adj_y in adjacent_positions:
        if 0 <= adj_x < s.COLS and 0 <= adj_y < s.ROWS:
            if field[adj_x][adj_y] == 1 and not is_in_dangerous_area((adj_x,adj_y,),bombs,field,explosion_map):
                return True
    return False

def can_be_attacked(position,others,crates):
    a=b=False
    if others:
        for other in others:
            _,_,_,(x,y)=other
            d = np.sum(np.abs(np.subtract((x,y), position)))
            if np.any(d<=s.BOMB_POWER and (x==position[0] or y==position[1])):
                a=True

    if crates:
        for crate in crates:
            d = np.sum(np.abs(np.subtract(crate, position)))
            if np.any(d<=s.BOMB_POWER and (crate[0]==position[0] or crate[1]==position[1])):
                b=True
    return (a or b)


#----------------------------------
# def to_nearest_safe_place(self_position,bombs,field,explosion_map):
#     """
#     Check if the agent can reach a safe place.
#     If yes,return the path.
#     If no, return WAIT
#     """
#     free_tiles = np.argwhere(np.array(field) == 0)
#     safe_tiles= np.argwhere(np.array(explosion_map) != 1)
#     free_tiles = [tuple(coord) for coord in free_tiles]
#     safe_tiles = [tuple(coord) for coord in  safe_tiles]
#     safe_places=set(free_tiles).intersection(set(safe_tiles))
#     distances = []
#     actions = []
#     best_d=float('inf')
#     if safe_places:
#         for (x,y) in safe_places:
#             d=np.sum(np.abs(np.subtract((x,y),self_position)))
#             if d<best_d:
#                 nearest=(x,y)
#                 best_d=d
#         dis,path,reachable=dis_path(field,self_position,nearest)
#         if reachable:
#             distances.append(dis)
#             if nearest!= self_position:
#                 actions.append(path)
#             else:
#                 actions.append(['WAIT'])
#         if distances:
#              return actions[np.argmin(distances)]
#         else:
#             return ['WAIT']
#     else:
#         return['WAIT']








    #     if (not is_in_dangerous_area((x,y),bombs,field,explosion_map)):
    #         dis,path,reachable=dis_path(field,self_position,(x,y))
    #         if reachable:
    #             distances.append(dis)
    #             if (x,y)!=self_position:
    #                 actions.append(path)
    #             else:
    #                 actions.append(['WAIT'])
    # if distances:
    #     return actions[np.argmin(distances)]
    # else:
    #     return ['WAIT']


def dis_path(field, self_coor, position):
    """
    Given a position(x,y), check if it can be reached in 5 steps.If yes, return distance, path and True.
    Otherwise, return 1000,None and False.(TO SEARCH A SAFE PLACE)
    """
    visited = set()
    queue = deque([(self_coor, 0, [])])

    while queue:
        (x, y), distance, path = queue.popleft()
        visited.add((x, y))

        if (x, y) == position:
            if len(path)<=4:
                return distance, path, True

        for dx, dy, direction in [(1, 0, 'RIGHT'), (-1, 0, 'LEFT'), (0, 1, 'DOWN'), (0, -1, 'UP')]:
            new_x, new_y = x + dx, y + dy
            if (
                    0 <= new_x < len(field) and
                    0 <= new_y < len(field[0]) and
                    field[new_y][new_x] == 0 and
                    (new_x, new_y) not in visited
            ):
                new_path = path + [direction]
                queue.append(((new_x, new_y), distance + 1, new_path))

    return 1000, ['WAIT'], False



#-----------------------------
# def look_for_targets(free_space, start, targets):
#
#     if len(targets) == 0: return None
#
#     frontier = [start]
#     parent_dict = {start: start}
#     dist_so_far = {start: 0}
#     best = start
#     best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
#
#     while len(frontier) > 0:
#         current = frontier.pop(0)
#         # Find distance from current position to all targets, track closest
#         d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
#         if d + dist_so_far[current] <= best_dist:
#             best = current
#             best_dist = d + dist_so_far[current]
#         if d == 0:
#             # Found path to a target's exact position, mission accomplished!
#             best = current
#             break
#         # Add unexplored free neighboring tiles to the queue in a random order
#         x, y = current
#         neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
#         #shuffle(neighbors)
#         for neighbor in neighbors:
#             if neighbor not in parent_dict:
#                 frontier.append(neighbor)
#                 parent_dict[neighbor] = current
#                 dist_so_far[neighbor] = dist_so_far[current] + 1
#     # Determine the first step towards the best found target tile
#     current = best
#     while True:
#         if parent_dict[current] == start: return current
#         current = parent_dict[current]
#
#
# def escape(position, field, bombs, others,
#                        explosion_map):
#
#     free_space = field == 0
#     # exclude bombs and others
#     for o in [xy for (n, s, b, xy) in others]:
#         free_space[o] = False
#     for b in [xy for (xy, c) in bombs]:
#         free_space[b] = False
#     # get targets
#     targets = []
#
#     for i in range(np.shape(field)[0]):
#         for j in range(np.shape(field)[1]):
#             # if not wall
#             if field[i][j] != -1:
#                 is_dangerous = is_in_dangerous_area((i,j),bombs,field,explosion_map)
#                 if is_dangerous == False:
#                     targets.append((i, j))
#
#     d = look_for_targets(free_space, position, targets)
#     if d == (position[0], position[1] - 1):
#         return 'UP'
#     elif d == (position[0], position[1] + 1):
#         return 'DOWN'
#     elif d == (position[0] - 1, position[1]):
#         return 'LEFT'
#     elif d == (position[0] + 1, position[1]):
#         return RIGHT
#     elif d == (position[0], position[1]):
#         is_dangerous = is_in_dangerous_area(position,bombs,field,explosion_map)
#         if is_dangerous == False:
#             return 'WAIT'
#         else:
#             return 'BOMB'
#     else:
#         return 'WAIT'

def safe_area_coordinates(field,explosion_map):
    safe_coordinates = []

    for y in range(len(explosion_map)):
        for x in range(len(explosion_map[y])):
            if explosion_map[x][y] and field[x][y]== 0:
                safe_coordinates.append((x, y))
    return safe_coordinates


def is_in_dangerous_area(position,bombs,field,explosion_map):
    """
    Given a position (x,y), check if it is in explosion area
    """
    if position in dangerous_area(bombs,field,explosion_map):
        return True
    return False

def is_in_explosion_area(field,explosion_map):
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if explosion_map[x][y]!=0:
                return True
    return False

def to_nearest_crate(position,crates,bombs,field,hasbomb):
    free_space=field==0
    x, y = position
    for bomb in bombs:
        coor, t = bomb
        free_space[coor] = False
    direction=look_for_targets(free_space,position,crates)
    if direction == (x, y - 1):
        return 'UP'
    elif direction == (x, y + 1):
        return 'DOWN'
    elif direction == (x - 1, y):
        return 'LEFT'
    elif direction == (x + 1, y):
        return 'RIGHT'
    elif direction == (x, y):
        if hasbomb:
            return 'BOMB'
        else:
            return 'WAIT'
    else:
        return 'WAIT'


        #x_dir = 'RIGHT' if dx > 0 else 'LEFT' if dx < 0 else ''
        #y_dir = 'DOWN' if dy > 0 else 'UP' if dy < 0 else ''

        #return [x_dir, y_dir] if x_dir and y_dir else [x_dir + y_dir]
def to_nearest_coin(position,coins,field):
    free_space=field==0
    x,y=position
    direction=look_for_targets(free_space,position,coins)
    if direction==(x,y-1):
        return 'UP'
    elif direction==(x,y+1):
        return 'DOWN'
    elif direction==(x-1,y):
        return 'LEFT'
    elif direction==(x+1,y):
        return 'RIGHT'
    else:
        if field[x+1][y]==1 or field[x-1][y]==1 or field[x][y+1]==1 or field[x][y-1]==1:
            return 'BOMB'
        else:
            return 'WAIT'

def get_crate_coordinates(field):
    crate_coordinates = []

    for y in range(s.ROWS):
        for x in range(s.COLS):
            if field[x][y] == 1:
                crate_coordinates.append((x, y))

    return crate_coordinates
def to_nearest_safe_place(position,bombs,field,safe_places):
    free_space=field==0
    x,y=position
    for bomb in bombs:
        coor,t=bomb
        free_space[coor]=False
    direction = look_for_targets(free_space, position, safe_places)
    if direction == (x, y - 1):
        return 'UP'
    elif direction == (x, y + 1):
        return 'DOWN'
    elif direction == (x - 1, y):
        return 'LEFT'
    elif direction == (x + 1, y):
        return 'RIGHT'
    else:
        return 'WAIT'

def to_nearest_others(position,others,field,explosion_map):
    x,y=position
    free_space=field==0
    for other in others:
        _,_,b,coor=other
        if is_in_explosion_area((x,y),explosion_map) or b:
            free_space[coor]=False
    for other in others:
        _, _, b, coor = other
        d = np.sum(np.abs(np.subtract(coor, position))).min()
        if d<=3:
            return 'BOMB'
    others_coor=[other[3] for other in others]
    direction=look_for_targets(free_space,position,others_coor)
    if direction == (x, y - 1):
        return 'UP'
    elif direction == (x, y + 1):
        return 'DOWN'
    elif direction == (x - 1, y):
        return 'LEFT'
    elif direction == (x + 1, y):
        return 'RIGHT'
    else:
        return 'WAIT'






























