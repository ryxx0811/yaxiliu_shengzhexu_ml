import os
import random
from collections import namedtuple, deque
import concurrent.futures
import pickle
from typing import List

import numpy as np
import multiprocessing

from torch import optim

import events as e
from .callbacks import (explosion_area,state_to_features,is_in_dangerous_area,is_in_explosion_area,to_nearest_coin,get_crate_coordinates,
                        to_nearest_others,safe_area_coordinates,to_nearest_crate,to_nearest_safe_place)
import settings as s

import environment as env
from environment import Agent
import torch.nn as nn
import torch
import torch.nn.functional
counter=1
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
ACTIONS=['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

#events
SURVIVE='SURVIVE'
ATTACK='ATTACK'

SAFE_ZONE_TO_DANGEROUS_ZONE = 'SAFE_ZONE_TO_DANGEROUS_ZONE'
DANGEROUS_ZONE_TO_SAFE_ZONE = 'DANGEROUS_ZONE_TO_SAFE_ZONE'
STAY_SAFE='STAY_SAFE'
STAY_DANGEROUS='STAY_DANGEROUS'

APPROACHING_TO_BOMB = 'APPROACHING_BOMB'
AWAY_FROM_BOMB = 'AWAY_FROM_BOMB'

APPROACHING_TO_COIN = 'APPROACHING_COIN'
AWAY_FROM_COIN = 'AWAY_FROM_COIN'

APPROACHING_TO_OTHERS='APPROACHING_TO_OTHERS'
AWAY_FROM_OTHERS='AWAY_FROM_OTHERS'

APPROACHING_TO_CRATE='APPROACHING_TO_CRATE'
AWAY_FROM_CRATE='AWAY_FROM_CRATE'

INEFFECTIVE_BOMB='INEFFECTIVE_BOMB'


SAFELY_DROPPING='SAFELY_DROPPING'
DANGEROUSLY_DROPPING='DANGEROUSLY_DROPPING'

UNNECESSARY_WAIT='UNNECESSARY_WAIT'


DESTROY_CRATES='DESTROY_CRATES'



class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 17 * 17, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


    field=old_game_state['field']
    explosion_map=old_game_state['explosion_map']
    self_position=old_game_state['self'][3]
    self_hasbomb=old_game_state['self'][2]
    bombs=old_game_state['bombs']
    coins=old_game_state['coins']
    others=old_game_state['others']



    explosion=explosion_area(bombs,field,explosion_map)
    safe_places=safe_area_coordinates(field,explosion_map)
    crates=get_crate_coordinates(field)
    danger_crates=list(set(crates).intersection(set(explosion)))
    safe_crates=[x for x in crates if x not in danger_crates]
    danger_others=[]
    for other in others:
        if not is_in_explosion_area(other[3],bombs,field,explosion_map):
            danger_others.append(other)
    safe_others = [x for x in others if x not in danger_others]
    danger_coins=list(set(coins).intersection(set(explosion)))
    safe_coins=[x for x in coins if x not in danger_coins]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        other_direction=executor.submit(to_nearest_others,field,self_position,self_hasbomb,safe_others)
        crate_direction=executor.submit(to_nearest_crate,field,self_position,self_hasbomb,safe_crates)
        coin_direction=executor.submit(to_nearest_coin,field,self_position,safe_coins)
        safe_direction=executor.submit(to_nearest_safe_place,field,self_position,safe_places)


        other_direction=other_direction.result()
        crate_direction= crate_direction.result()
        coin_direction=coin_direction.result()
        safe_direction=safe_direction.result()
    #print(safe_crates)
    #print([safe_direction,coin_direction,crate_direction,other_direction])
    #print(is_in_explosion_area(self_position,bombs,field,explosion_map))

    new_field=new_game_state['field']
    new_bombs=new_game_state['bombs']
    new_self_position=new_game_state['self'][3]
    new_explosion_map=new_game_state['explosion_map']
    new_safe_place=safe_area_coordinates(new_field,new_explosion_map)
    new_safe_direction=to_nearest_safe_place(new_field,new_self_position,new_safe_place)
    if is_in_explosion_area(self_position,bombs,field,explosion_map):
        if self_action==safe_direction:
            events.append(AWAY_FROM_BOMB)
        else:
            if self_action=='BOMB':
                if new_safe_direction == 'WAIT':
                    events.append(DANGEROUSLY_DROPPING)
                else:
                    events.append(SAFELY_DROPPING)

            events.append(STAY_DANGEROUS)
        if not is_in_explosion_area(new_self_position,new_bombs,new_field,new_explosion_map):
            events.append(DANGEROUS_ZONE_TO_SAFE_ZONE)

    #
    else:
        if is_in_explosion_area(new_self_position,new_bombs,new_field,new_explosion_map):
            if self_action!='BOMB':
                events.append(SAFE_ZONE_TO_DANGEROUS_ZONE)
        else:
            events.append(STAY_SAFE)
        if old_game_state['step']==1 and self_action=='BOMB':
            events.append(DANGEROUSLY_DROPPING)

        if other_direction=='None':
            if coin_direction=='None':
                if self_action==crate_direction:
                    if self_action=='BOMB':
                        if is_in_explosion_area(new_self_position,new_bombs,new_field,new_explosion_map):
                            events.append(DANGEROUSLY_DROPPING)
                        else:
                            events.append(SAFELY_DROPPING)
                        events.append(DESTROY_CRATES)
                    else:
                        events.append(APPROACHING_TO_CRATE)
                else:
                    if self_action=='WAIT':
                        events.append(UNNECESSARY_WAIT)
                    if self_action=='BOMB':
                        if is_in_explosion_area(new_self_position,new_bombs,new_field,new_explosion_map):
                            events.append(DANGEROUSLY_DROPPING)
                        else:
                            events.append(SAFELY_DROPPING)
                        events.append(INEFFECTIVE_BOMB)
                    else:
                        events.append(AWAY_FROM_CRATE)
            else:
                if self_action==coin_direction:
                    events.append(APPROACHING_TO_COIN)
                else:
                    if self_action=='WAIT':
                        events.append(UNNECESSARY_WAIT)
                    if self_action=='BOMB':
                        if is_in_explosion_area(new_self_position,new_bombs,new_field,new_explosion_map):
                            events.append(DANGEROUSLY_DROPPING)
                        else:
                            events.append(SAFELY_DROPPING)
                        events.append(INEFFECTIVE_BOMB)
                    else:
                        events.append(AWAY_FROM_COIN)
        else:
            if self_action==other_direction:
                if self_action=='BOMB':
                    if is_in_explosion_area(new_self_position, new_bombs, new_field, new_explosion_map):
                        events.append(DANGEROUSLY_DROPPING)
                    else:
                        events.append(SAFELY_DROPPING)
                    events.append(ATTACK)
                else:
                    events.append(APPROACHING_TO_OTHERS)
            else:
                events.append(AWAY_FROM_OTHERS)




    if (e.GOT_KILLED and e.KILLED_SELF) not in events:
        events.append(SURVIVE)
    targets = safe_crates + safe_coins + [safe_other[3] for safe_other in safe_others]
    if targets:
        d = np.sum(np.abs(np.subtract(targets, self_position)), axis=1).min()
        if d <= 4:
            if self_action == 'WAIT':
                events.append(UNNECESSARY_WAIT)



    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.killed_enemies += events.count('KILLED_OPPONENT')
    self.round_reward += reward_from_events(self, events)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    self.model.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.model.transitions.append(Transition(state_to_features(last_game_state), last_action, np.zeros((6,17,17)),reward_from_events(self, events)))
    round=last_game_state['round']
    if len(self.model.transitions) >= self.model.batch:

        # transitions_index = np.random.choice(len(self.transitions), len(self.transitions), replace=True)
        # return ACTIONS[torch.argmax(q).item()]

        transitions_index = np.random.choice(len(self.model.transitions), self.model.batch, replace=False)
        states = np.array([self.model.transitions[i].state for i in transitions_index])

        rewards = np.array([self.model.transitions[i].reward for i in transitions_index])
        actions = np.array([self.model.transitions[i].action for i in transitions_index])
    # print(actions)
        next_states = np.array([self.model.transitions[i].next_state for i in transitions_index])

        rewards = torch.FloatTensor(rewards).to(torch.device('cuda'))
        feature = torch.FloatTensor(states).to(torch.device('cuda'))

        next_feature = torch.FloatTensor(next_states).to(torch.device('cuda'))

        q_next = self.model.target_net.forward(next_feature).max(1)[0]
        q_val = self.model.q_net.forward(feature).max(1)[0]
        q_target_val = q_val.clone()
        # print(q_val)

        q_target_val = rewards + self.model.gamma * q_next
    # print(q_target_val)

    # fo  actions_index = ACTIONS.index(actions[i])
        #     #      q_target_val[i][actions_index] = rewards[i] + self.gamma * torch.max(q_next[i])
        #     # print(q_target_val)
        #     # self.loss=nn.functional.smooth_l1_loss(q_val, q_target_val).to(torch.device('cuda'))
        #     # print(q_val)
        #     # print(q_target_val)r i in range(len(transitions_index)):
    #
        self.model.loss = nn.MSELoss()(q_val, q_target_val)


        #self.model.loss.requires_grad_(True)

        torch.nn.utils.clip_grad_norm_(self.model.q_net.parameters(), max_norm=0.1)
        # for name, param in self.model.q_net.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
        self.model.optimizer.zero_grad()
        self.model.loss.backward()

        self.model.optimizer.step()

    if round%100==0:
        print(self.model.loss)

    if round%2000==0:
        self.model.target_net.load_state_dict(self.model.q_net.state_dict())

    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.killed_others += events.count('KILLED_OPPONENT')
    self.round_reward += reward_from_events(self, events)
    self.round += 1
    # Total score in this game.
    self.total_reward += np.sum(self.round_reward)

    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    # Store the model

    checkpoint = {
        'epoch': round,
        'model_state_dict': self.model.q_net.state_dict(),
        'optimizer_state_dict': self.model.optimizer.state_dict(),
        'learning_rate': self.model.optimizer.param_groups[0]['lr']
    }
    #torch.save(checkpoint, 'my_saved_model.pt')
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_DOWN: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_UP: 0,
        e.MOVED_RIGHT: 0,
        e.INVALID_ACTION: -1,
        e.WAITED: 0,

        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 0.2,
        e.COIN_FOUND: 0.5,
        e.COIN_COLLECTED: 1,

        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,

        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 5,
        e.SURVIVED_ROUND: 5,

        DANGEROUS_ZONE_TO_SAFE_ZONE: 2,
        SAFE_ZONE_TO_DANGEROUS_ZONE: -2,

        UNNECESSARY_WAIT: -2,


        INEFFECTIVE_BOMB: -0.5,

        APPROACHING_TO_COIN: 0.4,
        AWAY_FROM_COIN: -0.4,

        APPROACHING_TO_BOMB: -2,
        AWAY_FROM_BOMB: 0.5,

        APPROACHING_TO_OTHERS: 0.2,
        AWAY_FROM_OTHERS: -0.2,

        SAFELY_DROPPING: 0.5,
        DANGEROUSLY_DROPPING: -3,

        APPROACHING_TO_CRATE: 0.1,
        AWAY_FROM_CRATE: -0.1,

        STAY_SAFE: 0.2,
        STAY_DANGEROUS: -3,

        SURVIVE: 0.1,
        ATTACK:0.3,
        DESTROY_CRATES: 0.1,


    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

class dqnnet():
    def __init__(self):
        self.input_size = 5
        self.output_size = len(ACTIONS)
        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
        self.q_net = DQN().to(torch.device('cuda'))
        self.target_net = DQN().to(torch.device('cuda'))
        self.alpha = 0.0001
        self.gamma = 0.9
        self.counter = 1
        self.epsilon =0.2
        self.batch = 500

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)

    def action(self,feature):


        feature_input=torch.FloatTensor(np.array([feature])).to(torch.device('cuda'))
        # todo Exploration vs exploitation
        #if np.random.uniform()>self.epsilon:

            #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        q=self.q_net(feature_input).to(torch.device('cuda'))
        return q









