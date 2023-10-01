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
from .callbacks import (state_to_features,is_in_dangerous_area,explosion_area,to_nearest_coin,get_crate_coordinates,
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
NECESSARY_WAIT='NECESSARY_WAIT'

SAFELY_DROPPING='SAFELY_DROPPING'
DANGEROUSLY_DROPPING='DANGEROUSLY_DROPPING'

UNNECESSARY_WAIT='UNNECESSARY_WAIT'


DESTROY_CRATES='DESTROY_CRATES'



class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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
    feature_old=state_to_features(old_game_state)
    feature_new=state_to_features(new_game_state)

    new_safe_places = safe_area_coordinates(new_game_state['field'], new_game_state['explosion_map'])
    if feature_old[0]==1:
        if feature_new[0]==0:
            events.append(DANGEROUS_ZONE_TO_SAFE_ZONE)
        if feature_old[1]!=WAIT:
            if ACTIONS.index(self_action)==feature_old[1]:
                events.append(AWAY_FROM_BOMB)

        elif self_action=='BOMB' or self_action=='WAIT':
            events.append(STAY_DANGEROUS)

        else:
            events.append(APPROACHING_TO_BOMB)
    else:
        if old_game_state['step']==1 and self_action=='BOMB':
            events.append(DANGEROUSLY_DROPPING)
        if feature_new[0]==1:
            if self_action!='BOMB':
                events.append(SAFE_ZONE_TO_DANGEROUS_ZONE)
        if ACTIONS.index(self_action)==feature_old[2]:
            events.append(APPROACHING_TO_COIN)
            if self_action=='BOMB' and feature_new[1]==WAIT:
                events.append(DANGEROUSLY_DROPPING)
        else:
            if ACTIONS.index(self_action)==feature_old[3]:
                events.append(APPROACHING_TO_CRATE)
                if self_action=='BOMB' and feature_new[1]==WAIT:
                    events.append(DANGEROUSLY_DROPPING)

            else:
                events.append(AWAY_FROM_CRATE)
                events.append(AWAY_FROM_COIN)
    if e.KILLED_SELF not in events and e.GOT_KILLED not in events:
        events.append(SURVIVE)
    if old_game_state['others'] or old_game_state['coins']:
        if self_action=='WAIT':
            events.append(UNNECESSARY_WAIT)



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
    self.model.transitions.append(Transition(state_to_features(last_game_state), last_action, np.zeros(5),reward_from_events(self, events)))
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

        rewards = torch.FloatTensor(rewards)#.to(torch.device('cuda'))
        feature = torch.FloatTensor(states)#.to(torch.device('cuda'))

        next_feature = torch.FloatTensor(next_states)#.to(torch.device('cuda'))

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


    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

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

        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 0.5,
        e.COIN_COLLECTED: 1,

        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,

        e.GOT_KILLED: -10,
        e.OPPONENT_ELIMINATED: 5,
        e.SURVIVED_ROUND: 5,

        DANGEROUS_ZONE_TO_SAFE_ZONE: 4,
        SAFE_ZONE_TO_DANGEROUS_ZONE: -2,

        UNNECESSARY_WAIT: -2,


        INEFFECTIVE_BOMB: -0.5,

        APPROACHING_TO_COIN: 0.4,
        AWAY_FROM_COIN: -0.4,

        APPROACHING_TO_BOMB: -2,
        AWAY_FROM_BOMB: 2,

        APPROACHING_TO_OTHERS: 0.2,
        AWAY_FROM_OTHERS: -0.2,

        SAFELY_DROPPING: 0.5,
        DANGEROUSLY_DROPPING: -3,

        APPROACHING_TO_CRATE: 0.1,
        AWAY_FROM_CRATE: -0.1,

        STAY_SAFE: 0.2,
        STAY_DANGEROUS: -3,

        SURVIVE: 0.1,



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
        self.q_net = DQN(self.input_size,self.output_size)#.to(torch.device('cuda'))
        self.target_net = DQN(self.input_size,self.output_size)#.to(torch.device('cuda'))
        self.alpha = 0.001
        self.gamma = 0.9
        self.counter = 1
        self.epsilon =0.2
        self.batch = 300

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)

    def action(self,feature):


        feature_input=torch.FloatTensor(np.array([feature]))#.to(torch.device('cuda'))
        # todo Exploration vs exploitation
        #if np.random.uniform()>self.epsilon:

            #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        q=self.q_net(feature_input)#.to(torch.device('cuda'))
        return q









