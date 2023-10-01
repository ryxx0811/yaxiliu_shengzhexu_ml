import random
from collections import namedtuple, deque

import pickle
from typing import List

from torch import optim

import events as e
from .callbacks import state_to_features
import callbacks
import environment as env
from environment import Agent
import torch.nn as nn
import torch
import numpy as np
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
ACTIONS=['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


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

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_DOWN:0,
        e.MOVED_LEFT:0,
        e.MOVED_UP:0,
        e.MOVED_RIGHT:0,
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.COIN_FOUND:0.5,
        e.BOMB_DROPPED:1,
        e.CRATE_DESTROYED:1,
        e.KILLED_SELF:-1000,
        e.WAITED:0,
        e.GOT_KILLED:-1000  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    def forward(self,state):
        state = nn.functional.relu(self.fc1(state))
        state = nn.functional.relu(self.fc2(state))
        return self.fc3(state) #Q-value

def training(self,agent:Agent,n_episodes=20,alpha=0.1,gamma=0.99,input_size=8,output_size=6,):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dqn=DQN(input_size,output_size).to(device)
    DQN_target=DQN(input_size,output_size)
    DQN_target.load_state_dict(dqn.state_dict())
    DQN_target.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=alpha)
    criterion = nn.MSELoss()
    callbacks.setup(self)
    if self.train:
        setup_training(self)
    for episode in range(n_episodes):

        for step in range(400):
            state=env.BombeRLeWorld.get_state_for_agent(self,agent)
            state=state_to_features(state)
            Q=dqn(state)
            if self.train and not self.dead:
                action=callbacks.act(self,Q)
                index=ACTIONS.index(action)
                Q_value=Q[index].item()
                env.BombeRLeWorld.perform_agent_action(self,agent,action)
                next_state=env.BombeRLeWorld.get_state_for_agent(self,agent)
                events=env.BombeRLeWorld.send_game_events(self)
                game_events_occurred(self,state,action,next_state,events)
                reward=reward_from_events(self,events)
                Q_target = DQN_target(next_state)
                Q_value_target = reward + gamma * Q_target.max(1)[0]

                loss = criterion(Q_value, Q_value_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if self.train:
            end_of_round(self,state,action,next_state,events)
            env.BombeRLeWorld.end_round(self)

        if episode % 200 == 0:
            DQN_target.load_state_dict(dqn.state_dict())
    path='lyx_agent.pt'
    torch.save(dqn.state_dict(), path)

