from collections import namedtuple
import os
import pickle
from typing import List

import settings as s
import events as e
import numpy as np
from agent_code.xsz_agent.callbacks import (state_to_features)
FILENAME = "self_agent"
HISTORY_RECORD = f"{FILENAME}_record.pt"
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5

FEATURES = ['distance_to_crates', 'distance_to_coin', 'distance_to_opponents']
distance_to_crates = 0
distance_to_coin = 1
distance_to_opponents = 2

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
BATCH_SIZE = 500
FEATURE_NUM = 8
NUM_EPISODES = 10
input_shape = (FEATURE_NUM,)
num_actions = len(ACTIONS)
epsilon = 0.1
gamma = 0.9
learning_rate = 0.01
max_step = 400
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
# Events:
# FOR COIN
GET_COIN_BY_BOMB = "GET_COIN_BY_BOMB"
APPROACH_COIN = "APPROACH_COIN"
FAILED_COIN = "FAILED_COIN"
MOVE_AWAY_FROM_COIN = "MOVE_AWAY_FROM_COIN"
# PUBLIC USE
UNSUITED_BOMB = "UNSUITED_BOMB"
# FOR OPPONENT
BLOW_UP_OPPONENT = "BLOW_UP_OPPONENT"
FAILED_BLOW_UP_OPPONENT = "FAILED_BLOW_UP_OPPONENT"
APPROACH_OPPONENT = "APPROACH_OPPONENT"
SUCCESS_AVOID_OPPONENT_BOMB = "SUCCESS_AVOID_OPPONENT_BOMB"
MOVE_AWAY_FROM_OPPONENT = "MOVE_AWAY_FROM_OPPONENT"
# FOR CRATES
SUCCESS_BLOW_UP_CRATE = "SUCCESS_BLOW_UP_CRATE"
WAIT_NEAR_CRATE = "WAIT_NEAR_CRATE"
APPROACH_CRATE = "APPROACH_CRATE"
FAILED_WAIT_CRATE = "FAILED_WAIT_CRATE"
MOVE_AWAY_FROM_CRATE = "MOVE_AWAY_FROM_CRATE"
FAILED_CRATE = "FAILED_CRATE"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    if os.path.isfile(HISTORY_RECORD):
        with open(HISTORY_RECORD, "rb") as file:
            self.his_record = pickle.load(file)
        self.game_session = max(self.his_record['games']) + 1
    else:
        self.his_record = {'coins': [], 'score': [], 'opponent': [], 'crates': [], 'games': []}
        self.game_session = 1

    self.transition_len = 0
    self.coins_nr = 0
    self.score_point = 0
    self.blow_opponent = 0
    self.blow_crates = 0


def verify_action(action: int, self_action: str) -> bool:
    # to verify if the agent action is correct
    action_mapping = {
        UP: 'UP',
        RIGHT: 'RIGHT',
        DOWN: 'DOWN',
        LEFT: 'LEFT',
        WAIT: 'WAIT',
        BOMB: 'BOMBS'
    }
    return action_mapping.get(action) == self_action


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
    if old_game_state:
        old_state = state_to_features(old_game_state)
        old_state_dangerous = True
        if old_state_dangerous:
            coin_old = old_state[0][distance_to_coin]
            if verify_action(coin_old,self_action):
                events.append(APPROACH_COIN)
            else:
                events.append(MOVE_AWAY_FROM_COIN)
        else:
            # when the previous state is safe, there have many choices
            # Although blowing up the opponent is the highest score, we take a gentle approach,
            # choosing coins first,then crates, and finally the opponent
            # self.logger.debug("take coin action: %s", ACTIONS[dir_to_coin_old])
            coin_old = old_state[0][distance_to_coin]
            crates_old = old_state[0][distance_to_crates]
            opp_old = old_state[0][distance_to_opponents]

            if verify_action(coin_old, self_action):
                # check action for coins
                if coin_old == BOMB:
                    events.append(GET_COIN_BY_BOMB)
                else:
                    events.append(APPROACH_COIN)
            else:
                if self_action == "BOMB":
                    events.append(UNSUITED_BOMB)
                else:
                    events.append(FAILED_COIN)

                if verify_action(opp_old, self_action):
                    # check action for opponent
                    if self_action == 'BOMB':
                        events.append(BLOW_UP_OPPONENT)
                    else:
                        events.append(APPROACH_OPPONENT)
                else:
                    if self_action == 'BOMB':
                        events.append(UNSUITED_BOMB)

                    else:
                        if opp_old == BOMB:
                            events.append(SUCCESS_AVOID_OPPONENT_BOMB)
                        else:
                            events.append(MOVE_AWAY_FROM_OPPONENT)

                    if verify_action(crates_old, self_action):
                        # check action for crates
                        if self_action == 'BOMB':
                            events.append(SUCCESS_BLOW_UP_CRATE)
                        elif self_action == 'WAIT':
                            events.append(WAIT_NEAR_CRATE)
                        else:
                            events.append(APPROACH_CRATE)
                    else:
                        if self_action == 'BOMB':
                            events.append(UNSUITED_BOMB)
                            if crates_old == WAIT:
                                events.append(MOVE_AWAY_FROM_CRATE)
                        else:
                            if crates_old == BOMB:
                                events.append(FAILED_CRATE)
                            elif crates_old == WAIT:
                                events.append(FAILED_WAIT_CRATE)
                            else:
                                events.append(MOVE_AWAY_FROM_CRATE)

    #   events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if old_game_state:
        # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
        self.transition_length = self.model.store_transition(state_to_features(old_game_state)[0], self_action,
                                                             reward_from_events(self, events),
                                                             state_to_features(new_game_state)[0])

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


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

    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    self.transition_length = self.model.store_transition(state_to_features(last_game_state)[0], last_action,
                                                         reward_from_events(self, events),
                                                         state_to_features()[0])
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    if 'CRATE_DESTROYED' in events:
        self.blow_crates += events.count('CRATE_DESTROYED')
    if 'COIN_COLLECTED' in events:
        self.coins_nr += 1
    if 'KILLED_OPPONENT' in events:
        self.blow_opponentt += events.count('KILLED_OPPONENT')
    self.score_per_round += reward_from_events(self, events)
    sum_score = np.sum(self.score_per_round)

    data_to_append = [sum_score, self.blow_crates, self.coins_nr, self.blow_opponent, self.game_session]
    keys_to_append = ['score', 'crates', 'coins', 'opponent', 'games']
    for key, value in zip(keys_to_append, data_to_append):
        self.historic_data[key].append(value)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    # Rules for agents TIMEOUT = 0.5,TRAIN_TIMEOUT = float("inf"),REWARD_KILL = 5,REWARD_COIN = 1
    coin = s.REWARD_COIN  # 1
    kill = s.REWARD_KILL  # 5
    crate = 0.1

    game_rewards = {

        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: crate,  # 1
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: coin,  # 1
        e.KILLED_OPPONENT: kill,  # 5
        e.KILLED_SELF: -kill,  # -1
        e.GOT_KILLED: -kill,  # -1
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: 0,

        # for coin
        GET_COIN_BY_BOMB: 0.5 * coin,
        APPROACH_COIN: 0.1 * coin,
        FAILED_COIN: -0.1 * coin,
        # for public
        UNSUITED_BOMB: -0.5 * kill,
        # FOR OPPONENT
        BLOW_UP_OPPONENT: 0.4 * kill,
        FAILED_BLOW_UP_OPPONENT: -0.4 * kill,
        APPROACH_OPPONENT: 0.05 * kill,
        SUCCESS_AVOID_OPPONENT_BOMB: 0.4 * kill,
        MOVE_AWAY_FROM_OPPONENT: - 0.1 * kill,
        # FOR CRATES
        SUCCESS_BLOW_UP_CRATE: 0.1 * coin,
        WAIT_NEAR_CRATE: 0.05 * coin,
        APPROACH_CRATE: 0.5 * crate,
        FAILED_WAIT_CRATE: -0.05 * crate,
        MOVE_AWAY_FROM_CRATE: -0.5 * crate,
        FAILED_CRATE: -0.5 * crate,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum
