# Imports.
import numpy as np
import numpy.random as npr
from collections import deque
from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.memory_capacity = 100  # Tune this hyperparameter
        self.state_index = 0

        # if using epsilon greedy exploration, this specifies prob of random
        # action
        self.epsilon = 1

        # specify the discount factor
        self.gamma = .1
        self.D = deque(maxlen=self.memory_capacity)
        self.Q = self.Q_obj()

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.memory_capacity = 100  # Tune this hyperparameter
        self.state_index = 0

        # if using epsilon greedy exploration, this specifies prob of random
        # action
        self.epsilon = 1

        # specify the discount factor
        self.gamma = .1
        self.D = deque(maxlen=self.memory_capacity)
        self.Q = self.Q_obj()

    class Q_obj(object):

        def __init__(self):
            pass
            # TODO: random initialization of Q

        def get(self, s, a):
            # TODO
            return 0

        def update(self):
            # s, a, r, s_prime are all in D
            pass

    def policy(self, state):
        # For now, use an epsilon greedy policy with constant epsilon
        if npr.rand() < self.epsilon:
            # Choose an action uniformly at random
            action = npr.rand() < .5
        else:
            action = np.argmax([self.Q.get(state, 0), self.Q.get(state, 1)])
        return action

    def process_state(self, state):
        # Process the state returned by the world into a format
        # compatible with the learning algorithm
        processed = np.zeros(7)
        processed[0] = state['score']
        processed[1] = state['tree']['dist']
        processed[2] = state['tree']['top']
        processed[3] = state['tree']['bot']
        processed[4] = state['monkey']['vel']
        processed[5] = state['monkey']['top']
        processed[6] = state['monkey']['bot']
        return processed

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # state = s'
        # self.last_action = a
        # self.last_state = s
        # self.last_reward = r(s,a)
        # update Q
        new_state = self.process_state(state)

        if self.last_action is None:
            self.last_action = npr.rand() < 0.1  # CHANGE
            self.last_state = self.process_state(state)
            return self.last_action

        else:
            self.D.append((self.last_state, self.last_action,
                           self.last_reward, new_state))

            # Internally, this update will take into account the last
            # transition in D
            self.Q.update()

            # The policy abstracts how the learner should choose the next action
            # For example, epsilon greedy or something similar.
            new_action = self.policy(new_state)
            self.last_action = new_action
            self.last_state = new_state

            self.state_index += 1
            return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             # Display the epoch on screen.
                             text="Epoch %d" % (ii),
                             # Make game ticks super fast.
                             tick_length=t_len,
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    return


if __name__ == '__main__':

        # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 20, 10)

    # Save history.
    np.save('hist', np.array(hist))
