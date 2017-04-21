# Imports.
import numpy as np
import numpy.random as npr
from collections import deque
from SwingyMonkey import SwingyMonkey
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD

state_dim = 7 # Number of features in each state
batch_size = 500
gamma = .8


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.memory_capacity = 1000  # Tune this hyperparameter
        self.state_index = 0

        # if using epsilon greedy exploration, this specifies prob of random
        # action
        self.epsilon = lambda i=self.state_index: .2 ** i

        # specify the discount factor

        self.D = deque(maxlen=self.memory_capacity)
        self.Q = self.Q_obj()

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None



    class Q_obj(object):

        def __init__(self):
            self.model = self.define_model()

        def define_model(self):
            model = Sequential()
            # Input size is the number of dims in state plus action variable
            model.add(Dense(100,input_dim=state_dim + 1))
            model.add(ELU())
            model.add(Dropout(.8))
            model.add(Dense(32))
            model.add(ELU())
            model.add(Dropout(.8))
            model.add(Dense(1))
            sgd = SGD(lr=.001)
            model.compile(loss='mse', optimizer='rmsprop')
            print "Model has been constructed"
            print model.summary()
            return model
        

        def get(self, s, a):
            return self.model.predict(np.array([np.concatenate([s,[a]])]))

        def update(self, D):
            # s, a, r, s_prime are all in D

            if len(D) < batch_size:
                last_transition = D[-1]
                s, a, r, s_prime = last_transition
                if r == -10 or r == -5:
                    # print "terminal"
                    y = r
                else:
                    y = r + gamma * np.max([self.get(s_prime,0),self.get(s_prime,1)])
                x = np.array([np.concatenate([s,[a]])])
                self.model.train_on_batch(x,[y])
            else:
                # Definitely update Q on the last transition,
                # complete the batch with others from D
                randomindx = np.random.choice(
                    np.array(list(D)[:-1]).shape[0], size=(batch_size -1,)
                )
                sample = np.array(list(np.array(D)[randomindx]) + [D[-1]])
                y = np.zeros(batch_size)
                i = 0
                x = []
                for s, a, r, s_prime in sample:
                    x.append(np.concatenate([s,[a]]))
                    if r == -10 or r == -5:
                        print "terminal"
                        y[i] = r
                    else:
                        y[i] = r + gamma * np.max([self.get(s_prime,0),self.get(s_prime,1)])
                    i += 1
                # Not sure exactly how to do this update if I am using two q functions
                loss = self.model.train_on_batch(np.array(x), y)
                # print "loss is {}".format(loss)

    def policy(self, state):
        # For now, use an epsilon greedy policy with constant epsilon
        if npr.rand() < self.epsilon:
            # Choose an action uniformly at random
            action = npr.rand() < .1
        else:
            action = np.argmax([self.Q.get(state, 0), self.Q.get(state, 1)])
        return action

    def process_state(self, state):
        # Process the state returned by the world into a format
        # compatible with the learning algorithm
        processed = np.zeros(7)
        processed[0] = state['score']
        processed[1] = state['tree']['dist'] / 600.
        processed[2] = state['tree']['top'] / 400.
        processed[3] = state['tree']['bot'] / 400.
        processed[4] = state['monkey']['vel']
        processed[5] = state['monkey']['top'] / 400.
        processed[6] = state['monkey']['bot'] / 400.
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
            self.Q.update(self.D)

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
    run_games(agent, hist, 200,10)

    # Save history.
    np.save('hist', np.array(hist))
