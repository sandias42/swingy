# Imports.
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from collections import deque
from SwingyMonkey import SwingyMonkey
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD

state_plus_action_dim = 28 # Number of features in each state
batch_size = 500
gamma = .93
optim = "adam"
mem_cap = 3000


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.memory_capacity = 500  # Tune this hyperparameter
        self.state_index = 0
        self.epoch_index = 0
        # if using epsilon greedy exploration, this specifies prob of random
        # action

        self.epsilon = lambda i: .95 ** i
        # specify the discount factor

        self.D = deque(maxlen=self.memory_capacity)
        self.Q = self.Q_obj()

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.epoch_index += 1
        



    class Q_obj(object):

        def __init__(self):
            self.model = self.define_model()

        def define_model(self):
            model = Sequential()
            # Input size is the number of dims in state plus action variable
            model.add(Dense(400,input_dim=state_plus_action_dim))
            model.add(ELU())
            model.add(Dropout(.3))
            model.add(Dense(100))
            model.add(ELU())
            model.add(Dropout(.3))
            model.add(Dense(100))
            model.add(ELU())
            model.add(Dropout(.3))
            model.add(Dense(32))
            model.add(ELU())
            model.add(Dropout(.2))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer="adam")
            print "Model has been constructed"
            print model.summary()
            return model
        

        def get(self, s, a):
            return self.model.predict(np.array([np.concatenate([s,[a]])]))

        def update(self, D):
            # s, a, r, s_prime are all in D

            if len(D) < 4:
                """
                last_transition = D[-1]
                s, a, r, s_prime = last_transition
                if r == -10 or r == -5:
                    # print "terminal"
                    y = r
                else:
                    y = r + gamma * np.max([self.get(s_prime,0),self.get(s_prime,1)])
                x = np.array([np.concatenate([s,[a]])])
                self.model.train_on_batch(x,[y])
                """
                pass
            elif 4 <= len(D) < batch_size:
                last_transition = D[-1]
                index = -1
                # get previous 3 in addition to last_action
                s_prev1, a_prev1, __, __ = D[index -1]
                s_prev2, a_prev2, __, __ = D[index -2]
                s_prev3, a_prev3, __, __ = D[index -3]
                s, a, r, s_prime = last_transition
                #print s.shape
                x = np.array([list(s_prev3) + [a_prev3] + list(s_prev2) + [a_prev2] + list(s_prev1) + [a_prev1] + list(s) + [a]])
                s_prime = np.array(list(s_prev2) + [a_prev2]+ list(s_prev1) + [a_prev1]  + list(s) + [a] + list(s_prime))
                if r == -10 or r == -5:
                    # print "terminal"
                    y = r
                else:
                    y = r + gamma * np.max([self.get(s_prime,0),self.get(s_prime,1)])
                loss = self.model.train_on_batch(x,[y])
                #print "loss is {}".format(loss)
            else:
                # Definitely update Q on the last transition,
                # complete the batch with others from D
                randomindx = np.concatenate([np.random.choice(
                    np.array(list(D)[:-1]).shape[0], size=(batch_size -1,)
                ), [-1]])
                
                #sample = np.array(list(np.array(D)[randomindx]) + [D[-1]])
                y = np.zeros(batch_size)
                i = 0
                x = []
                for index in randomindx:
                    last_transition = D[index]
                    # get previous 3 in addition to last_action
                    s_prev1, a_prev1, __, __ = D[index -1]
                    s_prev2, a_prev2, __, __ = D[index -2]
                    s_prev3, a_prev3, __, __ = D[index -3]
                    s, a, r, s_prime = last_transition
                    x.append(np.array(list(s_prev3) + [a_prev3] + list(s_prev2) + [a_prev2] + list(s_prev1) + [a_prev1] + list(s) + [a]))
                    s_prime = np.array(list(s_prev2) + [a_prev2] + list(s_prev1) + [a_prev1] + list(s) + [a] + list(s_prime))
                    if r == -10 or r == -5:
                        # print "terminal"
                        y[i] = r
                    else:
                        y[i] = r + gamma * np.max([self.get(s_prime,0),self.get(s_prime,1)])
                    i += 1
                # Not sure exactly how to do this update if I am using two q functions
                loss = self.model.train_on_batch(np.array(x), y)
                #print "loss is {}".format(loss)

    def policy(self):
        # For now, use an epsilon greedy policy with constant epsilon
        ep = self.epsilon(self.epoch_index)
        if (npr.rand() < ep) or (len(self.D) < 4):
            # Choose an action uniformly at random
            #print "choosing randomly"
            action = npr.rand() < .1
        else:
            state = self.get_last_states()
            #print "calculating optimal action"

            action = np.argmax([self.Q.get(state, 0), self.Q.get(state, 1)])
        return action

    def process_state(self, state):
        # Process the state returned by the world into a format
        # compatible with the learning algorithm
        processed = np.zeros(6)
#        processed[0] = state['score']
        processed[0] = state['tree']['dist'] / 600.
        processed[1] = state['tree']['top'] / 400.
        processed[2] = state['tree']['bot'] / 400.
        processed[3] = state['monkey']['vel']
        processed[4] = state['monkey']['top'] / 400.
        processed[5] = state['monkey']['bot'] / 400.
        return processed
    
    def get_last_states(self):
        index = -1
        last_transition = self.D[index]
        # get previous 3 in addition to last_action
        s_prev1, a_prev1, __, __ = self.D[index -1]
        s_prev2, a_prev2, __, __ = self.D[index -2]
        s, a, r, s_prime = last_transition
        return np.array(list(s_prev2) + [a_prev2] + list(s_prev1) + [a_prev1] +list(s)+ [a] + list(s_prime))
    
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
            new_action = self.policy()
            self.last_action = new_action
            self.last_state = new_state

            self.state_index += 1
            return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if reward == 1:
            reward = 7
        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=1):
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
    iter = 2000
    run_games(agent, hist, iter,10)

    plt.plot(hist, marker='*')
    plt.title('first 100:{}, last 100:{}, max:{} '.format(
    np.sum(hist[:100]),
    np.sum(hist[-100:]),
    np.max(hist)
    ))
    plt.xlabel('iteration of the game')
    plt.ylabel('score')
    plt.savefig('batch size{}_gamma{}_mem_cap{}_optim_{}_iter.png'.format(batch_size,
    gamma,
    mem_cap,
    optim,
    iter))
    plt.show()
    f = open('./run_stats.txt','a')
    f.write('batch size{}_gamma{}_mem_cap{}_optim_{}_iter{}.png'.format(batch_size,
    gamma,
    mem_cap,
    optim,
    iter) + '\n'+ 'first 100:{}, last 100:{}, max:{} '.format(
    np.sum(hist[:100]),
    np.sum(hist[-100:]),
    np.max(hist)
    ) + '\n\n')
    
    # Save history.
    np.save('hist', np.array(hist))
