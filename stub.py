# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None 
        
    class Q_obj():
        def __init__(self):
            pass
            #TODO: random initialization of Q
            
        def get_quality(self, s, a):
            #TODO
            return
            
        def update(self, s, a, r, s_prime, a_prime):
            updated_Q = self
            #TODO
            return updated_Q

    Q = Q_obj()
        
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
        
        if self.last_action is None:
            new_action = npr.rand() < 0.1 # CHANGE
            new_state  = state

            self.last_action = new_action
            self.last_state  = new_state
            return self.last_action
            
        else:
            
            Q = Q.update()
            # here epsilon griddy with the updated Q and the current state
            return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
        


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
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
	np.save('hist',np.array(hist))


