# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        # bot (0 - 200) + top (200 - 400) + dist (600 - -200) + vel (-60 - 40) + bot (-40 - 460) + top (0 - 500)
        self.w0 = [0] * (5 * 5 * 5 * 5 * 5 * 5)
        self.w1 = [0] * (5 * 5 * 5 * 5 * 5 * 5)
        self.discount = .9
        self.eta = .1
        self.epochs = 0

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.epochs += 1

    def __get_state_num(self, state):
        total = 0
        tree_bot_scaled = int(state['tree']['bot'] / 50)
        total += max(min(tree_bot_scaled, 5), 0)
        total *= 5
        tree_top_scaled = int((state['tree']['top'] - 200) / 50)
        total += max(min(tree_top_scaled, 5), 0)
        total *= 5
        tree_dist_scaled = int(state['tree']['dist'] / 150)
        total += max(min(tree_dist_scaled, 5), 0)
        total *= 5
        mon_vel_scaled = int((state['monkey']['vel'] + 40) / 16)
        total += max(min(mon_vel_scaled, 5), 0)
        total *= 5
        mon_bot_scaled = int((state['monkey']['bot'] + 50) / 100)
        total += max(min(mon_bot_scaled, 5), 0)
        total *= 5
        mon_top_scaled = int((state['monkey']['top'] + 50) / 100)
        total += max(min(mon_top_scaled, 5), 0)
        return total


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # get index of old state
        old_state_num = -1

        if self.last_state is not None:
            old_state_num = self.__get_state_num(self.last_state)
            # print old_state_num

        # get old reward
        old_rew = self.last_reward
        if self.last_reward is None:
            old_rew = 0

        # get index of new state
        new_state_num = self.__get_state_num(state)

        # get the better of the two actions
        better_action = 0
        if self.w0[new_state_num] < self.w1[new_state_num]:
            better_action = 1

        # set next action to take
        new_action = better_action

        # update Q values
        if self.last_action == 0:
            if better_action == 0:
                self.w0[old_state_num] -= self.eta * (self.w0[old_state_num] -
                                                      (old_rew + self.discount * self.w0[new_state_num]))
                # print self.eta * (self.w0[old_state_num] -
                #                                       (old_rew + self.discount * self.w0[new_state_num]))
            else:
                self.w0[old_state_num] -= self.eta * (self.w0[old_state_num] -
                                                      (old_rew + self.discount * self.w1[new_state_num]))
                # print self.eta * (self.w0[old_state_num] -
                #                                       (old_rew + self.discount * self.w1[new_state_num]))
            # print self.w0[old_state_num]
        else:
            if better_action == 0:
                self.w1[old_state_num] -= self.eta * (self.w0[old_state_num] -
                                                      (old_rew + self.discount * self.w0[new_state_num]))
                # print old_rew + self.discount * self.w0[new_state_num]
                # print self.eta * (self.w0[old_state_num] -
                #                                       (old_rew + self.discount * self.w0[new_state_num]))
            else:
                self.w1[old_state_num] -= self.eta * (self.w0[old_state_num] -
                                                      (old_rew + self.discount * self.w1[new_state_num]))
                # print old_rew + self.discount * self.w1[new_state_num]
                # print self.eta * (self.w0[old_state_num] -
                #                                       (old_rew + self.discount * self.w1[new_state_num]))
            # print self.w1[old_state_num]

        # with 10% probability, explore
        if npr.rand() < .1:
            new_action = abs(1 - better_action)

        new_state = state

        self.last_action = new_action
        self.last_state = new_state
        # print new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
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
    run_games(agent, hist, 500, 50)

    print hist

    # Save history.
    np.save('hist', np.array(hist))
    print sum(hist[0:100])
    print sum(hist[100:200])
    print sum(hist[200:300])
    print sum(hist[300:400])
    print sum(hist[400:500])