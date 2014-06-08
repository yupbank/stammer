#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
em.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2014-05-29
'''
import copy
import numpy as np
from __init__ import GaussianMixture

class Hmm(object):
    def __init__(self, state_no, gm, observations):
        self.state_no = state_no
        self.states = range(state_no)
        self.a = np.zeros([state_no, state_no])
        self.pi = np.array([1.0/state_no]*state_no)
        self.gm = gm
        self.T = len(observations) 
        self.observations = observations
    #emit, b
    def emit(self, time):
        observation = self.observations[time]
        prob = 0.0
        for state in self.states:
            for component in gm.component:
                prob += self.gm.weights[state][component] * self.gm.distributions[state][component].prob(observation)
        return prob

    def emit_with_state(self, time, state):
        print time, '!!', self.observations
        observation = self.observations[time]
        prob = 0.0
        for component in self.gm.component:
            prob += self.gm.weights[state][component] * self.gm.distributions[state][component].prob(observation)
        return prob

    def emit_with_condition(self, time, state, component):
        observation = self.observations[time]
        return self.gm.weights[state][component]*self.gm.distributions[state][component].prob(observation)
    
    # p(o_0,...,o_t|lambda, q_t=state), forward
    def alpha(self, time, state):
        print time, '---'
        if time == 0:
            return self.pi[state]*self.emit_with_state(state, time)
        else:
            total = 0
            for _state in self.states:
                total += self.alpha(time-1, _state)*self.a[_state][state]
            return total*self.emit_with_state(time, state)
    
    #p(o_t,...,o_T|lambda, q_t=state), backword
    def beta(self, time, state):
        if time == self.T-1:
            return 1
        else:
            total = 0
            for _state in self.states:
                total += self.a[state][_state]*self.emit_with_state(time+1, _state)*self.beta(time+1, _state)
            return total

    #p(q_t = state|O, lambda)
    def gamma(self, time, state):
        print time, 'xxx'
        nor = self.alpha(time, state)*self.beta(time, state)
        denor = 0.0
        for _state in self.states:
            denor += self.alpha(time, _state)*self.beta(time, _state)
        return nor/denor

    def gamma_with_component(self, time, state, component):
        nor = self.gamma(time, state)*self.gm.weights[state][component]*self.emit_with_condition(time, state, component)
        denor = self.emit_wih_state(time, state)
        return nor/denor
    
    def xi(self, time, state_one, state_two):
        nor, denor = 0, 0
        nor = gamma_with_state(self, time, state_one)*self.a[state_one][state_two]*self.emit_with_state(time+1, state_two)*self.beta(time+1, state_two)
        denor = self.beta(time,  state_one)
        return nor/denor

    def update_pi(self, state):
        return self.gamma(0, state)
   

    def update_a(self, first, second):
        nor, denor = 0, 0
        for time, observation in enumerate(self.observations):
            nor += self.xi(time, first, second)
            denor += self.gamma(time, first)
        return nor/denor
   
    def update_weight(self, state, component):
        nor, denor = 0, 0
        for time, ob in enumerate(self.observations):
            nor += self.gamma_with_component(time, state, component)
            denor += self.gamma(time, state)
        return nor/denor

    def update_mean(self, state, component):
        nor = np.zeros_like(self.gm.distribution[state][component].means)
        demor = 0
        for time, ob in enumerate(self.observations):
            nor += ob*self.gamma_with_component(time, state, component)
            denor += self.gamma_with_component(time, state, component)
        return nor/denor
    
    def update_cov(self, state, component):
        nor = np.zeros_like(self.gm.distribution[state][component].cov)
        for time, ob in enumerate(self.observations):
            nor += np.outer(ob-self.gm.distribution[state][component].means, ob-self.gm.distribution[state][component].means)*self.gamma_with_component(time, state, component)
            denor += self.gamma_with_component(time, state, component)
        return nor/denor


    def run(self, run_num):
        for run in range(run_num):
            print 'step', run
            new_pi = copy.copy(self.pi)
            new_a = copy.copy(self.a)
            new_mean = []
            new_cov = []
            new_weight = []
            for state in  self.states:
                new_pi[state] = self.update_pi(state)
                for second_state in sel.states:
                    new_a[state][second_state] = self.update_a(state, second_state)
                _mean, _cov, _weight = [], [], []
                for component in self.gm.component:
                    mean = self.update_mean(state, component)
                    cov = self.update_cov(state, component)
                    weight = self.update_weight(state, component)
                    _weight.append(weight)
                    _mean.append(mean)
                    _cov.append(cov)
                new_mean.append(_mean)
                new_cov.append(_cov)
                new_weight.append(_weight)
            self.pi = new_pi
            self.a = new_a
            for state in self.states:
                for component in self.gm.component:
                    self.gm.weights[state][component]=new_weight[state][component]
                    self.gm.distributions[state][component].update_mean(new_mean[state][component])
                    self.gm.distributions[state][component].update_cov(new_cov[state][component])

            
def predict(hmm, observations):
    hmm.observations = observations
    res = 0
    for state in hmm.states:
        res += hmm.alpha(len(observations), state)
    return res

def main():
    observations = [[1, 2, 3], [1, 2, 4]]
    gm = GaussianMixture(component_no=4, states=4, vector_len=3)
    hmm = Hmm(4, gm, observations)
    hmm.run(10)

if __name__ == '__main__':
    main()
