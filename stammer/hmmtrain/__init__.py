#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
__init__.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2014-06-04
'''
import numpy as np
import numpy
from scipy.stats import multivariate_normal

class MultiGaussianDistribution(object):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.mean = np.zeros([dimensions])
        self.cov = np.ones([dimensions, dimensions])
        #self.cov = np.dot(self.cov, self.cov.T)
        self.distri = multivariate_normal(mean=self.mean, cov=self.cov)
    
    def prob(self, observation):
        return self.distri.pdf(observation)
    
    def update_mean(self, mean):
        self.mean = mean
        self.distri.mean = mean

    def update_cov(self, cov):
        self.cov = cov
        self.distri.cov = cov
    
    def update_distri(self):
        self.distri = multivariate_normal(mean=self.mean, cov=self.cov)


class GaussianMixture(object):
    def __init__(self, component_no, states, vector_len):
        self.state = range(states)
        self.component_no = component_no
        self.component = range(component_no)
        self.distributions = [[MultiGaussianDistribution(vector_len) for i in range(component_no)] for j in range(states)]
        self.weights = np.array([[1.0/component_no for i in range(component_no)] for i in range(states)])
    
    def save_model(self):
        for component in self.component:
            for state in self.state:
                print 'component: %s, state: %s'%(component, state)
                print self.distributions[state][component].mean
                print self.distributions[state][component].cov

def init_a(state_no):
    #a = np.random.random([state_no, state_no])
    a = np.ones([state_no, state_no])
    for i in xrange(state_no):
        a[i] = a[i]/sum(a[i])
    return a

def get_key(prefix, *args, **kwargs):
    return prefix + ':' + '-'.join(map(lambda x:str(x), args))

class Hmm(object):
    def __init__(self, state_no, gm, sequence, pi, a, component_no):
        self.state_no = state_no
        self.a = a
        self.pi = pi
        self.gm = gm
        self.components = component_no
        self.observations = sequence
        self.T = len(sequence)
        self.cache={}

    def do_cache(fn):
        def _(inst, *args, **kwargs):
            key = get_key(fn.func_name, *args, **kwargs)
            if key not in inst.cache:
                res = fn(inst, *args, **kwargs)
                inst.cache[key] = res
            return inst.cache[key]
        return _
    
    @do_cache
    def alpha(self, t, state):
        if t == 0:
            return self.pi[state]*self.b(state, self.observations[0])
        else:
            total = 0.0
            for _state in xrange(self.state_no):
                total += self.alpha(t-1, _state)*self.a[_state][state]
            return total*self.b(state, self.observations[t])

    @do_cache
    def beta(self, t, state):
        if t == self.T - 1:
            return 1
        else:
            total = 0.0
            for _state in xrange(self.state_no):
                total += self.a[state][_state]*self.b(_state, self.observations[t+1])*self.beta(t+1, _state)
            return total

    @do_cache
    def b(self, state, ob):
        return sum([self.b_component(state, com, ob) for com in xrange(self.components)])

    @do_cache
    def b_component(self, state, component, ob):
        return self.gm.weights[state][component]*self.gm.distributions[state][component].prob(ob)
    

    @do_cache
    def gamma_component(self, t, state, component):
        ob = self.observations[t]
        nor = self.gm.weights[state][component]*self.gm.distributions[state][component].prob(ob)
        denor = 0.0
        for i in xrange(self.components):
            denor += self.gm.weights[state][i]*self.gm.distributions[state][i].prob(ob)
        return self.gamma(t, state)*nor/denor

    @do_cache
    def gamma(self, t, state):
        nor, denor = 0.0, 0.0
        nor = self.alpha(t, state)*self.beta(t, state)
        for _state in xrange(self.state_no):
            denor +=self.alpha(t, _state)*self.beta(t, _state)
        return nor/denor

    @do_cache
    def xi(self, t, state_one, state_two):
        nor, denor = 0, 0
        ob = self.observations[t+1]
        nor = self.gamma(t, state_one)*self.a[state_one][state_two]*self.b(state_two, ob)*self.beta(t+1, state_two)
        denor = self.beta(t, state_one)
        return nor/denor
    
    def gamma_sum(self, state):
        return sum([self.gamma(t, state) for t in xrange(self.T)])

    def gamma_component_sum(self, state, component):
        return sum([self.gamma_component(t, state, component) for t in xrange(self.T)])

    def gamma_component_observation_sum(self, state, component):
        return sum([numpy.multiply(self.observations[t], self.gamma_component(t, state, component)) for t in xrange(self.T)])
    
    def gamma_component_cov_sum(self, state, component):
        cov = lambda t:numpy.outer(self.observations[t]-self.gm.distributions[state][component].mean, self.observations[t]-self.gm.distributions[state][component].mean)
        return sum([numpy.multiply(cov(t), self.gamma_component(t, state, component)) for t in xrange(self.T)])
    
    def xi_sum(self, state_one, state_two):
        return sum([self.xi(t, state_one, state_two) for t in xrange(self.T-1)])
    
    def predict(self):
        res = 0.0
        for _state in xrange(self.state_no):
            res += self.alpha(self.T-1, _state)
        return numpy.log(res)

class Trainner(object):
    def __init__(self, state_no, component_no, sequences):
        self.a = init_a(state_no)
        self.state_no = state_no
        self.pi = np.array([1.0/state_no for i in range(state_no)])
        self.component_no = component_no
        self.sequences = sequences
        self.E = len(sequences)
        self.gm = GaussianMixture(component_no=component_no, states=state_no, vector_len=len(self.sequences[0][0]))
        self.hmms = []
        for sequence in sequences:
            self.hmms.append(Hmm(state_no, self.gm, sequence, self.pi, self.a, component_no))
    
    def update(self):
        self.gm = GaussianMixture(component_no=self.component_no, states=self.state_no, vector_len=len(self.sequences[0][0]))
        self.gm.weights = self.weight
        for i in xrange(self.state_no):
            for j in xrange(self.component_no):
                self.gm.distributions[i][j].update_mean(self.mean[i][j])
                self.gm.distributions[i][j].update_cov(self.cov[i][j])
                self.gm.distributions[i][j].update_distri()

        for n, sequence in enumerate(self.sequences):
            #self.hmms[n] = Hmm(self.state_no, self.gm, sequence, self.new_pi, self.new_a, self.component_no)
            self.hmms[n].pi = self.pi
            self.hmms[n].a = self.a
            self.hmms[n].cache = {}
            self.hmms[n].gm = self.gm

    def fit(self, num=100, threashold=0.0001):
        for step in xrange(num):
            new_pi = np.zeros_like(self.pi)
            new_weight = []
            new_cov = []
            new_mean = []
            new_a = np.zeros_like(self.a)
            _hmm = self.hmms[0]
            old = []
            error = 0
            for hmm in self.hmms:
                old.append(hmm.predict())
            for state in xrange(self.state_no):
                _weights = []
                _mean = []
                _cov = []
                for hmm in self.hmms:
                    new_pi[state] += hmm.gamma(0, state)
                new_pi[state] /= self.E

                gamma_denor = np.zeros_like(_hmm.gamma_sum(state))

                for hmm in self.hmms:
                    gamma_denor += hmm.gamma_sum(state)

                for component in xrange(self.component_no):
                    gamma_component_nor = np.zeros_like(_hmm.gamma_component_sum(state, component))
                    gamma_component_observation_sum = np.zeros_like(_hmm.gamma_component_observation_sum(state, component))
                    gamma_component_cov_sum = np.zeros_like(_hmm.gamma_component_cov_sum(state, component))
                    for hmm in self.hmms:
                        gamma_component_nor += hmm.gamma_component_sum(state, component)
                        gamma_component_observation_sum += hmm.gamma_component_observation_sum(state, component)
                        gamma_component_cov_sum += hmm.gamma_component_cov_sum(state, component)
                    _weights.append(gamma_component_nor/gamma_denor)
                    _mean.append(numpy.multiply(gamma_component_observation_sum, 1/gamma_component_nor))
                    _cov.append(numpy.multiply(gamma_component_cov_sum, 1/gamma_component_nor))

                new_mean.append(_mean)
                new_weight.append(_weights)
                new_cov.append(_cov)

                for _state in xrange(self.state_no):
                    xi_sum = np.zeros_like(_hmm.xi_sum(state, state))
                    for hmm in self.hmms:
                        new_a[state][_state] += hmm.xi_sum(state, _state)
                    new_a[state][_state] /= gamma_denor

            print '--------------'
            for n, hmm in enumerate(self.hmms):
                error += old[n]-hmm.predict()
            if error <= threashold and step > 1:
                print error
                return
            else:
                print error
                
            self.a = new_a
            self.pi = new_pi
            self.mean = new_mean
            self.cov = new_cov
            self.weight = new_weight
            self.update()

    def predict(self, ob):
        gm = GaussianMixture(component_no=self.component_no, states=self.state_no, vector_len=len(ob[0]))
        gm.weights = self.weight
        for i in xrange(self.state_no):
            for j in xrange(self.component_no):
                gm.distributions[i][j].update_mean(self.mean[i][j])
                gm.distributions[i][j].update_cov(self.cov[i][j])
                gm.distributions[i][j].update_distri()

            hmm = Hmm(self.state_no, gm, ob, self.pi, self.a, self.component_no)
        return hmm.predict()
            
                
                
if __name__ == "__main__":
    observation_one = [[1, 2, 3], [3, 2, 4], [1, 3, 4], [2, 4, 5], [3, 2, 5]]
    observation_two = [[2, 2, 3], [3, 2, 4], [1, 3, 4], [2, 4, 5], [3, 2, 5]]
    observation_three = [[3, 2, 3], [3, 2, 4], [1, 3, 4], [2, 4, 5], [3, 2, 5]]
    a = []
    a.extend(observation_one)
    a.extend(observation_two)
    mean = numpy.mean(a)
    cov = numpy.cov(a)

    obs = [observation_one, observation_two]
    train = Trainner(2, 2, obs)
    train.fit(20)
    print train.predict(obs[0])
    print train.predict(obs[1])
    print train.predict(observation_three)





