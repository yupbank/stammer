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
    def __init__(self, dimensions, mean=None, cov=None):
        self.dimensions = dimensions
        if mean is None:
            self.mean = np.random.random([dimensions])
        else:
            self.mean = mean

        if cov is None:
            self.cov = np.random.random([dimensions, dimensions])
            self.cov = np.dot(self.cov, self.cov.T)
        else:
            self.cov = cov

        self.distri = multivariate_normal(mean=self.mean, cov=self.cov)
    
    def prob(self, observation):
        return self.distri.pdf(observation)
    
    def update_mean(self, mean):
        self.mean = mean
        self.distri.mean = mean

    def update_cov(self, cov):
        self.cov = cov
        self.distri.cov = cov


def init_a(state_no):
    a = np.random.random([state_no, state_no])
    for i in xrange(state_no):
        a[i] = a[i]/sum(a[i])
    return a

def get_key(prefix, *args, **kwargs):
    return prefix + ':' + '-'.join(map(lambda x:str(x), args))

class Hmm(object):
    def __init__(self, state_no, distributions, sequence, pi, a):
        self.state_no = state_no
        self.a = a
        self.pi = pi
        self.distributions = distributions
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
        return self.distributions[state].prob(ob)


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

    def gamma_observation_sum(self, state):
        return sum([numpy.multiply(self.observations[t], self.gamma(t, state)) for t in xrange(self.T)])
    
    def gamma_component_cov_sum(self, state):
        cov = lambda t:numpy.outer(self.observations[t]-self.distributions[state].mean, self.observations[t]-self.distributions[state].mean)
        return sum([numpy.multiply(cov(t), self.gamma(t, state)) for t in xrange(self.T)])
    
    def xi_sum(self, state_one, state_two):
        return sum([self.xi(t, state_one, state_two) for t in xrange(self.T-1)])
    
    def predict(self):
        res = 0.0
        for _state in xrange(self.state_no):
            res += self.alpha(self.T-1, _state)
        return res

class Trainner(object):
    def __init__(self, state_no, sequences, distributions):
        self.a = init_a(state_no)
        self.state_no = state_no
        self.pi = np.array([1.0/state_no for i in range(state_no)])
        self.sequences = sequences
        self.E = len(sequences)
        self.hmms = []
        self.distributions = distributions
        for sequence in sequences:
            self.hmms.append(Hmm(state_no, distributions, sequence, self.pi, self.a))
    
    def update(self):
        for i in xrange(self.state_no):
            print self.new_mean[i], '-----'
            self.distributions[i].update_mean(self.new_mean[i])
            self.distributions[i].update_cov(self.new_cov[i])

        for n, sequence in enumerate(self.sequences):
            self.hmms[n] = Hmm(self.state_no, distributions, sequence, self.new_pi, self.new_a)

    def fit(self, num=100):
        for step in xrange(num):
            new_pi = []
            new_cov = []
            new_a = []
            new_mean = []
            for state in xrange(self.state_no):
                _weights = []
                _mean = []
                _cov = []
                _a = []
                _pi = 0.0
                print '------------'
                for hmm in self.hmms:
                    print hmm.predict(), '!!'
                    _pi += hmm.gamma(0, state)
                    gamma_denor += hmm.gamma_sum(state)
                    gamma_observation_sum += hmm.gamma_observation_sum(state)
                    gamma_component_cov_sum = +hmm.gamma_component_cov_sum(state)
                _mean += numpy.multiply(gamma_observation_sum, 1/gamma_denor))
                _cov.append(numpy.multiply(gamma_component_cov_sum, 1/gamma_denor))
                for _state in xrange(self.state_no):
                    xi_sum = hmm.xi_sum(state, _state)
                    _a.append(xi_sum/gamma_denor)
                new_pi.append(_pi/self.E)
                new_mean.append(_mean)
                new_cov.append(_cov)
                new_a.append(_a)
            self.new_a = new_a
            self.new_pi = new_pi
            self.new_mean = new_mean
            self.new_cov = new_cov
            self.update()

    def predict(self, ob):
        distributions = []
        for i in xrange(self.state_no):
            distributions[i].update_mean(self.new_mean[i])
            distributions[i].update_cov(self.new_cov[i])

            hmm = Hmm(self.state_no, distributions, ob, self.new_pi, self.new_a, self.component_no)
        return hmm.predict()
            
                
                
if __name__ == "__main__":
    observation_one = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 4, 5], [3, 2, 5]]
    observation_two = [[2, 2, 3], [3, 2, 4], [1, 3, 4], [2, 4, 5], [3, 2, 5]]
    a = []
    a.extend(observation_one)
    a.extend(observation_two)
    a = map(lambda x: numpy.array(x), a)
    mean = numpy.multiply(sum(a), 1.0/len(a))
    cov = np.random.random([3, 3])
    cov = np.dot(cov, cov.T)

    obs = [observation_one, observation_two]
    state_no = 2
    #distri = multivariate_normal(mean=mean, cov=cov)
    distributions=[MultiGaussianDistribution(3, mean=mean, cov=cov) for i in xrange(state_no)]
    train = Trainner(state_no, obs, distributions)
    train.fit(10)
    print train.predict(obs[0])





