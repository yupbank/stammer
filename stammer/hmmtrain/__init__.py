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
from scipy.stats import multivariate_normal

class MultiGaussianDistribution(object):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.mean = np.ones([dimensions])
        self.cov = np.ones([dimensions, dimensions])
        self.distri = multivariate_normal(mean=self.mean, cov=self.cov)
    
    def prob(self, observation):
        return self.distri.pdf(observation)
    
    def update_mean(self, mean):
        self.mean = mean
        self.distri.mean = mean

    def update_cov(self, cov):
        self.cov = cov
        self.distri.cov = cov

class GaussianMixture(object):
    def __init__(self, component_no, states, vector_len):
        self.component_no = component_no
        self.component = range(component_no)
        self.distributions = [[MultiGaussianDistribution(vector_len)]*component_no]*states
        self.weights = np.array([[1.0/component_no]*component_no]*states)
    
    def init_weight(self, observations):
        self.weights = np.zeros([self.component_no, len(observations)])
        for n, observation in enumerate(observations):
            weights[n] = [1.0/len(observation)]*len(observation)


if __name__ == '__main__':
    main()
