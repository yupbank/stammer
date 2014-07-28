#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
demo.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2014-06-30
'''
from __init__ import Trainner

def main():
    observation_one = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 4, 5], [3, 2, 5]]
    observation_two = [[2, 2, 3], [3, 2, 4], [1, 5, 4], [2, 4, 5], [3, 2, 5]]
    obs = [observation_one, observation_two]
    train = Trainner(2, 2, obs)
    train.fit(10)
    print train.predict(obs[0])
    obs = [[0.3,0.3], [0.1,0.1], [0.2,0.2]]
    train = Trainner(2, 2, [obs])
    train.fit(10)

if __name__ == '__main__':
    main()
