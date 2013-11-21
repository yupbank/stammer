#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
my.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2013-11-21
'''

from __future__ import with_statement
import re
import os
import json
import sys
from collections import defaultdict

from stammer.common import require_initialize
MIN_FLOAT=-3.14e100

PROB_START_P = "prob_start.json"
PROB_TRANS_P = "prob_trans.json"
PROB_EMIT_P = "prob_emit.json"
START_P = {}
TRANS_P = {}
EMIT_P = {}
INIT_HMMMODEL = False

PrevStatus = {
    'B':('E','S'),
    'M':('M','B'),
    'S':('S','E'),
    'E':('B','M')
}

#global START_P
#global TRANS_P
#global EMIT_P 
#global INIT_HMMMODEL 

def load_model():
    global START_P
    global TRANS_P
    global EMIT_P 
    _curpath=os.path.normpath( os.path.join( os.getcwd(), os.path.dirname(__file__) )  )

    start_p = {}
    abs_path = os.path.join(_curpath, PROB_START_P)
    with open(abs_path, mode='rb') as f:
        START_P = json.load(f)
    f.closed
    
    trans_p = {}
    abs_path = os.path.join(_curpath, PROB_TRANS_P)
    with open(abs_path, 'rb') as f:
        TRANS_P = json.load(f)
    f.closed
    
    emit_p = {}
    abs_path = os.path.join(_curpath, PROB_EMIT_P)
    with file(abs_path, 'rb') as f:
        EMIT_P = json.load(f)
    f.closed


def initialize_hmm():
    load_model()
    global INIT_HMMMODEL
    INIT_HMMMODEL = True
    

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}] #tabular
    path = {}
    for y in states: #init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0],MIN_FLOAT)
        path[y] = [y]
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t],MIN_FLOAT)
            (prob,state ) = max([(V[t-1][y0] + trans_p[y0].get(y,MIN_FLOAT) + em_p ,y0) for y0 in PrevStatus[y] ])
            V[t][y] =prob
            newpath[y] = path[state] + [y]
        path = newpath
    
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in ('E','S')])
    
    return path[state]


@require_initialize(signal=INIT_HMMMODEL, init=initialize_hmm)
def __hmmcut(sentence):
    pos_list =  viterbi(sentence, ('B','M','E','S'), START_P, TRANS_P, EMIT_P)
    begin, next = 0, 0
    #print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos=='B':
            begin = i
        elif pos=='E':
            yield sentence[begin:i+1]
            next = i+1
        elif pos=='S':
            yield char
            next = i+1
    if next<len(sentence):
        yield sentence[next:]


