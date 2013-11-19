#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
__init__.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2013-11-20
'''
from __future__ import with_statement
__version__ = '0.1'
__license__ = 'MIT'

import re
import os
import sys
import tempfile
import marshal
from math import log
import random
import threading
from functools import wraps
import logging
import time

import finalseg

DICTIONARY = "dict.txt"
DICT_LOCK = threading.RLock()
TRIE = None # to be initialized
FREQ = {}
MIN_FREQ = 0.0
TOTAL =0.0
user_word_tag_tab={}
INITIALIZED = False


log_console = logging.StreamHandler(sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_console)


def setLogLevel(log_level):
    logger.setLevel(log_level)


def gen_trie(f_name):
    lfreq = {}
    trie = {}
    ltotal = 0.0
    with open(f_name, 'rb') as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            try:
                freq, _, words = line.split('-')
                freq = float(freq)
                lfreq[words] = freq
                ltotal = ltotal + freq
                p = trie
                words = words.split()
                for c in words:
                    if c not in p:
                        p[c] ={}
                    p = p[c]
                p['']='' #ending flag
            except ValueError, e:
                logger.debug('%s at line %s %s' % (f_name,  lineno, line))
                raise ValueError, e
    lfreq = dict([(k,log(float(v)/ltotal)) for k,v in lfreq.iteritems()]) #normalize
    return trie, lfreq, ltotal


def initialize(*args):
    global TRIE, FREQ, TOTAL, INITIALIZED
    _curpath=os.path.normpath( os.path.join(os.getcwd(), os.path.dirname(__file__)))
    abs_path = os.path.join(_curpath, DICTIONARY)
    TRIE, FREQ, TOTAL = gen_trie(abs_path)
    MIN_FREQ = min(FREQ.itervalues())
    INITIALIZED = True


def require_initialized(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if INITIALIZED:
            return fn(*args, **kwargs)
        else:
            initialize(DICTIONARY)
            return fn(*args, **kwargs)

    return wrapped


@require_initialized
def get_DAG(words):
    N = len(words)
    i, j=0, 0
    p = TRIE
    DAG = {}
    while i<N:
        c = words[j]
        if c in p:
            p = p[c]
            if '' in p:
                if i not in DAG:
                    DAG[i]=[]
                DAG[i].append(j)
            j+=1
            if j >= N:
                i += 1
                j = i
                p = TRIE
        else:
            p = TRIE
            i += 1
            j = i

    for i in xrange(N):
        if i not in DAG:
            DAG[i] =[i]
    return DAG


def __cut_DAG(words):
    DAG = get_DAG(words)
    route ={}
    calc(words, DAG, 0, route=route)
    x = 0
    buf =u''
    N = len(words)
    while x<N:
        y = route[x][1]+1
        l_word = words[x:y]
        if y-x==1:
            buf = buf + ' '+ ' '.join(l_word)
        else:
            if len(buf)>0:
                if len(buf)==1:
                    yield buf
                    buf=u''
                else:
                    if (buf not in FREQ):
                        regognized = finalseg.cut(buf)
                        for t in regognized:
                            yield t
                    else:
                        for elem in buf:
                            yield elem
                    buf=u''
            yield ' '.join(l_word)        
        x = y

    if len(buf)>0:
        if len(buf)==1:
            yield buf
        else:
            if (buf not in FREQ):
                regognized = finalseg.cut(buf)
                for t in regognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

def calc(sentence, DAG, idx, route):
    N = len(sentence)
    route[N] = (0.0,'')
    for idx in xrange(N-1,-1,-1):
        candidates = [(FREQ.get(' '.join(sentence[idx:x+1]), MIN_FREQ) + route[x+1][0],x ) for x in DAG[idx] ]
        route[idx] = max(candidates)


#def __cut_DAG_NO_HMM(sentence):
#    re_eng = re.compile(ur'[a-zA-Z0-9]',re.U)
#    DAG = get_DAG(sentence)
#    route ={}
#    calc(sentence,DAG,0,route=route)
#    x = 0
#    N = len(sentence)
#    buf = u''
#    while x<N:
#        y = route[x][1]+1
#        l_word = sentence[x:y]
#        if re_eng.match(l_word) and len(l_word)==1:
#            buf += l_word
#            x =y
#        else:
#            if len(buf)>0:
#                yield buf
#                buf = u''
#            yield l_word        
#            x =y
#    if len(buf)>0:
#        yield buf
#        buf = u''


def cut(sentence, HMM=True):
    re_word, re_skip = re.compile(ur"([*(,|\n)]+)", re.U), re.compile(ur"(\r\n|\s)", re.U)
    blocks = re_word.split(sentence)
    
    cut_block = __cut_DAG
    
    for blk in filter(lambda x: x, map(lambda x: x.strip(), blocks)):
        if re.match(r'[(\w+)]+', blk):
            for word in cut_block(blk.split()):
                if word.strip():
                    yield word
        else:
            if any(blk):
                yield blk
