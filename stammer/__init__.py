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
from math import log
import threading
import logging
from collections import defaultdict
from stammer.common import _cut, require_initialize
from finalseg import __hmmcut
from functools import partial

DICTIONARY = "dict.txt"
DICT_LOCK = threading.RLock()
TRIE = None # to be initialized
FREQ = {}
MIN_FREQ = 0.0
TOTAL = 0.0
user_word_tag_tab = {}
INITIALIZED = False


log_console = logging.StreamHandler(sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(log_console)


def setLogLevel(log_level):
    logger.setLevel(log_level)


def build_trie(atree):
    def _(words):
        pointer = atree
        for word in words:
            if word not in pointer:
                pointer[word] = {}
            pointer = pointer[word]
        pointer[''] = '' #end tag
    return _

def gen_trie(f_name):
    lfreq, trie, ltotal = {}, {}, 0.0
    tree_building = build_trie(trie)
    with open(f_name, 'r') as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            try:
                freq, _, words = line.split('-')
                freq = float(freq)
                lfreq[words] = freq
                ltotal = ltotal + freq
                words = words.split()
                tree_building(words)
            except ValueError, e:
                logger.debug('%s at line %s %s' % (f_name,  lineno, line))
                raise ValueError, e

    lfreq = dict([(k,log(float(v)/ltotal)) for k,v in lfreq.iteritems()]) #normalize
    return trie, lfreq, ltotal


def initialize(*args):
    global TRIE, FREQ, TOTAL, INITIALIZED
    _curpath = os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    abs_path = os.path.join(_curpath, DICTIONARY)
    TRIE, FREQ, TOTAL = gen_trie(abs_path)
    MIN_FREQ = min(FREQ.itervalues())
    INITIALIZED = True




def _travel_depth(words, node):
    if not words:
        return 0
    if words[0] == '':
        return 1
    elif words[0] in node:
        return 1+_travel_depth(words[1:], node[words[0]])
    else:
        return 0


@require_initialize(signal=INITIALIZED, init=initialize, iargs=(DICTIONARY,))
def _segment(words):
    seg = defaultdict(list)
    for no, word in enumerate(words):
        s = _travel_depth(words[no:], TRIE)
        if s > 0:
            seg[no].append(s+no-1)
        else:
            seg[no].append(no)
    return seg


hmmcut = partial(_cut, cut_block=__hmmcut)

def handler_buf(buf):
    buf = buf.strip()
    if buf:
        if len(buf.split()) == 1:
            yield buf
        else:
            recognized = hmmcut(buf)
            for t in recognized:
                yield t


def _dict_cut(words):
    segment = _segment(words)
    route = calc(words, segment)
    aggregate = []
    for begin, end in route.iteritems():
        aggregate.append([begin, end[1]+1 if end[1] else begin+1])
    t = 0
    for i, j in aggregate:
        if i < j-1:
            for res in handler_buf(' '.join(words[t:i])):
                yield res
            yield(' '.join(words[i:j]))
            t = j
        elif j == len(words)-1:
            for res in handler_buf(' '.join(words[t:])):
                yield res


def calc(sentence, DAG):
    route = {}
    N = len(sentence)
    route[N] = (0.0, '')
    for idx in xrange(N-1,-1,-1):
        candidates = [(FREQ.get(' '.join(sentence[idx:x+1]), MIN_FREQ) + route[x+1][0], x ) for x in DAG[idx] ]
        route[idx] = max(candidates)
    return route




def _travel_depth(words, node):
    if not words:
        return 0
    if words[0] == '':
        return 1
    elif words[0] in node:
        return 1+_travel_depth(words[1:], node[words[0]])
    else:
        return 0



def cut(sentence):
    for word in _cut(sentence, _dict_cut):
        yield word
