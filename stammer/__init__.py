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
from collections import defaultdict

import finalseg


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


def require_initialize(signal=INITIALIZED, init=initialize, args=(DICTIONARY,)):
    def _(fn):
        @wraps(fn)
        def __(*args, **kwargs):
            if not signal:
                init(*args)
            return fn(*args, **kwargs)
        return __
    return _


def _travel_depth(words, trie):
    if not words:
        return 0
    if words[0] == '':
        return 1
    elif words[0] in trie:
        return 1+_travel_depth(words[1:], trie[words[0]])
    else:
        return 0


@require_initialize()
def _segment(words):
    seg = defaultdict(list)
    for no, word in enumerate(words):
        s = _travel_depth(words[no:], TRIE)
        if s > 0:
            seg[no].append(s+no-1)
        else:
            seg[no].append(no)
    return seg


def handler_buf(buf):
    buf = buf.strip()
    if buf:
        if len(buf.split()) == 1:
            yield buf
        else:
            regognized = finalseg.cut(buf)
            for t in regognized:
                yield t

def _cut(words):
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


def cut(sentence, cut_block=_cut):
    re_word, re_skip = re.compile(ur"([.*(,|\n)]+)", re.U), re.compile(ur"(\r\n|\s)", re.U)
    blocks = re_word.split(sentence)
    
    cut_block = _cut
    
    for blk in filter(lambda x: x.strip(), blocks):
        blk = blk.strip()
        if re.match(r'[(\w+)]+', blk):
            for word in cut_block(blk.split()):
                if word.strip():
                    yield word
        else:
            if blk:
                yield blk


def _travel_depth(words, trie):
    if not words:
        return 0
    if words[0] == '':
        return 1
    elif words[0] in trie:
        return 1+_travel_depth(words[1:], trie[words[0]])
    else:
        return 0

@require_initialize()
def _segment(words):
    seg = defaultdict(list)
    for no, word in enumerate(words):
        s = _travel_depth(words[no:], TRIE)
        if s > 0:
            seg[no].append(s+no-1)
        else:
            seg[no].append(no)
    return seg

