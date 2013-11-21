#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
common.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2013-11-21
'''

import re
from functools import wraps


def _cut(sentence, cut_block):
    re_word, re_skip = re.compile(ur"([.*(,|\n)]+)", re.U), re.compile(ur"(\r\n|\s)", re.U)
    blocks = re_word.split(sentence)
    
    for blk in filter(lambda x: x.strip(), blocks):
        blk = blk.strip()
        if re.match(r'[(\w+)]+', blk):
            for word in cut_block(blk.split()):
                if word.strip():
                    yield word
        else:
            if blk:
                yield blk


def require_initialize(signal, init, iargs=()):
    def _(fn):
        @wraps(fn)
        def __(*args, **kwargs):
            if not signal:
                if any(iargs):
                    init(*iargs)
                else:
                    init()
            return fn(*args, **kwargs)
        return __
    return _
