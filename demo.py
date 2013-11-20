#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
demo.py
Author: yupbank
Email:  yupbank@gmail.com

Created on
2013-11-20
'''
from stammer import cut

def main():
    for word in cut('we and anyone else Happy New Year and New York Jesus, Happy New Year, which is better than now'):
        print '|term|', word

if __name__ == '__main__':
    main()
