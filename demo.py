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
    for word in cut('We are in New York, we love New Year.'):
        print word

if __name__ == '__main__':
    main()
