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
    print 'we and anyone else Happy New Year and New York is a great place to stay, Happy New Year, which is better than now'
    print '|'.join(cut('we and anyone else Happy New Year and New York is a great place to stay, Happy New Year, which is better than now ?'))
    print 'i love dim sum.'
    print '|'.join(cut('i love dim sum.'))

if __name__ == '__main__':
    main()
