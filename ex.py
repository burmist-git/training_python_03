#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Date        : Wed Oct 21 15:38:27 CEST 2020
Autor       : Leonid Burmistrov
Description : Simple reminder-training example.
'''

import numpy as np

def test_expand_dims():
    l = [4,2,3,5]
    x = np.array(l)
    y = np.expand_dims(x, axis=0)
    z = np.expand_dims(x, axis=(0, 1))
    k = np.expand_dims(x, axis=(2, 0))
    print('np.newaxis = ',np.newaxis)
    print('x.ndim     = ',x.ndim)
    print('x.shape    = ',x.shape)
    print('y.ndim     = ',y.ndim)
    print('y.shape    = ',y.shape)
    print('z.ndim     = ',z.ndim)
    print('z.shape    = ',z.shape)
    print('k.ndim     = ',k.ndim)
    print('k.shape    = ',k.shape)
    print('x          = ',x)
    print('y          = ',y)
    print('z          = ',z)
    print('k          = ',k)
    print(x[np.newaxis])
    print(x[:,np.newaxis])
    print("np.newaxis is None : ", np.newaxis is None)

def test_squeeze_dims():
    x = np.array([[[0], [1], [2]]])
    y = np.squeeze(x)
    print(x.shape)
    print(y.shape)
    
def main():
    test_expand_dims()
    test_squeeze_dims()
    
if __name__ == "__main__":
    main()
