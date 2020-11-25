#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Date        : Wed Oct 21 15:38:27 CEST 2020
Autor       : Leonid Burmistrov
Description : Simple reminder-training example.
'''

import numpy as np
import time

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
    print('x[np.newaxis]      = ',x[np.newaxis])
    print('x[:,np.newaxis]    = ',x[:,np.newaxis])
    print("np.newaxis is None : ", np.newaxis is None)

def test_squeeze_dims():
    x = np.array([[[0], [1], [2]]])
    y = np.squeeze(x)
    print(x.shape)
    print(y.shape)

def fill_np_arr_random(np_arr):
    return np.append(np_arr, np.random.random(size=(n_per_job,)))

def fill_np_arr_append(np_arr):
    tic = time.time()
    np_arr=np.append([],
              [np.random.random(size=(n_per_job,)),
               np.random.random(size=(n_per_job,)),
               np.random.random(size=(n_per_job,)),
               np.random.random(size=(n_per_job,)),
               np.random.random(size=(n_per_job,)),
               np.random.random(size=(n_per_job,))])
    toc = time.time()
    print('{} {:.2f} s'.format(len(np_arr),toc - tic))
    time.sleep(3)
    
def fill_np_arr_for(np_arr):
    tic = time.time()
    for _ in range(0,njobs):
        np_arr = fill_np_arr_random(np_arr=np_arr)
    toc = time.time()
    print('{} {:.2f} s'.format(len(np_arr),toc - tic))
    time.sleep(3)
    
def fill_np_arr_in_one_shot(np_arr):
    tic = time.time()
    np_arr=np.random.random(size=(n_per_job*njobs,))
    toc = time.time()
    print('{} {:.2f} s'.format(len(np_arr),toc - tic))
    time.sleep(3)

def main():
    #test_expand_dims()
    #test_squeeze_dims()

    np_arr = np.array([])
    #fill_np_arr_in_one_shot(np_arr)
    #fill_np_arr_for(np_arr)
    fill_np_arr_append(np_arr)
    
njobs=6
n_per_job=int(1e8)
    
if __name__ == "__main__":
    main()
