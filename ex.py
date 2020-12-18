#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Date        : Wed Oct 21 15:38:27 CEST 2020
Autor       : Leonid Burmistrov
Description : Simple reminder-training example.
'''

import numpy as np
import time

def printinfo(func):
    def wrapper():
        print("")
        print("Simple reminder-training example. Function name : {} --> ".format(func.__name__))
        func()
    return wrapper

@printinfo
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

@printinfo    
def test_squeeze01_dims():
    x = np.array([[[0], [1], [2]],[3,4,5]])
    y = np.squeeze(x)
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)

@printinfo    
def test_squeeze02_dims():
    x = np.array([[[0], [1], [2]]])
    y = np.squeeze(x)
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)

@printinfo
def test_cros_dot_product():
    nPoints=3
    x = np.array([i+1. for i in range(nPoints)])
    y = np.array([i+1. for i in range(nPoints)])

    r_x_y=x*y
    r_x_tr_y=x*np.transpose(y)
    r_s=np.dot(x,np.transpose(y))
    r_m=np.dot(np.transpose(y),x)

    print(' ')
    print('x.ndim  = ', x.ndim)
    print('x.shape = ', x.shape)
    print('x       = ', x)

    print(' ')
    print('y.ndim  = ', y.ndim)
    print('y.shape = ', y.shape)
    print('y       = ', y)

    print(' ')
    print('r_x_y=x*y')
    print('r_x_y.ndim  = ', r_x_y.ndim)
    print('r_x_y.shape = ', r_x_y.shape)
    print('r_x_y       = ', r_x_y)

    print(' ')
    print('r_x_tr_y=x*np.transpose(y)')
    print('r_x_tr_y.ndim  = ', r_x_tr_y.ndim)
    print('r_x_tr_y.shape = ', r_x_tr_y.shape)
    print('r_x_tr_y       = ', r_x_tr_y)

    print(' ')
    print('r_s=np.dot(x,np.transpose(y))')
    print('r_s.ndim  = ', r_s.ndim)
    print('r_s.shape = ', r_s.shape)
    print('r_s       = ', r_s)

    print(' ')
    print('r_m=np.dot(np.transpose(y),x)')
    print('r_m.ndim  = ', r_m.ndim)
    print('r_m.shape = ', r_m.shape)
    print('r_m       = ', r_m)
    
@printinfo
def test_cros_dot_product_expand_dims():
    nPoints=3
    x = np.expand_dims(np.array([i+1. for i in range(nPoints)]),axis=0)
    y = np.expand_dims(np.array([i+1. for i in range(nPoints)]),axis=0)

    r_x_y=x*y
    r_x_tr_y=x*np.transpose(y)
    r_s=np.dot(x,np.transpose(y))
    r_m=np.dot(np.transpose(y),x)
    
    print(' ')
    print('x.ndim  = ', x.ndim)
    print('x.shape = ', x.shape)
    print('x       = ', x)

    print(' ')
    print('y.ndim  = ', y.ndim)
    print('y.shape = ', y.shape)
    print('y       = ', y)

    print(' ')
    print('r_x_y = x*y')
    print('r_x_y.ndim  = ', r_x_y.ndim)
    print('r_x_y.shape = ', r_x_y.shape)
    print('r_x_y       = ', r_x_y)

    print(' ')
    print('r_x_tr_y=x*np.transpose(y)')
    print('r_x_tr_y.ndim  = ', r_x_tr_y.ndim)
    print('r_x_tr_y.shape = ', r_x_tr_y.shape)
    print('r_x_tr_y       = ', r_x_tr_y)

    print(' ')
    print('r_s=np.dot(x,np.transpose(y))')
    print('r_s.ndim  = ', r_s.ndim)
    print('r_s.shape = ', r_s.shape)
    print('r_s       = ', r_s)

    print(' ')
    print('r_m=np.dot(np.transpose(y),x)')
    print('r_m.ndim  = ', r_m.ndim)
    print('r_m.shape = ', r_m.shape)
    print('r_m       = ', r_m)

@printinfo
def test_cros_dot_stack():
    nPoints=3
    x = np.array([i+1. for i in range(nPoints)])
    y = np.array([i+4. for i in range(nPoints)])

    r = np.stack((x,y))
    rr=np.stack(np.array([np.array([i+j*nPoints+1. for i in range(nPoints)]) for j in range(4)]))
    rrr=np.vstack(np.array([np.array([i+j*nPoints+1. for i in range(nPoints)]) for j in range(4)]))
    rrrr=np.hstack(np.array([np.array([i+j*nPoints+1. for i in range(nPoints)]) for j in range(4)]))
    
    print(' ')
    print('x.ndim  = ', x.ndim)
    print('x.shape = ', x.shape)
    print('x       = ', x)

    print(' ')
    print('y.ndim  = ', y.ndim)
    print('y.shape = ', y.shape)
    print('y       = ', y)

    print(' ')
    print('r = np.stack((x,y))')
    print('r.ndim  = ', r.ndim)
    print('r.shape = ', r.shape)
    print('type(r) = ', type(r))
    print('r       = ', r)
    
    print(' ')
    print('stack + list comprehension')
    print('rr.ndim  = ', rr.ndim)
    print('rr.shape = ', rr.shape)
    print('type(rr) = ', type(rr))
    print('rr       = ', rr)

    print(' ')
    print('vstack + list comprehension')
    print('rrr.ndim  = ', rrr.ndim)
    print('rrr.shape = ', rrr.shape)
    print('type(rrr) = ', type(rrr))
    print('rrr       = ', rrr)

    print(' ')
    print('hstack + list comprehension')
    print('rrrr.ndim  = ', rrrr.ndim)
    print('rrrr.shape = ', rrrr.shape)
    print('type(rrrr) = ', type(rrrr))
    print('rrrr       = ', rrrr)

@printinfo
def test_cros_dot_stack_expand_dims():
    nPoints=3
    x = np.expand_dims( np.array([i+1. for i in range(nPoints)]), axis=0)
    y = np.expand_dims( np.array([i+4. for i in range(nPoints)]), axis=0)

    r = np.stack((x,y))
    #rr=np.stack(np.expand_dims(np.array([np.array([i+j*nPoints+1. for i in range(nPoints)])) for j in range(4)]))
    #rrr=np.vstack(np.array([np.array([i+j*nPoints+1. for i in range(nPoints)]) for j in range(4)]))
    #rrrr=np.hstack(np.array([np.array([i+j*nPoints+1. for i in range(nPoints)]) for j in range(4)]))
    
    print(' ')
    print('x.ndim  = ', x.ndim)
    print('x.shape = ', x.shape)
    print('x       = ', x)

    print(' ')
    print('y.ndim  = ', y.ndim)
    print('y.shape = ', y.shape)
    print('y       = ', y)

    print(' ')
    print('r = np.stack((x,y))')
    print('r.ndim  = ', r.ndim)
    print('r.shape = ', r.shape)
    print('type(r) = ', type(r))
    print('r       = ',r)
    print('r[0]    = ',r[0])
    print('r[1]    = ',r[1])
    print('r[0][0] ',r[0][0])
    print('r[1][0] ',r[1][0])
    print('r[0][0][0] ',r[0][0][0])
    print('r[1][0][0] ',r[1][0][0])

    #print(' ')
    #print('stack + list comprehension')
    #print('rr.ndim  = ', r.ndim)
    #print('rr.shape = ', r.shape)
    #print('type(rr) = ', type(rr))
    #print('rr       = ',rr)
    #print('rr[0]    = ',r[0])
    #print('r[1]    = ',r[1])
    #print('r[0][0] ',r[0][0])
    #print('r[1][0] ',r[1][0])
    #print('r[0][0][0] ',r[0][0][0])
    #print('r[1][0][0] ',r[1][0][0])

    '''
    print(' ')
    print('stack + list comprehension')
    print('rr.ndim  = ', rr.ndim)
    print('rr.shape = ', rr.shape)
    print('type(rr) = ', type(rr))
    print('rr       = ', rr)

    print(' ')
    print('vstack + list comprehension')
    print('rrr.ndim  = ', rrr.ndim)
    print('rrr.shape = ', rrr.shape)
    print('type(rrr) = ', type(rrr))
    print('rrr       = ', rrr)

    print(' ')
    print('hstack + list comprehension')
    print('rrrr.ndim  = ', rrrr.ndim)
    print('rrrr.shape = ', rrrr.shape)
    print('type(rrrr) = ', type(rrrr))
    print('rrrr       = ', rrrr)
    '''
    
def fill_np_arr_random(np_arr):
    return np.append(np_arr, np.random.random(size=(n_per_job,)))

def fill_np_arr_append(np_arr):
    tic = time.time()
    np_arr=np.append([],
              [np.random.random(size=(n_per_job,)),
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
    test_expand_dims()
    test_squeeze01_dims()
    test_squeeze02_dims()
    test_cros_dot_product()
    test_cros_dot_product_expand_dims()
    test_cros_dot_stack()
    test_cros_dot_stack_expand_dims()
    
    #np_arr = np.array([])
    #fill_np_arr_in_one_shot(np_arr)
    #fill_np_arr_for(np_arr)
    #fill_np_arr_append(np_arr)
    
njobs=2
n_per_job=int(1e5)
    
if __name__ == "__main__":
    main()
