#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:12:48 2025

@author: guoyijun

Course1 2025-03-11
"""

import random
import numpy as np
import pylab as pl
import scipy .signal as signal
from scipy import fftpack
import math


#矩形函数
def rect(t):    

    temp = np.logical_and(t>-0.5, t<0.5)
    r = temp*1
    return r

#信号内积
def inner(x,y,dt):    
    r=np.inner(x,y)*dt
    return r

#信号卷积
def convolve(x,y,dt):    
    r=np.convolve(x,y,mode='same')*dt
    return r


#傅立叶级数
def FourierSeries(t,x,dt,N,T):
    n=np.arange(-N/2,N/2,1)
    sn=np.zeros(len(n),dtype=np.complex_)
    for i in range(0,N): 
        f0=n[i]/T   #复单频信号的频率
        z=np.cos(2*math.pi*f0*t)-1j*np.sin(2*math.pi*f0*t)   #复单频信号
        sn[i]=np.inner(x,z)*dt/T #计算傅立叶级数系数
    return n,sn

#傅立叶变换
def FourierTransfrom(t,st):
    dt=t[1]-t[0] #时间间隔
    T=t[-1]-t[0]+dt  #总的时间长度
    df=1/T
    N=len(st)
    f=np.arange(-N/2*df,N/2*df,df)
    st_shift=np.fft.ifftshift(st)
    Sf0=np.fft.fft(st_shift)*T/N  
    Sf=np.fft.fftshift(Sf0) 
    return f,Sf

#傅立叶反变换
def RFourierTransfrom(f,Sf):
    df=f[1]-f[0]
    Fmax=f[-1]-f[0]+df
    dt=1/Fmax
    N=len(Sf)
    T=N*dt
    t=np.arange(-N/2*dt,N/2*dt,dt)
    Sf_shift=np.fft.ifftshift(Sf)
    st0=np.fft.ifft(Sf_shift)*N/T
    st=np.fft.fftshift(st0) 
    return t,st







