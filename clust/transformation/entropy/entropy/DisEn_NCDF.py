# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 01:30:26 2021

@author: coals
"""
# This function calculates dispersion entropy (DisEn) of a univariate
# signal x, using normal cumulative distribution function (NCDF)
#
# Inputs:
#
# x: univariate signal - a vector of size 1 x N (the number of sample points)
# m: embedding dimension
# nc: number of classes (it is usually equal to a number between 3 and 9 - we used c=6 in our studies)
# tau: time lag (it is usually equal to 1)
# type : select MDE or MCRDE (MDE : 0, MCRDE : 1)
#
# Outputs:
#
# Out_DisEn: scalar quantity - the DisEn of x
# npdf: a vector of length nc^m, showing the normalized number of disersion patterns of x
#
# Ref:
#
# [1] H. Azami, M. Rostaghi, D. Abasolo, and J. Escudero, "Refined Composite Multiscale Dispersion Entropy and its Application to Biomedical
# Signals", IEEE Transactions on Biomedical Engineering, 2017.
# [2] M. Rostaghi and H. Azami, "Dispersion Entropy: A Measure for Time-Series Analysis", IEEE Signal Processing Letters. vol. 23, n. 5, pp. 610-614, 2016.
#
# If you use the code, please make sure that you cite references [1] and [2].
#
# Hamed Azami, Mostafa Rostaghi, and Javier Escudero Rodriguez
# hamed.azami@ed.ac.uk, rostaghi@yahoo.com, and javier.escudero@ed.ac.uk
#
#  20-January-2017

import numpy as np
from scipy.stats import norm
from entropy.MCRDE.cumulativeFunc import cumulativeFunc


def DisEn_NCDF(x,m,nc,tau,Type):
    
    N = len(x);
    sigma = np.std(x);
    mu = np.mean(x);
    
    # x mapping NCDF
    y = norm.cdf(x, loc=mu, scale=sigma);
   
    for i_N in range(N):
        
        if y[i_N] == 1:
            y[i_N] = 1 - np.exp(-10);
        if y[i_N] == 0:
            y[i_N] = np.exp(-10);
    
    z = np.rint(y*nc + 0.5);
    l = np.arange(1,nc+1);
    all_patterns = l[:,np.newaxis];
    
    
    for f in range(1,m):
        temp = all_patterns;
        all_patterns = np.array([]);
        j=0;
        for w in range(nc):
            [a,b] = temp.shape;
            if w==0:
                all_patterns =  np.append(temp, (w+1)*np.ones((a,1)), axis=1);
            else:
                all_patterns =  np.append(all_patterns, 
                                          np.append(temp, (w+1)*np.ones((a,1)), axis=1),
                                          axis=0);
            j = j+a;
    

    key = [];
    for i in range(nc**m):
        key = np.append(key, 0);
        for i_r in range(m):
            key[i] = key[i]*10 + all_patterns[i,i_r];
    
    embd2 = np.zeros((N-(m-1)*tau,1));
    for i in range(m):
        a = z[i*tau:N-(m-i-1)*tau];
        a_T = a[:,np.newaxis];
        embd2 = a_T * 10**(m-i-1) + embd2;
    

    pdf = np.zeros((1,nc**m));
    
    for id in range(nc**m):
        [R,C] = np.where(embd2==key[id]);
        pdf[0][id] = len(R);
    
    npdf = pdf / (N-(m-1)*tau);
    
    Out_DisEn = 0;
    
    # if type=0, MDE  
    if np.sign(Type) == 0:
        p = npdf[npdf != 0];
        Out_DisEn = -np.sum(np.dot(p,np.log(p)));
    
    # if type=1, MCRDE
    elif np.sign(Type) == 1:
        cmf = cumulativeFunc(npdf);
        rsd_cmf = np.zeros((1,len(cmf)));
        rsd_cmf = 1 - cmf[0][:];
        rsd = rsd_cmf[rsd_cmf != 0];
        # save ncpdf
        Out_DisEn = -np.sum(np.dot(rsd,np.log(np.abs(rsd))));
    else:
        print('Error: Undefined type');
    
  
    return Out_DisEn