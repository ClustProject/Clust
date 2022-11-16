# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 01:35:23 2021

@author: coals
"""
 # This function calculates the multiscale dispersion entropy (MDE) of a univariate signal x

 # Inputs:

 # x: univariate signal - a vector of size 1 x N (the number of sample points)
 # m: embedding dimension
 # c: number of classes (it is usually equal to a number between 3 and 9 - we used c=6 in our studies)
 # tau: time lag (it is usually equal to 1)
 # Scale: number of scale factors

 # Outputs:

 # Out_MDE: a vector of size 1 * Scale - the MDE of x

 # Ref:
 # [1] H. Azami, M. Rostaghi, D. Abasolo, and J. Escudero, "Refined Composite Multiscale Dispersion Entropy and its Application to Biomedical
 # Signals", IEEE Transactions on Biomedical Engineering, 2017.
 # [2] M. Rostaghi and H. Azami, "Dispersion Entropy: A Measure for Time-Series Analysis", IEEE Signal Processing Letters. vol. 23, n. 5, pp. 610-614, 2016.

 # If you use the code, please make sure that you cite references [1] and [2].

 # Hamed Azami and Javier Escudero Rodriguez
 # Emails: hamed.azami@ed.ac.uk and javier.escudero@ed.ac.uk

 #  20-January-2017
import numpy as np
from entropy.DisEn_NCDF import DisEn_NCDF 
from entropy.DisEn_NCDF_ms import DisEn_NCDF_ms
from entropy.Multi import Multi


def MDE(x,m,c,tau,scale):
       Out_MDE =  np.nan * np.ones((1,scale))
       
       # When Scale=1, MDE value 
       Out_MDE[0][0] =  DisEn_NCDF(x,m,c,tau,0)
       
       sigma = np.std(x)
       mu = np.mean(x)
       
       # MDE value when Scale 2~25(coarse-graining using mean)
       for j in range(1,scale):
           xs = Multi(x,j+1)
           Out_MDE[0][j] = DisEn_NCDF_ms(xs,m,c,mu,sigma,tau,0)
                   
       return Out_MDE