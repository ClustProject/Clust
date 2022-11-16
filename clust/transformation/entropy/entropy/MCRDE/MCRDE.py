
########################################################################################

# This function calculates the multiscale cumulative residual dispersion entropy (MCRDE) of a univariate signal x

 # Inputs:

 # x: univariate signal - a vector of size 1 x N (the number of sample points)
 # m: embedding dimension
 # c: number of classes (it is usually equal to a number between 3 and 9 - we used c=6 in our studies)
 # tau: time lag (it is usually equal to 1)
 # Scale: number of scale factors

 # Outputs:

 # Out_MCRDE: a vector of size 1 * Scale - the MCRDE of x

########################################################################################


import numpy as np

from entropy.DisEn_NCDF import DisEn_NCDF
from entropy.DisEn_NCDF_ms import DisEn_NCDF_ms
from entropy.Multi import Multi

def MCRDE(x,m,c,tau,scale):
        
       Out_MCRDE =  np.nan * np.ones((1,scale))
       
       # When Scale=1, MCRDE value 
       Out_MCRDE[0][0] =  DisEn_NCDF(x,m,c,tau,1)
       
       sigma = np.std(x)
       mu = np.mean(x)
       
       # MultiScale Entropy(coarse-graining mean)
       for j in range(1,scale):
           xs = Multi(x,j+1)
           Out_MCRDE[0][j] = DisEn_NCDF_ms(xs,m,c,mu,sigma,tau,1)
         
       return Out_MCRDE