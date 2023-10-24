# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 03:23:39 2021

@author: coals
"""
import numpy as np
from math import floor
from math import ceil

def Multi(Data, S):
     """
     generate the consecutive coarse-grained time series

     Args:
        Data (np.array) : time series data
        S (integer) : the scale factor

    Returns:
        np.array : M_Data
     
     """
     J = 0;
     M_Data = [];
     L = len(Data);
     
     if L/S > 0 :
         J = floor(L/S);
     elif L/S < 0 :
         J = ceil(L/S);
     else:
         J = 0;
    
     for i in range(J):
         M_Data = np.append(M_Data, np.mean(Data[i*S:(i+1)*S]));
     
     
     return M_Data
        
         