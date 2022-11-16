
############################################

# input
# pdf: probability density function
#
# output
# output: cumulative probability density function

############################################

import numpy as np

def cumulativeFunc(pdf):
    
    length = len(pdf[0])
    output = np.zeros((1,length))
    
    for i in range(length):
        output[0][i] = np.sum(pdf[0][:i+1])
    return output