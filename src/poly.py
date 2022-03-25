import torch

'''
This definition is not backward friendly. 
#TODO: Fix this
'''

def chebyshev(x, degree):
    '''
    Uses the recursion T_{n+1}(x) = 2x T_{n}(x) - T_{n-1}(x)
    '''

    retvar = torch.zeros(x.size(0), degree+1).type(x.type())
    retvar[:, 0] = x * 0 + 1
    if degree > 0:
        retvar[:, 1] = x
        for ii in range(1, degree):
            retvar[:, ii+1] = 2 * x * retvar[:, ii] -  retvar[:, ii-1]

    return retvar
