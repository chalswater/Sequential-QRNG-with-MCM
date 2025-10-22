#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:06:29 2025

@author: carles
"""

import numpy as np
from matplotlib import pyplot as plt 

NN = 1000

number_of_samples = 10
vec_samples = list(range(2,number_of_samples+2))

d_stored_vec = []

for s in range(number_of_samples):
    
    N = vec_samples[s]
    
    vec = np.linspace(0.0,0.4,NN)
    Qvec = {}
    for i in range(N):
        Qvec[i] = []
    
    checked = False
    store_d = 0.0
    for j in range(NN):
        
        d = vec[j]
        
        Q = {}
        Q[0] = 0.5*(1.0+d**2)
        for i in range(1,N-1):
            Q[i] = 0.5*(1.0+d**2 / Q[i-1]**2 )
            
        Q[N-1] = d/np.prod([Q[i] for i in range(N-1)])
        Qlast = Q[N-1]
        Qfirsts = [Q[i] for i in range(N-1)]
    
        for i in range(N):
            Qvec[i] += [Q[i]]
            
        if all(Qlast >= q for q in Qfirsts) and checked == False:
            checked = True
            store_d = d
            d_stored_vec += [store_d]
        
       
plt.figure(figsize=(4,3))
plt.xlabel('Number of sequential measurements')
plt.ylabel(r'$\Delta$',size=15)
plt.grid()
plt.xticks(vec_samples)
plt.plot(vec_samples,d_stored_vec,marker='.',c='crimson')
plt.tight_layout()

# plt.savefig('Outputs/finding_N.pdf')



# for i in range(N):
#     plt.plot(vec,Qvec[i])
    
# plt.axvline(store_d)
    
