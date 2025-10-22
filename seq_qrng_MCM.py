#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:55:40 2025

@author: carles
"""

import numpy as np
import cvxpy as cp
from cvxpy import *
import chaospy 

id_2 = np.identity(2)
X = np.array([[0.0,1.0],[1.0,0.0]])
Y = np.array([[0.0,1j],[-1j,0.0]])
Z = np.array([[1.0,0.0],[0.0,-1.0]])

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Functions                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def embed(mat,D):
    
    """ Embed a d dimensional matrix "mat" into a D dimensional space """
    d = len(mat)
    if D >= d:
        
        M = np.zeros((D,D))
        M[:d,:d] = mat
        return M

    else:
        return ' d must be equal or smaller than D! ' 
    
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def deltaF(x,xx):

    """ Delta function """
    
    if x == xx:
        return 1.0
    else:
        return 0.0
    
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def generate_equidistant_states(nX,olap):
    
    """ Generates nX states in nX dimensions, all with the same overlap olap """
    
    rho = []
    
    state = []
    
    vec = np.zeros((nX,1))
    vec[0][0] = 1.0
    state += [vec]
    
    vec = np.zeros((nX,1))
    vec[0][0] = olap
    vec[1][0] = np.sqrt(1.0-vec[0][0]**2.0)
    state += [vec]

    for x in range(2,nX):
        
        A = np.array([ np.transpose(state[y])[0][:x] for y in range(0,x)])
        B = np.array([olap for y in range(0,x)])
                
        state_x = list(np.linalg.solve(A,B))
        state_x += [np.sqrt(1.0 - sum([state_x[u]**2.0 for u in range(len(state_x))]))]
        state_x = np.transpose(np.array([state_x]))

        final_state = np.zeros((nX,1))

        final_state[:len(state_x)] = state_x

        state += [final_state]
        
    for element in state:
        rho += [np.kron(element,np.transpose(element))]
    
    return rho

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def max_CB(rho,nX,nB,dim,Q):
    
    """ Maximum Confidence in Bob """
    
    # --------------
    # Variables
    # --------------
    
    M = {}
    for b in range(nB):
        M[b] = cp.Variable((dim,dim),complex=True)
            
    # --------------
    # Constraints
    # --------------
    
    ct = []
    
    ct += [ M[b] >> 0.0 for b in range(nB) ]
    ct += [ M[b] == M[b].H for b in range(nB) ]
    ct += [ sum([ M[b] for b in range(nB) ]) == np.identity(dim) ]
        
    ct += [ cp.real(cp.trace( M[b] @ rho[x] )) == Q for b in range(nX,nB) for x in range(nX) ]

    Conf = [ cp.trace( M[x] @ rho[x] )/(1.0-Q) for x in range(nX) ]

    goal = cp.real(sum([ Conf[x]/nX for x in range(nX) ]))

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Maximize(goal)
    prob = cp.Problem(obj,ct)
    
    output = []
    
    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------
    
    return goal.value

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def max_CC(rho,nX,nB,nC,dim,QB,ConfB_obs,QC):
    
    """ Maixmum confidence obtainable in Charlie, a sequential measurement """
    
    # --------------
    # Variables
    # --------------
    
    G = {}
    for b in range(nB):
        G[b] = {}
        for c in range(nC):
            G[b][c] = cp.Variable((dim,dim),complex=True)
            
    M = [ sum([ G[b][c] for c in range(nC) ]) for b in range(nB) ] # Bob's measurement
    N = [ sum([ G[b][c] for b in range(nB) ]) for c in range(nC) ] # Charlie's measurement
            
    # --------------
    # Constraints
    # --------------
    
    ct = []
    
    ct += [ G[b][c] >> 0.0 for b in range(nB) for c in range(nC) ]
    ct += [ G[b][c] == G[b][c].H for b in range(nB) for c in range(nC) ]
    ct += [ sum([ G[b][c] for b in range(nB) for c in range(nC) ]) == np.identity(dim) ]
        
    ct += [ cp.real(cp.trace( M[b] @ rho[x] )) == QB for b in range(nX,nB) for x in range(nX) ]
    ct += [ cp.real(cp.trace( N[b] @ rho[x] )) == QC for c in range(nX,nC) for x in range(nX) ]

    ConfB = [ cp.trace( M[x] @ rho[x] )/(1.0-QB) for x in range(nX) ]
    ConfC = [ cp.trace( N[x] @ rho[x] )/(1.0-QC) for x in range(nX) ]
    
    ct += [ ConfB_obs == ConfB[x] for x in range(nX) ]

    goal = cp.real(sum([ ConfC[x]/nX for x in range(nX) ]))

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Maximize(goal)
    prob = cp.Problem(obj,ct)
    
    output = []
    
    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------
    
    return goal.value

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Dual_Hmin_Bob(rho_in,CB,QB,nX,nB,dim,xstar):
    
    """ Min-entropy in Bob with a known state preparation """

    # --------------
    # Variables
    # --------------
    
    rho = [ embed(rho_in[x],dim) for x in range(nX) ]

    nL = nB
    
    H = {}
    for l in range(nL):
        H[l] = cp.Variable((dim,dim),complex=True)

    g = cp.Variable()
    f = cp.Variable()
    R = cp.Variable((dim,dim),complex=True)
    
    # --------------
    # Constraints
    # --------------
    
    ct = []

    ct += [ H[l] == H[l].H for l in range(nL) ]
    ct += [ deltaF(b,l)*rho[xstar] - \
            sum([ (g*deltaF(b,nX)+f*deltaF(b,x))*rho[x]/nX for x in range(nX) ])  + \
            H[l] - cp.trace(H[l])*np.identity(dim)/dim - R << 0.0 for b in range(nB) for l in range(nL) ]
        
    # --------------
    # Object function: Shannon entropy
    # --------------
                          
    pg = cp.real(g*QB + f*CB*(1.0-QB) + cp.trace(R))

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Minimize(pg)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1#,
           #     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e4
           #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": prec,
           #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": prec#,
           #     "MSK_DPAR_INTPNT_CO_TOL_INFEAS": prec
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------

    output = []
    
    if pg.value != None:
        output += [ -np.log2(pg.value) ]
    else:
        output += [ None ]
        
    
    return output

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Dual_Hmin_Charlie(rho_in,CB,QB,CC,QC,nX,nB,dim,xstar):
    
    """ Min-entropy in Charlie with a known state preparation """

    # --------------
    # Variables
    # --------------
    
    rho = [ embed(rho_in[x],dim) for x in range(nX) ]

    nL = nB
    
    H = {}
    for l in range(nL):
        H[l] = cp.Variable((dim,dim),complex=True)

    gc = cp.Variable()
    fc = cp.Variable()
    
    gb = cp.Variable()
    fb = cp.Variable()
    
    R = cp.Variable((dim,dim),complex=True)
    
    # --------------
    # Constraints
    # --------------
    
    ct = []

    ct += [ H[l] == H[l].H for l in range(nL) ]
    ct += [ deltaF(c,l)*rho[xstar] - \
            sum([ (gb*deltaF(b,nX)+fb*deltaF(b,x))*rho[x]/nX + (gc*deltaF(c,nX)+fc*deltaF(c,x))*rho[x]/nX for x in range(nX) ])  + \
            H[l] - cp.trace(H[l])*np.identity(dim)/dim - R << 0.0 for b in range(nB) for c in range(nC) for l in range(nL) ]
        
    # --------------
    # Object function: Shannon entropy
    # --------------
                          
    pg = cp.real(gb*QB + fb*CB*(1.0-QB) + gc*QC + fc*CC*(1.0-QC) + cp.trace(R))

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Minimize(pg)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1#,
           #     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e4
           #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": prec,
           #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": prec#,
           #     "MSK_DPAR_INTPNT_CO_TOL_INFEAS": prec
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------

    output = []
    
    if pg.value != None:
        output += [ -np.log2(pg.value) ]
    else:
        output += [ None ]
        
    
    return output

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Dual_Hmin_BC(rho_in,CB,QB,CC,QC,nX,nB,dim,xstar):
    
    """ Min-entropy in Bob & Charlie with a known state preparation """

    # --------------
    # Variables
    # --------------
    
    rho = [ embed(rho_in[x],dim) for x in range(nX) ]

    nL = nB
    
    H = {}
    for lb in range(nL):
        H[lb] = {}
        for lc in range(nL):
            H[lb][lc] = cp.Variable((dim,dim),complex=True)

    gc = cp.Variable()
    fc = cp.Variable()
    
    gb = cp.Variable()
    fb = cp.Variable()
    
    R = cp.Variable((dim,dim),complex=True)
    
    # --------------
    # Constraints
    # --------------
    
    ct = []

    ct += [ H[lb][lc] == H[lb][lc].H for lb in range(nL) for lc in range(nL) ]
    ct += [ deltaF(b,lb)*deltaF(c,lc)*rho[xstar] - \
            sum([ (gb*deltaF(b,nX)+fb*deltaF(b,x))*rho[x]/nX + (gc*deltaF(c,nX)+fc*deltaF(c,x))*rho[x]/nX for x in range(nX) ])  + \
            H[lb][lc] - cp.trace(H[lb][lc])*np.identity(dim)/dim - R << 0.0 for b in range(nB) for c in range(nB) for lb in range(nL) for lc in range(nL) ]
        
    # --------------
    # Object function: Shannon entropy
    # --------------
                          
    pg = cp.real(gb*QB + fb*CB*(1.0-QB) + gc*QC + fc*CC*(1.0-QC) + cp.trace(R))

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Minimize(pg)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1#,
           #     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e4
           #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": prec,
           #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": prec#,
           #     "MSK_DPAR_INTPNT_CO_TOL_INFEAS": prec
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------

    output = []
    
    if pg.value != None:
        output += [ -np.log2(pg.value) ]
    else:
        output += [ None ]
        
    
    return output

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Dual_H_is_Bob(m,w,t,rho_in,CB,QB,nX,nB,dim,xstar):
    
    """ Shannon entropy in Bob with a known state preparation """
    
    # --------------
    # Variables
    # --------------
    
    rho = [ embed(rho_in[x],dim) for x in range(nX) ]

    Hout = sum([w[i]/(t[i]*np.log(2.0)) for i in range(m)])

    tau = [ w[i]/(t[i]*np.log(2.0)) for i in range(m) ]
    cm = sum([tau[i] for i in range(m)])
    
    # for i in range(m):

        
    R = cp.Variable((dim,dim),complex=True)
 
    Q1 = {}
    Q2 = {}
    for i in range(m):
        Q1[i] = {}
        Q2[i] = {}
        for j in range(nC):
            Q1[i][j] = cp.Variable((dim,dim),complex=True)
            Q2[i][j] = cp.Variable((dim,dim),complex=True)
            
    gb = cp.Variable()
    fb = cp.Variable()
    
    D = {}
    F = {}
    for i in range(m):
        D[i] = {}
        F[i] = {}
        for j in range(nC):
            D[i][j] = {}
            F[i][j] = {}
            for b in range(nB):
                D[i][j][b] = cp.Variable((dim,dim),complex=True)
                F[i][j][b] = cp.Variable((dim,dim),complex=True)
                        
    # --------------
    # Constraints
    # --------------
    
    ct = []

    ct += [ sum([ D[i][j][b] for j in range(nB) ]) == sum([ (gb*deltaF(b,nX)+fb*deltaF(b,x))*rho[x]/nX for x in range(nX) ]) + R  for b in range(nB) for i in range(m) ] 

    for i in range(m):
        for j in range(nB):
            for b in range(nB):
        
                ct += [ F[i][j][b].H + F[i][j][b] == ( 2.0*tau[i]*deltaF(b,j)*rho[xstar] + Q1[i][j] - cp.trace(Q1[i][j])*np.identity(dim)/dim ) ]      
                L = tau[i]*rho[xstar]*( (1.0-t[i])*deltaF(j,b) + t[i] ) + Q2[i][j] - cp.trace(Q2[i][j])*np.identity(dim)/dim 
                
                matrix = cp.bmat([[  D[i][j][b]   , F[i][j][b] ],
                                  [  F[i][j][b].H , L       ]])                
                
                ct += [matrix >> 0.0]

    # --------------
    # Object function: Shannon entropy
    # --------------
                        
    H = cm - cp.real(gb*QB + fb*CB*(1.0-QB) + cp.trace(R))
    
    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Maximize(H)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1#,
           #     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e4
           #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": prec,
           #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": prec#,
           #     "MSK_DPAR_INTPNT_CO_TOL_INFEAS": prec
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
                    
        
    # --------------
    # Output
    # --------------
    
    output = []
    
    output += [H.value]
    
    return output

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Dual_H_is_Charlie(m,w,t,rho_in,CB,QB,CC,QC,nX,nB,nC,dim,xstar):
    
    """ Shannon entropy in Charlie with a known state preparation """
    
    # --------------
    # Variables
    # --------------
    
    rho = [ embed(rho_in[x],dim) for x in range(nX) ]

    Hout = sum([w[i]/(t[i]*np.log(2.0)) for i in range(m)])

    tau = [ w[i]/(t[i]*np.log(2.0)) for i in range(m) ]
    
    for i in range(m):
        string = f'\r Running: {np.round(i/m*100,2)}%\r'
        print(string,end="")
        
        R = cp.Variable((dim,dim),complex=True)
     
        Q1 = {}
        Q2 = {}
        for j in range(nC):
            Q1[j] = cp.Variable((dim,dim),complex=True)
            Q2[j] = cp.Variable((dim,dim),complex=True)
                
        gb = cp.Variable()
        fb = cp.Variable()

        gc = cp.Variable()
        fc = cp.Variable()
        
        D = {}
        F = {}
        for j in range(nC):
            D[j] = {}
            F[j] = {}
            for b in range(nB):
                D[j][b] = {}
                F[j][b] = {}
                for c in range(nC):
                    D[j][b][c] = cp.Variable((dim,dim),complex=True)
                    F[j][b][c] = cp.Variable((dim,dim),complex=True)
                            
        # --------------
        # Constraints
        # --------------
        
        ct = []
    
        ct += [ sum([ D[j][b][c] for j in range(nC) ]) == sum([ (gb*deltaF(b,nX)+fb*deltaF(b,x))*rho[x]/nX + (gc*deltaF(c,nX)+fc*deltaF(c,x))*rho[x]/nX for x in range(nX) ]) + R  for b in range(nB) for c in range(nC) ] 
    
        for j in range(nB):
            for b in range(nB):
                for c in range(nC):
            
                    ct += [ F[j][b][c].H + F[j][b][c] == ( 2.0*tau[i]*deltaF(c,j)*rho[xstar] + Q1[j] - cp.trace(Q1[j])*np.identity(dim)/dim ) ]      
                    L = tau[i]*rho[xstar]*( (1.0-t[i])*deltaF(j,c) + t[i] ) + Q2[j] - cp.trace(Q2[j])*np.identity(dim)/dim 
                    
                    matrix = cp.bmat([[  D[j][b][c]   , F[j][b][c] ],
                                      [  F[j][b][c].H , L             ]])                
                    
                    ct += [matrix >> 0.0]

        # --------------
        # Object function: Shannon entropy
        # --------------
                            
        H = -cp.real(gb*QB + fb*CB*(1.0-QB) + gc*QC + fc*CC*(1.0-QC) + cp.trace(R))
        
        # --------------
        # Run the SDP
        # --------------
        
        obj = cp.Maximize(H)
        prob = cp.Problem(obj,ct)
    
        output = []
    
        try:
            mosek_params = {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1#,
               #     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e4
               #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": prec,
               #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": prec#,
               #     "MSK_DPAR_INTPNT_CO_TOL_INFEAS": prec
                }
            prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
        except SolverError:
            something = 10
                        
        if H.value != None:
            Hout += H.value
        else:
            return None
        
    # --------------
    # Output
    # --------------
    
    print('\r \r'*len(string),end="")
    
    output = []
    
    output += [Hout]
    
    return output

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Dual_H_is_global(m,w,t,rho_in,CB,QB,CC,QC,nX,nB,nC,dim,xstar):
    
    """ Shannon entropy in Bob & Charlie with a known state preparation """
    
    # --------------
    # Variables
    # --------------
    
    rho = [ embed(rho_in[x],dim) for x in range(nX) ]

    Hout = sum([w[i]/(t[i]*np.log(2.0)) for i in range(m)])
    tau = [ w[i]/(t[i]*np.log(2.0)) for i in range(m) ]
    
    for i in range(m):
        string = f'\r Running: {np.round(i/m*100,2)}%\r'
        print(string,end="")
        
        R = cp.Variable((dim,dim),complex=True)
     
        Q1 = {}
        Q2 = {}
        for j in range(nB):
            Q1[j] = {}
            Q2[j] = {}
            for k in range(nC):
                Q1[j][k] = cp.Variable((dim,dim),complex=True)
                Q2[j][k] = cp.Variable((dim,dim),complex=True)
                
        gb = cp.Variable()
        fb = cp.Variable()
        
        gc = cp.Variable()
        fc = cp.Variable()
        
        D = {}
        F = {}
        for j in range(nB):
            D[j] = {}
            F[j] = {}
            for k in range(nC):
                D[j][k] = {}
                F[j][k] = {}
                for b in range(nB):
                    D[j][k][b] = {}
                    F[j][k][b] = {}
                    for c in range(nC):
                        D[j][k][b][c] = cp.Variable((dim,dim),complex=True)
                        F[j][k][b][c] = cp.Variable((dim,dim),complex=True)
                            
        # --------------
        # Constraints
        # --------------
        
        ct = []
    
        ct += [ sum([ D[j][k][b][c] for j in range(nB) for k in range(nC) ]) == sum([ (gb*deltaF(b,nX)+fb*deltaF(b,x))*rho[x]/nX + (gc*deltaF(c,nX)+fc*deltaF(c,x))*rho[x]/nX for x in range(nX) ]) + R  for b in range(nB) for c in range(nC) ] 
    
        ct += [ R == R.H ]
        ct += [ Q1[j][k] == Q1[j][k].H for j in range(nB) for k in range(nC)]
        ct += [ Q2[j][k] == Q2[j][k].H for j in range(nB) for k in range(nC)]
    
        for j in range(nB):
            for b in range(nB):
                for k in range(nC):
                    for c in range(nC):
                
                        ct += [ F[j][k][b][c].H + F[j][k][b][c] == ( 2.0*tau[i]*deltaF(b,j)*deltaF(c,k)*rho[xstar] + Q1[j][k] - cp.trace(Q1[j][k])*np.identity(dim)/dim ) ]      
                        L = tau[i]*rho[xstar]*( (1.0-t[i])*deltaF(j,b)*deltaF(k,c) + t[i] ) + Q2[j][k] - cp.trace(Q2[j][k])*np.identity(dim)/dim 
                        
                        matrix = cp.bmat([[  D[j][k][b][c]   , F[j][k][b][c] ],
                                          [  F[j][k][b][c].H , L             ]])                
                        
                        ct += [matrix >> 0.0]

        # --------------
        # Object function: Shannon entropy
        # --------------
                            
        H = -cp.real(gb*QB + fb*CB*(1.0-QB) + gc*QC + fc*CC*(1.0-QC) + cp.trace(R))
        
        # --------------
        # Run the SDP
        # --------------
        
        obj = cp.Maximize(H)
        prob = cp.Problem(obj,ct)
    
        output = []
    
        try:
            mosek_params = {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1#,
               #     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e4
               #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": prec,
               #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": prec#,
               #     "MSK_DPAR_INTPNT_CO_TOL_INFEAS": prec
                }
            prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
        except SolverError:
            something = 10
                        
        if H.value != None:
            Hout += H.value
        else:
            return None
        
    # --------------
    # Output
    # --------------
    
    print('\r \r'*len(string),end="")
    
    output = []
    
    output += [Hout]
    
    return output


#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        MAIN CODE                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

nX = 2 # Number of Inputs in Alice
nB = 3 # Number of Outputs in Bob
nC = 3 # Number of Outputs in Charlie
dim = 2 # Dimension
xstar = 0 # Setting from which we extract randomness

# Gauss-Radau quadrature weitghts (w) and nodes (t)
m_in = 2 # half of the quadrature limit ( m = m_in * 2 )
m = int(m_in*2) # quadrature limit
distribution = chaospy.Uniform(lower=1e-3, upper=1)
t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
t = t[0]

N = 70 # Number of datapoints

olap = 0.99
vec = np.linspace(olap+0.001,0.999,N)
# vec = np.linspace(0.001,0.999,N)

Hvec = [[],[]] # Shannon entropy in Bob & Charlie
HBvec = [[],[]] # Shannon entropy in Bob
HCvec = [[],[]] # Shannon entropy in Charlie

Hminvec = [[],[]] # Min-entropy in Bob & Charlie
HminBvec = [[],[]] # Min-entropy in Bob
HminCvec = [[],[]] # Min-entropy in Charlie

for j in range(N):
    
    r = 1.0#vec[j] # White noise

    QB = 1.0/3.0#vec[j] # Inconclusive rate in Bob
    QC = 1.0/3.0#olap/QB#vec[j]# r*olap/QB #1.0/3.0 # Inconclusive reate in Charlie
    
    # Alice's noisy state preparations
    rho = generate_equidistant_states(nX,olap)
    rho = [ embed(rho[x] ,dim) for x in range(nX) ]
    rho = [ r*rho[x] + (1.0-r)*np.identity(dim)/dim for x in range(nX) ] # White noise component

    # Compute maximum confidences compatible with noisy state preparations
    ConfB = max_CB(rho,nX,nB,dim,QB)
    ConfC = max_CC(rho,nX,nB,nC,dim,QB,ConfB,QC)

    # Alice's noiseless states (used for QRNG)
    rho = generate_equidistant_states(nX,olap)
    rho = [ embed(rho[x] ,dim) for x in range(nX) ]
    
    ## -------------------------
    ## Compute Shannon entropies
    ## -------------------------
    
    # Hout = Dual_H_is_global(m-1,w,t,rho,ConfB,QB,ConfC,QC,nX,nB,nC,dim,xstar)
    # HCout = Dual_H_is_Charlie(m-1,w,t,rho,ConfB,QB,ConfC,QC,nX,nB,nC,dim,xstar)
    HBout = Dual_H_is_Bob(m-1,w,t,rho,ConfB,QB,nX,nB,dim,xstar)
    
    print('SHANNON ENTROPY:',Hout[0],HBout[0],HCout[0])
    
    # Hvec[0] += [vec[j]]
    # Hvec[1] += [Hout[0]]
    
    HBvec[0] += [vec[j]]
    HBvec[1] += [HBout[0]]
    
    # HCvec[0] += [vec[j]]
    # HCvec[1] += [HCout[0]]
    
    ## ---------------------
    ## Compute min-entropies
    ## ---------------------
    
    # Hminout = Dual_Hmin_BC(rho,ConfB,QB,ConfC,QC,nX,nB,dim,xstar)
    # HminBout = Dual_Hmin_Bob(rho,ConfB,QB,nX,nB,dim,xstar)
    # HminCout = Dual_Hmin_Charlie(rho,ConfB,QB,ConfC,QC,nX,nB,dim,xstar)
    
    # print('MIN_ENTROPY',Hminout[0],HminBout[0],HminCout[0])
    
    # Hminvec[0] += [vec[j]]
    # Hminvec[1] += [Hminout[0]]
    
    # HminBvec[0] += [vec[j]]
    # HminBvec[1] += [HminBout[0]]
    
    # HminCvec[0] += [vec[j]]
    # HminCvec[1] += [HminCout[0]]
            
    











