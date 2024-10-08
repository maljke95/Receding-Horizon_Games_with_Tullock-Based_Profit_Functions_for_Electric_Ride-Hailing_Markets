# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from cvxpy.atoms.affine.wraps import psd_wrap

import os
from datetime import datetime 
import matplotlib.pyplot as plt
from matplotlib import cm
import tikzplotlib

def define_player_params(n_st, T, A, B, x0):
    
    lamb = np.ones(n_st)
    lamb[-1] = 0.0
    
    current_eq_r = []
    
    for l in range(T):
            
        current_l = np.hstack((-np.eye(n_st), np.zeros((n_st, (T-1)*n_st))))
        current_r = -np.linalg.matrix_power(A, l) @ x0
        
        current_eq_r.append(lamb.T @ current_r)
        
        if l > 0:
            
            for j in range(l):
                
                current_l = current_l[:,:-n_st]
                add_ = np.linalg.matrix_power(A, j) @ B
                current_l = np.hstack((add_, current_l))
        
        if l == 0:
            
            L = current_l
            r = current_r
            
            lamb_tilda = lamb
 
        else: 
            
            L = np.vstack((L, current_l))
            r = np.hstack((r, current_r))
                 
            lamb_tilda = np.array(block_diag(lamb_tilda, lamb))
            
    L_eq = np.hstack((-np.eye(T), lamb_tilda @ L ))
    r_eq = lamb_tilda @ r
    
    L2 = L
    L2 = np.vstack((L2, np.eye(T * n_st))) 
    L_inq = np.hstack((np.zeros((L2.shape[0],T)), L2))
    r_inq = np.hstack((r, np.zeros(T*n_st)))
    
    return L_inq, r_inq, L_eq, r_eq

def find_feasible_init_point(L_inq, r_inq, L_eq, r_eq):
    
    z = cp.Variable(L_inq.shape[1])
    
    prob = cp.Problem(cp.Minimize(1),
                  [L_inq    @ z >= r_inq,
                   L_eq @ z == r_eq])

    prob.solve()

    z_init = np.squeeze(np.array(z.value)) 
    
    return z_init

def calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, zi, zi_):
    
    fi_array = zi[:T]
    ui_array = zi[T:]
    
    fminusi_array = zi_[:T]
    uminusi_array = zi_[T:]
    
    gradient_fi = []
    
    for k in range(T):
        
        element = beta1_list[k] * (fminusi_array[k] + eps_list[k]) / (fi_array[k] + fminusi_array[k] + eps_list[k])**2.0
        gradient_fi.append(element)
        
        if k == 0:
            
            Bu = np.diag(n_st*[beta2_list[k]])
            
        else:
            
            Bu = np.array(block_diag(Bu, np.diag(n_st*[beta2_list[k]])))
    
    gradient_fi = np.array(gradient_fi)
    gradient_u = - 2*Bu @ ui_array -  Bu @ uminusi_array
    
    gradient = np.concatenate((gradient_fi, gradient_u))
    
    return gradient

def project(x_hat, L_inq, r_inq, L_eq, r_eq):
    
    P = np.eye(len(x_hat))
    
    x = cp.Variable(len(P))
    
    P = psd_wrap(P)
    q = x_hat
            
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - 2*q.T @ x),
                      [L_inq @ x >= r_inq,
                       L_eq @ x == r_eq])

    prob.solve()

    return np.squeeze(np.array(x.value))

def check_stationarity(zi, gradient, L_inq, r_inq, L_eq, r_eq):

    #----- Find active constraints -----
    
    list_of_active_idx = []
    list_of_active_constr = []
    
    list_of_inactive_idx = []
    
    delta = L_inq @ zi - r_inq
    
    for i in range(len(delta)):
        
        if delta[i] <= 0.0000001:
            
            list_of_active_idx.append(i)
            list_of_active_constr.append(-L_inq[i,:])
            
        else:
            
            list_of_inactive_idx.append(i)
    
    #----------
    
    if len(list_of_active_idx) == 0:
        
        H_u = L_eq.T
        len_u = H_u.shape[1]
        u = cp.Variable(len_u)
        
        P = H_u.T @ H_u 
        P = psd_wrap(P)

        q = H_u.T @ gradient     
        
        prob = cp.Problem(cp.Minimize(cp.quad_form(u, P) - 2*q.T @ u))
        prob.solve()
        u = np.squeeze(np.array(u.value))
        
    else:
        
        H_u = np.vstack((np.array(list_of_active_constr), L_eq)).T
        len_u = H_u.shape[1]
        u = cp.Variable(len_u)
        
        len_J = len(list_of_active_idx)
        L_inq_u = np.hstack((np.eye(len_J), np.zeros((len_J, L_eq.shape[0]))))
        r_inq_u = np.zeros(len_J)
        
        P = H_u.T @ H_u 
        P = psd_wrap(P)

        q = H_u.T @ gradient         
        prob = cp.Problem(cp.Minimize(cp.quad_form(u, P) - 2*q.T @ u),
                          [L_inq_u @ u >= r_inq_u])
        prob.solve()
        u = np.squeeze(np.array(u.value))

    f = -gradient + H_u @ u         
    
    return np.linalg.norm(f)

def find_output(n_st, T, A1, B1, x01, A2, B2, x02, beta1_list, beta2_list, eps_list, step, max_iter):

    #----- Init values -----
    
    L_inq1, r_inq1, L_eq1, r_eq1 = define_player_params(n_st, T, A1, B1, x01)
    z_init1 = find_feasible_init_point(L_inq1, r_inq1, L_eq1, r_eq1)

    L_inq2, r_inq2, L_eq2, r_eq2 = define_player_params(n_st, T, A2, B2, x02)
    z_init2 = find_feasible_init_point(L_inq2, r_inq2, L_eq2, r_eq2)

    #------------------------
    
    z1 = z_init1
    z2 = z_init2
    z = np.concatenate((z1, z2))
    
    #----- While loop for convergence -----
    
    current_iter = 0
    lenz = int(len(z)/2)

    list_of_norm = []
    list_of_f1 = []
    list_of_f2 = []
    
    f1 = 100.0
    f2 = 100.0
    
    while current_iter < max_iter or (f1>0.05 or f2>0.05):
        
        if current_iter % 100 == 0:
            print("Iter: ", current_iter)
        
        # ----- Classic projected gradient -----
        
        z = np.concatenate((z1, z2))

        gradient1 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, z1, z2)
        gradient2 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, z2, z1)  
        
        f1 = check_stationarity(z1, gradient1, L_inq1, r_inq1, L_eq1, r_eq1)
        f2 = check_stationarity(z2, gradient2, L_inq2, r_inq2, L_eq2, r_eq2)
        
        list_of_f1.append(f1)
        list_of_f2.append(f2)
        
        gradient = np.concatenate((gradient1, gradient2))
        
        z_hat = z + step * gradient
        
        z1_hat = z_hat[:lenz]
        z2_hat = z_hat[lenz:]
        
        z1_plus = project(z1_hat, L_inq1, r_inq1, L_eq1, r_eq1)
        z2_plus = project(z2_hat, L_inq2, r_inq2, L_eq2, r_eq2) 
        
        z_plus = np.concatenate((z1_plus, z2_plus))
        
        norma = np.linalg.norm(z-z_plus)
        list_of_norm.append(norma)
        
        z1 = z1_plus
        z2 = z2_plus
        
        current_iter += 1
        
    z1_solution = z1_plus
    z2_solution = z2_plus
    
    return z1_solution, z2_solution, list_of_norm, list_of_f1, list_of_f2

def simulation(x01, x02, demand_profile, price_profile, abandonments_profile, T, max_iter):

    A1 = np.array([[0,0,0],[1,0,0],[0,1,1]])
    B1 = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])
    
    A2 = A1
    B2 = B1
    
    step = 0.05 
    n_st = 3
    
    #----- Store data -----
    
    list_of_norm_final = []
    list_of_f1_final    = []
    list_of_f2_final    = []
    
    list_of_u1   = []
    list_of_u2   = []
    list_of_x1   = [x01]
    list_of_x2   = [x02]
    
    
    #----- Receding horizon simulation -----
    
    total_steps = len(demand_profile)-T+1
    
    counter = 0
    
    list_of_z1 = []
    list_of_z2 = []
    
    while counter<total_steps:
        
        print("Step: ",counter)
        
        beta1_list = []
        beta2_list = []
        eps_list   = []
        
        for i in range(counter, counter+T):
            
            beta1_list.append(demand_profile[i])
            beta2_list.append(price_profile[i])
            eps_list.append(abandonments_profile[i])
            
        print("beta1: ",beta1_list)
        print("beta2: ",beta2_list)
        print("eps  : ",eps_list)
            
        z1_solution, z2_solution, list_of_norm, list_of_f1, list_of_f2 = find_output(n_st, T, A1, B1, x01, A2, B2, x02, beta1_list, beta2_list, eps_list, step, max_iter)
        
        list_of_z1.append(z1_solution)
        list_of_z2.append(z2_solution)
        
        list_of_norm_final.append(list_of_norm[-1])
        list_of_f1_final.append(list_of_f1[-1])
        list_of_f2_final.append(list_of_f2[-1])
        
        print("f1: ",list_of_f1_final)
        print("f2: ",list_of_f2_final)
         
        if counter == total_steps - 1:
            
            for j in range(T):
                
                idx = T+j*n_st
                
                u1 = z1_solution[idx:idx+n_st]
                u2 = z2_solution[idx:idx+n_st]
                list_of_u1.append(u1)
                list_of_u2.append(u2)
                
                x01 = A1 @ x01 + B1 @ u1
                x02 = A2 @ x02 + B2 @ u2
        
                list_of_x1.append(x01)
                list_of_x2.append(x02)
                
        else:
            
            u1 = z1_solution[T:T+n_st]
            u2 = z2_solution[T:T+n_st]
        
            print("u1: ",u1)
            print("u2: ",u2)
        
            x01 = A1 @ x01 + B1 @ u1
            x02 = A2 @ x02 + B2 @ u2 
            
            list_of_u1.append(u1)
            list_of_u2.append(u2)
 
            list_of_x1.append(x01)
            list_of_x2.append(x02)
        
        counter += 1
        
        print(30*'-')
        
    #----- Save results -----
    
    current_folder = os.getcwd() + '/Results'
    
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)    

    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    
    name = current_folder + "/" + date_time
    
    if not os.path.isdir(name):
        os.makedirs(name)   
        
    np.save(name + '/list_of_u1.npy', np.array(list_of_u1))
    np.save(name + '/list_of_u2.npy', np.array(list_of_u2))
    
    np.save(name + '/list_of_x1.npy', np.array(list_of_x1))
    np.save(name + '/list_of_x2.npy', np.array(list_of_x2))
    
    np.save(name + '/list_of_norm_final.npy', np.array(list_of_norm_final))
    np.save(name + '/list_of_f1_final.npy', np.array(list_of_f1_final))
    np.save(name + '/list_of_f2_final.npy', np.array(list_of_f2_final))
    
    np.save(name + '/list_of_z1.npy', np.array(list_of_z1))
    np.save(name + '/list_of_z2.npy', np.array(list_of_z2))
    
    np.save(name + '/demand_profile.npy', np.array(demand_profile))    
    np.save(name + '/price_profile.npy', np.array(price_profile))
    np.save(name + '/abandonments_profile.npy', np.array(abandonments_profile))
    
    np.save(name + '/T.npy', np.array([T]))
        
    return list_of_u1, list_of_u2, list_of_x1, list_of_x2, list_of_norm_final, list_of_f1_final, list_of_f2_final, list_of_z1, list_of_z2

def prepare_plots(name):
    
    current_folder = os.getcwd() + '/Results'
    file = current_folder + "/" + name
    
    list_of_u1 = np.load(file + '/list_of_u1.npy')
    list_of_u2 = np.load(file + '/list_of_u2.npy')
    
    list_of_x1 = np.load(file + '/list_of_x1.npy')
    list_of_x2 = np.load(file + '/list_of_x2.npy')
    
    list_of_norm_final = np.load(file + '/list_of_norm_final.npy')
    list_of_f1_final = np.load(file + '/list_of_f1_final.npy')
    list_of_f2_final = np.load(file + '/list_of_f2_final.npy')
    
    list_of_z1 = np.load(file + '/list_of_z1.npy')
    list_of_z2 = np.load(file + '/list_of_z2.npy')    
    
    demand_profile = np.load(file + '/demand_profile.npy')    
    price_profile = np.load(file + '/price_profile.npy')
    abandonments_profile = np.load(file + '/abandonments_profile.npy')
    
    T = np.load(file + '/T.npy')[0]
    
    #----------
    
    len_x = len(list_of_x1)
    
    fig1, ax1 = plt.subplots(dpi=180)
    list_of_colors = ['green', 'orange', 'red', 'black']

    list_of_parked1 = list_of_x1[:len_x-1,2] - list_of_u1[:,2]
    
    k = np.arange(len_x)
    
    for i in range(len(list_of_colors)-1):
        
        color = list_of_colors[i]
        ax1.plot(k, list_of_x1[:,i], '*--', color=color, label=color)
        
    ax1.plot(k[:-1], list_of_parked1, '*--', color=list_of_colors[3], label='parked')
        
    ax1.grid('on')
    ax1.legend()
    ax1.set_xlabel("Iteration [k]")
    ax1.set_ylabel("Vehicle accumulation 1")
    ax1.set_xlim((0, len(k)-1))
    
    fig1.savefig(file + "/p1.jpg", dpi=180)
    tikzplotlib.save(file + "/p1.tex")  
    
    #----------
    
    fig2, ax2 = plt.subplots(dpi=180)
    
    list_of_parked2 = list_of_x2[:len_x-1,2] - list_of_u2[:,2]

    for i in range(len(list_of_colors)-1):
        
        color = list_of_colors[i]
        ax2.plot(k, list_of_x2[:,i], '*--', color=color, label=color)
        
    ax2.plot(k[:-1], list_of_parked2, '*--', color=list_of_colors[3], label='parked')
        
    ax2.grid('on')
    ax2.legend()
    ax2.set_xlabel("Iteration [k]")
    ax2.set_ylabel("Vehicle accumulation 2")
    ax2.set_xlim((0, len(k)-1))

    fig2.savefig(file + "/p2.jpg", dpi=180)
    tikzplotlib.save(file + "/p2.tex") 
    
    #----------
    
    fig3, ax3 = plt.subplots(dpi=180)

    for i in range(len(list_of_colors)-1):
        
        color = list_of_colors[i]
        ax3.plot(k[:-1], list_of_u1[:,i], '*--', color=color, label=color)    

    ax3.grid('on')
    ax3.legend()
    ax3.set_xlabel("Iteration [k]")
    ax3.set_ylabel("Control 1")
    ax3.set_xlim((0, len(k)-1))

    fig3.savefig(file + "/p3.jpg", dpi=180)
    tikzplotlib.save(file + "/p3.tex") 
    
    #----------
    
    fig4, ax4 = plt.subplots(dpi=180)

    for i in range(len(list_of_colors)-1):
        
        color = list_of_colors[i]
        ax4.plot(k[:-1], list_of_u2[:,i], '*--', color=color, label=color)    

    ax4.grid('on')
    ax4.legend()
    ax4.set_xlabel("Iteration [k]")
    ax4.set_ylabel("Control 2")
    ax4.set_xlim((0, len(k)-1))

    fig4.savefig(file + "/p4.jpg", dpi=180)
    tikzplotlib.save(file + "/p4.tex") 
    
    #-----------
    
    fig5, ax5 = plt.subplots(dpi=180)
    ax5.step(k[:-1], demand_profile, '--', where='post', color='black')
    #ax5.grid('on')
    #ax5.set_xlabel("Iteration [k]")
    #ax5.set_ylabel("Demand")
    ax5.set_xlim((0, len(k)-1))    

    fig5.savefig(file + "/p5.jpg", dpi=180)
    tikzplotlib.save(file + "/p5.tex") 

    fig6, ax6 = plt.subplots(dpi=180)
    ax6.step(k[:-1], price_profile, '--', where='post', color='black')
    #ax6.grid('on')
    #ax6.set_xlabel("Iteration [k]")
    #ax6.set_ylabel("Prices")
    ax6.set_xlim((0, len(k)-1)) 

    fig6.savefig(file + "/p6.jpg", dpi=180)
    tikzplotlib.save(file + "/p6.tex") 
    
    fig7, ax7 = plt.subplots(dpi=180)
    ax7.step(k[:-1], abandonments_profile, '--', where='post', color='black')
    #ax7.grid('on')
    #ax7.set_xlabel("Iteration [k]")
    #ax7.set_ylabel("Abandonments")
    ax7.set_xlim((0, len(k)-1))

    fig7.savefig(file + "/p7.jpg", dpi=180)
    tikzplotlib.save(file + "/p7.tex") 
    
    #----------
    
    list_of_profit1 = []
    list_of_profit2 = []
    list_of_lost    = []
    
    for i in range(len_x-1):
        
        fia = list_of_z1[0][i]
        fib = list_of_z2[0][i]
        
        pa = demand_profile[i]*(fia)/(fia + fib + abandonments_profile[i])
        pb = demand_profile[i]*(fib)/(fia + fib + abandonments_profile[i])
        pl = demand_profile[i]*(abandonments_profile[i])/(fia + fib + abandonments_profile[i])
        
        Bu = np.diag(3*[price_profile[i]])
        
        cha = - list_of_u1[i].T @ Bu @ (list_of_u1[i] + list_of_u2[i])
        chb = - list_of_u2[i].T @ Bu @ (list_of_u1[i] + list_of_u2[i])
        
        list_of_profit1.append(pa+cha)
        list_of_profit2.append(pb+chb)
        list_of_lost.append(pl)

    list_of_profit1 = np.array(list_of_profit1)
    list_of_profit2 = np.array(list_of_profit2)
    list_of_lost    = np.array(list_of_lost)
    
    fig8, ax8 = plt.subplots(dpi=180)
    
    ax8.step(k[:-1], list_of_profit1, '*--', color='purple',where='post', label='profit a')
    ax8.step(k[:-1], list_of_profit2, '*--', color='orange',where='post', label='profit b')
    ax8.step(k[:-1], list_of_lost   , '*--', color='black' ,where='post', label='profit b')

    ax8.grid('on')
    ax8.set_xlabel("Iteration [k]")
    ax8.set_ylabel("Profit")
    ax8.set_xlim((0, len(k)-1))

    fig8.savefig(file + "/p8.jpg", dpi=180)
    tikzplotlib.save(file + "/p8.tex") 
    
    print(np.sum(list_of_profit1))
    print(np.sum(list_of_profit2))
    print(np.sum(list_of_lost))
    
if __name__ == '__main__':
    
    #----- Parameters -----

    x01 = np.array([400, 50, 10])
    x02 = np.array([800, 50, 10])

    T = 9
    
    max_iter = 1000
    
    demand_profile       = [5000, 5000, 80000, 160000, 140000, 100000, 20000, 5000, 5000]
    price_profile        = [1  , 1  , 0.1  , 0.1   , 0.1   , 0.5  , 1.5 , 1.5 , 1.5] 
    abandonments_profile = [10 , 20 , 30,  50 , 50 , 40 , 20 , 10 , 10] 
    
    list_of_u1, list_of_u2, list_of_x1, list_of_x2, list_of_norm_final, list_of_f1_final, list_of_f2_final, list_of_z1, list_of_z2 = simulation(x01, x02, demand_profile, price_profile, abandonments_profile, T, max_iter)
    
    #----- PLOT -----

    # name1 = '/03_20_2024_00_51_11'
    # name2 = '/03_20_2024_00_54_02'
    # name3 = '/03_20_2024_00_58_18'

    # nameT3 = '/03_20_2024_01_20_33'
    # nameT6 = '/03_20_2024_01_22_14'
    # nameT9 = '/03_20_2024_01_23_01'
    
    # prepare_plots(nameT3)
    
    
    # #----- TESTING -----
    
    # A1 = np.array([[0,0,0],[1,0,0],[0,1,1]])
    # B1 = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])
    
    # A2 = A1
    # B2 = B1
    
    # step = 0.05
    # n_st = 3  

    # beta1_list = []
    # beta2_list = []
    # eps_list   = []
    
    # counter = 1
    
    # for i in range(counter, counter+T):
            
    #     beta1_list.append(demand_profile[i])
    #     beta2_list.append(price_profile[i])
    #     eps_list.append(abandonments_profile[i])

    # z1_solution, z2_solution, list_of_norm, list_of_f1, list_of_f2 = find_output(n_st, T, A1, B1, x01, A2, B2, x02, beta1_list, beta2_list, eps_list, step, max_iter)    
    
    # plt.plot(list_of_f1)  
    # plt.plot(list_of_f2)
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    

