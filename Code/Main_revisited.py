# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

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
    
    gradient_fi = []
    
    for k in range(T):
        
        element = beta1_list[k] * (fminusi_array[k] + eps_list[k]) / (fi_array[k] + fminusi_array[k] + eps_list[k])**2.0
        gradient_fi.append(element)
        
        if k == 0:
            
            Bu = np.diag(n_st*[beta2_list[k]])
            
        else:
            
            Bu = np.array(block_diag(Bu, np.diag(n_st*[beta2_list[k]])))
    
    gradient_fi = np.array(gradient_fi)
    gradient_u = - Bu @ ui_array
    
    gradient = np.concatenate((gradient_fi, gradient_u))
    
    return gradient

def calculate_f(T, n_st, beta1_list, beta2_list, eps_list, z1, z2, L_inq1, r_inq1, L_eq1, r_eq1, L_inq2, r_inq2, L_eq2, r_eq2):
    
    z = np.concatenate((z1, z2))
    
    L_eq_hat  = np.array(block_diag(L_eq1, L_eq2))
    L_inq_hat = np.array(block_diag(L_inq1, L_inq2))
    
    r_eq_hat   = np.concatenate((r_eq1, r_eq2))
    r_inq_hat = np.concatenate((r_inq1, r_inq2)) 
    
    gradient1 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, z1, z2)
    gradient2 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, z2, z1)
    
    gradient  = np.concatenate((gradient1, gradient2))
    
    Hl = np.vstack((L_inq_hat, L_eq_hat, -L_eq_hat))
    hr = np.hstack((r_inq_hat, r_eq_hat, -r_eq_hat))
    
    #----- Find active constraints -----
    
    list_of_active_idx = []
    list_of_inactive_idx = []
    
    delta = Hl @ z - hr
    
    for i in range(len(delta)):
        
        if delta[i] <= 0.0000001:
            
            list_of_active_idx.append(i)
            
        else:
            
            list_of_inactive_idx.append(i)
    
    H_x = Hl.T
    
    #----- Compute u -----
    
    len_u = H_x.shape[1]
    
    len_J       = len(list_of_active_idx)
    len_J_compl = len(list_of_inactive_idx)
    
    L_eq_u = np.zeros((len_J_compl, len_u))
    
    for i in range(len_J_compl):
        
        L_eq_u[i, list_of_inactive_idx[i]] = 1.0
        
    r_eq_u = np.zeros(len_J_compl)
    
    #-----
    
    L_inq_u = np.zeros((len_J, len_u))

    for i in range(len_J):
        
        L_inq_u[i, list_of_active_idx[i]] = 1.0
        
    r_inq_u = np.zeros(len_J)
    
    #-----
    
    P = H_x.T @ H_x + 0.000001 * np.eye(H_x.shape[1])
    q = H_x.T @ gradient
    
    u = cp.Variable(len_u)
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(u, P) - 2*q.T @ u),
                      [L_inq_u @ u >= r_inq_u,
                       L_eq_u @ u == r_eq_u]) 
    
    prob.solve()
    u = np.squeeze(np.array(u.value))
    
    #----- Clean up u -----
    
    for i in range(len(u)):
        if np.abs(u[i]) < 1e-6:
            u[i] = 0.0
            
    #----- Calculate f -----
    
    f = gradient + H_x @ u
    
    return f

def project(x_hat, L_inq, r_inq, L_eq, r_eq):
    
    P = np.eye(len(x_hat))
    x = cp.Variable(len(P))
    q = x_hat
            
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - 2*q.T @ x),
                      [L_inq @ x >= r_inq,
                       L_eq @ x == r_eq])

    prob.solve()

    return np.squeeze(np.array(x.value))
            
def find_step(z, f, tau0, alpha, T, n_st, beta1_list, beta2_list, eps_list, L_inq1, r_inq1, L_eq1, r_eq1, L_inq2, r_inq2, L_eq2, r_eq2):
    
    lenz = int(len(z)/2)
    
    m = 0.0
    step_found = False
    forced_stop = False
    
    while not step_found:
        
        step = alpha**m * tau0
        
        z_hat = z + step * f

        z1_hat = z_hat[:lenz]
        z2_hat = z_hat[lenz:]      
        
        # z1_plus = project(z1_hat, L_inq1, r_inq1, L_eq1, r_eq1)
        # z2_plus = project(z2_hat, L_inq2, r_inq2, L_eq2, r_eq2)
        
        z1_plus = z1_hat
        z2_plus = z2_hat
        
        z_plus = np.concatenate((z1_plus, z2_plus))

        f_ = calculate_f(T, n_st, beta1_list, beta2_list, eps_list, z1_plus, z2_plus, L_inq1, r_inq1, L_eq1, r_eq1, L_inq2, r_inq2, L_eq2, r_eq2)        
        
        if np.linalg.norm(f_) < np.linalg.norm(f):
            
            step_found = True
            
        else:
            
            m += 1
            
        if m == 20:
            
            print("Failed")
            forced_stop = True
            break;  
            
        delta = -np.linalg.norm(f_) + np.linalg.norm(f)
        
        return step, np.linalg.norm(f_), z_plus, f_, forced_stop, m, delta
        
if __name__ == '__main__':
    
    #----- Parameters -----
    
    A1 = np.array([[0,0,0],[1,0,0],[0,1,1]])
    B1 = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])
    x01 = np.array([300, 50, 50])
    
    A2 = A1
    B2 = B1
    x02 = np.array([500, 50, 50])
    
    n_st = 3
    T = 4
    
    beta1_list = [100,200,300,100]
    beta2_list = [50,2,1,60] 
    eps_list  = [10,20,20,10]
    
    max_iter = 5000
    
    tau0 = 0.001
    alpha = 0.8
    
    #----- Init values -----
    
    L_inq1, r_inq1, L_eq1, r_eq1 = define_player_params(n_st, T, A1, B1, x01)
    z_init1 = find_feasible_init_point(L_inq1, r_inq1, L_eq1, r_eq1)

    L_inq2, r_inq2, L_eq2, r_eq2 = define_player_params(n_st, T, A2, B2, x02)
    z_init2 = find_feasible_init_point(L_inq2, r_inq2, L_eq2, r_eq2)
    
    #------------------------
    
    z1 = z_init1
    z2 = z_init2
    z = np.concatenate((z1, z2))
    f = calculate_f(T, n_st, beta1_list, beta2_list, eps_list, z1, z2, L_inq1, r_inq1, L_eq1, r_eq1, L_inq2, r_inq2, L_eq2, r_eq2)
    
    #----- While loop for convergence -----
    
    current_iter = 0
    lenz = int(len(z)/2)
    
    list_of_z1 = []
    list_of_z2 = []
    list_of_f  = []
    list_of_norm = []
    list_of_m = []
    list_of_delta = []
    
    step = 0.001
    
    while current_iter < max_iter:
        
        print("Iter: ", current_iter)
        
        # list_of_f.append(f)
        # list_of_z1.append(z1)
        # list_of_z2.append(z2)
        
        # step, norma, z_plus, f_, forced_stop, m, delta = find_step(z, f, tau0, alpha, T, n_st, beta1_list, beta2_list, eps_list, L_inq1, r_inq1, L_eq1, r_eq1, L_inq2, r_inq2, L_eq2, r_eq2)
        
        # #----- clean up z -----

        # for i in range(len(z_plus)):
        #     if np.abs(z_plus[i]) < 1e-6:
        #         z_plus[i] = 0.0
        
        # #----------------------
        
        # z = z_plus
        # f = f_
        
        # z1 = z[:lenz]
        # z2 = z[lenz:]
        
        # list_of_norm.append(norma)
        # list_of_delta.append(delta)
        # list_of_m.append(m)
        
        # if forced_stop:
            
        #     print("Forced stop")
        #     break;
        
        # ----- Classic projected gradient -----
        
        z = np.concatenate((z1, z2))

        gradient1 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, z1, z2)
        gradient2 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_list, z2, z1)  
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    