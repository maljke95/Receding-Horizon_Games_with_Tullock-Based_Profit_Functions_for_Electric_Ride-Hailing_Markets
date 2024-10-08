# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import matrix_rank
import cvxpy as cp
from scipy.linalg import block_diag

def define_player_params(n_st, T, A, B, x0):
    
    lamb = np.ones(n_st)
    lamb[-1] = 0.0
    
    current_eq_r = []
    
    for l in range(T):
        
        current = -np.hstack((np.eye(n_st), np.zeros((n_st, (T-1)*n_st))))
        current_r = -np.linalg.matrix_power(A, l) @ x0
        
        
        select_phi = 0.0*np.ones(T)
        select_phi[l] = -1.0
        
        select_u = np.concatenate((-lamb.T @ np.eye(n_st), np.zeros((T-1)*n_st)))
        
        current_eq_r.append(lamb.T @ current_r)
        
        if l > 0:
            
            for j in range(l):
                
                current = current[:,:-n_st]
                add_ = np.linalg.matrix_power(A, j) @ B
                current = np.hstack((add_, current))
                
                select_u = select_u[:-n_st]
                add_eq   = lamb.T @ np.linalg.matrix_power(A, j) @ B
                select_u = np.concatenate((add_eq, select_u))
                
        current_eq = np.concatenate((select_phi, select_u))
        
        if l == 0:
            
            L = current
            r = current_r
            
            L_eq = current_eq
            
        else: 
            
            L = np.vstack((L, current))
            r = np.hstack((r, current_r))
            
            L_eq = np.vstack((L_eq, current_eq))
    
    L = np.hstack((np.zeros((L.shape[0],T)), L))           
    L = np.vstack((L, np.eye(T*n_st+T)))
    
    r = np.hstack((r, np.zeros(T*n_st+T)))
    
    # L = -L
    # r = -r

    r_eq = np.array(current_eq_r)
    
    return L, r, L_eq, r_eq

def calculate_gradient(T, n_st, beta1_list, beta2_list, eps_array, zi, fminusi_array):
    
    fi_array = zi[:T]
    ui_array = zi[T:]
    
    gradient = []
    
    for k in range(T):
        
        element = beta1_list[k]*(fminusi_array[k] + eps_array[k])/(fi_array[k] + fminusi_array[k] + eps_array[k])**2
        gradient.append(element)
        
        if k == 0:
            
            Bu = np.diag(n_st*[beta2_list[k]])
            
        else:
            
            Bu = np.array(block_diag(Bu, np.diag(n_st*[beta2_list[k]])))
        
    gradient = np.array(gradient)

    gradient_u = - Bu @ ui_array
    
    gradient = np.concatenate((gradient, gradient_u))
    
    return gradient

def calculate_f(zi, L, r, L_eq, r_eq, gradient):
    
    delta = L @ zi - r
    
    list_of_active_left  = []
    list_of_active_idx   = []
    
    list_of_active_right = []
    
    for i in range(len(L_eq)):
        
        list_of_active_left.append(np.squeeze(L_eq[i, :]))
        list_of_active_right.append(r_eq[i])
    
    for i in range(len(delta)):
        
        if delta[i] <= 0:
            
            list_of_active_idx.append(i)
            list_of_active_left.append(np.squeeze(L[i, :]))
            list_of_active_right.append(r[i])
            
    #----- Create H(x) -----
    
    list_of_ind_active = []
    list_of_chosen_idx = []
    list_of_ind_r_active = []
    
    for idx, constr in enumerate(list_of_active_left):
        
        if len(list_of_ind_active) == 0:
            
            list_of_ind_active.append(constr)
            list_of_chosen_idx.append(idx)
            list_of_ind_r_active.append(list_of_active_right[idx])
            
        if len(list_of_ind_active)>0:
            
            rank_before = len(list_of_ind_active)
            
            matrix = list_of_ind_active + [constr]
            matrix = np.array(matrix)
            rank = matrix_rank(matrix.T)
            
            if rank>rank_before:
                
                list_of_ind_active.append(constr)
                list_of_chosen_idx.append(idx)
                list_of_ind_r_active.append(list_of_active_right[idx])
                
    #Hx = np.array(list_of_ind_active).T
    #u = -np.linalg.inv(Hx.T @ Hx) @ Hx.T @ gradient
    
    #f = gradient + Hx @ u
    
    f = project_grad(gradient, np.array(list_of_ind_active), np.array(list_of_ind_r_active))
    
    f = np.squeeze(f)
    
    return f

def project(x_hat, L, r, L_eq, r_eq):
    
    P = np.eye(len(x_hat))
    x = cp.Variable(len(P))
    q = x_hat
            
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - 2*q.T @ x),
                      [L @ x >= r,
                       L_eq @ x == r_eq])

    prob.solve()

    return np.squeeze(np.array(x.value))

def project_grad(x_hat, L_eq, r_eq):
    
    P = np.eye(len(x_hat))
    x = cp.Variable(len(P))
    q = x_hat
            
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - 2*q.T @ x),
                      [L_eq @ x == r_eq])

    prob.solve()

    return np.squeeze(np.array(x.value))

def find_step(z1, z2, f1, f2, tau0, alpha, L1, r1, L_eq1, r_eq1, L2, r2, L_eq2, r_eq2, T, n_st, beta1_list, beta2_list, eps_array):
    
    m = 0.0       
    step_found = False
    forced_stop = False
    
    while not step_found:
        
        step = alpha**m * tau0
        
        z1_hat = z1 + step*f1
        z2_hat = z2 + step*f2
        
        #z1_plus = project(z1_hat, L1, r1, L_eq1, r_eq1)
        #z2_plus = project(z2_hat, L2, r2, L_eq2, r_eq2)
        
        z1_plus = z1_hat
        z2_plus = z2_hat

        z1_minusi_ = z2_plus[T:]
        z2_minusi_ = z1_plus[T:]
        gradient1_ = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_array, z1_plus, z1_minusi_)
        gradient2_ = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_array, z2_plus, z2_minusi_)
        
        f1_ = calculate_f(z1_plus, L1, r1, L_eq1, r_eq1, gradient1_)
        f2_ = calculate_f(z2_plus, L2, r2, L_eq2, r_eq2, gradient2_)
        
        if np.linalg.norm(np.concatenate([f1,f2]))>np.linalg.norm(np.concatenate([f1_,f2_])):
            
            step_found = True
            
        else:
            
            m += 1
            
        if m == 20:
            
            print("Failed")
            #forced_stop = True
            break;
            
        step_found = True
    
    delta = np.linalg.norm(np.concatenate([f1,f2])) - np.linalg.norm(np.concatenate([f1_,f2_]))
    
    return step, np.linalg.norm(np.concatenate([f1_,f2_])), z1_plus, z2_plus, f1_, f2_, gradient1_, gradient2_, forced_stop, m, delta

def find_feasible_init_point(L, r, L_eq, r_eq):
    
    z = cp.Variable(L.shape[1])
    q = np.ones(L.shape[1]).reshape(-1,1)
    
    prob = cp.Problem(cp.Minimize(1),
                  [L    @ z >= r,
                   L_eq @ z == r_eq])

    prob.solve()

    z_init = np.squeeze(np.array(z.value)) 
    
    return z_init

if __name__ == '__main__':
    
    A1 = np.array([[0,0,0],[1,0,0],[0,1,1]])
    B1 = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])
    x01 = np.array([300, 50, 50])
    
    A2 = A1
    B2 = B1
    x02 = np.array([500, 50, 50])
    
    n_st = 3
    T = 4
    
    beta1_list = [100,200,300,100]
    beta2_list = [10,10,10,10] 
    
    
    eps_array  = [10,20,20,10]
    
    max_iter = 3000
    
    tau0 = 0.01
    alpha = 0.5
    
    list_of_z1 = []
    list_of_z2 = []
    
    #----- Define params -----
    
    L1, r1, L_eq1, r_eq1 = define_player_params(n_st, T, A1, B1, x01)
    z_init1 = find_feasible_init_point(L1, r1, L_eq1, r_eq1)

    L2, r2, L_eq2, r_eq2 = define_player_params(n_st, T, A2, B2, x02)
    z_init2 = find_feasible_init_point(L2, r2, L_eq2, r_eq2)
    
    # Promenio sam model
    # Dodao konstraint u>=0
    
    current_iter = 0
    
    z1 = z_init1
    z2 = z_init2

    fminus1_array = z2[:T]
    gradient1 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_array, z1, fminus1_array)

    fminus2_array = z1[:T]
    gradient2 = calculate_gradient(T, n_st, beta1_list, beta2_list, eps_array, z2, fminus2_array)

    f1 = calculate_f(z1, L1, r1, L_eq1, r_eq1, gradient1)
    f2 = calculate_f(z2, L2, r2, L_eq2, r_eq2, gradient2)
    
    list_of_delta = []
    list_of_f1    = []
    list_of_f2    = []
    list_of_z1    = []
    list_of_z2    = []
    list_of_norm  = []
    list_of_forced_m = []
    list_of_step = []
    
    state1_list = []
    state2_list = []
    state1_list.append(x01)
    state2_list.append(x02)
    
    while current_iter < max_iter:
        
        print("Iter: ", current_iter)
        
        list_of_f1.append(f1)
        list_of_f2.append(f2)
        list_of_z1.append(z1)
        list_of_z2.append(z2)
        
        step, norma, z1_plus, z2_plus, f1_, f2_, gradient1_, gradient2_, forced_stop, m, delta = find_step(z1, z2, f1, f2, tau0, alpha, L1, r1, L_eq1, r_eq1, L2, r2, L_eq2, r_eq2, T, n_st, beta1_list, beta2_list, eps_array)
        
        list_of_norm.append(norma)
        list_of_forced_m.append(m)
        list_of_delta.append(delta)
        list_of_step.append(step)
        
        z1 = z1_plus
        z2 = z2_plus
        
        f1 = f1_
        f2 = f2_
        
        gradient1 = gradient1_
        gradient2 = gradient2_
        
        current_iter += 1
        
    
        
        
