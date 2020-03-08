import SS
import numpy as np
import scipy.optimize as opt
import scipy.stats as sts
import math
import scipy.special as spc
import numpy.linalg as lin
import scipy.integrate as intgr

def data_moments(hrs_data, tot_hrs_av):
    '''

    '''
    return hrs_data/tot_hrs_av

def model_moments(init_vals, args):
    '''
    '''
    l_tilde = args[3]
    ss_output = SS.get_SS(init_vals, args, graphs=False)

    n_ss = ss_output["n_ss"]

    return n_ss / l_tilde

def error_vec(hrs_data, tot_hrs_av, init_vals, args, simple):
    '''
    '''
    moms_data = data_moments(hrs_data, tot_hrs_av)
    moms_model = model_moments(init_vals, args)

    if simple:
        err_vec = moms_model - moms_data

    else:
        err_vec = (moms_model - moms_data) / moms_data


    return err_vec

def criterion(params, *args):
    '''
    '''
    beta_0, beta_1, beta_2, beta_3, beta_4, beta_5 = params
    hrs_data, tot_hrs_av, r_init, c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon, A, alpha,\
     delta, Bsct_Tol, Eul_Tol, EulDiff, xi, maxiter, W = args
    
    s_values = np.arange(1, S+1)

    chi_n_vec = beta_0 + beta_1 * s_values + beta_2 * s_values**2 + \
     beta_3 * s_values ** 3 + beta_4 * s_values ** 4 + beta_5 * s_values **5

    mod_args = (S, beta, sigma, l_tilde, b_ellip, upsilon, chi_n_vec, A, alpha,
     delta, Bsct_Tol, Eul_Tol, EulDiff, xi, maxiter)
    init_vals = (r_init, c1_init)
    err = error_vec(hrs_data, tot_hrs_av, init_vals, mod_args, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    print(crit_val)
    
    return crit_val



def calibrate_chi_n(hrs_data, tot_hrs_av, init_vals, args):
    '''
    '''
    r_init, c1_init = init_vals

    S, beta, sigma, l_tilde, beta_0,\
     beta_1, beta_2, beta_3, beta_4, beta_5, b_ellip, upsilon, A, alpha,\
     delta, Bsct_Tol, Eul_Tol, EulDiff, xi, maxiter = args

    params_init = np.array([beta_0, beta_1, beta_2, beta_3, beta_4, beta_5])
    W_hat = np.eye(S)
    gmm_args = (hrs_data,tot_hrs_av, r_init, c1_init, S, beta, sigma, l_tilde, b_ellip, upsilon, A, alpha,\
     delta, Bsct_Tol, Eul_Tol, EulDiff, xi, maxiter, W_hat)
    results = opt.minimize(criterion, params_init, args=gmm_args,
                       tol=1e-14, method='L-BFGS-B')
    
    results_betas = results.x

    return results_betas


