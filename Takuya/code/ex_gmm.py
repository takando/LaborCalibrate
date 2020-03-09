import gmm
import numpy as np
from numpy import loadtxt

import pickle
import os
import SS as ss
import TPI as tpi
import aggregates as aggr
import elliputil as elp
import utilities as utils

# Household parameters
S = int(80)
beta_annual = 0.96
beta = beta_annual ** (80 / S)
sigma = 2.5
l_tilde = 1.0

(beta_0, beta_1, beta_2, beta_3, beta_4, beta_5) = \
(1, 0, 0, 0, 0, 0)

# chi_n_vec = 1.0 * np.ones(S)

# Firm parameters
A = 1.0
alpha = 0.35
delta_annual = 0.05
delta = 1 - ((1 - delta_annual) ** (80 / S))
# SS parameters
SS_solve = True
SS_BsctTol = 1e-13
SS_EulTol = 1e-13
SS_graphs = True
SS_EulDiff = True
xi_SS = 0.15
SS_maxiter = 200

ellip_graph = False
b_ellip_init = 1.0
upsilon_init = 2.0
ellip_init = np.array([b_ellip_init, upsilon_init])
Frisch_elast = 0.9
CFE_scale = 1.0
cfe_params = np.array([Frisch_elast, CFE_scale])
b_ellip, upsilon = elp.fit_ellip_CFE(ellip_init, cfe_params, l_tilde,
                                     ellip_graph)

hrs_data = loadtxt("adj_hrs_data.csv", delimiter=",")

tot_hrs_av = 133

# Compute steady-state solution
ss_args = (S, beta, sigma, l_tilde, beta_0,\
           beta_1, beta_2, beta_3, beta_4, beta_5, b_ellip,\
           upsilon, A, alpha, delta, SS_BsctTol,\
           SS_EulTol, SS_EulDiff, xi_SS, SS_maxiter)


if SS_solve:
    rss_init = 0.06
    c1_init = 0.1
    init_vals = (rss_init, c1_init)

    print('Estimating chi_n by GMM')
    chi_n_vec_gmm = gmm.calibrate_chi_n(hrs_data, tot_hrs_av, init_vals, ss_args)

    print(chi_n_vec)
    