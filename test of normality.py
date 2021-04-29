# -*- coding: utf-8 -*-
"""
File: 
Created on Tue Apr 20 09:55:49 2021

@author: kratz
@purpose: 
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import distributions as iid
from scipy.optimize import minimize
from scipy.special import factorial2
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import progressbar

import sympy as sym
from sympy.stats import E, Normal
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl
from wolframclient.language import wlexpr
import warnings


##############################################
#                Functions
##############################################

def dgp(N, mu, sigma, dist = 'normal'):
    """Generate a series of X observations from a type dist distribution.

    Satisfies model:
        X ~iid Normal(mu, sigma)

    Each element of the series is a single observation of X

    Inputs include
    - N :: number of observations to sample
    - mu :: Expectation of X
    - sigma :: Variance of X
    - dist :: one of ['normal', 'cauchy', 'poisson']
    """
    if dist=='normal':
        return iid.norm(loc=mu, scale=sigma).rvs(size=(N,1))
    if dist=='cauchy':
        return iid.cauchy(loc=mu, scale=sigma).rvs(size=(N,1))
    if dist=='poisson':
        return iid.poisson(mu).rvs(size=(N,1))

    
def moment_restriction(b, x, k):
    """k^th moment condition evaluated at the observation x
    
    note: k%2 = "k modulo 2" = remainder of k/2
          so k%2=0 if k is even
          (k%2==0) is an indicator function (=1 if k%2=0, 0 otherwise)
    """
    mu, sigma = b
    return (x - mu)**k - (k%2==0)*sigma**k*factorial2(k-1)
    

def gj(b,K,x):
    """The observation-level moment restriction for normal distribution null hypothesis.
    Observations of g_j(b). gj is a single observation
    
    K = # of moments to consider

    This defines the deviations from the predictions of our model; i.e.,
    k=1: e1 = x - mu,               where E[e1] = E[x - mu] = 0 under the null
    k=2: e2 = (x - mu)^2 - sigma^2, where E[e2] = E[(x-mu)^2 - sigma^2] = 0 under the null
    ...
    
    """
    gj = moment_restriction(b, x, np.arange(1,K+1))
    return gj


def gN(b,K,data):
    """Averages of g_j(b) over all data.
    """
    e = gj(b,K,data)

    # Check to see more obs. than moments.
    assert e.shape[0] > e.shape[1]
    
    # Return sample average of gj(b)'s
    gN = e.mean(axis=0)
    return gN



def Times(a,b):
    return a*b

def Rational(a,b):
    return a/b

def Power(a,b):
    return a**b

def translate(wl_obj):
    s = str(wl_obj)
    s = s.replace('[', '{').replace(']', '}')
    s = s.replace('(', '[').replace(')', ']')
    s = s.replace('{', '(').replace('}', ')')
    s = s.replace('Global`', '')
    return s

def OmegaInv(K):
    """Return a function that will evaluate the asymptotic covariance matrix
    under the null hypotheis that X comes from a Normal distribution.
    
    The returned function will return a matrix when evaluated at specific
    values of the paramters.
    
    Use wolfram to analytically evaluate the inverse of the expectation of 
    gj*gj' symbolically, and return a function to evaluate the matrix at 
    particular values of mu and sigma.
    """
    # print(f'\rFinding asymptotic covariance matrix for {K} moments, assuming normality.', end='')
    # Create list of moment functions (to use in Mathematica)
    moments = ','.join([ '{M[x, '+f'{k}'+']}' for k in range(1,K+1) ])
    # Filepath to mathematica wolfram kernal
    WL_path = "D:\\Programs\\Wolfram\\12.1\\WolframKernel.exe"
    # Use wolfram to solve for symbolic inverse of covariance matrix
    with WolframLanguageSession(WL_path) as session:
        session.evaluate( wlexpr(f'''
        Ieven[m_] := Boole[EvenQ[m]]
        M[x_, m_] := (x - mu)^m - Ieven[m] sigma^m (m - 1)!!
        gigi[x_] = ({{{moments}}}).Transpose[({{{moments}}})]
        o[mu_, sigma_] = Expectation[gigi[x],
                            x \[Distributed] NormalDistribution[mu, sigma]]
        '''))
        Omegai = session.evaluate( wlexpr('''Inverse[o[mu, sigma]]'''))
    # Convert Wolfram syntax to python syntax
    string = translate(Omegai)
    # Return function which evaluates the matrix at the given sigma
    return lambda s: np.array(eval(string.replace('sigma',f'{s}')))


def Omegahat(b,K,data):
    """Estimate of the covariance matrix E[gN'gN]"""
    e = gj(b,K,data)
    # Recenter! We have Eu=0 under null.
    # Important to use this information.
    e = e - e.mean(axis=0) 
    return e.T@e/e.shape[0]


def J(b,K,W,data):
    m = gN(b,K,data) # Sample moments @ b
    N = len(data)
    # Criterion Function: N * (gN)^T * W * gN
    return N*m.T@W@m # Scale by sample size


def cu_gmm(data, beta, K, Omega_inv_fun):
    """Continuously-Updated GMM
    In each step of the minimization, the covariance matrix estimate is updated
    with the new parameter value
    
    The lambda function creates a function J_b(b)
    where J_b(b) = J(b, K, W(b,K,data) data)
    
    The minimize function is used instead of the minimize_scalar because the
    minimize function can minimize over a vector of parameters instead
    of just a single parameter.
    """
    J_b = lambda b: J(b, K, Omega_inv_fun(b[1]), data)
    J_min = minimize(J_b, x0=beta)
    return J_min


def plots(type_list, proportions_dict, D=1000, title_size = 20, fig_size=(8,5)):
    """Plot the proportion of simulations for which the null hypothesis is rejected,
    depending on the number of moment conditions that were included in gj.
    
    A plot is created for each distribution in type_list. 
    
    type_list :: list of strings with names of distributions
    proportions_dict :: dictionary, each key is a distribution name, and value is the
        dataset of proportion of MC draws where the null hypothesis was rejected.
    """    
    # Create figure
    f, ax = plt.subplots(1, 1, figsize=fig_size)
    
    # For each distribution, create a plot of rejections vs moments included
    for dist in type_list:
        # Get data from dictionary
        y = np.array(proportions_dict[dist])*100 # convert portion to %
        x = proportions_dict['moments']
        # Generate plot of % rejected vs # of moments included in gj
        ax.plot(x, y, label=dist)
        ax.set_xlabel('number of moment conditions (K)')
        ax.set_ylabel(r'% of MC simulations that led to rejected H$_0$')
        
    # Add title
    ax.set_title(f'% of {D} Monte Carlo draws from distributions\nthat led to a rejected null hypothesis at 5% confidence', 
                    fontsize=15)
    # Better x-tick marks (on even ticks)
    plt.xticks(range(2,max(x)+1,2))
    plt.grid(True)
    # Add legend
    ax.legend(shadow=True, fancybox=True)
    plt.show()


    
##############################################
# Main: Monte Carlo simulations for 
##############################################

def run_moments(beta=(5,3), N=1000, D=100, k_max=20, alpha=0.05):
    """
    Want to see how the average null hypothesis rejection rate changes when
    including different numbers of moments. For each set of moments, run D
    Monte Carlo simulations and calculate portion of simulations where the 
    null hypothesis (that the data comes from a Normal dist) is rejected. Run
    through 2 to k_max moments.

    Parameters
    ----------
    beta : 2-tuple, true parameters (mean, sd)
    N : int, number of obs to use in each simulation
    D : int, number of simulations to use for each moment
    k_max : int, max number of moments to include
    alpha : float, significance level of hypothesis test

    Returns
    -------
    dictionary of rejection data

    """
    # True parameters
    mu, sigma = beta
    
    # Run some Monte Carlo!
    # Try pulling data from different distributions in type_list
    # For each distribution type, try many different #s of moment conditions
    # For each # of moment conditions, run a Monte Carlo set of simulations to
    #     calculate the portion of the MC simulations that lead to a rejected 
    #     null hypothesis that the data came from a Normal distribution
    
    type_list = ['normal', 'cauchy', 'poisson']
    reject_portions = {} 
    for dist in type_list:
        print(f'Working on {dist} distribution')
        rejections = []
        # Try different numbers of moment conditions (2 - k_max)
        for K in progressbar.progressbar(range(2, k_max+1)):
            print()
            P_values = []
            # Find Asymptotic Covariance matrix under null
            Omega_inv_fun = OmegaInv(K) 
            # Omega_inv_fun is a function of the paramters
            # Monte Carlo: simulate data D times
            for d in range(1,D+1):
                print(f'\rMoments = {K}/{k_max}, MC draw = {d}/{D}', end='')
                # added a try-except loop because sometimes
                # the data leads to a singular covariance matrix
                soltn = None
                n=0
                while soltn is None:
                    n+=1
                    try: 
                        # for each MC draw of data, get the GMM Solution
                        # passing the Omega Inverse function for this set of K moments
                        soltn = cu_gmm(dgp(N, mu, sigma, dist = dist), 
                                       (mu, sigma), K, Omega_inv_fun)
                        # Minimized criterion function value
                        J_min = soltn.fun
                        # Calculate the prob. of observing this J value or greater if
                        # the true distribution is normal (assym. J = 0)
                        # p-value = 1 - CDF(J-value)
                        P_values.append(1 - iid.chi2.cdf(J_min, K-2, loc=0, scale=1))
                    except TypeError:
                        print(f'\rTypeError {n}', end='')                    
                        
                    except:
                        pass
    
            # Calculate the portion of MC draws where we would reject the null
            reject_portion = sum(1 for i in P_values if i < alpha) / D
            rejections.append(reject_portion)
        # Add proportions of rejections to dictionary for this sampling distribution
        reject_portions[dist] = rejections
    
    
    reject_portions['moments'] = [k for k in range(2, k_max+1)]

    return reject_portions #dictionary of data

if 1==1:
    # print table of proportions of MC draws that lead to a rejected null 
    # hypothesis (of normality)
    num_of_simulations = 10
    reject_portions = run_moments(D=num_of_simulations)
    print(pd.DataFrame(reject_portions, index=reject_portions['moments']))
            
            
    # How does the proportion of rejected MC draws change with the number of 
    # moment conditions?
    # For each of the distributions tested, we can see how the number of 
    #    moment conditions used changes how often we reject the null hypothesis. 
    #    Ideally, we reject the null very little if we are pulling from a normal 
    #    distribution, and we reject the null very often if we are pulling from 
    #    a non-normal distribution. 
    # For a given # of moments used, we have calculated the portion of the MC 
    #    simulations that led to a rejected hypotheis. So let's plot 
    #    (% MC simulations rejected) vs (# of moments used).
    plots(['normal', 'cauchy', 'poisson'], reject_portions, D=100)
    
    
    # In loop, just save max. output (J, mu, sigma) for plotting later
    # can plot distribution of J or p-values at each moment for any 
    # level of significance
    
    # optimal # of moments ==> what's the optimal subset of moments of that size?
    # +/- 1 moment (if optimal is 4, look at all combinations of 3, 4, and 5 moments
    # for all moments less than 30)
    
    # Can I use conda to install an env in my home directory where I can install other packages and upload the wolfram kernal?


bhats = []
shats = []
Js = []

# =============================================================================
# Not formatted well: Looking at distributions of mu, sigma, J at different N
# =============================================================================
if 1==0:
    cols = ['N','b','s','J']
    data = pd.DataFrame(columns=['N','b','s','J'],dtype=float)
    Ns=[round(10**(k/2)) for k in range(2,12)]
    D=10
    K=4
    Omega_inv_fun = OmegaInv(K)
    for N in Ns:
        try:
            for d in range(D):
                print(f'\rRunning {N} observations', end='')
                soltn = cu_gmm(dgp(N, mu=5, sigma=3, dist = 'normal'), (mu,sigma), K, Omega_inv_fun)
                data1 = pd.DataFrame([[N, soltn.x[0], soltn.x[1], soltn.fun]], columns=cols)
                data = data.append(data1)
        except:
            raise
    data['logN'] = np.log(data['N'])
    # J
    data.groupby('logN').mean()[['J']].plot(title=r'Mean $J(\hat\mu_{gmm},\hat\sigma_{gmm})$ '+f'over {D} simulations for {K} moments')
    plt.show()
    
    # beta, sigma
    data.groupby('logN').mean()[['b','s']].plot(title=r'Mean $\hat\mu_{gmm}$, $\hat\sigma_{gmm}$ '+f'over {D} simulations for {K} moments')
    plt.legend([fr'mean $\hat\mu_{{gmm}}$ $(\mu={mu})$', fr'mean $\hat\sigma_{{gmm}}$ $(\sigma={sigma})$'])
    
    # var beta, sigma
    data.groupby('logN').var()[['b','s']].plot(title=r'Variance of $\hat\mu_{gmm}$, $\hat\sigma_{gmm}$ '+f'over {D} simulations for {K} moments')
    plt.legend([r'Var$\hat\mu_{gmm}$', r'Var$\hat\sigma_{gmm}$'])
