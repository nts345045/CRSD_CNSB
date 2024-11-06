"""
:module: model.rate_and_state
:auth: Nathan T. Stevens
:email: ntsteven@uw.edu
:org: Pacific Northwest Seismic Network
:license: Creative Commons 4.0 Attribution (CC-BY 4.0)
:purpose: This module contains scripts for conducting rate-and-state friction modeling.

:references:
Dietrich, J.H. Modeling of rock friction 1. Experimental results and contitutive equations.
    Journal of Geophysical Research: Solid Earth (1978--2012), 84(B5), 2161-2168.

Leeman, J.R., May, R., Marone, C., Saffer, D. Modeling Rate-and-State Friction with Python. 2016
    GitHub: SciPy Scientific Programming. Talk (https://github.com/jrleeman/rsfmodel/tree/master)

Ruina, A. Slip instability and state variable friction laws. Journal of Geophysical Research: Solid Earth
    (1978--2012), 88(B12), 10359-10370.

Skarbek, R.M., McCarthy, C., Savage, H.M. Oscillatory loading can alter the velocity dependence of ice-on-rock friction.
    Geochem. Geophys. Geosyst. 2022;

Zoet LK, Iverson NR, Andrews L, Helanow C. Transient evolution of basal drag during glacier slip.
    Journal of Glaciology. 2022;68(270):741-750. doi:10.1017/jog.2021.131

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import odeint

spy = 365.24*24.*3600.

def rsf_mu(mu0, a, b, Dc, V, V0, theta):
    """
    Rate and state friction formula of Dietrich (1979)
    and Ruina (1983). 
    
    Equation (1) in Zoet and others (2022)
    Equation (2) in Skarbek and others (2022)

    :math:`\mu_0 + a log \left(\frac{V}{V_0}\right) + b log \left(\frac{V \theta}{D_c} \right)`

    :param mu0: reference coefficient of friction [dimensionless]
    :type mu0: float-like
    :param a: direct effect proportionality constant [dimensionless]
    :type a: float-like
    :param b: evolution effect proportionality constant [dimensionless]
    :type b: float-like
    :param Dc: characteristic slip distance [length]
    :type Dc: float-like
    :param V: measured slip velocity [length time**-1]
    :type V: float-like
    :param V0: reference slip velocity [length time**-1]
    :type V0: float-like
    :param theta: state variable [dimensionless]
    :type theta: float-like 

    :returns:
     - **mu** (*float*) - modeled friction [dimensionless]
    """
    mu = mu0 + a*np.log(V/V0) + b*np.log((V*theta)/Dc)
    return mu

def hooke_dmudt(k, sigma, Vl, V):
    """Calculate the rate of change in friction
    according to Hooke's law (linear elasticity)

    Equation 4 in Skarbek and others (2022)

    :math:`\frac{\partial \mu}{\partial t} = \frac{k}{\sigma}(V_l - V)`

    :param k: elastic stiffness [length**-1]
    :type k: float
    :param sigma: normal stress
    :param Vl: load-point velocity [length time**-1]
    :type Vl: float
    :param V: sliding velocity [length time**-1]
    :type V: float
    :return: first time derivative of friction [dimensionless]
    :rtype: float
    """    
    dmudt = k*(Vl - V)
    return dmudt


def rsf_tau(N, mu0, V, V0, a, b, Dc, theta):
    """Calculate the shear stress for a given effecitve pressure **N**
    and parameterized rate-and-state friction equation

    :param N: effective pressure [mass length**-1 time**-2]
    :type N: float
    :param mu0: _description_ [dimensionless]
    :type mu0: float
    :param V: measured velocity [length time**-1]
    :type V: float
    :param V0: reference velocity [length time**-1]
    :type V0: float
    :param a: direct effect scaling parameter [dimensionless]
    :type a: float
    :param b: evolution effect scaling parameter [dimensionless]
    :type b: float
    :param Dc: characteristic evolution length [length]
    :type Dc: float
    :param theta: state parameter [dimensionless]
    :type theta: float
    :return: shear stress [mass length**-1 time**-2]
    :rtype: float
    """    
    mu = rsf_mu(mu0, a, b, Dc, V, V0, theta)
    tau = N*mu
    return tau

def dietrich_state_evolution(V, theta, Dc):
    """State evolution equation from Dietrich (1979)
    "aging law"

    Equation 2 in Zoet and others (2022)

    .. math::`\\frac{\partial \\theta}{\partial t} = 1 - \\frac{V \\theta}{D_c}`

    :param V: measured slip velocity
    :type V: float-like
    :param theta: state parameter
    :type theta: float-like
    :param Dc: characteristic slip distance
    :type Dc: float-like
    :return:
     - **dOdt** (*float*) - change 
    :rtype: _type_
    """    
    dOdt = 1. - (V*theta)/Dc
    return dOdt

def zoet_state_evolution(V, theta, Dc, p=1):
    """
    Modification of the Dietrich (1979) state evolution
    equation in Zoet and others (2022) [Eqn. 2] that
    incorporates an exponent **p** related to ice flow

    :math:`\dot{\theta} = 1 - {\left \frac{V \Theta}{D_c} \right}^p

    :param V: measured slip velocity [length time**-1]
    :type V: float
    :param theta: state parameter [dimensionless]
    :type theta: float
    :param Dc: characteristic evolution distance [length]
    :type Dc: float
    :param p: ice flow related exponent [dimensionless], defaults to 1.
      Parameter based on 
    :type p: float, optional
    :return: time derivative of state [time**-1]
    :rtype: float
    """
    dOdt = 1. - ((V*theta)/Dc)**p
    return dOdt

def ruina_state_evolution(V, theta, Dc):
    """State evolution equation from Ruina (1983)
    "slip law"

    Equation (3) in Skarbek and others (2022)
    
    :math:`\frac{\partial \Theta}{\partial t} = - \frac{V \Theta}{D_c} log \left( \frac{V \Theta}{D_c} \right)`

    :param V: measured slip velocity [length time**-1]
    :type V: float
    :param theta: state parameter [dimensionless]
    :type theta: float
    :param Dc: characteric slip distance [length]
    :type Dc: float
    :return: first time derivative of state parameter [time**-1]
    :rtype: float
    """    
    dOdt = -1.*((V*theta)/Dc)*np.log((V*theta)/Dc)
    return dOdt


def calc_veff_chatgpt(N, N0, V0, alpha):
    """Calculate the effective velocity of a rate-and-state
    system given an effecive pressure perturbation

    :math:`V_{eff} = V_0 exp\left(\frac{N - N_0}{\alpha}\right)`

    Formulation proposed by ChatGPT based on it's interpretation
    of 

    :param N: _description_
    :type N: _type_
    :param N0: _description_
    :type N0: _type_
    :param V0: _description_
    :type V0: _type_
    :param alpha: _description_
    :type alpha: _type_
    :return: _description_
    :rtype: _type_
    """    
    veff = V0*np.exp((N - N0)/alpha)
    return veff


def calc_v_equivalent(N, N0, V0, n=3.):
    """Calculate the equivalent velocity that would
    produce the same cavity geometry in Lliboutry/Kamb
    stead-state cavity modeling as an effective pressure
    relative to a reference effective pressure, velocity,
    and flow-law exponent. Uses a rearrangement of equation
    (A3) in Stevens and others (in review), reducing to:

    :math:`V = V_0 \left(\frac{N}{N_0}\right)^n`

    :param N: perturbed effective pressure [pascals]
    :type N: float
    :param N0: reference effective pressure [pascals]
    :type N0: float
    :param V0: reference slip velocity
    :type V0: float
    :param n: ice flow-law exponent, defaults to 3.
    :type n: float, optional
    :return: equivalent velocity
    :rtype: float
    """    
    V = V0*(N/N0)**n
    return V

def calc_tau_rsf_veff(N, veff, mu0, V0, a, b, theta, Dc):
    tau = N*(mu0 + a*np.log(veff/V0) + b*np.log((V0*theta)/Dc))
    return tau


def model_rsf_tau(params, Nvect, Tvect, dt, T0, N0, V0, n, p):
    a, b, Dc = params
    # Calculate reference drag from reference shear stress and effective stress
    mu0 = T0/N0
    # Calculate Velocity equivalent predicted by steady-state modeling
    Vvect = calc_v_equivalent(Nvect, N0, V0, n=n)

    # Populate theta vector with steady-state estimates
    theta = np.ones_like(Nvect)*Dc/V0
    Tmodel = np.zeros_like(Tvect)
    
    for _e in range(1, len(Nvect)):
        # Get current velocity
        V = Vvect[_e]
        # Get past state
        theta_old = theta[_e - 1]
        # Calculate state evolution from prior state
        dOdt = zoet_state_evolution(V, theta_old, Dc, p=p)
        # Update current state
        theta[_e] += dOdt*dt
        # Run RSF prediction
        Tmodel[_e] = rsf_tau(N0, mu0, V, V0, a, b, Dc, theta[_e])

    return (Tmodel, theta)

def objective(params, Nvect, Tvect, dt, T0, N0, V0=15./spy, n=3, p=1):
    model = model_rsf_tau(params, Nvect, Tvect, dt, T0, N0, V0, n, p)
    # Calculate sum of squares of residuals
    residual = np.sum((Tvect - model) **2)
    
    return residual

def fit_rsf(Nvect, Tvect, dt=1., p0=[0.1, 0.184, 0.194], T0=140e3, N0=340e3, V0=15./spy, n=3, p=1):
    result = minimize(objective, p0, args=(Nvect, Tvect, dt, T0, N0, V0, n, p), method='Nelder-Mead')
    return result


