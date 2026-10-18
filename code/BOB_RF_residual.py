#A couple notes
# 1. This code is not optimized
# 2. I assume the time series is with respect to the news peak. So t = 0 is the time of the maximum amplitude of the unfiltered news.
# 3. We use mpmath because our functions have complex arguments and mpmath is great for this stuff
# This code requires 3 NR values mf, chif, and A_p (peak amplitude of the unfiltered news)
# a negative value of chif should indicate that the final spin is in the -z direction
# !!!!!!!Very Important!!!!!!!!!!!!!!!
# The finite N result is an exact result, the large N result is an approximation that has mixed accuracy
# If you want to use these results, use the finite N result. It is much better.
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qnm
qnm.download_data()

def get_mod_bessel_2ndkind(a,z):
    return mp.besselk(a,z)

#simplified BOB functions
def get_key_BOB_values(mode_number,mf,chif):
    #we assume chi_f is the dimensionless spin between [-1,1]
    #A positive number indicates the remnant spin is in the +z direction
    # A negative number indicates the remnant spin is in the -z direction

    if(chif>=0):
        grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        omega_qnm, A, C = grav_220(a=chif)
        omega_qnm = np.conj(omega_qnm)/mf
    else:
        grav_220 = qnm.modes_cache(s=-2,l=2,m=-2,n=0)
        omega_qnm, A, C = grav_220(a=-chif)
        omega_qnm = omega_qnm/mf

    #for the (2,2) mode
    A = 0.148271
    B = 0.024652
    C = 0.038459
    omega_0 = 2*( A + B*chif + C*(chif**3))

    Omega_QNM = omega_qnm.real/mode_number
    Omega_0 = omega_0/mode_number
    tau = 1./np.abs(omega_qnm.imag)
    return Omega_QNM,Omega_0,tau

def BOB_amplitude(mode_number,mf,chif,A_p,t):
    #A_\bob(t) = A_p\, \text{sech}\bigg(\frac{t-t_p}{\tau}\bigg),
    _,_,tau = get_key_BOB_values(mode_number,mf,chif)
    return A_p/np.cosh(t/tau)

def BOB_simple_frequency(mode_number,mf,chif,t):
    '''
    \Omega_\bob = \frac{\Omega_{\qnm} + \Omega_0}{2} + \frac{\Omega_\qnm-\Omega_0}{2}\tanh{\bigg(\frac{t-t_p-t_\text{shift}}{\tau}\bigg)}

    # t_{\text{shift}} = -\tau \text{arctanh}\bigg[\frac{2\Omega_p - (\Omega_\qnm +\Omega_0)}{\Omega_{\qnm}-\Omega_0}\bigg]
    '''

    Omega_QNM, Omega_0, tau = get_key_BOB_values(mode_number,mf,chif)
    TERM1 = (Omega_QNM + Omega_0)/2
    TERM2 = (Omega_QNM-Omega_0)/2
    TERM3 = np.tanh((t)/tau)
    return mode_number*(TERM1 + TERM2*TERM3)

def BOB_simple_phase(mode_number,mf,chif,t):
    #\Phi_\bob = \frac{\Omega_\qnm + \Omega_0}{2}t + \frac{\Omega_\qnm - \Omega_0}{2}\tau \ln (\cosh(s))

    Omega_QNM, Omega_0, tau = get_key_BOB_values(mode_number,mf,chif)
    s = (t)/tau
    TERM1 = (Omega_QNM + Omega_0)/2*t 
    TERM2 = (Omega_QNM - Omega_0)/2
    TERM3 = tau*np.log(np.cosh(s))
    return mode_number*(TERM1 + TERM2*TERM3)

#rational f1j*lambdater results
def get_finite_N_sl0(mode_number,mf,chif,t,A_p = 1,N=7):
    '''
    \mathcal{N}_{DW,\bob, s<0} (t)  = &iA_pe^{-im\bar{\Omega}(t_p + t_\shift + s\tau)}2^{1+i\lambda}e^s\\   \bigg[&e^{is\lambda}(-i)\frac{\Gamma(2+N+i\lambda)\Gamma(i\lambda-N)}{\Gamma(1+i\lambda)^2}F(i\lambda+N+2,i\lambda-N,1+i\lambda,-e^{2s}) \notag \\  
    + & e^{-is\lambda} (-1)^{N+1/2}\frac{\Gamma(i\lambda)\Gamma(2+N)}{\Gamma(1+N)\Gamma(1+i\lambda)}F(-N,N+2,1-i\lambda,-e^{2s})\bigg] 
    \label{eq:N_sl0}
    '''
    Omega_QNM, Omega_0,tau = get_key_BOB_values(mode_number,mf,chif)
    lam = mode_number*tau*(Omega_QNM-Omega_0)/2.
    Omega_bar = (Omega_QNM + Omega_0)/2.
    s = (t)/tau


    z           = -mp.e**(2*s)
    pref        = 1j*A_p*mp.mpf(2)**(1+1j*lam)*(mp.e**s)*mp.e**(-1j*mode_number*Omega_bar*(s*tau))
    Fm          = mp.hyp2f1(1j*lam+N+2,1j*lam-N,1+1j*lam,z)
    gamma_pref  = mp.gamma(2+N+1j*lam)*mp.gamma(1j*lam-N)/mp.gamma(1+1j*lam)**2
    term1       = mp.e**(1j*s*lam)*(-1j)*gamma_pref*Fm

    Ff          = mp.hyp2f1(-N,N+2,1-1j*lam,z)
    gamma_pref2 = mp.gamma(1j*lam)*mp.gamma(2+N)/(mp.gamma(1+N)*mp.gamma(1+1j*lam))
    term2       = mp.e**(-1j*s*lam)*(-1)**(N+mp.mpf(1)/2)*gamma_pref2*Ff

    return complex(pref*(term1+term2))

def get_finite_N_sg0(mode_number,mf,chif,t,A_p=1,N=7):
    Omega_QNM, Omega_0,tau= get_key_BOB_values(mode_number,mf,chif)
    lam = mode_number*tau*(Omega_QNM-Omega_0)/2.
    Omega_bar = (Omega_QNM + Omega_0)/2.
    s = (t)/tau
    
    pref       = 1j*A_p*mp.mpf(2)**(1j*lam+1)*(-1)**(mp.mpf(5)/2+N)*mp.e**(-1j*s*lam)*mp.e**(-s*(3+2*N))*mp.e**(-1j*mode_number*Omega_bar*(s*tau))
    gamma_pref = mp.gamma(2+N)*mp.gamma(1j*lam+2+N)/(mp.gamma(3+2*N)*mp.gamma(1+1j*lam))
    hyp        = mp.hyp2f1(2+N+1j*lam,2+N,3+2*N,-mp.e**(-2*s))

    return complex(pref*gamma_pref*hyp)

def get_large_N_amplitude(mode_number,mf,chif,t,A_p=1,N=7):
    #N only comes in through the form of a time shift here
    '''
    A(t)_{DW,\bob, s<0} (t) &= 2A_p\sqrt{\frac{\sinh(\pi\lambda)}{\pi \lambda}} e^{s+\ln(2N)} K_{i\lambda}(e^{s+\ln(2N)})
    '''
    Omega_QNM, Omega_0,tau = get_key_BOB_values(mode_number,mf,chif)
    lam = mode_number*tau*(Omega_QNM-Omega_0)/2.
    s = (t)/tau
    sbar = s+mp.log(2*N)

    return 2*A_p*mp.sqrt(mp.sinh(mp.pi*lam)/(mp.pi*lam))*mp.e**(sbar)*get_mod_bessel_2ndkind(1j*lam,mp.e**sbar)

def get_large_N_phase(mode_number,mf,chif,t,N=7):
    '''
    \Phi_{DW,\bob} (t) = \frac{1}{4}\bigg(3\Omega_\qnm + \Omega_0 \bigg)t + \frac{\Omega_\qnm-\Omega_0}{4}\tau \ln \bigg(\cosh\bigg[\frac{t-t_p-t_\shift}{\tau}\bigg]\bigg) + \Phi_0
    '''
    Omega_QNM, Omega_0,tau = get_key_BOB_values(mode_number,mf,chif)
    s = (t)/tau

    TERM1 = (1./4.)*(3*Omega_QNM+Omega_0)*t
    TERM2 = (1./4.)*(Omega_QNM-Omega_0)*tau*mp.log(mp.cosh(s))

    return mode_number*(TERM1+TERM2)
    
def get_large_N_frequency(mode_number,mf,chif,t,N=7):
    '''
    \Omega_{DW,\bob} (t) &= \frac{1}{4}\Omega_0 + \frac{3}{4}\Omega_\qnm +\frac{1}{4}(\Omega_\qnm-\Omega_0)\tanh(s)
    '''
    Omega_QNM, Omega_0,tau = get_key_BOB_values(mode_number,mf,chif)
    s = (t)/tau

    TERM1 = (1./4.)*(3*Omega_QNM+Omega_0)
    TERM2 = (1./4.)*(Omega_QNM-Omega_0)*mp.tanh(s)
    return mode_number*(TERM1+TERM2)

def get_large_N_damping(mode_number,mf,chif,t,A_p=1,N=7):
    '''
    \Omega_I = \frac{1}{\tau}\bigg(1 - e^{s+\ln(2N)} \frac{\Re[K_{1+i\lambda}(e^{s+\ln(2N)})]}{K_{i\lambda}(e^{s+\ln(2N)})}\bigg)
    '''
    Omega_QNM, Omega_0,tau = get_key_BOB_values(mode_number,mf,chif)
    lam = mode_number*tau*(Omega_QNM-Omega_0)/2.
    s = (t)/tau
    sbar = s+mp.log(2*N)

    omega_I = (1./tau)*(1-mp.e**(sbar)*mp.re(get_mod_bessel_2ndkind(1+1j*lam,mp.e**sbar))/get_mod_bessel_2ndkind(1j*lam,mp.e**sbar))
    #the damping time is the inverse of omega_I
    return omega_I

def get_D_BOB(index,mode_number,mf,chif,t,A_p=1):
    '''
    \mathcal{D}_{\bob,n}(t) = \mathcal{N}_\bob(t) \frac{\tau(1+e^{-2s})^{1+i\lambda}\kappa_4}{2(n+1)}
    \kappa_4 = \,_2F_1(1+i\lambda,n+1,n+2,-e^{-2s})
    \lambda = (m\tau/2)(\Omega_\qnm - \Omega_0)
    '''
    Omega_QNM, Omega_0,tau = get_key_BOB_values(mode_number,mf,chif)
    lam = mode_number*tau*(Omega_QNM-Omega_0)/2.
    s = (t)/tau
    z = mp.e**(-2*s)
    kappa_4 = mp.hyp2f1(1+1j*lam,index+1,index+2,-z)
    amp = BOB_amplitude(mode_number,mf,chif,A_p,t)
    phase = BOB_simple_phase(mode_number,mf,chif,t)
    BOB = amp*mp.e**(-1j*phase)
    return BOB*tau*((1+z)**(1+1j*lam))*kappa_4 / (2*(index+1))