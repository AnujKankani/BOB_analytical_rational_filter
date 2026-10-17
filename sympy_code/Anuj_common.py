import sympy as sp

s      = sp.symbols("s", real=True)          # s = (t - t_p)/tau
t      = sp.symbols("t", real=True)          # time
omega  = sp.symbols("omega", real=True)      # GW angular frequency
c      = sp.symbols("c", positive=True)      # positive stand-in for cosh(s) > 0
x      = sp.symbols("x", positive=True)      # substitution variable x = e^{2s} > 0
m      = sp.symbols("m", positive=True)      # GW mode number
Ap, tau, Omega_0, Omega_QNM, t_p = sp.symbols(
    "Ap tau Omega_0 Omega_QNM t_p", positive=True)

Om_bar = (Omega_0 + Omega_QNM) / 2          
lamb   = m * tau * (Omega_QNM - Omega_0) / 2 
