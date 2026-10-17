#In arxiv V1, this is equation D7.
import sympy as sp

s   = sp.symbols("s", real=True)
t   = sp.symbols("t", real=True)
lam = sp.symbols("lambda", real=True)
m   = sp.symbols("m", real=True)
tau = sp.symbols("tau", positive=True)
Omb = sp.symbols("Omegabar", real=True)                 # carrier Ombar (a symbol for now)
Om_qnm, Om_0 = sp.symbols("Omega_qnm Omega_0", real=True)
I = sp.I

sep = lambda x: print("\n" + "-"*68 + f"\n{x}\n" + "-"*68)


g  = t*(1 + sp.exp(-2*s)*t)**(-(2 + I*lam))                  #between D19 and D20 in arxiv v1
F  = sp.log(t) + sp.log(1 - t) - sp.log(1 + sp.exp(-2*s)*t)   #between D19 and D20 in arxiv v1
t0 = sp.exp(2*s)*(sp.sqrt(1 + sp.exp(-2*s)) - 1)              # D20 in arxiv v1
lam_def = m*tau/2*(Om_qnm - Om_0)                             # D21 in arxiv v1
Omb_def = (Om_qnm + Om_0)/2                                   # D22 in arxiv v1

print("GIVENS:")
print("  g(t)  =", g)
print("  F(t)  =", F)
print("  t0    =", t0)
print("  lam   = (m tau/2)(Om_qnm - Om_0)     Ombar = (Om_qnm + Om_0)/2")
print("  omega(s) = m Ombar - (1/tau) d/ds arg[ e^{-i lam s} e^{-s(2N+3)} F ]   ")

# ===================================================================
# DERIVE 1 -- t0 is a stationary point of F  (checks the givens are consistent)
# ===================================================================
sep("DERIVE 1:  F'(t0) = 0   (t0 given is the max of F given)")
Fp_at_t0 = sp.simplify(sp.diff(F, t).subs(t, t0))
print("  F'(t0) =", Fp_at_t0, "  -> stationary:", Fp_at_t0 == 0)

# ===================================================================
# DERIVE 2 -- the base of g's complex factor at the saddle
#   In g(t0) the only complex piece is (1 + e^{-2s} t0)^{-(2+i lam)}.
# ===================================================================
sep("DERIVE 2:  1 + e^{-2s} t0  =  sqrt(1 + e^{-2s})")
base0 = sp.simplify(1 + sp.exp(-2*s)*t0)
print("  1 + e^{-2s}*t0 =", base0)

# ===================================================================
# DERIVE 3 -- arg[g(t0)].
#   the integral ~ (real prefactors) * g(t0) * e^{N F(t0)},
#   and e^{N F(t0)} = t0^{2N} (real, t0>0), F''(t0)=-2/(t0(1-t0)) (real), and the
#   Gamma prefactor is real => arg[kappa_3] = arg[g(t0)].
#   g(t0) = t0 * base0^{-(2+i lam)} with t0>0 and base0>0, so only the i lam part is
#   imaginary.  NOTE this is the CONTINUOUS (unwrapped) branch of the log, not the
#   principal one: arg[g(t0)] = -(lam/2) ln(1+e^{-2s}) leaves (-pi,pi] once s is
#   negative enough.  The continuous branch is the one an instantaneous frequency
#   needs, and the 2 pi k branch constant is s-independent, so it drops under d/ds.
# ===================================================================
sep("DERIVE 3:  arg[g(t0)] = -lam * ln(1 + e^{-2s} t0)")
T, P = sp.symbols("T P", positive=True)                       # stand-ins: T=t0>0, P=base0>0
#=ln(T*P^{-(2+i'lambda)})
ln_g = sp.log(T) - (2 + I*lam)*sp.log(P)                      # ln[ t0 * base0^{-(2+i lam)} ]
arg_g = sp.im(ln_g)                                           # sympy: T,P>0 -> -lam*log(P)
print("  arg[g(t0)] (with base0 -> P):", arg_g)
arg_g_s = arg_g.subs(P, base0)                               # put base0 = sqrt(1+e^{-2s}) back
print("  arg[g(t0)] (s):", sp.simplify(arg_g_s))

# ===================================================================
# DERIVE 4 -- arg[I_B].
#   I_B = e^{-i lam s} * e^{-s(2N+3)} * kappa_3.
#   arg[e^{-i lam s}] = -lam s ;  e^{-s(2N+3)} is real -> 0 ;  arg[kappa_3] = arg[g(t0)].
# ===================================================================
sep("DERIVE 4:  arg[I_B] = -lam s + arg[g(t0)]")
arg_IB = -lam*s + arg_g_s
print("  arg[I_B](s) =", sp.simplify(arg_IB))

# ===================================================================
# DERIVE 5 -- the frequency  omega(s) = m Ombar - (1/tau) d/ds arg[I_B]
# ===================================================================
sep("DERIVE 5:  omega(s)")
omega = sp.simplify(m*Omb - (1/tau)*sp.diff(arg_IB, s))
print("  omega(s) =", omega)
print("           = m Ombar + (lam/tau)/(1 + e^{-2s})  ?  ",
      sp.simplify(omega - (m*Omb + (lam/tau)/(1 + sp.exp(-2*s)))) == 0)

# ===================================================================
# DERIVE 6 -- the s=0 value, then substitute the draft definitions of lam, Ombar
# ===================================================================
sep("DERIVE 6:  omega(s=0)")
omega0 = sp.simplify(omega.subs(s, 0))
print("  omega(0) =", omega0, "   (= m Ombar + lam/(2 tau))")
omega0_phys = sp.simplify(omega0.subs({Omb: Omb_def, lam: lam_def}))
target = sp.Rational(3, 4)*m*Om_qnm + sp.Rational(1, 4)*m*Om_0
print("  omega(0) with draft defs =", omega0_phys)
ok = sp.simplify(omega0_phys - target) == 0
print("  == (3/4) m Om_qnm + (1/4) m Om_0 :", ok)

print("\n" + "="*68)
print("RESULT:", "frequency + s=0 value "
      if (Fp_at_t0 == 0 and ok) else "FAILED")
omega_compact = m*Omb + sp.simplify(omega - m*Omb)   # pull the carrier out; remainder stays genuine sympy
omega_tanh = m*Omb + sp.simplify((sp.simplify(omega - m*Omb)).rewrite(sp.tanh))  # same result, tanh spelling
print("  omega(s) =", omega_compact)
print("           =", omega_tanh, "   (tanh form)")
# the draft displays this per mode (omega = m Omega) and with lam, Ombar substituted
Omega_tanh = sp.collect(sp.expand((omega_tanh/m).subs({Omb: Omb_def, lam: lam_def})), sp.tanh(s))
print("  Omega(s) = omega(s)/m =", Omega_tanh, "   ")
print("  omega(0) =", omega0_phys)
print("="*68)
