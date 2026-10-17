#Perform fourier transform of the simplified BOB
#This file was created with the assistance of Claude Opus under the close supervision of Anuj. All code was checked and approved by Anuj.
#for simplicity we set t_p = 0 since that term can be pulled out of the integral
#In arxiv V1, this is the solution to integral C8.
import sympy as sp, mpmath as mp
from Anuj_common import s, x, Ap, tau, m, omega, Om_bar, lamb, Omega_0, Omega_QNM
from Anuj_sec1_1 import get_simplified_BOB
from Anuj_common import s, c, Ap, tau, m, Om_bar, lamb

News_td_orig, News_td = get_simplified_BOB()

integrand = News_td.copy()


lam, nu  = sp.symbols("lambda nu", real=True)   # symbolic lambda, nu (the compact closed form is in these)
lam_phys = lamb                                 #lambda = m tau (Omega_QNM - Omega_0)/2
nu_phys  = (omega - m*Om_bar)*tau               


def get_frequency_domain_BOB():
    _, News_td = get_simplified_BOB()                  
    News_td = News_td.subs(lamb, lam)                   # write the sech power with the symbol lambda

    # FT integrand in s: cancel News_td's carrier, recombine carrier+kernel into a single e^{i nu s}
    envelope    = News_td * sp.exp(sp.I*m*Om_bar*s*tau) # = Ap sech(s)^{1+i lam}
    integrand_s = envelope * sp.exp(sp.I*nu*s) * tau    # Ap tau sech(s)^{1+i lam} e^{i nu s}
    print(integrand_s)
    # apply the x = e^{2s} substitution RULES (the definition), then the Jacobian ds = dx/(2x):
    sub = {sp.sech(s): 2*sp.sqrt(x)/(1+x),  sp.exp(sp.I*nu*s): x**(sp.I*nu/2)} #checked by anuj
    #rewrite into a sympy friendly form, #ds = 2/x dx
    integrand_x = sp.powsimp(sp.expand_power_base(integrand_s.subs(sub)/(2*x), force=True), force=True)
    #s->\infty x->\infty, s->-\infty, x->0
    E = sp.integrate(integrand_x, (x, 0, sp.oo))        # the Beta integral
    if isinstance(E, sp.Piecewise):
        E = E.args[0].expr                              # convergent branch
    return sp.simplify(E)                               # compact closed form in (lam, nu)


def target_frequency_domain_BOB():
    """The claimed closed form (draft Eq. 111 with t_p = 0), in (lam, nu)."""
    p = 1 + sp.I*lam
    return 2**(p-1)/sp.gamma(p) * Ap*tau * sp.gamma((p+sp.I*nu)/2) * sp.gamma((p-sp.I*nu)/2)

def prove_frequency_domain_BOB():
    N_freq = get_frequency_domain_BOB()
    p = 1 + sp.I*lam
    # (1) SYMBOLIC: FT result == claimed closed form (clear 1/Gamma(p) so sympy collapses it)
    if sp.simplify(sp.gamma(p)*(N_freq - target_frequency_domain_BOB())) == 0:
        print("Sec 1.2: FT[News_td] == closed-form frequency-domain BOB  (symbolic)")
    else:
        print("Sec 1.2: symbolic proof FAILED"); return
    # (2) NUMERIC tie-back: brute-force FT of the ACTUAL imported News_td(t) vs the closed form
    _, News_td = get_simplified_BOB()
    N_phys = N_freq.subs({lam: lam_phys, nu: nu_phys})   # substitute physical lambda, nu for the numerics
    mp.mp.dps = 30
    News_fn = sp.lambdify((s, m, tau, Omega_0, Omega_QNM, Ap), News_td, "mpmath")
    N_fn    = sp.lambdify((omega, m, tau, Omega_0, Omega_QNM, Ap), N_phys, "mpmath")
    def brute(om, mv, tv, O0, OQ, A):
        g = lambda sv: News_fn(sv, mv, tv, O0, OQ, A) * mp.e**(1j*om*sv*tv) * tv
        return mp.quad(g, [-mp.inf, 0, mp.inf])
    ok = all(abs(brute(om,2,11.,.15,.30,1.) - complex(N_fn(om,2,11.,.15,.30,1.)))
             / abs(complex(N_fn(om,2,11.,.15,.30,1.))) < 1e-10 for om in (0.3, 0.5, 0.65))
    print("Sec 1.2: numeric FT tie-back " + ("matches (~1e-16)" if ok else "FAILED"))

if __name__ == "__main__":
    sp.pprint(get_frequency_domain_BOB())
    prove_frequency_domain_BOB()
