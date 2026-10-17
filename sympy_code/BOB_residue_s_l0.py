#This file was created with the assistance of Claude Opus under the close supervision of Anuj. All code was checked and approved by Anuj.
#for simplicity we set t_p = 0 since that term can be pulled out of the integral
#In arxiv V1, this is equation D6.
import sympy as sp

nu, lam = sp.symbols('nu lambda', real=True)
w, k = sp.symbols('w k')
#Eq. 4 in arxiv v1 ignoring the prefactors
Ehat = ((2**(sp.I*lam)/sp.gamma(1+sp.I*lam))
        * sp.gamma((1+sp.I*lam+sp.I*nu)/2)
        * sp.gamma((1+sp.I*lam-sp.I*nu)/2))


def spectrum_residue_mirror(n):
    """The closed form we want to prove."""
    #slightly simplify the prefactors in the arxiv version
    return -2*sp.I*(-1)**n*2**(sp.I*lam)*sp.gamma(n+1+sp.I*lam)/(sp.factorial(n)*sp.gamma(1+sp.I*lam))


def derive_symbolic(n):
    """Derive Res_{nu0} Ehat for general symbolic n, from first principles (no plugged answer)."""
    # (1) Res_w Gamma(w) at w = -n via the functional equation (sympy does the product for symbolic n)
    #\Gamma(w+n) = \Gamma(w)*\prod_{k=0}^{n-1}(w+k)
    #easy to prove or https://mathworld.wolfram.com/RisingFactorial.html
    reduced = sp.gamma(w+n+1) / sp.product(w+k, (k, 0, n-1))   # = (w+n) Gamma(w)
    Res_w = sp.simplify(reduced.subs(w, -n))                   # = (-1)^n / n!

    # (2) chain rule: the singular Gamma has argument w = (1+i lam+i nu)/2, so dw/dnu = i/2
    dwdnu = sp.diff((1+sp.I*lam+sp.I*nu)/2, nu)
    Res_nu_gamma = Res_w / dwdnu                              # Res_nu[Gamma_mirror]

    # (3) regular remainder (prefactor * the OTHER Gamma) at nu0 = -lam + i(2n+1)
    nu0 = -lam + sp.I*(2*n+1) #location of pole
    other_arg = sp.simplify((1+sp.I*lam-sp.I*nu0)/2)          # collapses to n+1+i lam
    remainder = 2**(sp.I*lam)/sp.gamma(1+sp.I*lam) * sp.gamma(other_arg)

    # (4) assemble
    return sp.simplify(Res_nu_gamma * remainder)


def prove_general_n():
    n = sp.symbols('n', integer=True, positive=True)
    derived = derive_symbolic(n)
    closed = sp.simplify(spectrum_residue_mirror(n))
    print("General symbolic n:")
    print("  derived by sympy =", derived)
    print("  closed form      =", closed)
    proven = sp.simplify(derived - closed) == 0
    print("  PROVEN (derived == closed for all n):", proven)
    return proven


def check_concrete_n(nmax=6):
    print("\nIndependent cross-check vs sp.residue(Ehat, nu, nu0), concrete n:")
    all_ok = True
    for nn in range(nmax):
        nu0 = -lam + sp.I*(2*nn+1)
        direct = sp.simplify(sp.residue(Ehat, nu, nu0))       # sympy's own residue engine
        formula = sp.simplify(spectrum_residue_mirror(nn))
        ok = sp.simplify(direct - formula) == 0
        all_ok &= ok
        print("  n=%d : sp.residue == closed form : %s" % (nn, ok))
    print("ALL concrete n match:", all_ok)
    return all_ok


if __name__ == "__main__":
    p = prove_general_n()
    c = check_concrete_n()
    print("\nRESULT:", "spectrum_residue_mirror PROVEN (general n) and cross-checked"
          if (p and c) else "PROOF FAILED")
