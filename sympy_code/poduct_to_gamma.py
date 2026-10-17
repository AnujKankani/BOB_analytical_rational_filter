#This file was created with the assistance of Claude Opus under the close supervision of Anuj. All code was checked and approved by Anuj.
#In arxiv V1, this is equation D9 and D12

import sympy as sp


def prove_filter_product_to_gamma():
    """Prove, for general symbolic N, that the filter value at the mirror pole

        prod_{m=0}^{N} (i*lam + n + m + 1) / (i*lam + n - m)
            = Gamma(n+N+2+i*lam) * Gamma(n-N+i*lam) / Gamma(n+1+i*lam)^2

    sympy evaluates the symbolic product itself (as rising factorials), so this is a
    derivation, not a numeric spot-check.
    """
    n, N, m = sp.symbols("n N m", integer=True, nonnegative=True)
    lam = sp.symbols("lambda", real=True)
    a = sp.I * lam

    # the product, exactly as written
    P = sp.Product((a + n + m + 1) / (a + n - m), (m, 0, N)).doit()

    # the claimed closed form
    target = sp.gamma(n + N + 2 + a) * sp.gamma(n - N + a) / sp.gamma(n + 1 + a) ** 2

    proven = sp.simplify((P / target).rewrite(sp.gamma)) == 1

    print("sympy product  =", P)
    print("gamma target   =", target)
    print("P / target     =", sp.simplify((P / target).rewrite(sp.gamma)))
    print("PROVEN (product == gamma form for all n, N):", proven)
    return proven


def prove_filter_qnm_to_gamma():
    """Prove, for general symbolic N, that the filter value at the QNM (s>0) tail pole

        prod_{m=0}^{N} (n - m) / (n + m + 1)
            = Gamma(n+1)^2 / [Gamma(n-N) * Gamma(n+N+2)]

    The lambda-independent counterpart of prove_filter_product_to_gamma() (at the QNM
    pole nu - lam = -i(2n+1) carries no lambda).
    """
    n, N, m = sp.symbols("n N m", integer=True, nonnegative=True)

    # the product, exactly as written
    P = sp.Product((n - m) / (n + m + 1), (m, 0, N)).doit()

    # the claimed closed form
    target = sp.gamma(n + 1) ** 2 / (sp.gamma(n - N) * sp.gamma(n + N + 2))

    proven = sp.simplify((P / target).rewrite(sp.gamma)) == 1

    print("sympy product  =", P)
    print("gamma target   =", target)
    print("P / target     =", sp.simplify((P / target).rewrite(sp.gamma)))
    print("PROVEN (product == gamma form for all n, N):", proven)
    return proven


if __name__ == "__main__":
    print("Mirror pole (s<0):")
    prove_filter_product_to_gamma()
    print()
    print("QNM tail pole (s>0):")
    prove_filter_qnm_to_gamma()
