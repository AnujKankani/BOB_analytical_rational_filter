import sympy as sp
from Anuj_common import s, c, Ap, tau, m, Om_bar, lamb
#s = (t-t_p)/tau 

def get_simplified_BOB():
    #BOB news with simplified frequency
    amp = Ap*sp.sech(s)
    Phi = m*Om_bar*s*tau + lamb*sp.log(sp.cosh(s))

    News_td_orig = amp*sp.exp(-sp.I*Phi)
    News_td = Ap*sp.sech(s)**(1+sp.I*lamb)*sp.exp(-sp.I*m*Om_bar*s*tau)

    return News_td_orig, News_td

def test_simplified_BOB():
    News_td_orig, News_td = get_simplified_BOB()
    #we need to tell sympy that cosh(s) is always >=0
    reps = {sp.cosh(s): c, sp.sech(s): 1/c}
    News_td_orig = News_td_orig.subs(reps).rewrite(sp.exp)
    News_td = News_td.subs(reps).rewrite(sp.exp)
    if News_td_orig.equals(News_td):
        print("News_td_orig -> News_td simplification is proven")
    else:
        print("News_td_orig -> News_td simplification is NOT proven")

if __name__ == "__main__":
    test_simplified_BOB()