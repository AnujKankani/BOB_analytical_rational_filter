#!/usr/bin/env python3
r"""The filter-identity decomposition of the direct wave, for one system.

The NR news has 3 prograde QNMs, 3 retrograde and the (3,2,0) removed; BOB
removes the same prograde overtones, so N_FILTER = 2.

TOP PANEL     |Re| of three curves that should coincide:
                NR residual           from the catalogue
                BOB finite-N          the module's closed form
                net decomposition     the expanded operator product

BOTTOM PANEL  the two signed PARTS the decomposition adds up to:
                unfiltered BOB        the simplified BOB news
                filter effects        every non-empty term of that product
              Each part is far larger than their sum: the filtered residual is
              what survives a near-cancellation between them.

Alignment follows the rest of the pipeline.  NR comes from the catalogue, so it
already carries the filter-delay correction measured 1000 M before the peak.
BOB finite-N comes from the BOB catalogue, so it already carries the time shift
fitted against NR on [-10, 0].  The decomposition is given that SAME shift and
the SAME overlap phase, so the top panel shows whether the identity holds, not
whether it can be made to fit.

The assembly follows BOB.tex and BOB2_appendix.tex: the filtered waveform is
the operator product

    Psi_res(t) = [ prod_{n=0}^{N} (1 + a_n D_n) ] Psi(t),   a_n = 2 Im(w_22n),

expanded term by term, with every nested composition collapsed by the
divided-difference identity

    D_{n_1}...D_{n_p} = sum_{m in S} D_m / prod_{n in S, n != m} i(w*_m - w*_n).

The empty subset is Psi itself, the unfiltered news; every non-empty subset
contributes to the filter term.  Only the removed FREQUENCIES enter the
weights; no BOB waveform expression is rewritten.  Before anything is plotted
the assembled result is checked against the module's closed form, and the
script aborts if they disagree.
"""
import itertools
import pickle

import numpy as np
import mpmath as mp
import sxs
from kuibit.timeseries import TimeSeries
import matplotlib.pyplot as plt

import BOB_RF_residual as BOB

# ---- figure style (copied verbatim from figure_style.py) --------------------
plt.style.use("paper.mplstyle")
pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
            "CQG": {"onecol": 374.*pt}, # CQG is only one column
            # Add more journals below. Can add more properties to each journal
            }

my_width = jour_sizes["PRD"]["onecol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2
# -----------------------------------------------------------------------------

M_MODE = 2
CONFIG = "_3pro_3retro_320"   # the filter configuration, named as the catalogue names it
CASE = "SXS:BBH:0178"
N_FILTER = 2                  # BOB removes the 3 prograde overtones, so N = 3 - 1

XLIM = (-20.0, 25.0)
PHASE_WINDOW = (-10.0, 0.0)   # the window the amplitude shift was fitted on
# |Re| dives to whatever the 0.1 M grid samples at each zero crossing, so the
# top panel's range is set from the largest curve in the window rather than
# letting those samples set it, and is shared with the bottom panel.
DECADES, HEADROOM = 4.0, 1.6
# BOB's own frame, wide enough that T_GRID - finite_time_shift stays inside it
MODEL_GRID = np.arange(-70.0, 25.0 + 1e-9, 0.1)
STRAIN_DT = 0.1               # the step the catalogue used for the news
LEGEND_FONTSIZE = 10          # overrides paper.mplstyle's 6; labels and ticks still come from it
NET_LEGEND_Y = -0.190         # the centred entry sits below the two-column block
AGREEMENT_TOLERANCE = 1e-14   # backward error, relative to the summed term magnitudes

mp.mp.dps = 30

nr = pickle.load(open(f"data/NR_catalog{CONFIG}.pkl", "rb"))
bob = pickle.load(open(f"data/BOB_catalog_N{N_FILTER}{CONFIG}.pkl", "rb"))
assert np.all(nr["sxs_id"] == bob["sxs_id"]), "catalogue files are not row-matched"
assert bob["N_filter"] == N_FILTER, "BOB catalogue has a different filter order"
assert np.array_equal(bob["t_grid"], nr["t_grid"]), "catalogue time grids differ"
t = nr["t_grid"]


def ladder(tau, Omega_QNM):
    """The removed frequencies and their weights, from the eikonal ladder.

    The removed modes are omega_k = m Omega_QNM - i(2k+1)/tau, so the weights
    are a_k = 2 Im(omega_k) = -2(2k+1)/tau, which is BOB2_appendix.tex's a_n.
    What is RETURNED is the CONJUGATE omega*_k = m Omega_QNM + i(2k+1)/tau --
    hence the plus sign below -- because that is the frequency both the
    divided-difference denominators i(omega*_m - omega*_n) and the D_n kernels
    are written in.
    """
    omega_star = [mp.mpf(float(M_MODE*Omega_QNM)) + 1j*(2*k + 1)/tau
                  for k in range(N_FILTER + 1)]
    a_k = [-2*(2*k + 1)/tau for k in range(N_FILTER + 1)]
    return omega_star, a_k


def filter_term(D, omega_star, a_k):
    """Every non-empty term of the expanded product, given D_m at one time.

    Expanding prod_n (1 + a_n D_n) gives one term per SUBSET of the removed
    modes.  The empty subset is Psi itself, handled by the caller; a subset S
    contributes prod_{n in S} a_n times the composition of its operators, which
    the divided-difference identity collapses to a single sum over S.

    Also returns the sum of the term magnitudes, which is how well-conditioned
    this sum is.
    """
    total = 0
    magnitude = 0.0
    for size in range(1, N_FILTER + 2):
        for subset in itertools.combinations(range(N_FILTER + 1), size):
            weight = 1
            for n in subset:
                weight *= a_k[n]
            collapsed = 0
            for m in subset:
                denominator = 1
                for n in subset:
                    if n != m:
                        denominator *= 1j*(omega_star[m] - omega_star[n])
                collapsed += D[m]/denominator
            total += weight*collapsed
            magnitude += abs(complex(weight*collapsed))
    return total, magnitude


def strain_peak_time(sxs_id, stored_news_peak):
    """Where the (2,2) STRAIN peaks, measured from the (2,2) news peak.

    The load and resample follow catalog_residual_and_BOB.py::load_news exactly
    -- same reference time, same step -- so the news peak found here must
    reproduce the one the catalogue stored.  The assert is what makes the
    difference meaningful rather than a comparison of two conventions.
    """
    simulation = sxs.load(sxs_id)
    strain = simulation.h
    reference_index = strain.index_closest_to(simulation.metadata.reference_time)
    h22 = strain[reference_index:][:, strain.index(2, M_MODE)]
    h22 = TimeSeries(h22.t, h22.data).fixed_timestep_resampled(STRAIN_DT)

    def peak_time(series):
        return float(np.asarray(series.t)[int(np.argmax(np.abs(np.asarray(series.y))))])

    news_peak = peak_time(h22.differentiated(1))
    assert abs(news_peak - stored_news_peak) < 1e-6, (
        f"{sxs_id}: news peak {news_peak} does not reproduce the catalogue's "
        f"{stored_news_peak} -- the two time conventions have drifted apart")
    return peak_time(h22) - news_peak


def build():
    """Every curve for the chosen simulation, on the catalogue's time grid."""
    row = list(nr["sxs_id"]).index(CASE)
    mf = float(nr["mf"][row])
    chif = float(nr["chif"][row])
    A_peak = float(nr["A_peak"][row])
    finite_time_shift = float(bob["finite_time_shift"][row])
    nr_wave = nr["residual"][row]
    finite_wave = bob["finite_residual"][row]

    print(f"{CASE}:  m_f={mf:.4f}  chi_f={chif:.4f}  A_peak={A_peak:.5f}")
    print(f"  NR filter delay      {nr['nr_time_shift'][row]:+.2f} M "
          f"(measured at align_ref = {nr['align_ref']:.0f} M)")
    print(f"  BOB amplitude shift  {finite_time_shift:+.2f} M "
          f"(fitted on {tuple(bob['bob_fit_window'])})")

    Omega_QNM, _, tau = BOB.get_key_BOB_values(M_MODE, mf, chif)
    omega_star, a_k = ladder(mp.mpf(float(tau)), Omega_QNM)
    print(f"  expanding the product over {2**(N_FILTER + 1)} subsets; "
          f"a_n = " + ", ".join(f"{float(a):+.4f}" for a in a_k))

    news = np.empty(len(MODEL_GRID), dtype=complex)
    operator_term = np.empty(len(MODEL_GRID), dtype=complex)
    module_form = np.empty(len(MODEL_GRID), dtype=complex)
    conditioning = np.empty(len(MODEL_GRID))
    for i, time in enumerate(MODEL_GRID):
        # the unfiltered simplified BOB news, exactly as get_D_BOB builds it
        news[i] = complex(BOB.BOB_amplitude(M_MODE, mf, chif, A_peak, time)
                          * mp.e**(-1j*BOB.BOB_simple_phase(M_MODE, mf, chif, time)))
        D = [BOB.get_D_BOB(k, M_MODE, mf, chif, time, A_peak)
             for k in range(N_FILTER + 1)]
        term, conditioning[i] = filter_term(D, omega_star, a_k)
        operator_term[i] = complex(term)
        module_form[i] = complex(
            BOB.get_finite_N_sl0(M_MODE, mf, chif, time, A_peak, N_FILTER)
            if time < 0 else
            BOB.get_finite_N_sg0(M_MODE, mf, chif, time, A_peak, N_FILTER))

    decomposition = news + operator_term

    # The expansion sums many large terms down to a small one, so the error is
    # normalised by the SUM OF THE MAGNITUDES rather than by the answer.  That
    # is the standard backward-error test: it asks whether the terms were added
    # correctly, which is the identity's content.  Normalising by the answer
    # would only measure the cancellation, which reaches 1e8 at N = 6.
    scale = np.maximum(conditioning, np.abs(news))
    backward = np.abs(decomposition - module_form)/scale
    print(f"  identity holds to {backward.max():.3e} of the summed term "
          f"magnitudes (float64 is ~2e-16)")
    print(f"  conditioning sum|term|/|net|: median "
          f"{np.median(conditioning/np.abs(decomposition)):.3e}")
    print(f"  cancellation |news|/|sum|: median "
          f"{np.median(np.abs(news)/np.abs(module_form)):.1f}")
    if backward.max() > AGREEMENT_TOLERANCE:
        raise SystemExit(
            f"ABORT: the assembled terms disagree with the module by "
            f"{backward.max():.3e} of their summed magnitudes, well above "
            f"float64 -- the identity, not the conditioning, is at fault.")

    def to_catalogue_grid(values):
        """Same shift the catalogue applied to the finite-N model, same sign."""
        return (np.interp(t - finite_time_shift, MODEL_GRID, values.real)
                + 1j*np.interp(t - finite_time_shift, MODEL_GRID, values.imag))

    # One phase, from the complex overlap against NR on the window the shift was
    # fitted on, applied identically to every model curve -- so the top panel
    # still compares the identity rather than three separate fits.
    fitted = (t >= PHASE_WINDOW[0]) & (t <= PHASE_WINDOW[1])
    rotation = np.exp(1j*np.angle(np.sum(np.conj(finite_wave[fitted])*nr_wave[fitted])))

    strain_peak = strain_peak_time(CASE, float(nr["t_peak"][row]))
    print(f"  (2,2) strain peak at t = {strain_peak:+.2f} M")
    # s = 0 is where BOB's tanh frequency is centred and where its own
    # unfiltered news peaks.  It is a model-frame time, so it picks up the same
    # fitted shift to_catalogue_grid applies to the curves.
    print(f"  s = 0 at t = {finite_time_shift:+.2f} M")

    return dict(nr=nr_wave,
                finite=finite_wave*rotation,
                net=to_catalogue_grid(decomposition)*rotation,
                news=to_catalogue_grid(news)*rotation,
                operator=to_catalogue_grid(operator_term)*rotation,
                strain_peak=strain_peak, s_zero=finite_time_shift)


curves = build()

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(my_width, my_width*1.15),
                       gridspec_kw={"hspace": 0.0})

# The net decomposition (top) and the sum (bottom) are the SAME curve, so they
# share one style and one legend entry.  NR therefore gives up black for grey.
net_style = dict(color="black", lw=1)

# draw order sets what covers what: NR underneath, then net, then finite N
nr_line, = ax[0].semilogy(t, np.abs(curves["nr"].real), color="0.55", lw=1.6)
net_line, = ax[0].semilogy(t, np.abs(curves["net"].real), **net_style)
finite_line, = ax[0].semilogy(t, np.abs(curves["finite"].real),
                              color="red", ls=(0, (1, 5)), lw=1)
ax[0].set_ylabel(r"$|\Re\,\mathcal{N}_{22}|$")

# SIGNED, on a linear axis.  Taking the modulus here put a spike at every zero
# crossing -- twice per carrier cycle for each curve, at unrelated times --
# which reads as a cancellation between the curves when it is only the
# oscillation passing through zero.
unfiltered_line, = ax[1].plot(t, curves["news"].real, color="#2f6b8c", lw=1,
                              ls=(0, (1, 1.5)))
ax[1].plot(t, curves["net"].real, **net_style)
# drawn last, so it sits on top of the net curve where the two coincide
filter_line, = ax[1].plot(t, curves["operator"].real, color="forestgreen", lw=1,
                          ls=(0, (5, 4)))
ax[1].axhline(0.0, color="0.55", lw=0.5, ls="--", zorder=0)
# the (2,2) strain peak, which leads the news peak the axis is measured from
strain_line = ax[1].axvline(curves["strain_peak"], color="black", lw=0.6, ls="--",
                            zorder=0)
s_zero_line = ax[1].axvline(curves["s_zero"], color="maroon", lw=0.6, ls="--",
                            zorder=0)
ax[1].set_ylabel(r"$\Re\,\mathcal{N}_{22}$")
ax[1].set_xlabel(r"$t-t_{\rm peak}^{\mathcal{N}_{22}}$   $[M]$")

window = (t >= XLIM[0]) & (t <= XLIM[1])
top = max(np.abs(curves[k].real)[window].max()
          for k in ("nr", "net", "finite", "news", "operator"))
for panel in ax:
    panel.set_xlim(*XLIM)
ax[0].set_ylim(top*10**(-DECADES), top*HEADROOM)
ax[1].set_ylim(-top*HEADROOM, top*HEADROOM)   # linear and symmetric, since it is signed

# Two legends rather than one.  A single legend with a padding handle reserves a
# cell for the blank, leaving white space beside the long entry; instead the six
# short entries go in a 2-column block, filled column-first, and the long one
# gets its own centred legend underneath.
short = [(nr_line, "NR residual"), (finite_line, r"BOB finite $N$"),
         (strain_line, r"$t_p^{h_{22}}$"), (unfiltered_line, "unfiltered BOB"),
         (filter_line, "filter effects"), (s_zero_line, r"$s=0$")]
fig.legend([h for h, _ in short], [l for _, l in short], loc="upper center",
           bbox_to_anchor=(0.5, -0.005), ncol=2, fontsize=LEGEND_FONTSIZE)
fig.legend([net_line], ["net decomposition (unfiltered BOB + filter effects)"],
           loc="upper center", bbox_to_anchor=(0.5, NET_LEGEND_Y), ncol=1,
           fontsize=LEGEND_FONTSIZE)

output = f"fig_decomposition{CONFIG}_{CASE.split(':')[-1]}.pdf"
fig.savefig(output, bbox_inches="tight")
print(f"  wrote final_code/{output}")
