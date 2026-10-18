#!/usr/bin/env python3
"""Simplified BOB against NR for the (2,2) news, three SXS systems side by side.

Each panel draws Im[N_22], NR in black and simplified BOB dashed.  The model is
neither rescaled nor shifted in time -- A_p is read off the NR news peak -- so
its only freedom is a single constant phase, fitted over [0, TMAX].  An inset
repeats the late-time window, where the two curves are hardest to separate.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from kuibit.timeseries import TimeSeries
from gwBOB import BOB_utils

import BOB_RF_residual as BOB

#https://duetosymmetry.com/code/latex-mpl-fig-tips/
plt.style.use("paper.mplstyle")
pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
            "CQG": {"onecol": 374.*pt}, # CQG is only one column
            # Add more journals below. Can add more properties to each journal
            }

my_width = jour_sizes["PRD"]["twocol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2

fig = plt.figure(figsize = (my_width, my_width/golden))
# -----------------------------------------------------------------------------

L, M_MODE = 2, 2
XMIN, TMAX = -20.0, 100.0    # plotted range, M from the news peak
ZOOM = (60.0, 100.0)         # inset window
CASES = ["SXS:BBH:2477", "SXS:BBH:0305", "SXS:BBH:0178"]   # chi_f = 0.19, 0.69, 0.95
NR_STYLE = dict(color="black", linewidth=1)
BOB_STYLE = dict(color="cyan", linestyle=(0, (5, 5)), linewidth=1)


def load_nr_news(sxs_id):
    """The NR (2,2) news, with time measured from its own peak."""
    sim = BOB_utils.BOB()
    sim.initialize_with_sxs_data(sxs_id, l=L, m=M_MODE)
    sim.what_should_BOB_create = "news"
    sim.set_start_before_tpeak = -100
    sim.set_end_after_tpeak = TMAX + 15
    news = TimeSeries(*sim.get_news_data(l=L, m=M_MODE)).fixed_timestep_resampled(0.1)

    t, nr = np.asarray(news.t), np.asarray(news.y)
    peak = np.argmax(np.abs(nr))
    return t - t[peak], nr, np.abs(nr[peak]), sim.mf, sim.chif


def simplified_bob(mf, chif, A_p, t):
    """A_BOB exp(-i Phi_BOB), both from the BOB module; t runs from the news peak."""
    amplitude = BOB.BOB_amplitude(M_MODE, mf, chif, A_p, t)
    phase = BOB.BOB_simple_phase(M_MODE, mf, chif, t)
    return amplitude*np.exp(-1j*phase)


def align_phase(model, nr, window):
    """Rotate the model by the constant phase that best matches NR on `window`."""
    phi = np.angle(np.sum(np.conj(model[window])*nr[window]))
    return model*np.exp(1j*phi), np.degrees(phi)


axes = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.22))

print(f"  {'case':<8}{'chi_f':>8}{'A_p':>10}{'phase':>9}{'resid':>9}")
for ax, sxs_id in zip(axes, CASES):
    t, nr, A_p, mf, chif = load_nr_news(sxs_id)
    fit_window = (t >= 0.0) & (t <= TMAX)        # align on the ringdown only
    model, phi = align_phase(simplified_bob(mf, chif, A_p, t), nr, fit_window)

    shown = (t >= XMIN) & (t <= TMAX)
    ax.plot(t[shown], np.imag(nr[shown]), label=sxs_id, **NR_STYLE)
    ax.plot(t[shown], np.imag(model[shown]), label="BOB", **BOB_STYLE)
    ax.set_xlim(XMIN, TMAX)

    drawn = np.concatenate([np.imag(nr[shown]), np.imag(model[shown])])
    ax.set_ylim(1.15*drawn.min(), 1.15*drawn.max())
    ax.set_box_aspect(1)     # square panels; savefig's tight bbox trims the spare height
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend()

    inset = inset_axes(ax, width="42%", height="30%", loc="lower right", borderpad=0.8)
    zoomed = (t >= ZOOM[0]) & (t <= ZOOM[1])
    inset.plot(t[zoomed], np.imag(nr[zoomed]), **NR_STYLE)
    inset.plot(t[zoomed], np.imag(model[zoomed]), **BOB_STYLE)
    inset.set_xlim(*ZOOM)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)

    for corner, edge in zip([(0, 1), (1, 1)], ZOOM):
        fig.add_artist(ConnectionPatch(
            xyA=corner, coordsA=inset.transAxes,
            xyB=(edge, np.interp(edge, t, np.imag(nr))), coordsB=ax.transData,
            color="0.62", linewidth=0.3, zorder=0.5))

    residual = (np.sqrt(np.mean(np.abs(model[fit_window] - nr[fit_window])**2))
                / np.sqrt(np.mean(np.abs(nr[fit_window])**2)))
    print(f"  {sxs_id.split(':')[-1]:<8}{chif:>8.4f}{A_p:>10.5f}{phi:>+9.2f}{residual:>9.3f}")

axes[0].set_ylabel(r"$\Im[\mathcal{N}_{22}(t)]$")
axes[1].set_xlabel(r"$t-t^{\mathcal{N}_{22}}_{\rm peak}\;\;[M]$")

fig.savefig("simpleBOB_vs_NR.pdf", bbox_inches="tight", pad_inches=0.01)
print("\n  saved final_code/simpleBOB_vs_NR.pdf")
