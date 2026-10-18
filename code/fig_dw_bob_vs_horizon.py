#!/usr/bin/env python3
"""The NR direct wave measured against BOB, and against the horizon, stacked.

Two panels on common axes, so the two comparisons can be read directly against
each other.  Each trajectory is one simulation, traced over WINDOW and coloured
by its signed remnant spin:

    top     x = Re omega_res^NR - Re omega_res^BOB      BOB, finite N
            y = -Im omega_res^NR + Im omega_res^BOB

    bottom  x = Re omega_res^NR - |m Omega_H|           the horizon
            y = -Im omega_res^NR - kappa

Everything is in 1/M, so perfect agreement would collapse every trajectory onto
the origin.  The differences keep their sign, and the dashed cross marks zero.

BOB's finite-N waveform has no closed-form frequency, so its frequency and
damping rate come from the log derivative of the waveform, the same way the NR
ones do.
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

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
CONFIG = "_3pro_3retro_320"   # NR removes 3 prograde QNMs, 3 retrograde and the (3,2,0)
N_FILTER = 2                  # BOB removes the same 3 prograde overtones, so N = 3 - 1
WINDOW = (-7.5, 7.5)          # traced interval, M from the unfiltered news peak

COLOUR_MAP = "managua"
COLOUR_RANGE = (-0.4, 1.0)    # the remnants occupy chi_f in [-0.354, +0.951]
COLOUR_CENTRE = 0.69          # where the colour map's neutral midpoint is pinned
COLOUR_TICKS = [-0.4, 0.0, 0.5, 0.69, 0.75, 0.9, 1.0]
XLIM, YLIM = (-0.3, 0.4), (-0.4, 0.3)


def frequency_and_damping(t, waveform):
    """Instantaneous frequency and damping time of a complex waveform.

    With N = A exp(-i Phi), Ndot/N = Adot/A - i omega, so omega = -Im(Ndot/N)
    and, for A ~ exp(-t/tau), tau = -1/Re(Ndot/N).
    """
    log_derivative = np.gradient(waveform, t, axis=-1)/waveform
    return -np.imag(log_derivative), -1.0/np.real(log_derivative)


nr = pickle.load(open(f"data/NR_catalog{CONFIG}.pkl", "rb"))
bob = pickle.load(open(f"data/BOB_catalog_N{N_FILTER}{CONFIG}.pkl", "rb"))

# Every catalogue in data/ holds the same 591 rows in the same order, so the
# filter order alone cannot tell two BOB files apart.  
assert np.all(nr["sxs_id"] == bob["sxs_id"]), "catalogue files are not row-matched"
assert bob["N_filter"] == N_FILTER, "BOB file has a different filter order"
assert np.array_equal(bob["t_grid"], nr["t_grid"]), "catalogue time grids differ"
assert np.array_equal(bob["chif"], nr["chif"]), "catalogue remnant spins differ"

chif = nr["chif"]
root = np.sqrt(1.0 - chif**2)
# chif is the signed z-component, negative for 43 of the 591 remnants. 
# Both are written in the remnant's own mass, so both are divided by mf.
omega_horizon = np.abs(M_MODE*chif/(2.0*(1.0 + root))/nr["mf"])
surface_gravity = root/(2.0*(1.0 + root))/nr["mf"]

bob_omega, bob_tau = frequency_and_damping(nr["t_grid"], bob["finite_residual"])
traced = (nr["t_grid"] >= WINDOW[0]) & (nr["t_grid"] <= WINDOW[1])
frequency = nr["omega"][:, traced]
rate = 1.0/nr["tau"][:, traced]          # tau is the damping TIME, so the rate is 1/tau

PANELS = [
    (frequency - bob_omega[:, traced],
     rate - 1.0/bob_tau[:, traced],
     r"$\Re\,\omega_{\rm res}^{\rm NR} - \Re\,\omega_{\rm res}^{\rm BOB}$",
     r"$-\Im\,\omega_{\rm res}^{\rm NR} + \Im\,\omega_{\rm res}^{\rm BOB}$",
     "BOB (finite $N$)"),
    (frequency - omega_horizon[:, None],
     rate - surface_gravity[:, None],
     r"$\Re\,\omega_{\rm res}^{\rm NR} - |m\Omega_H|$",
     r"$-\Im\,\omega_{\rm res}^{\rm NR} - \kappa$",
     "horizon"),
]

fig, axes = plt.subplots(2, 1, figsize=(my_width, my_width*0.62*2), sharex=True,
                         gridspec_kw={"hspace": 0.38})
colour_map = plt.get_cmap(COLOUR_MAP)
# the midpoint is pinned at COLOUR_CENTRE and the two sides stretched unequally,
# so the bar still spans the full range
norm = TwoSlopeNorm(vmin=COLOUR_RANGE[0], vcenter=COLOUR_CENTRE, vmax=COLOUR_RANGE[1])

print(f"  window {WINDOW[0]:+.1f} to {WINDOW[1]:+.1f} M, {len(chif)} simulations")
for ax, (x, y, xlabel, ylabel, name) in zip(axes, PANELS):
    for zero_line in (ax.axhline, ax.axvline):
        zero_line(0.0, color="0.65", linestyle=(0, (6, 4)), linewidth=0.5, zorder=1)

    # drawn in order of increasing spin, so the fastest remnants land on top
    # rather than wherever the catalogue happened to put them
    for row in np.argsort(chif):
        colour = colour_map(norm(chif[row]))
        ax.plot(x[row], y[row], color=colour, alpha=0.6, zorder=2)
        # A triangle marks where each trajectory ends.  It points away from the
        # track: up when the endpoint sits in the upper half of that
        # trajectory's own y range, down when it sits in the lower half.
        midpoint = 0.5*(y[row].min() + y[row].max())
        ax.plot(x[row, -1], y[row, -1],
                marker="^" if y[row, -1] >= midpoint else "v",
                markerfacecolor=colour, markeredgecolor=colour,
                markersize=3, markeredgewidth=0.5,
                linestyle="none", alpha=0.6, zorder=3)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    # the two panels measure the direct wave against different references, so
    # each x axis is labelled and ticked in full; sharex only locks the ranges
    ax.tick_params(labelbottom=True)
    print(f"    {name:18s} median x {np.median(x):+.4f}   median y {np.median(y):+.4f}")

# the top panel keeps its own x tick labels, whose last one runs to the right
# edge, so the colour bar is padded clear of them
colour_bar = fig.colorbar(ScalarMappable(norm=norm, cmap=colour_map),
                          ax=axes.tolist(), pad=0.10)
colour_bar.set_label(r"$\chi_f$")
colour_bar.set_ticks(COLOUR_TICKS)

fig.savefig("dw_bob_vs_horizon.pdf", bbox_inches="tight", pad_inches=0.01)
print("\n  saved final_code/dw_bob_vs_horizon.pdf")
