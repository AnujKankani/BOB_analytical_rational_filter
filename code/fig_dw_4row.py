#!/usr/bin/env python3
"""NR direct wave against both BOB approximations

Columns are three simulations ordered by remnant spin; rows are the real part,
the amplitude, the frequency and the damping rate of the direct wave.

The NR direct wave is read from the catalogue, which has 3 prograde QNMs, 3
retrograde and the (3,2,0) removed.  Two BOB curves are drawn against it:

    finite N : get_finite_N_sl0 / get_finite_N_sg0, whichever is valid at each s
    large N  : get_large_N_amplitude x exp(-i get_large_N_phase), with
               get_large_N_frequency and get_large_N_damping for rows 3 and 4

Both are aligned independently -- each gets its own time shift from the
amplitude residual on FIT_WINDOW, and its own phase from the complex overlap on
the same window -- so neither inherits the other's alignment.

Rows 3 and 4 carry reference levels rather than dividing by them: the
fundamental QNM and the remnant's horizon scales, all in 1/M.
"""
import pickle

import numpy as np
import qnm
import sxs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import BOB_RF_residual as BOB

# ---- figure style (copied verbatim from figure_style.py) --------------------
plt.style.use("paper.mplstyle")
pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
            "CQG": {"onecol": 374.*pt}, # CQG is only one column
            # Add more journals below. Can add more properties to each journal
            }

my_width = jour_sizes["PRD"]["twocol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2

page_height = 650.*pt        # PRD text height: this figure fills the page
# -----------------------------------------------------------------------------

M_MODE = 2
CONFIG = "_3pro_3retro_320"            # named the way the catalogue names it
NUM_PROGRADE, NUM_RETROGRADE, ADD_32 = 3, 3, True
N_BOB = NUM_PROGRADE - 1     # the closed forms remove n = 0 ... N_BOB
# the large-N forms are an asymptotic in N, so they are evaluated far from
# N_BOB; only the amplitude and damping use it, the frequency ignores it
N_LARGE = 7

CASES = ["SXS:BBH:2477", "SXS:BBH:0305", "SXS:BBH:0178"]   # by remnant spin
XLIM = (-20.0, 20.0)
FIT_WINDOW = (-10.0, 0.0)              # both shifts are fitted on the amplitude here
FREQUENCY_YLIM = (0.0, 1.0)   # row 3; wide enough for every reference line
# the lower bound is negative so a sign change stays visible; the upper bound
# clips the late-time rise rather than letting it set the scale
DAMPING_YLIM = (-0.1, 0.6)    # row 4
AMP_FLOOR = 0.0              # no amplitude cut: everything is drawn, including
                             # where the log-derivative is amplifying noise
MODEL_GRID = np.arange(-110.0, 35.0 + 1e-9, 0.1)   # BOB's own frame, padded so
SHIFT_MAX, SHIFT_STEP = 60.0, 0.05                # the shift search cannot run off

NR_STYLE = dict(color="black", linewidth=1)
LARGE_N_STYLE = dict(color="green", linestyle=(0, (5, 10)), linewidth=1)
FINITE_N_STYLE = dict(color="red", linestyle=(0, (1, 1.5)), linewidth=1)
ROW_LABELS = [r"$|\Re[\mathcal{N}^{\rm res}_{22}]|$",
              r"$|\mathcal{N}^{\rm res}_{22}|$",
              r"$\Re\,\omega_{\rm res}$",
              r"$-\Im\,\omega_{\rm res}$"]

# the first prograde overtone neither filter removes; available as a reference
# level, though the figure as it stands does not draw it
FIRST_SURVIVING_N = max(NUM_PROGRADE, N_BOB + 1)
OVERTONE_FREQUENCY_LABEL = rf"$\Re\,\omega_{{22{FIRST_SURVIVING_N}}}$"
OVERTONE_DAMPING_LABEL = rf"$-\Im\,\omega_{{22{FIRST_SURVIVING_N}}}$"


def prograde_qnm_frequency(n, mf, chif):
    """Complex omega_{22n} in units of 1/M, as get_key_BOB_values defines it.

    Transcribed from that function so the overtone levels sit in exactly the
    same convention as the omega_220 ones: qnm only accepts a in [0, 1), so a
    remnant spinning along -z is looked up at |chi_f| with m -> -m, and the +z
    branch is conjugated (qnm returns Im omega < 0, BOB wants Im omega > 0).
    Only the +z branch is conjugated, so take abs() of the imaginary part.
    """
    if chif >= 0:
        omega, _, _ = qnm.modes_cache(s=-2, l=2, m=2, n=n)(a=chif)
        return np.conj(omega)/mf
    omega, _, _ = qnm.modes_cache(s=-2, l=2, m=-2, n=n)(a=-chif)
    return omega/mf


nr = pickle.load(open(f"data/NR_catalog{CONFIG}.pkl", "rb"))
# the configuration is named in the filename, but it is also recorded inside, so
# the two cannot drift apart unnoticed
assert nr["num_prograde_qnms"] == NUM_PROGRADE, "catalogue removes a different prograde count"
assert nr["num_retrograde_qnms"] == NUM_RETROGRADE, "catalogue removes a different retrograde count"
assert nr["add_32_mode"] == ADD_32, "catalogue disagrees on the (3,2,0) mode"
t = nr["t_grid"]
shown = (t >= XLIM[0]) & (t <= XLIM[1])
ids = list(nr["sxs_id"])

print(f"  NR from data/NR_catalog{CONFIG}.pkl: {NUM_PROGRADE} prograde, "
      f"{NUM_RETROGRADE} retrograde{' and the (3,2,0)' if ADD_32 else ''}")
for sxs_id in CASES:
    print(f"    {sxs_id:<16} filter delay "
          f"{nr['nr_time_shift'][ids.index(sxs_id)]:+7.2f} M")

# the mass ratio is not in the catalogue pickle, so it comes from the dataframe
catalogue = sxs.load("dataframe", tag="3.0.0").loc[CASES]
MASS_RATIO = {sxs_id: (max(row["reference_mass1"], row["reference_mass2"])
                       / min(row["reference_mass1"], row["reference_mass2"]))
              for sxs_id, row in catalogue.iterrows()}


def bob_curves(mf, chif, A_peak):
    """Both BOB approximations on MODEL_GRID, in BOB's own time frame.

    MODEL_GRID is measured from the unfiltered news peak, the convention every
    function in the BOB module expects.  The finite-N form has two
    representations either side of s = 0, so each sample uses the valid one.
    """
    _, _, tau_bob = BOB.get_key_BOB_values(M_MODE, mf, chif)

    finite = np.empty(len(MODEL_GRID), dtype=complex)
    large = np.empty(len(MODEL_GRID), dtype=complex)
    omega = np.empty(len(MODEL_GRID))
    damping_rate = np.empty(len(MODEL_GRID))
    for i, time in enumerate(MODEL_GRID):
        if time < 0:
            finite[i] = BOB.get_finite_N_sl0(M_MODE, mf, chif, time, A_peak, N_BOB)
        else:
            finite[i] = BOB.get_finite_N_sg0(M_MODE, mf, chif, time, A_peak, N_BOB)
        amplitude = complex(BOB.get_large_N_amplitude(
            M_MODE, mf, chif, time, A_peak, N_LARGE)).real
        phase = float(BOB.get_large_N_phase(M_MODE, mf, chif, time, N_LARGE))
        large[i] = amplitude*np.exp(-1j*phase)
        omega[i] = float(BOB.get_large_N_frequency(M_MODE, mf, chif, time, N_LARGE))
        # K_{i lam}(x) is real for real x, but mpmath returns the mpc type
        damping_rate[i] = -complex(BOB.get_large_N_damping(
            M_MODE, mf, chif, time, A_peak, N_LARGE)).real

    # the finite-N form has no closed-form frequency or damping, so they come
    # from differentiating it -- done here, in the model frame, so that a time
    # shift later is a pure translation of them
    derivative = np.gradient(finite, MODEL_GRID)/finite
    return dict(finite=finite, large=large, omega=omega, rate=damping_rate,
                finite_omega=-np.imag(derivative),
                finite_rate=-np.real(derivative),
                tau_bob=tau_bob)


def sample(values, shift):
    """A model-frame array read at the plot grid, moved later in time by `shift`."""
    if np.iscomplexobj(values):
        return (np.interp(t - shift, MODEL_GRID, values.real)
                + 1j*np.interp(t - shift, MODEL_GRID, values.imag))
    return np.interp(t - shift, MODEL_GRID, values)


def align(curves, prefix, nr_wave):
    """(shift, phase) for one BOB curve, fitted on the amplitude over FIT_WINDOW.

    Fitting the amplitude cannot settle on a neighbouring cycle, since it is
    blind to the carrier.  The phase then follows in closed form from the
    complex overlap at the chosen shift.
    """
    fitted = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1])
    model_wave = curves[prefix]
    shifts = np.arange(-SHIFT_MAX, SHIFT_MAX + 1e-9, SHIFT_STEP)

    target = np.abs(nr_wave)[fitted]
    # absolute, not squared, matching best_bob_shift in the catalogue builder:
    # every sample counts in proportion to how far off it is, so the loud
    # samples near the peak do not dominate the fit
    cost = [np.sum(np.abs(np.abs(sample(model_wave, shift))[fitted] - target))
            for shift in shifts]
    shift = float(shifts[int(np.argmin(cost))])
    phase = np.angle(np.sum(np.conj(sample(model_wave, shift)[fitted])
                            *nr_wave[fitted]))
    return shift, phase


print(f"  BOB at N = {N_BOB}, removing {N_BOB + 1} prograde QNMs")
MODELS = {sxs_id: bob_curves(nr["mf"][ids.index(sxs_id)],
                             nr["chif"][ids.index(sxs_id)],
                             nr["A_peak"][ids.index(sxs_id)])
          for sxs_id in CASES}

fig, axes = plt.subplots(4, 3, figsize=(my_width, page_height), sharex=True)
legend_entries = None

for column, sxs_id in enumerate(CASES):
    row_index = ids.index(sxs_id)
    mf, chif = nr["mf"][row_index], nr["chif"][row_index]
    nr_wave = nr["residual"][row_index]
    nr_tau = nr["tau"][row_index]
    curves = MODELS[sxs_id]

    # each approximation is aligned on its own, not inherited from the other
    finite_shift, finite_phase = align(curves, "finite", nr_wave)
    large_shift, large_phase = align(curves, "large", nr_wave)
    finite_wave = sample(curves["finite"], finite_shift)*np.exp(1j*finite_phase)
    large_wave = sample(curves["large"], large_shift)*np.exp(1j*large_phase)

    Omega_QNM, _, tau_qnm = BOB.get_key_BOB_values(M_MODE, mf, chif)
    omega_220 = M_MODE*Omega_QNM
    root = np.sqrt(1.0 - chif**2)
    omega_horizon = M_MODE*abs(chif)/(2*(1.0 + root))/mf
    # the row 4 references carry no factor of m, unlike the frequencies above
    damping_220 = 1.0/tau_qnm
    surface_gravity = root/(2*(1.0 + root))/mf

    amplitude = np.abs(nr_wave[shown])
    weak = amplitude < AMP_FLOOR*amplitude.max()

    def drawn(values, shift, mask_weak=False, stop_at_s0=True):
        """Sampled at the plot grid, optionally stopping at s = 0.

        The large-N amplitude and damping come from the Bessel envelope, which
        the draft derives for s < 0 only, so they stop there.  The large-N
        frequency is the tanh law, which carries no such restriction, and the
        finite-N form is valid either side, so both are drawn through.
        """
        out = np.array(values[shown],
                       dtype=complex if np.iscomplexobj(values) else float)
        bad = weak if mask_weak else np.zeros(out.shape, dtype=bool)
        if stop_at_s0:
            bad = bad | (((t - shift)/curves["tau_bob"])[shown] > 0)
        out[bad] = np.nan
        return out

    def finite(values, mask_weak=False):
        return drawn(values, finite_shift, mask_weak, stop_at_s0=False)

    def large(values, mask_weak=False, stop_at_s0=True):
        return drawn(values, large_shift, mask_weak, stop_at_s0=stop_at_s0)

    rows = [(np.abs(nr_wave[shown].real),
             np.abs(large(large_wave).real),
             np.abs(finite(finite_wave).real)),
            (amplitude,
             np.abs(large(large_wave)),
             np.abs(finite(finite_wave))),
            (np.where(weak, np.nan, nr["omega"][row_index][shown]),
             # shifted like the amplitude above, so every large-N panel shows
             # the same waveform in the same frame
             large(sample(curves["omega"], large_shift), True, stop_at_s0=False),
             finite(sample(curves["finite_omega"], finite_shift), True)),
            (np.where(weak, np.nan, 1.0/nr_tau[shown]),
             large(sample(curves["rate"], large_shift), True),
             finite(sample(curves["finite_rate"], finite_shift), True))]

    # one mass throughout: rows 3 and 4 are in 1/M, so the time axis is in M too
    plot_time = t[shown]
    for row, (nr_values, large_values, finite_values) in enumerate(rows):
        ax = axes[row][column]
        # large N is drawn last so it sits on top where the curves overlap
        nr_line, = ax.plot(plot_time, nr_values, **NR_STYLE)
        finite_line, = ax.plot(plot_time, finite_values, **FINITE_N_STYLE)
        large_line, = ax.plot(plot_time, large_values, **LARGE_N_STYLE)
        ax.grid(False)

    frequency_axis = axes[2][column]
    freq_220 = frequency_axis.axhline(omega_220, color="lightgreen", ls="--",
                                      lw=0.5, zorder=1)
    freq_horizon = frequency_axis.axhline(omega_horizon, color="maroon", ls=":",
                                          lw=0.5, zorder=1)
    frequency_axis.set_ylim(*FREQUENCY_YLIM)

    damping_axis = axes[3][column]
    # same colour coding as row 3 -- green for the QNM, maroon for the horizon --
    # but dash-dotted, so the legend entries stay distinguishable
    damp_220 = damping_axis.axhline(damping_220, color="lightgreen", ls="-.",
                                    lw=0.5, zorder=1)
    damp_horizon = damping_axis.axhline(surface_gravity, color="maroon", ls="-.",
                                        lw=0.5, zorder=1)
    damping_axis.set_ylim(*DAMPING_YLIM)

    top = axes[0][column]
    real_part = np.abs(nr_wave[shown].real)
    # the zero crossings dive to whatever the grid samples, so the floor is set
    # four decades below the peak rather than by those samples
    top.set_yscale("log")
    top.set_ylim(1e-4*real_part.max(), 2.0*real_part.max())
    # usetex is on, so the id goes in \texttt verbatim rather than as math
    top.set_title(r"\texttt{" + sxs_id + r"}" + "\n"
                  + rf"$q = {MASS_RATIO[sxs_id]:.2f}$, "
                  + rf"$\chi_f = {chif:.2f}$")
    if column == 0:
        # matplotlib fills the legend column by column; the blank keeps each
        # reference level's frequency above its own damping
        legend_entries = [(nr_line, "NR"),
                          (large_line, r"BOB (large $N$)"),
                          (finite_line, r"BOB (finite $N$)"),
                          (Line2D([], [], linestyle="none"), ""),
                          (freq_220, r"$\Re\,\omega_{220}$"),
                          (damp_220, r"$-\Im\,\omega_{220}$"),
                          (freq_horizon, r"$\omega_H$"),
                          (damp_horizon, r"$\kappa$")]

    print(f"    {sxs_id:<16}finite {finite_shift:+7.2f} / "
          f"{np.degrees(finite_phase):+7.1f}      large {large_shift:+7.2f} / "
          f"{np.degrees(large_phase):+7.1f}")

for row in range(4):
    axes[row][0].set_ylabel(ROW_LABELS[row])
for column in range(3):
    axes[3][column].set_xlim(*XLIM)
    axes[3][column].set_xlabel(r"$t-t^{\mathcal{N}_{22}}_{\rm peak}\;\;[M]$")

fig.legend([artist for artist, _ in legend_entries],
           [name for _, name in legend_entries],
           loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=4,
           frameon=False, handlelength=2.8, fontsize=9)
fig.subplots_adjust(hspace=0.12, wspace=0.28)

output = f"fig_dw_4row{CONFIG}_largeN{N_LARGE}.pdf"
fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
print(f"\n  saved final_code/{output}")
