"""Rational filter for SXS waveforms.

Removes a chosen set of quasinormal modes from a waveform by multiplying its
spectrum by one all-pass factor per mode.

The caller chooses how many prograde (l, m, n) overtones to remove, how many
retrograde (l, -m, n) overtones, and whether to also remove the prograde
(3, 2, 0) mode.

    from NR_apply_filter import apply_filter
    filtered = apply_filter(news, num_prograde_qnms=7, num_retrograde_qnms=2,
                            mf=mf, chif=chif, add_32_mode=True)

"""
import numpy as np
import qnm
from kuibit.timeseries import TimeSeries as kuibit_ts

PAD_DAMPING_TIMES = 300   # zero padding, in fundamental damping times


def get_qnm_frequency(l, m, n, chi):
    """Complex Kerr QNM frequency omega_{lmn} in units of the remnant mass.

    A negative chi means the remnant spins along -z, while the waveform's m is
    defined against +z.  qnm only accepts a in [0, 1), and reversing the spin
    direction is equivalent to reversing the sign of m, so such a mode is
    looked up at |chi| with m -> -m.  Both filter factors below go through
    this function, so prograde and retrograde swap over together, as they
    should.  Same convention as get_key_BOB_values in BOB_RF_residual.py.
    """
    if chi < 0:
        m, chi = -m, -chi
    grav_mode = qnm.modes_cache(s=-2, l=l, m=m, n=n)
    omega, _, _ = grav_mode(a=chi)
    return omega


def get_prograde_filter(freq, l, m, n, mf, chi):
    """All-pass factor whose zero sits on the prograde (l, m, n) mode."""
    # numpy has the opposite fft convention to the directwave convention
    freq = -2*np.pi*freq
    omega_lmn = get_qnm_frequency(l, m, n, chi)/mf
    return (freq - omega_lmn)/(freq - np.conj(omega_lmn))


def get_retrograde_filter(freq, l, m, n, mf, chi):
    """All-pass factor whose zero sits on the retrograde (l, -m, n) mode."""
    # numpy has the opposite fft convention to the directwave convention
    freq = -2*np.pi*freq
    omega_lmn = get_qnm_frequency(l, -m, n, chi)/mf
    return (freq + np.conj(omega_lmn))/(freq + omega_lmn)


def apply_filter(waveform, num_prograde_qnms, num_retrograde_qnms, mf, chif,
                 l=2, m=2, add_32_mode=False, l_max=None):
    """Filter `waveform`, returning a TimeSeries on the same time axis.

    Parameters
    ----------
    waveform : kuibit TimeSeries
        Complex SXS waveform, regularly sampled.
    num_prograde_qnms : int
        Remove prograde overtones n = 0 ... num_prograde_qnms-1.
    num_retrograde_qnms : int
        Remove retrograde overtones n = 0 ... num_retrograde_qnms-1.
    mf, chif : float
        Remnant mass and dimensionless spin.
    l, m : int
        Mode being filtered.
    add_32_mode : bool
        Also remove the prograde (3, 2, 0) mode.
    l_max : int or None
        If set, repeat the whole prograde and retrograde removal for every
        harmonic l ... l_max at the SAME m, not just for l itself.  The (2,2)
        data carries mode-mixing contamination from the higher l at the same m,
        of which the (3,2,0) handled by add_32_mode is the leading piece; this
        removes that whole family.  l_max = 6 with three overtones each way
        removes 5 * 3 * 2 = 30 modes.

    Returns
    -------
    kuibit TimeSeries
        The filtered waveform.

    Notes
    -----
    The waveform is zero-padded by PAD_DAMPING_TIMES fundamental damping times
    on each side (at least its own length), so the filter's response decays
    inside the padding rather than wrapping around.  No taper is applied.
    """
    if not waveform.is_regularly_sampled():
        raise ValueError("apply_filter needs a regularly sampled waveform; "
                         "resample it first, e.g. fixed_timestep_resampled(0.1)")

    t = np.asarray(waveform.t)
    y = np.asarray(waveform.y)
    dt = t[1] - t[0]

    omega_fund = get_qnm_frequency(l, m, 0, chif)/mf
    tau_max = 1.0/abs(np.imag(omega_fund))
    pad_len = max(int(PAD_DAMPING_TIMES*tau_max/dt), len(y))

    y_padded = np.pad(y, (pad_len, pad_len), mode="constant")
    y_fft = np.fft.fft(y_padded, norm="ortho")
    freqs = np.fft.fftfreq(len(y_padded), d=dt)

    ells = [l] if l_max is None else list(range(l, l_max + 1))
    if add_32_mode and 3 in ells:
        raise ValueError("add_32_mode would remove the (3, 2, 0) mode a second "
                         f"time: l_max={l_max} already covers l = 3")

    H_filter = np.ones(len(freqs), dtype=complex)
    for ell in ells:
        for n in range(num_prograde_qnms):
            H_filter = H_filter*get_prograde_filter(freqs, l=ell, m=m, n=n, mf=mf, chi=chif)
        for n in range(num_retrograde_qnms):
            H_filter = H_filter*get_retrograde_filter(freqs, l=ell, m=m, n=n, mf=mf, chi=chif)
    if add_32_mode:
        H_filter = H_filter*get_prograde_filter(freqs, l=3, m=2, n=0, mf=mf, chi=chif)

    y_filtered = np.fft.ifft(y_fft*H_filter, norm="ortho")[pad_len:-pad_len]
    return kuibit_ts(t, y_filtered)
