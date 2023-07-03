import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt


def estimate_background(x, y, **kwargs):  # log_width = 0.01 default
    width = kwargs.get('width', 0.01)
    log_width = width
    count = np.zeros(len(x), dtype=int)
    bkg = np.zeros_like(x)
    x0 = np.log10(x[0])
    n = len(x)
    while x0 < np.log10(x[n-1]):
        m = np.abs(np.log10(x) - x0) < log_width
        bkg[m] += np.median(y[m])
        count[m] += 1
        x0 += 0.5 * log_width
    result = bkg/count
    return result


def estimate_background2(x, y, **kwargs):  # width = 100 default
    width = kwargs.get('width', 100)
    log_width = width
    count = np.zeros(len(x), dtype=int)
    bkg = np.zeros_like(x)
    x0 = x[0]
    n = len(x)
    while x0 < x[n-1]:
        m = np.abs(x - x0) < log_width
        bkg[m] += np.median(y[m])
        count[m] += 1
        x0 += 0.5 * log_width
        result = bkg/count
    return result


def find_peaks(z, **kwargs):
    # compare both sides, more bigger values are alive.
    # make arrange = np.arange(start, stop, step) // default step = 1
    # extract order of arrange = np.argsort
    value = kwargs.get('value', 'max')
    if value == 'max':
        #        print('\n', 'Finding maximum peaks')
        peak_inds = (z[1:-1] > z[:-2]) * (z[1:-1] > z[2:])
    if value == 'min':
        #        print('\n', 'Finding minimum peaks')
        peak_inds = (z[1:-1] < z[:-2]) * (z[1:-1] < z[2:])

    peak_inds = np.arange(1, len(z)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(z[peak_inds])][::-1]
    print('Complete : ' + value + ' points')
    return peak_inds


def nu_max(x, y, **kwargs):
    value = kwargs.get('value', 0)
    width = kwargs.get('width', 0.01)
    smooth = kwargs.get('smooth', 10)
    bkg = estimate_background(x, y, width=width)
    df = x[20] - x[19]
    #    i = 2
    # while df < 0.01:
    #     df = x[i]-x[i-1]
    #     i += 1
    smoothed_ps = gaussian_filter(y / bkg, smooth / df)
 #   print(df, smooth, smooth/df)
    peak_freqs = np.float64(x[find_peaks(smoothed_ps)])
 #   print(x[find_peaks(smoothed_ps)])
    numax = peak_freqs[peak_freqs > 5][0]
 #   print(peak_freqs)
    print('nu_max : ', format(numax, '.3f'))
    if value == 1:
        return numax, smoothed_ps
    if value == 0:
        return numax


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def acor_function(x):
    x = np.atleast_1d(x)
    n = next_pow_two(len(x))
    # print(n)
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]

    return acf


def delta_nu_acf(x, y, **kwargs):
    # delta_nu using autocorrelation
    value = kwargs.get('value', 0)
    width = kwargs.get('width', 0.01)
    smooth = kwargs.get('smooth', 0.2)
    numax = kwargs.get('numax', 0)
    numax = np.float(numax)
    if numax == 0:
        numax = nu_max(x, y, width=width)

    bkg = estimate_background(x, y, width=width)
    df = x[20] - x[19]
#    print(df)
    # And the autocorrelation function of a lightly smoothed power spectrum
    acor = acor_function(gaussian_filter(y / bkg, smooth / df))
    lags = df*np.arange(len(acor))
    data0 = pd.DataFrame({'acor': acor, 'lags': lags})
    # Expected delta_nu: stello et al (2009)
    dnu_expected = 0.263 * numax ** 0.773
    print('Expected Dnu : ', format(dnu_expected, ".3f"))
    acor = acor[lags < dnu_expected*1.2]
    lags = lags[lags < dnu_expected*1.2]
    acor = acor[lags > dnu_expected*0.8]
    lags = lags[lags > dnu_expected*0.8]

    peak_lags = (lags[find_peaks(acor)])
    # print(find_peaks(acor), peak_lags)
    peak_acor = (acor[find_peaks(acor)])
    # print(peak_acor, np.where(peak_acor == max(peak_acor)))
    # if len(peak_acor) == 1:
    #     del00 = peak_lags[0]
    # else:

    del00 = peak_lags[np.where(peak_acor == max(peak_acor))]
    # print(del00)
    # # print(peak_lags)
    # if len(lags[find_peaks(acor)]) == 1:
    #     deltanu = peak_lags[0]
    # else:
    deltanu = del00[0]
    # deltanu = peak_lags[np.argmin(np.abs(peak_lags - dnu_expected))]
    print('Delta nu : ', format(deltanu, '.3f'))
    data = pd.DataFrame({'lags': lags, 'acor': acor})
    if value == 0:
        return deltanu
    if value == 1:
        return deltanu, data0, data, bkg, peak_lags, peak_acor


def delta_nu_ps2(x, y, **kwargs):
    width = kwargs.get('width', 0.01)
    fwhm = kwargs.get('fwhm', 0)
    numax = kwargs.get('numax', 0)
    numax = np.float(numax)
    value = kwargs.get('value', 0)
    fwhm /= 2
    if numax == 0:
        numax = nu_max(x, y, width=width)
    if fwhm == 0:
        fwhm = 0.0098*numax**1.47/4/2

    est_del = 0.263*numax**(0.773)
    print('Expected Dnu : ', format(est_del, ".3f"))

    # range PSPS
    w0 = np.where((x >= numax - fwhm*3) & (x <= numax + fwhm*3))
    x_ps = x[w0]
    y_ps = y[w0]
    # plt.plot(x_ps, y_ps)
    # plt.draw()
    # plt.pause(0.001)
    # PSPS
    f_ps, p_ps = LombScargle(
        x_ps, y_ps, normalization='psd').autopower(nyquist_factor=1)
    t_ps = 1./f_ps  # time to freq

    wh = np.where((t_ps < est_del/2*1.3) & (t_ps > est_del/2*0.7))
    t_ps1 = t_ps[wh]
    p_ps1 = p_ps[wh]

    z0 = find_peaks(p_ps1)
    tz0 = t_ps1[z0]
    pz0 = p_ps1[z0]
    # deltanu = t_ps[z0[np.argmin(abs(t_ps[z0]-est_del/2))]]*2

    deltanu = np.float(tz0[np.where(pz0 == max(pz0))]*2)

    # deltanu = t_ps[p_ps[z0].index(max(p_ps[z0]))]*2
    # print(deltanu)
    # abs(t_ps[z0]-est_del/2)

    # print(deltanu, est_del/2)
    z1 = find_peaks(p_ps, value='min')

    e = t_ps[z1]-deltanu/2

    a = e > 0

    e1 = min(e[a])
    a = e < 0
    e2 = max(e[a])
    err = e1-e2

    del_nu = np.array([deltanu, err])

    if value == 1:
        print('Delta nu : ', format(deltanu, '.3f'))
        va = pd.DataFrame({'time': t_ps1, 'power': p_ps1})
        return va, z0

    print('Delta nu : ', format(deltanu, '.3f'))
    return del_nu
