# 2021-4-13
# level2
# median filter 0 range 6000
import time
import datetime
from astropy.io import fits
from astropy.time import Time
from astropy.timeseries import LombScargle
import os
import numpy as np
import math
import julian
import pandas as pd


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares, leastsq
import scipy.stats as stats
from lmfit import Minimizer, Parameters, report_fit

from kb import kbreduct  # , binspec
from kbseism import estimate_background, find_peaks
from kbseism import delta_nu_acf, delta_nu_ps2, next_pow_two
from background.bgm import kbgauss, xyfitglog, kbparameters, param_value, harvey1


start = time.time()
npi = 1./np.sqrt(2.*np.pi)
bgm_path = 'test'
################################################
############# Kepler ###########################
################################################
path = '/media/kim/datafold/data/virgo_etc/activites/'
rf = pd.read_csv(path+'solflux_monthly_average.txt', skiprows=2, delimiter=',',
                 names=['y', 'm', 'obf', 'adf', 'abf'])
rf_year = pd.DataFrame({'year': rf.y})
rf.y = rf.y + round((rf.m - 0.5)/12, 3)
rf = pd.concat([rf, rf_year], axis=1)

path0 = '/media/kim/datafold/data/virgo/level2/'
file_list = os.listdir(path0)
EXT = '_dat'
filen = [file for file in file_list if file.endswith(EXT)]
filen.sort()
nam = ['y', 'm', 'd', 'day', 'flx']

sphp = pd.read_csv('sph.csv', delimiter=',')
t_sph = np.array(sphp.tsph)
td_sph = np.array(sphp.tdsph)
sph = np.array(sphp.sph)

df = pd.read_csv(path0+filen[0], delimiter=',',
                 names=nam)  # , na_values=[9999.])
df.flx = df.flx * 10
# df.flx = kbreduct.kbnan(df.flx)
# df.day = kbreduct.kbdisx(df.day, dic=1./24./60.)
# df.flx = kbreduct.detrend(df.day, df.flx, 10, 3)
# 1998.6M ~ 10M, 12M ~ 1999.2M
day2 = pd.DataFrame({'day2': df.y+(df.m-1.)/12 + df.d/365.})

df = pd.concat([df, day2], axis=1)
# df = df[0:495360]
# df = df[0:1020959]

sunrot = 25.
amp = []
amps = []
numax = []
numaxs = []
width = []
widths = []
whn = []
whns = []
tsei = []
dsei = []
sphq = []
t_sphq = []
td_sphq = []
dfy = df.groupby('y')
uHz_conv = 1e-6 * 24. * 60. * 60.

tmax23 = np.array(df.day)
fmax23 = np.array(df.flx)
dmax23 = np.array(df.day2)
rf = rf.drop([285, 286, 287])
rf0 = np.array(rf.adf)
rft = np.array(rf.y)
rfy = np.array(rf.m)
rfm = []
rfmy = []
rfmt = []
lcss = []
lcee = []
for m in range(0, len(rf0), 6):
    rfm = np.append(rfm, np.mean(rf0[0+m:3+m]))
    rfmy = np.append(rfmy, np.mean(rfy[0+m:3+m]))
    rfmt = np.append(rfmt, np.mean(rft[0+m:3+m]))

rfp = pd.DataFrame({'rfmt': rfmt, 'rfmy': rfmy, 'rfm': rfm})
#rfp.to_csv('results6m/rfm6m_half.csv', float_format='%.4f', index=False)
yy = 0
fname = 'figures/figure1h.pdf'
with PdfPages(fname) as pdf:

    lcss = 1996
    lc_s = 1996.0
    lc_e = 1996.5
    while lc_e <= 1997.1:
        lcss = np.append(lcss, lc_s)
        lcee = np.append(lcee, lc_e)
        print('Seqeunce : ', yy)
        yy += 1
        whe_sph = np.where((td_sph >= lc_s) & (td_sph <= lc_e))
        t_sphq = np.append(t_sphq, np.mean(t_sph[whe_sph]))
        td_sphq = np.append(td_sphq, np.mean(td_sph[whe_sph]))
        sphq = np.append(sphq, np.mean(sph[whe_sph]))

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax = ax.ravel()

        whe = np.where((dmax23 >= lc_s) & (dmax23 <= lc_e))
        t1 = tmax23[whe]
        f1 = fmax23[whe]
        d1 = dmax23[whe]
        f1 = kbreduct.detrend(t1, f1, 10, 3)
        f1 = kbreduct.kboutlier(f1, 3)
        # ax[0].plot(d1, f1, c='k', label='original')
        # ax[0].set_xlabel('Date')
        # ax[0].set_ylabel('flux')
        # ax[0].legend()

        tsei = np.append(tsei, np.mean(t1))
        dsei = np.append(dsei, np.mean(d1))
        num = len(t1)
        freq = np.linspace(1, 8000, 100000)
        freq0 = freq * uHz_conv
        model = LombScargle(t1, f1)
        power = model.power(freq0, method='fast', normalization='psd')
        power *= uHz_conv / num

        bg = estimate_background(freq, power, width=0.01)
        power_bg = power/bg
        wheb = np.where((freq > 5530) & (freq < 5580))
        nb = len(wheb[0])
        power_bgg = power_bg[min(wheb[0])-nb:min(wheb[0])]
        power_bg[wheb] = power_bgg
        power1 = power_bg * bg
        # ax[1].plot(freq, power_f0, c='black')
        # ax[1].plot(freq, bg, c='grey', label='background', linewidth=3)
        # ax[1].set_xlabel(r"Frequency [$\mu$Hz]")
        # ax[1].set_ylabel(r"Power $(ppm^2 / \mu Hz)$")
        # ax[1].set_yscale('log')
        # ax[1].set_xlim(1, 7000)
        # ax[1].set_ylim(5e-4, 2)
        # ax[1].legend()

        bin0 = freq[10]-freq[9]
        wid = int(10/bin0)
        power_f0 = kbreduct.smoothing(power1, wid)
        bg = estimate_background(freq, power_f0, width=0.01)

        power_f = power_f0 / bg

        whef = np.where((freq > 1000.) & (freq < 6000.))
        freq_l = freq[whef]
        power_l = power_f[whef]
        bg_l = bg[whef]
        pg0 = [50,  3100, 200, 1.]
        pg = kbparameters(pg0)

        minner = Minimizer(kbgauss, pg, fcn_args=(freq_l, power_l))
        bg_res = minner.minimize()
        report_fit(bg_res)
        res_g = kbgauss(bg_res.params.valuesdict(),
                        freq_l, power_l, value=1)
        resg = param_value(bg_res)
        ax.plot(freq, power_f0, c='black', linewidth=0.5)
        # ax.plot(freq, power_f, c='black')
        ax.set_yscale('log')
        ax.set_ylim(2e-3, 2)
        ax.set_xlim(1000, 5500)
        ax.set_xticks(np.arange(1000, 5500+1, 1000))
        # ax[1].set_title('[median - Smoothed Power Spectrum]/[Background] - '+str(name))
        ax.set_xlabel(r"Frequency ($\mu$Hz)", size=20)
        # [ppm$^2$/$\mu$Hz]")
        ax.set_ylabel(r"Power (ppm$^2 / \mu$Hz)", size=20)
        ax.plot(freq_l, res_g.fv*bg_l, c="k", linewidth=2)
        ax.tick_params(axis='both', direction='in',
                       top=True, right=True, labelsize=20)
        ax.tick_params(axis='both', which='minor', direction='in',
                       top=True, right=True)
        # ax.plot(freq_l, res_g.pex, ':', c="grey",
        #         label='p-mode excess', linewidth=2)
        # ax.plot(freq_l, res_g.whn, '--', c="grey",
        #         label='white noise', linewidth=2)
        pdf.savefig()
        plt.close()

        # ax[1].plot(freq_l, power_l, c='black')
        # ax[2].plot(freq, power_f, c='black')
        # # pt.yscale('log')
        # ax[2].set_ylim(-1, 25)
        # ax[2].set_xlim(1, 7000)
        # # ax[1].set_title('[median - Smoothed Power Spectrum]/[Background] - '+str(name))
        # ax[2].set_xlabel(r"Frequency [$\mu$Hz]")
        # ax[2].set_ylabel(r"Power")  # [ppm$^2$/$\mu$Hz]")
        # ax[2].plot(freq_l, res_g.fv, "grey",
        #            label='total model', linewidth=3)
        # ax[2].plot(freq_l, res_g.pex, ':', c="grey",
        #            label='p-mode excess', linewidth=2)
        # ax[2].plot(freq_l, res_g.whn, '--', c="grey",
        #            label='white noise', linewidth=2)
        # pdf.savefig()
        # plt.close()
        amp = np.append(amp, resg.p[0])
        numax = np.append(numax, resg.p[1])
        width = np.append(width, resg.p[2])
        amps = np.append(amps, resg.st[0])
        numaxs = np.append(numaxs, resg.st[1])
        widths = np.append(widths, resg.st[2])
        whn = np.append(whn, resg.p[3])
        whns = np.append(whns, resg.st[3])
        lc_s = lc_e
        lc_e += 0.5

results = pd.DataFrame(
    {'tsei': tsei, 'dsei': dsei, 'amp': amp, 'amps': amps, 'numax': numax, 'numaxs': numaxs, 'width': width, 'widths': widths, 'whn': whn, 'whns': whns, 'sphq': sphq, 'tsphq': t_sphq, 'tdsphq': td_sphq})
# results.to_csv('results6m/results6m_half.csv',
#                float_format='%.5f', index=False)

end = time.time()
print('\033[01;31;43m', format(end-start, ".3f"), 's\033[00m')
