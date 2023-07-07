from background.bgm import kbgauss, xyfitmodel, kbparameters, param_value, harvey3_4, bgm2
from kbseism import delta_nu_acf, delta_nu_ps2, next_pow_two
from kbseism import estimate_background, find_peaks
from kb import kbreduct  # , binspec
from lmfit import Minimizer, Parameters, report_fit, Model
from scipy import signal
import scipy.stats as stats
from scipy.optimize import least_squares, leastsq
from scipy.ndimage.filters import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import julian
import math
import numpy as np
import os
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.io import fits
import datetime
import time


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = "12"
plt.rcParams["axes.xmargin"] = "0.1"
plt.rcParams["axes.ymargin"] = "0.1"

start = time.time()
npi = 1./np.sqrt(2.*np.pi)
uhz_conv = 1e-6 * 24. * 60. * 60.

pth = '/media/kim/datafold/data/CoRoT/HD49933/'
file_list = os.listdir(pth)
file_list.sort()
hdu = fits.open(pth+file_list[0])
amp = []
amps = []
numax = []
width = []
fshift = []
fshift1 = []
fshift2 = []
tsei = []
sph = []
sphq = []
t_sph = []
t_sphq = []
hdu = fits.open(pth+file_list[0])
t0 = hdu[3].data.DATEBARREGTT
f0 = hdu[3].data.FLUXBARREG
t_min = min(t0)
c = 0
ps = pd.DataFrame()
ps1 = pd.DataFrame()
ps2 = pd.DataFrame()
df = pd.DataFrame()
tt = []
ff = []
for f in range(0, 3):
    hdu = fits.open(pth+file_list[0])
    t0 = hdu[3].data.DATEBARREGTT
    f0 = hdu[3].data.FLUXBARREG

    hdu = fits.open(pth+file_list[1])
    t1 = hdu[3].data.DATEBARREGTT
    f1 = hdu[3].data.FLUXBARREG

    hdu = fits.open(pth+file_list[2])
    t2 = hdu[3].data.DATEBARREGTT
    f2 = hdu[3].data.FLUXBARREG

    dt = 2400000
    dte = np.max(t1)-np.min(t1)+6
    # dts = str(dt)
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.35)
    ax[0].plot(t0, f0/1e6, 'k')
    ax[0].set_xlabel('Time (JD-2400000)', size=15)
    ax[0].set_ylabel('Flux ($10^6$ $e^{-}/s$)', size=15)
    ax[0].set_xticks(np.arange(54140, 54130+dte, 40))
    ax[0].set_yticks(np.arange(9.17, 9.23, 0.02))
    ax[0].set_xlim(np.min(t0)-6, np.min(t0)+dte)
    ax[0].set_ylim(9.16, 9.24)
    ax[0].tick_params(axis='both', direction='in',
                      top=True, right=True, labelsize=15)

    ax[1].plot(t1, f1/1e6, 'k')
    ax[1].set_xlabel('Time (JD-2400000)', size=15)
    ax[1].set_ylabel('Flux ($10^6$ $e^{-}/s$)', size=15)
    ax[1].set_xlim(np.min(t1)-6, np.min(t1)+dte)
    ax[1].set_ylim(9.26, 9.355)
    ax[1].set_xticks(np.arange(54400, 54560, 40))
    ax[1].set_yticks(np.arange(9.27, 9.36, 0.03))
    ax[1].tick_params(axis='both', direction='in',
                      top=True, right=True, labelsize=15)
    ax[2].plot(t2, f2/1e6, 'k')
    ax[2].set_xlabel('Time (JD-2400000)', size=15)
    # ax[2].set_xlabel('Time (JD \N{MINUS SIGN} 2,400,000)', size=15)
    ax[2].set_ylabel('Flux ($10^6$ $e^{-}/s$)', size=15)
    ax[2].set_yticks(np.arange(8.53, 8.63, 0.03))
    ax[2].set_xticks(np.arange(55945, 55933+dte, 40))
    ax[2].set_xlim(np.min(t2)-6, np.min(t2)+dte)
    ax[2].tick_params(axis='both', direction='in',
                      top=True, right=True, labelsize=15)
    c += 1
plt.savefig("figure1.pdf")
plt.close()
end = time.time()
print('\033[01;31;43m', format(end-start, ".3f"), 's\033[00m')
