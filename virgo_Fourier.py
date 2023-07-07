# 2021-08-25
# Level2 analysis
# version 4 free
# + time series 3
# + log scale 4
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
from scipy.interpolate import interp1d, splrep, splev

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import least_squares, leastsq
import scipy.stats as stats
from lmfit import Minimizer, Parameters, report_fit, Model

from kb import kbreduct
from kbbgm import kfunc, kbgm
from kbseism import estimate_background, find_peaks
from kbseism import delta_nu_acf, delta_nu_ps2, acor_function
from background.bgm import xyfit, xyfitp, kbparameters, param_value, xyfitmodel
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = "12"
plt.rcParams["axes.xmargin"] = "0.1"
plt.rcParams["axes.ymargin"] = "0.1"
csfont = {'fontname': 'Times New Roman'}
start = time.time()
npi = 1./np.sqrt(2.*np.pi)
#plt.rcParams["font.family"] = "Times New Roman"

###############################################
##################  Read data  ################
###############################################
path0 = '../bg0_rev/results/'
filen = os.listdir(path0)
filen.sort()
for fi in range(len(filen)):
    dv0 = pd.read_csv(path0+filen[fi])
dv0.fwhm = dv0.fwhm * 2 * np.sqrt(2*np.log(2))
dv0.fwhms = dv0.fwhms * 2 * np.sqrt(2*np.log(2))


path1 = '../li0_again/results/'
filen = os.listdir(path1)
filen.sort()
dvv = pd.DataFrame()
for fi in range(len(filen)):
    dv1 = pd.read_csv(path1+filen[fi])
#     dvv = pd.concat([dvv, dv1], ignore_index=True)
# dv1 = dvv
dv1 = dv1[0:19]
dv1.fwhm = dv1.fwhm * 2 * np.sqrt(2*np.log(2))
dv1.fwhms = dv1.fwhms * 2 * np.sqrt(2*np.log(2))

path = '/media/kim/datafold/data/virgo_etc/activites/'
rf = pd.read_csv(path+'solflux_monthly_average.txt', skiprows=2, delimiter=',',
                 names=['y', 'm', 'obf', 'adf', 'abf'])
rf = rf[:228]
rf_year = pd.DataFrame({'year': rf.y})
rf.y = rf.y + round((rf.m - 0.5)/12, 3)
rf = pd.concat([rf, rf_year], axis=1)
rfy = rf.groupby(['year']).mean()
mrf = rf.rolling(window=13, min_periods=1,
                 win_type='gaussian', center=True).mean(std=10)
rffm = mrf.adf-min(mrf.adf)
rffm /= max(rffm)

rfym = rfy.adf-min(rfy.adf)
rfym /= max(rfym)
###################################################
rfy.y = rfy.y - 0.5
amp0 = dv0.amp-min(dv0.amp)
ampm = max(amp0)
amp0 /= max(amp0)
amps0 = dv0.amps/ampm


fwhm0 = dv0.fwhm-min(dv0.fwhm)
fwhmm = max(fwhm0)
fwhm0 /= max(fwhm0)
fwhms0 = dv0.fwhms/fwhmm


numax0 = dv0.numax-min(dv0.numax)
numax0 /= max(numax0)

slope0 = dv1.p1-min(dv1.p1)
slope0 /= max(slope0)


l0 = 'median filter > divide'

dv = [dv0]
amp = [amp0]
la = [l0]

lw0 = 4
lw1 = 1
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax[0].scatter(rfy.adf, dv0.amp, c='black', s=s)
ax.errorbar(rfy.adf, dv0.amp, yerr=dv0.amps, c='black',
            fmt='o', capsize=2, ms=lw0)


ax.set_ylabel(r'amplitude (arbitrary units)')  # ($ppm^2$)')

# ax[0].set_errorbar(dv.amps, fmt='bo-', elinewidth=0.5)

mini = [-10000, -10000]
p0 = [5, -0.005]
pg = kbparameters(p0, mini=mini)
minner = Minimizer(xyfit, pg, fcn_args=(rfy.adf, dv0.amp))
bg_res = minner.minimize()
# report_fit(bg_res)
res_a = xyfit(bg_res.params.valuesdict(),
              rfy.adf, dv0.amp, value=1)
resa = param_value(bg_res)

pg = kbparameters(p0, mini=mini)
minner = Minimizer(xyfit, pg, fcn_args=(rfy.adf, dv0.fwhm))
bg_res = minner.minimize()
# report_fit(bg_res)
res_e = xyfit(bg_res.params.valuesdict(),
              rfy.adf, dv0.fwhm, value=1)
rese = param_value(bg_res)

pg = kbparameters(p0, mini=mini)
minner = Minimizer(xyfit, pg, fcn_args=(rfy.adf, dv0.numax))
bg_res = minner.minimize()
# report_fit(bg_res)
res_n = xyfit(bg_res.params.valuesdict(),
              rfy.adf, dv0.numax, value=1)
resn = param_value(bg_res)

# ax.plot(rfy.adf, res_a.fv, c='black', linewidth=lw1)
# ax.set_ylabel(r'amplitude (arbitrary units)')
# ax.set_xlabel('F10.7 ($sfu$)')

# # ($ppm^2$)')
# plt.savefig("figure2_dummy.pdf")
# plt.close()

fs = 15
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# ax.errorbar(rfy.y, amp0, c='k', marker='o')  # marker='o', ms=5
ax.errorbar(rfy.y, dv0.amp, yerr=dv0.amps, fmt='-o', c='black',
            capsize=2, ms=lw0)
ax.set_ylabel(r'$A$ (arbitrary units)', fontsize=fs)
ax.set_xlabel('Date (years)', fontsize=fs)
ax.set_xlim(1995, 2015)
ax.tick_params(axis='x', labelsize=fs, direction='in')
ax.tick_params(axis='y', labelsize=fs, direction='in')
# ax.set_yticklabels(['', '1.4', '', '1.6', '', '1.8', '', '2.0', ''])
ax.set_xticks(np.arange(1997, 2015, 4))
ax.set_yticks(np.arange(1.7, 2.4, 0.2))
axa = ax.twinx()
axa.plot(rfy.y, rfy.adf, c='k', ls='--')
axa.set_ylabel('F10.7 (sfu)', fontsize=fs)
axa.tick_params(axis='y', labelsize=fs, direction='in')
# axa.set_yticklabels(['', '', '100', '', '140', '', '180'])
axa.set_yticks(np.arange(80, 180, 30))
# axa.set_xticks(np.arange(1996, 2015, 3))
plt.savefig("figure2.pdf", dpi=1200)
plt.close()


# fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# ax.errorbar(rfy.adf, dv0.fwhm, yerr=dv0.fwhms, c='black',
#             fmt='-o', capsize=2, ms=lw0)
# ax.plot(rfy.adf, res_e.fv, c='black', linewidth=lw1)
# ax.set_ylabel(r'$\delta \nu_{env}$ $(\mu Hz$)')
# ax.set_xlabel('F10.7 ($sfu$)')
# plt.savefig("figure3_dummy.pdf")
# plt.close()

fs = 15
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# ax.errorbar(rfy.y, amp0, c='k', marker='o')  # marker='o', ms=5
ax.errorbar(rfy.y, dv0.fwhm, yerr=dv0.fwhms, fmt='-o', c='black',
            capsize=2, ms=lw0)
ax.set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', fontsize=fs)
ax.set_xlabel('Date (years)', fontsize=fs)
ax.set_xlim(1995, 2015)
ax.tick_params(axis='x', labelsize=fs, direction='in')
ax.tick_params(axis='y', labelsize=fs, direction='in')
# ax.set_yticklabels(['', '700', '', '750', '', '800', '', '850', ''])
ax.set_xticks(np.arange(1997, 2015, 4))
ax.set_yticks(np.arange(700, 910, 50))
ax.set_ylim(670, 910)
axa = ax.twinx()
axa.plot(rfy.y, rfy.adf, c='k', ls='--')
axa.set_ylabel('F10.7 (sfu)', fontsize=fs)
axa.tick_params(axis='y', labelsize=fs, direction='in')
axa.set_yticks(np.arange(80, 200, 30))
# axa.set_yticklabels(['', '', '100', '', '140', '', '180'])
plt.savefig("figure4.pdf", dpi=1200)
plt.close()


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_ylabel(r'$\nu_{\rm max}$ $(\mu Hz$)')
ax.set_xlabel('F10.7 ($sfu$)')

ax.errorbar(rfy.adf, dv0.numax, yerr=dv0.numaxs, c='black',
            fmt='o', capsize=2, ms=lw0)
ax.plot(rfy.adf, res_n.fv, c='black', linewidth=lw1)
ax.set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu Hz$)')
ax.set_ylabel(r'$\nu_{\rm max}$ $(\mu Hz$)')
ax.set_xlabel('F10.7 ($sfu$)')
plt.savefig("figure2_3.pdf")
plt.close()

s = 20

# fig.colorbar(sc0, ax=ax[0])
# fig.colorbar(sc1, ax=ax[2])
# fig.colorbar(sc2, ax=ax[1])
p0 = [0, 0]
pg = kbparameters(p0, mini=mini)
minner = Minimizer(xyfit, pg, fcn_args=(
    dv0.fwhm, dv0.numax, dv0.numaxs))
bg_res = minner.minimize()
# report_fit(bg_res)
res_w = xyfit(bg_res.params.valuesdict(),
              dv0.fwhm, dv0.numax, value=1)
resw = param_value(bg_res)

p0 = [0, 0]
pg = kbparameters(p0, mini=mini)
minner = Minimizer(xyfit, pg, fcn_args=(dv0.amp, dv0.numax, dv0.numaxs))
bg_res = minner.minimize()
# report_fit(bg_res)
res_a = xyfit(bg_res.params.valuesdict(),
              dv0.amp, dv0.numax, value=1)
resa = param_value(bg_res)

# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax[0].errorbar(dv0.fwhm, dv0.numax, xerr=dv0.fwhms, yerr=dv0.numaxs, fmt='o', c='black',
#                capsize=2, ms=lw0)
# ax[0].plot(dv0.fwhm, res_w.fv, color='k')
# ax[0].set_xlabel(r'$\delta \nu_{env}$ $[\mu Hz$]', fontsize=fs)
# ax[0].set_ylabel(r'$\nu_{max}$ $[\mu Hz]$', fontsize=fs)
# ax[0].tick_params(axis='x', labelsize=fs, direction='in')
# ax[0].tick_params(axis='y', labelsize=fs, direction='in')

# ax[1].errorbar(dv0.amp, dv0.numax, xerr=dv0.amps, yerr=dv0.numaxs, fmt='o', c='black',
#                capsize=2, ms=lw0)
# ax[1].plot(dv0.amp, res_a.fv, color='k')
# ax[1].set_xlabel(r'Amplitude [arbitrary units]', fontsize=fs)
# ax[1].set_ylabel(r'$\nu_{max}$ $[\mu Hz]$', fontsize=fs)
# ax[1].tick_params(axis='x', labelsize=fs, direction='in')
# ax[1].tick_params(axis='y', labelsize=fs, direction='in')
# plt.savefig("figure4_dummy.pdf")
# plt.close()


fmod = Model(xyfitmodel)
p0 = [3100, 0.1]
# sig = np.arange(len(dv0.numax))
lmresutf = fmod.fit(dv0.fwhm, x=dv0.amp,
                    a=p0[0], b=p0[1], weights=1.0/(dv0.fwhm))
# print(lmresut.fit_report())
modelf = param_value(lmresutf)
amod = Model(xyfitmodel)
p0 = [3100, 0.1]
# sig = np.arange(len(dv0.numax))
lmresuta = amod.fit(dv0.numax, x=dv0.amp,
                    a=p0[0], b=p0[1], weights=1.0/(dv0.numaxs))
# print(lmresut.fit_report())
modela = param_value(lmresuta)

#dvd = dv0.drop([7])

dmod = Model(xyfitmodel)
p0 = [3100, 0.1]
lmresutd = dmod.fit(dv0.numax, x=dv0.amp,
                    a=p0[0], b=p0[1], weights=1.0/(dv0.numaxs))
modeld = param_value(lmresutd)

xx = np.linspace(min(dv0.amp)-0.05, max(dv0.amp)+0.05, 20)
yy = modelf.p[0] + modelf.p[1]*xx

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].errorbar(dv0.amp, dv0.fwhm, xerr=dv0.amps, yerr=dv0.fwhms, fmt='o', c='black',
               capsize=2, ms=lw0)
# ax[0].plot(dv0.fwhm, res_w.fv, color='k')
# ax[0].plot(dv0.fwhm, lmresutf.best_fit, color='k')
ax[0].plot(xx, yy, color='k')
ax[0].set_xlabel(r'$A$ (arbitrary units)', fontsize=fs)
ax[0].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', fontsize=fs)
ax[0].tick_params(axis='x', labelsize=fs, direction='in')
ax[0].tick_params(axis='y', labelsize=fs, direction='in')
ax[0].set_yticks(np.arange(650, 950, 50))
ax[0].set_xticks(np.arange(1.6, 2.5, 0.2))
ax[0].set_xlim(1.55, 2.35)
ax[0].set_ylim(670, 905)
xx = np.linspace(min(dv0.amp)-0.05, max(dv0.amp)+0.05, 20)
yy = modeld.p[0] + modeld.p[1]*xx

ax[1].errorbar(dv0.amp, dv0.numax, xerr=dv0.amps, yerr=dv0.numaxs, fmt='o', c='black',
               capsize=2, ms=lw0)
# ax[1].plot(dv0.amp, res_a.fv, color='k')
# ax[1].plot(dv0.amp, lmresuta.best_fit, color='k')
# ax[1].plot(dvd.amp, lmresutd.best_fit, '--', color='k', dashes=[5, 10])
ax[1].plot(xx, yy, color='k')
ax[1].set_xlabel(r'$A$ (arbitrary units)', fontsize=fs)
ax[1].set_ylabel(r'$\nu_{\rm max}$ $(\mu$Hz)', fontsize=fs)
ax[1].tick_params(axis='x', labelsize=fs, direction='in')
ax[1].tick_params(axis='y', labelsize=fs, direction='in')
ax[1].set_yticks(np.arange(3140, 3220, 20))
ax[1].set_xticks(np.arange(1.6, 2.5, 0.2))
ax[1].set_ylim(3130, 3210)
ax[1].set_xlim(1.55, 2.35)


plt.savefig("figure5.pdf", dpi=300)
plt.close()


fmod = Model(xyfitmodel)
p0 = [3100, 0.1]
# sig = np.arange(len(dv0.numax))
lmresutf = fmod.fit(dv0.numax, x=dv0.fwhm,
                    a=p0[0], b=p0[1], weights=1.0/(dv0.numaxs))
# print(lmresut.fit_report())
modelf = param_value(lmresutf)
amod = Model(xyfitmodel)
p0 = [3100, 0.1]
# sig = np.arange(len(dv0.numax))
lmresuta = amod.fit(dv0.numax, x=dv0.amp,
                    a=p0[0], b=p0[1], weights=1.0/(dv0.numaxs))
# print(lmresut.fit_report())
modela = param_value(lmresuta)

# dvd = dv0.drop([7])

# dmod = Model(xyfitmodel)
# p0 = [3100, 0.1]
# lmresutd = dmod.fit(dvd.numax, x=dvd.amp,
#                     a=p0[0], b=p0[1], weights=1.0/(dvd.numaxs))
# modeld = param_value(lmresutd)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))

xx = np.linspace(min(dv0.amp)-0.05, max(dv0.amp)+0.05, 20)
yy = modela.p[0] + modela.p[1]*xx

ax[0].errorbar(dv0.amp, dv0.numax, xerr=dv0.amps, yerr=dv0.numaxs, fmt='o', c='black',
               capsize=2, ms=lw0)
# ax[1].plot(dv0.amp, res_a.fv, color='k')
ax[0].plot(xx, yy, color='k')
# ax[1].plot(dvd.amp, lmresutd.best_fit, '--', color='k', dashes=[5, 10])
# ax[1].plot(xx, yy, '--', color='k')
ax[0].set_xlabel(r'$A$ (arbitrary units)', fontsize=fs)
ax[0].set_ylabel(r'$\nu_{\rm max}$ $(\mu$Hz)', fontsize=fs)
ax[0].tick_params(axis='x', labelsize=fs, direction='in')
ax[0].tick_params(axis='y', labelsize=fs, direction='in')
ax[0].set_xticks(np.arange(1.6, 2.5, 0.2))
ax[0].set_xlim(1.55, 2.35)


xx = np.linspace(min(dv0.fwhm)-20, max(dv0.fwhm)+20, 20)
yy = modelf.p[0] + modelf.p[1]*xx
ax[1].errorbar(dv0.fwhm, dv0.numax, xerr=dv0.fwhms, yerr=dv0.numaxs, fmt='o', c='black',
               capsize=2, ms=lw0)
# ax[0].plot(dv0.fwhm, res_w.fv, color='k')
# ax[0].plot(dv0.fwhm, lmresutf.best_fit, color='k')
ax[1].plot(xx, yy, color='k')
ax[1].set_xlabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', fontsize=fs)
ax[1].set_ylabel(r'$\nu_{\rm max}$ $(\mu$Hz)', fontsize=fs)
ax[1].tick_params(axis='x', labelsize=fs, direction='in')
ax[1].tick_params(axis='y', labelsize=fs, direction='in')
ax[1].set_xticks(np.arange(650, 950, 50))
ax[1].set_xlim(670, 905)

plt.savefig("figure5v2.pdf", dpi=300)
plt.close()
# solar cycle23 : 1996 ~ 2008, solar cycle24 : 2009~

# dvd = dv0.drop([7])
# s23 = dv0[0:13]
# s24 = dv0[13:]

# fmod = Model(xyfitmodel)
# p0 = [3100, 0.1]
# lmresut23f = fmod.fit(s23.numax, x=s23.fwhm,
#                       a=p0[0], b=p0[1], weights=1.0/(s23.numaxs)**2)
# model23f = param_value(lmresut23f)
# lmresut24f = fmod.fit(s24.numax, x=s24.fwhm,
#                       a=p0[0], b=p0[1], weights=1.0/(s24.numaxs)**2)
# model24f = param_value(lmresut24f)

# amod = Model(xyfitmodel)
# p0 = [3100, 0.1]
# lmresut23a = amod.fit(s23.numax, x=s23.amp,
#                       a=p0[0], b=p0[1], weights=1.0/(s23.numaxs)**2)
# model23a = param_value(lmresut23a)

# lmresut24a = amod.fit(s24.numax, x=s24.amp,
#                       a=p0[0], b=p0[1], weights=1.0/(s24.numaxs)**2)
# model24a = param_value(lmresut24a)

# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax[0].errorbar(s23.fwhm, s23.numax, xerr=s23.fwhms, yerr=s23.numaxs, fmt='o', c='black',
#                capsize=2, ms=lw0)
# ax[0].errorbar(s24.fwhm, s24.numax, xerr=s24.fwhms, yerr=s24.numaxs, fmt='o', c='red',
#                capsize=2, ms=lw0)

# ax[0].plot(s23.fwhm, lmresut23f.best_fit, color='k')
# ax[0].plot(s24.fwhm, lmresut24f.best_fit, color='r')
# ax[0].set_xlabel(r'$\delta \nu_{env}$ $(\mu Hz$)', fontsize=fs)
# ax[0].set_ylabel(r'$\nu_{max}$ $(\mu Hz)$', fontsize=fs)
# ax[0].tick_params(axis='x', labelsize=fs, direction='in')
# ax[0].tick_params(axis='y', labelsize=fs, direction='in')

# ax[1].errorbar(s23.amp, s23.numax, xerr=s23.amps, yerr=s23.numaxs, fmt='o', c='black',
#                capsize=2, ms=lw0)
# ax[1].errorbar(s24.amp, s24.numax, xerr=s24.amps, yerr=s24.numaxs, fmt='o', c='r',
#                capsize=2, ms=lw0)
# ax[1].plot(s23.amp, lmresut23a.best_fit, color='k')
# ax[1].plot(s24.amp, lmresut24a.best_fit, color='r')

# ax[1].set_xlabel(r'Amplitude (arbitrary units)', fontsize=fs)
# ax[1].set_ylabel(r'$\nu_{max}$ $(\mu Hz)$', fontsize=fs)
# ax[1].tick_params(axis='x', labelsize=fs, direction='in')
# ax[1].tick_params(axis='y', labelsize=fs, direction='in')
# plt.savefig("figure5X.pdf", dpi=300)
# plt.close()

col = stats.spearmanr(dv1.y, dv1.p1)

fs = 15
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
# ax.errorbar(rfy.y, amp0, c='k', marker='o')  # marker='o', ms=5
ax.errorbar(dv1.y, dv1.p1, yerr=dv1.p1s, fmt='-o', c='black',
            capsize=2, ms=lw0)
ax.set_ylabel(r'$s$', fontsize=fs)
ax.set_xlabel('Date (years)', fontsize=fs)
ax.set_xlim(1995, 2015)
ax.set_ylim(-3.9e-4, -3.2e-4)
ax.tick_params(axis='x', labelsize=fs, direction='in')
ax.tick_params(axis='y', labelsize=fs, direction='in')
ax.set_yticks(np.arange(-0.00038, -0.00031, 0.00002))
# ax.set_yscale('log')
ax.set_yticklabels(['$-3.8$', '$-3.6$', '$-3.4$', '$-3.2$'])
ax.set_xticks(np.arange(1997, 2015, 4))
ax.yaxis.exponent = 4
axa = ax.twinx()
axa.plot(rfy.y, rfy.adf, c='k', ls='--')
axa.set_ylabel('F10.7 (sfu)', fontsize=fs)
axa.tick_params(axis='y', labelsize=fs, direction='in')
# axa.set_yticklabels(['', '', '100', '', '140', '', '180'])

axa.set_yticks(np.arange(80, 200, 30))
plt.savefig("figure6.pdf", dpi=1200)
plt.close()


# fig, ax = plt.subplots(5, 1, figsize=(10, 13))
# resid, fit = kbreduct.detrend(rfy.y, dv0.amp, 19, 10, fits=1)
# fft = np.fft.fft(resid) / len(resid)
# fft_magnitude = abs(fft)
# freq = np.fft.fftfreq(19, 1)
# acor = acor_function(resid)
# lag = np.arange(len(acor))
# freq0, power0 = LombScargle(lag, resid).autopower()
# ax[0].plot(rfy.y, dv0.amp, label='amp')
# ax[0].set_xlabel('time')
# ax[0].set_ylabel('ampitude')
# ax[0].plot(rfy.y, fit, label='fit')
# ax[1].plot(rfy.y, resid, '-o', label='residual')
# ax[1].set_xlabel('time')
# ax[1].set_ylabel('residual amplitude')
# ax[2].plot(freq, fft_magnitude, '-o', label='FFT')
# ax[2].set_xlabel('frequency')
# ax[2].set_ylabel('fft')
# ax[3].plot(lag, acor, '-o', label='Autocorrelation')
# ax[3].set_xlabel('lag')
# ax[3].set_ylabel('Autocorrelation')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# ax[4].plot(freq0, power0, '-o')
# plt.savefig('figureA_dmmy.pdf')
# plt.close()
# resid0 = resid
# fit0 = fit
# fig, ax = plt.subplots(3, 1, figsize=(10, 10))

# fft = np.fft.fft(dv0.amp) / len(dv0.amp)
# fft_magnitude = abs(fft)
# freq = np.fft.fftfreq(19, 1)
# acor = acor_function(dv0.amp)
# lag = np.arange(len(acor))

# ax[0].plot(rfy.y, dv0.amp, label='amp')
# ax[0].set_xlabel('time')
# ax[0].set_ylabel('ampitude')
# # ax[0].plot(rfy.y, fit, label='fit')
# ax[1].plot(freq, fft_magnitude, '-o', label='FFT')
# ax[1].set_xlabel('frequency')
# ax[1].set_ylabel('fft')
# ax[2].plot(lag, acor, '-o', label='Autocorrelation')
# ax[2].set_xlabel('lag')
# ax[2].set_ylabel('Autocorrelation')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# # ax[4].plot(freq0, power0, '-o')
# plt.savefig('figureA_2dummy.pdf')
# plt.close()


spl = splrep(rfy.y, dv0.amp, k=4)
i_y = np.linspace(min(rfy.y), max(rfy.y), 1800)
interp = splev(i_y, spl)


resid, fit = kbreduct.detrend(i_y, interp, 19, 5, fits=1)

inter_amp = []
for v in rfy.y:
    # print(v)
    spla = splrep(i_y, fit, k=4)
    aaa = splev(v, spla)
    inter_amp = np.append(inter_amp, aaa)

fft = np.fft.fft(resid) / len(resid)
fft_magnitude = abs(fft)
freq = np.fft.fftfreq(1900, 0.01)
acor = acor_function(resid)
lag = np.arange(len(acor))

freq0 = np.linspace(0.0001, 2, 6000)
model = LombScargle(i_y, resid)
power0 = model.power(freq0, method='fast', normalization='psd')
# power0 /= len(freq0)

freq00 = np.linspace(0.0001, 2, 6000)
model00 = LombScargle(i_y, interp)
power00 = model00.power(freq00, method='fast', normalization='psd')

freq000 = np.linspace(0.0001, 2, 6000)
model000 = LombScargle(rfy.y, dv0.amp)
power000 = model000.power(freq000, method='fast', normalization='psd')

# freq0, power0 = LombScargle(i_y, resid).autopower()
fig, ax = plt.subplots(3, 1, figsize=(8, 10))

ax[0].plot(rfy.y, dv0.amp, color='k', label='orgin')
ax[0].plot(i_y, fit, ':', color='k', label='fit')
ax[0].tick_params(axis='x', direction='in')
ax[0].tick_params(axis='y', direction='in')
ax[0].set_xlabel('Date (years)')
ax[0].set_ylabel('$A$ (arbitrary units)')
ax[0].set_xlim(1995, 2015)

# ax[1].plot(i_y, resid, ':', color='k', label='residual')
ax[1].plot(rfy.y, dv0.amp-inter_amp, color='k')
ax[1].set_xlabel('Date (years)')
ax[1].set_ylabel('Residual')
ax[1].tick_params(axis='x', direction='in')
ax[1].tick_params(axis='y', direction='in')
ax[1].set_xlim(1995, 2015)
# ax[1].set_yticklabels(['', '', '-0.1', '', '0.0', '', '0.1', ''])

ax[2].plot(freq0, power0, color='k', label='Lomb-Scargle')
ax[2].set_xlabel(r'Frequency (1/year)')
ax[2].set_ylabel('Power (ppm$^2$)')
ax[2].set_xlim(0, 0.7)
ax[2].set_ylim(0, 1.95)
ax[2].tick_params(axis='x', direction='in')
ax[2].tick_params(axis='y', direction='in')
# ax[2].set_yticklabels(['', '', '0.5', '', '1.0', '', '1.5', ''])

plt.savefig('figure3.pdf', dpi=1200)
plt.close()

fig, ax = plt.subplots(4, 1, figsize=(8, 14))

ax[0].plot(rfy.y, dv0.amp, color='k', label='orgin')
ax[0].plot(i_y, fit, ':', color='k', label='fit')
ax[0].tick_params(axis='x', direction='in')
ax[0].tick_params(axis='y', direction='in')
ax[0].set_xlabel('Date (years)')
ax[0].set_ylabel('$A$ (arbitrary units)')
ax[0].set_xticks(np.arange(1997, 2015, 4))
ax[0].set_yticks(np.arange(1.7, 2.4, 0.2))
ax[0].set_xlim(1995, 2015)
# ax[1].plot(i_y, resid, ':', color='k', label='residual')
ax[1].plot(rfy.y, dv0.amp-inter_amp, color='k')
ax[1].set_xlabel('Date (years)')
ax[1].set_ylabel('Residual')
ax[1].tick_params(axis='x', direction='in')
ax[1].tick_params(axis='y', direction='in')
ax[1].set_xlim(1995, 2015)
ax[1].set_xticks(np.arange(1997, 2015, 4))

# ax[1].set_yticklabels(['', '', '-0.1', '', '0.0', '', '0.1', ''])

ax[2].plot(freq0, power0, color='k', label='Lomb-Scargle')
ax[2].set_xlabel(r'Frequency (1/year)')
ax[2].set_ylabel('Power (ppm$^2$)')
ax[2].set_xlim(0, 0.7)
ax[2].set_ylim(0, 5)
ax[2].tick_params(axis='x', direction='in')
ax[2].tick_params(axis='y', direction='in')
# ax[2].set_yticklabels(['', '', '0.5', '', '1.0', '', '1.5', ''])

ax[3].plot(freq00, power00, color='k', label='Lomb-Scargle')
ax[3].set_xlabel(r'Frequency (1/year)')
ax[3].set_ylabel('Power (ppm$^2$)')
ax[3].set_xlim(0, 0.7)
ax[3].set_ylim(0, 5)
ax[3].tick_params(axis='x', direction='in')
ax[3].tick_params(axis='y', direction='in')

# ax[4].plot(freq000, power000, color='k', label='Lomb-Scargle')
# ax[4].set_xlabel(r'Frequency (1/year)')
# ax[4].set_ylabel('Power (ppm$^2$)')
# ax[4].set_xlim(0, 0.7)
# ax[4].set_ylim(0, 0.05)
# ax[4].tick_params(axis='x', direction='in')
# ax[4].tick_params(axis='y', direction='in')

plt.savefig('figure3_phd.pdf', dpi=1200)
plt.close()


# spl = splrep(rfy.y, rfy.adf, k=4)
# i_y = np.linspace(min(rfy.y), max(rfy.y), 1900)
# interp = splev(i_y, spl)
# resid, fit = kbreduct.detrend(i_y, interp, 19, 10, fits=1)

# inter_r = []
# for v in rfy.y:
#     # print(v)
#     spla = splrep(i_y, fit, k=4)
#     aaa = splev(v, spla)
#     inter_r = np.append(inter_r, aaa)

# freq0 = np.linspace(0.0001, 2, 6000)
# model = LombScargle(i_y, resid)
# power0 = model.power(freq0, method='fast', normalization='psd')
# fig, ax = plt.subplots(3, 1, figsize=(8, 10))
# ax[0].plot(rfy.y, rfy.adf, color='k', label='orgin')
# ax[0].plot(i_y, fit, ':', color='k', label='fit')
# # ax[0].plot(rfy.y, fit0)
# # ax[0].plot(i_y, i_x, label='interpolate')
# ax[0].set_xlabel('Time ($years$)')
# ax[0].set_ylabel('F10.7 ($sfu$)')

# ax[1].plot(i_y, resid, ':', color='k', label='residual')
# ax[1].plot(rfy.y, rfy.adf-inter_r, color='k')

# ax[1].set_xlabel('Time ($years$)')
# ax[1].set_ylabel('Residual ($F10.7 - trend$)')
# ax[2].plot(1/freq0, power0, color='k', label='Lomb-Scargle')
# ax[2].set_xlabel(r'Period $(years)$')
# ax[2].set_ylabel('Power ($ppm^2$)')
# ax[2].set_xlim(0, 10)
# plt.savefig('figureF_dummy.pdf')
# plt.close()


spl = splrep(rfy.y, dv0.fwhm, k=4)
i_y = np.linspace(min(rfy.y), max(rfy.y), 1900)
interp = splev(i_y, spl)
resid, fit = kbreduct.detrend(i_y, interp, 19, 5, fits=1)

inter_r = []
for v in rfy.y:
    # print(v)
    spla = splrep(i_y, fit, k=4)
    aaa = splev(v, spla)
    inter_r = np.append(inter_r, aaa)

freq0 = np.linspace(0.0001, 2, 6000)
model = LombScargle(i_y, resid)
power0 = model.power(freq0, method='fast', normalization='psd')
fig, ax = plt.subplots(3, 1, figsize=(8, 10))
ax[0].plot(rfy.y, dv0.fwhm, color='k', label='orgin')
ax[0].plot(i_y, fit, ':', color='k', label='fit')
# ax[0].plot(i_y, interp)
ax[0].set_xlabel('Time ($years$)')
ax[0].set_ylabel(r'$\delta \nu_{env}$ $(\mu Hz)$')
ax[0].tick_params(axis='x', direction='in')
ax[0].tick_params(axis='y', direction='in')
ax[0].set_xlim(1995, 2020)

ax[1].plot(i_y, resid, ':', color='k', label='residual')
ax[1].plot(rfy.y, dv0.fwhm-inter_r, color='k')
ax[1].set_xlabel('Time ($years$)')
ax[1].set_ylabel(r'Residual ($\delta \nu_{env} - trend$)')
ax[1].tick_params(axis='x', direction='in')
ax[1].tick_params(axis='y', direction='in')
ax[1].set_xlim(1995, 2020)

ax[2].plot(1/freq0, power0, color='k', label='Lomb-Scargle')
ax[2].set_xlabel(r'Period $(years)$')
ax[2].set_ylabel('Power ($ppm^2$)')
ax[2].set_xlim(0, 10)
ax[2].set_ylim(0, 600000)
ax[2].tick_params(axis='x', direction='in')
ax[2].tick_params(axis='y', direction='in')
plt.savefig('figureW_dummy.pdf')
plt.close()


# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# p0 = [0, 0]
# pg = kbparameters(p0, mini=mini)
# minner = Minimizer(xyfit, pg, fcn_args=(rfy.adf, dv0.numax))
# bg_res = minner.minimize()
# # report_fit(bg_res)
# res_p = xyfit(bg_res.params.valuesdict(),
#               rfy.adf, dv0.numax, value=1)
# resp = param_value(bg_res)
# ax.scatter(rfy.adf, dv0.numax, c='k')  # marker='o', ms=5
# ax.plot(rfy.adf, res_p.fv, c='black', linewidth=lw1)
# ax.set_xlabel(r'F10.7 (sfu)')
# ax.set_ylabel(r'$\nu_{max}$')
# plt.savefig("figureX.pdf")

# dvd = dv0.drop([7])
# rfyd = rfy.drop([2003])


# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# p0 = [0, 0]
# pg = kbparameters(p0, mini=mini)
# minner = Minimizer(xyfit, pg, fcn_args=(dvd.amp, dvd.numax))
# bg_res = minner.minimize()
# # report_fit(bg_res)
# res_p = xyfit(bg_res.params.valuesdict(),
#               rfyd.adf, dvd.numax, value=1)
# resp = param_value(bg_res)
# ax.scatter(rfyd.adf, dvd.numax, c='k')  # marker='o', ms=5
# ax.plot(rfyd.adf, res_p.fv, c='black', linewidth=lw1)
# ax.set_xlabel(r'F10.7 ($sfu$)')
# ax.set_ylabel(r'$\nu_{max}$ - drop[2003]')
# plt.savefig("figureX_2.pdf")
# plt.close()

end = time.time()
print('\033[01;31;43m', format(end-start, ".3f"), 's\033[00m')
