# for analysis calculated data
#

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import math
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d, splrep, splev
import scipy.stats as stats
from lmfit import Minimizer, Parameters, report_fit, Model
from background.bgm import xyfitmodel, param_value

start = time.time()

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = "12"
plt.rcParams["axes.xmargin"] = "0.1"
plt.rcParams["axes.ymargin"] = "0.1"
csfont = {'fontname': 'Times New Roman'}

pth = '/home/kim/research/amplitude_fs/'
pth_r = pth+'Sun02/results6m/'

dfg = pd.read_csv(pth_r+'results6m_half.csv', delimiter=',')
# dfg = dfg.loc[1:]
# dfg = dfg.drop([10])
dfg.width = dfg.width * 2 * np.sqrt(2*np.log(2))
dfg.widths = dfg.widths * 2 * np.sqrt(2*np.log(2))
dfr = pd.read_csv(pth_r+'rfm6m_half.csv', delimiter=',')
ddd = np.int64(dfr.rfmt)+((dfr.rfmt - np.int64(dfr.rfmt))*12 - 0.5)/12
spl = splrep(ddd, dfr.rfm, k=5)
i_time = dfg.dsei
rdi = splev(i_time, spl)

day = np.array(dfg.dsei)
amp = np.array(dfg.amp)
amps = np.array(dfg.amps)
numax = np.array(dfg.numax)
numaxs = np.array(dfg.numaxs)
width = np.array(dfg.width)
widths = np.array(dfg.widths)
sph = np.array(dfg.sphq)
# for i in range(22):
#     print(dfp.KIC[i]-dfc.Ki])3.

# max 23 = 2001. 11
max23 = 2001+(11-1)/12
# max 24 = 2014. 4
max24 = 2014+(4-1)/12
# min 23 1996. 08
min23 = 1996+(8-1)/12
# min 23-24 2008. 12
min34 = 2008+(12-1)/12
# min 24 2020. 5
min24 = 2020+(5-1)/12

whe01 = np.where(((day >= min23) & (day <= min23+2))
                 | ((day >= min34-2) & (day <= min34+2))
                 | (day >= min24-2) & (day <= min24))

whe02 = np.where(((day >= min23) & (day <= min23+4))
                 | ((day >= min34) & (day <= min34+4)))

whe03 = np.where(((day >= max23-2) & (day <= max23+2))
                 | ((day >= max24-2) & (day <= max24+2)))

whe04 = np.where(((day >= max23) & (day <= max23+4))
                 | ((day >= max24) & (day <= max24+4)))

day01 = day[whe01]
amp01 = amp[whe01]
amps01 = amps[whe01]
sph01 = sph[whe01]
rdi01 = rdi[whe01]
numax01 = numax[whe01]
width01 = width[whe01]
numaxs01 = numaxs[whe01]
widths01 = widths[whe01]

day02 = day[whe02]
amp02 = amp[whe02]
amps02 = amps[whe02]
sph02 = sph[whe02]
rdi02 = rdi[whe02]
numax02 = numax[whe02]
width02 = width[whe02]
numaxs02 = numaxs[whe02]
widths02 = widths[whe02]

day03 = day[whe03]
amp03 = amp[whe03]
amps03 = amps[whe03]
sph03 = sph[whe03]
rdi03 = rdi[whe03]
numax03 = numax[whe03]
width03 = width[whe03]
numaxs03 = numaxs[whe03]
widths03 = widths[whe03]

day04 = day[whe04]
amp04 = amp[whe04]
amps04 = amps[whe04]
sph04 = sph[whe04]
rdi04 = rdi[whe04]
numax04 = numax[whe04]
width04 = width[whe04]
numaxs04 = numaxs[whe04]
widths04 = widths[whe04]

# fname = pth+'Sun02/analysis/fig_analysis_Sun6m.pdf'

""" group plot """
ms = 5
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].errorbar(rdi, amp, fmt='o', yerr=amps, ms=ms,
               c='k', label='All')

ax[0].set_xlabel(r'F10.7 (sfu)', size=15)
ax[0].set_ylabel(r'$A$ (arbitrary units)', size=15)
ax[0].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
# ax[0].set_yticklabels(['', '1.2', '', '1.6', '', '1.8', ''])
# ax[0].set_yticks(np.arange(0.8, 2.2, 0.4))
# ax[0].legend()
col1 = stats.pearsonr(rdi, amp)
coef1 = '$\it r$ = ' + '%.3f' % col1[0]
pval1 = '$\it P$ =  ' + '%.3f' % col1[1]
pval1 = '$\it P$ = ' + r'3$\times 10^{-6}$'

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(amp, x=rdi,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps)))
ax[0].plot(rdi, lmr0.best_fit, color='k')
ax[0].text(0.980, 0.93, coef1, fontsize=10,
           ha='right', va='top', transform=ax[0].transAxes)
ax[0].text(0.989, 0.88, pval1, fontsize=10,
           ha='right', va='top', transform=ax[0].transAxes)
ax[1].errorbar(sph, amp, fmt='o', yerr=amps, ms=ms,
               c='k', label='All')
ax[1].set_xlabel(r'$S_{\rm ph}$ (ppm)', size=15)
ax[1].set_ylabel(r'$A$ (arbitrary units)', size=15)
# ax[1].legend()
col1 = stats.pearsonr(sph, amp)
coef1 = '$\it r$ = ' + '%.3f' % col1[0]
pval1 = '$\it P$ =  ' + '%.3f' % col1[1]
pval1 = '$\it P$ = ' + r'2$\times 10^{-8}$'

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(amp, x=sph,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps)))
ax[1].plot(sph, lmr0.best_fit, color='k')
ax[1].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
ax[1].text(0.980, 0.93, coef1, fontsize=10,
           ha='right', va='top', transform=ax[1].transAxes)
ax[1].text(0.991, 0.88, pval1, fontsize=10,
           ha='right', va='top', transform=ax[1].transAxes)
# ax[1].set_yticklabels(['', '1.2', '', '1.6', '', '1.8', ''])
# ax[0].set_xticklabels(['', '60', '', '100', '', '140', '', '180', ''])
# ax[1].set_xticklabels(['', '60', '', '100', '', '140', ''])
ax[0].set_yticks(np.arange(1.2, 2.4, 0.4))
ax[0].set_xticks(np.arange(60, 220, 40))
ax[1].set_yticks(np.arange(1.2, 2.4, 0.4))
ax[1].set_xticks(np.arange(60, 220, 40))
ax[1].set_xlim(40, 175)
plt.savefig("figure2.pdf")
plt.close()


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.errorbar(rdi, width, fmt='o', yerr=widths, ms=ms,
            c='k', label='All')
ax.set_xlabel(r'F10.7 (sfu)', size=18)
ax.set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', size=18)
# ax.legend()
col1 = stats.pearsonr(rdi, width)
coef1 = '$\it r$ = ' + '%.3f' % col1[0]
pval1 = '$\it P$ = ' + '%.3f' % col1[1]
# ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
#         transform=ax.transAxes, fontsize=8)
p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(width, x=rdi,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(widths)))
ax.plot(rdi, lmr0.best_fit, color='k')
ax.tick_params(axis='both', direction='in',
               top=True, right=True, labelsize=18)
ax.set_ylim(655, 870)
# ax.set_yticklabels(['', '', '700', '', '750', '', '800', '', '850'])
# ax.set_xticklabels(['', '60', '', '100', '', '140', '', '180', ''])
ax.set_yticks(np.arange(700, 900, 50))
ax.set_xticks(np.arange(60, 220, 40))
ax.set_xlim(50, 210)
ax.text(0.977, 0.97, coef1, fontsize=13,
        ha='right', va='top', transform=ax.transAxes)
ax.text(0.98, 0.94, pval1, fontsize=13,
        ha='right', va='top', transform=ax.transAxes)
plt.savefig('figure3.pdf')
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.errorbar(numax, width, fmt='o', xerr=numaxs, yerr=widths,
            ms=ms, c='k', label='All')
ax.set_xlabel(r'$\nu_{\rm max}$ $(\mu$Hz)', size=18)
ax.set_ylabel(r'$\delta \nu_{ \rm env}$ $(\mu$Hz)', size=18)
# ax.legend()
col1 = stats.pearsonr(numax, width)
coef1 = '$\it r$ = ' + '%.3f' % col1[0]
pval1 = '$\it P$ = ' + '%.3f' % col1[1]
# ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
#         transform=ax.transAxes, fontsize=8)
p0 = [0.0, 1.0]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(width, x=numax,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(widths)))
ax.plot(numax, lmr0.best_fit, color='k')
ax.tick_params(axis='both', direction='in',
               top=True, right=True, labelsize=18)
ax.set_ylim(655, 870)
ax.set_xlim(3100, 3210)
# ax.set_yticklabels(['', '', '700', '', '750', '', '800', '', '850'])
# ax.set_xticklabels(['', '3120', '', '3160', '', '3200', ''])
ax.set_yticks(np.arange(700, 900, 50))
ax.set_xticks(np.arange(3120, 3210, 40))
ax.text(0.979, 0.97, coef1, fontsize=13,
        ha='right', va='top', transform=ax.transAxes)
ax.text(0.98, 0.94, pval1, fontsize=13,
        ha='right', va='top', transform=ax.transAxes)
plt.savefig("figure4.pdf")
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.ravel()
# ms = 3
ax[0].plot(rdi01, sph01, 'o', c='k', label=r'min $\pm$ 2yrs', ms=ms)
ax[1].plot(rdi02, sph02, 'o', c='k', label=r'min + 4yrs', ms=ms)
ax[2].plot(rdi03, sph03, 'o', c='k', label=r'Max $\pm$ 2yrs', ms=ms)
ax[3].plot(rdi04, sph04, 'o', c='k', label=r'Max + 4yrs', ms=ms)
# ax[4].plot(rdi, sph, 'o', c='gray', alpha=0.5, label='All')
ax[0].set_xlabel(r'F10.7 (sfu)')
ax[0].set_ylabel(r'$S_{\rm ph}$ (ppm)')
ax[1].set_xlabel(r'F10.7 (sfu)')
ax[1].set_ylabel(r'$S_{\rm ph}$ (ppm)')
ax[2].set_xlabel(r'F10.7 (sfu)')
ax[2].set_ylabel(r'$S_{\rm ph}$ (ppm)')
ax[3].set_xlabel(r'F10.7 (sfu)')
ax[3].set_ylabel(r'$S_{\rm ph}$ (ppm)')
ax[0].set_ylim(53, 79)
ax[1].set_ylim(51, 125)
ax[1].set_xlim(54, 200)
# ax[2].set_ylim(50, 125)
# ax[2].set_xlim(55, 220)
# ax[3].set_ylim(50, 125)
# ax[3].set_xlim(55, 220)
# ax[0].set_yticklabels(['', '55', '', '65', '', '75', ''])
# ax[0].set_xticklabels(['', '60', '', '80', '', '100', ''])
ax[0].set_yticks(np.arange(55, 80, 10))
ax[0].set_xticks(np.arange(70, 125, 20))
# ax[1].set_yticklabels(['', '60', '', '80', '', '100', '', '120'])
# ax[1].set_xticklabels(['', '75', '', '125', '', '175', ''])
ax[1].set_yticks(np.arange(60, 140, 20))
ax[1].set_xticks(np.arange(75, 200, 50))

# ax[2].set_yticklabels(['', '80', '', '120', '', '160'])
# ax[2].set_xticklabels(['', '100', '', '140', '', '180', ''])
ax[2].set_yticks(np.arange(80, 200, 40))
ax[2].set_xticks(np.arange(110, 250, 40))

# ax[3].set_yticklabels(['', '60', '', '100', '', '140', ''])
# ax[3].set_xticklabels(['', '75', '', '125', '', '175', ''])
ax[3].set_yticks(np.arange(70, 250, 40))
ax[3].set_xticks(np.arange(80, 250, 40))

# ax[0].set_ylim(50, 90)

ax[0].legend(frameon=False, markerscale=0, fontsize=12)
ax[1].legend(frameon=False, markerscale=0, fontsize=12)
ax[2].legend(frameon=False, markerscale=0, fontsize=12)
ax[3].legend(frameon=False, markerscale=0, fontsize=12)
col1 = stats.pearsonr(rdi01, sph01)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]
col2 = stats.pearsonr(rdi02, sph02)
coef2 = 'C. : ' + '%.4f' % col2[0]
pval2 = ' P. : ' + '%.4f' % col2[1]
col3 = stats.pearsonr(rdi03, sph03)
coef3 = 'C. : ' + '%.4f' % col3[0]
pval3 = ' P. : ' + '%.4f' % col3[1]
col4 = stats.pearsonr(rdi04, sph04)
coef4 = 'C. : ' + '%.4f' % col4[0]
pval4 = ' P. : ' + '%.4f' % col4[1]
# ax[0].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
#            transform=ax[0].transAxes, fontsize=8)
# ax[1].text(0.98, 0.85, coef2 + pval2, ha='right', va='top',
#            transform=ax[1].transAxes, fontsize=8)
# ax[2].text(0.98, 0.85, coef3 + pval3, ha='right', va='top',
#            transform=ax[2].transAxes, fontsize=8)
# ax[3].text(0.98, 0.85, coef4 + pval4, ha='right', va='top',
#            transform=ax[3].transAxes, fontsize=8)
p0 = [0.0, 1.0]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(sph01, x=rdi01,
                a=p0[0], b=p0[1])  # , weights=np.sqrt(1.0/(amps)))
res0 = param_value(lmr0)
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(sph02, x=rdi02,
                a=p0[0], b=p0[1])  # , weights=np.sqrt(1.0/(amps)))
res1 = param_value(lmr1)
mod2 = Model(xyfitmodel)
lmr2 = mod2.fit(sph03, x=rdi03,
                a=p0[0], b=p0[1])  # , weights=np.sqrt(1.0/(amps)))
res2 = param_value(lmr2)
mod3 = Model(xyfitmodel)
lmr3 = mod3.fit(sph04, x=rdi04,
                a=p0[0], b=p0[1])  # , weights=np.sqrt(1.0/(amps)))
res3 = param_value(lmr3)
inc0 = '%.3f' % res0.p[0]
slp0 = '%.3f' % res0.p[1]
inc1 = '%.3f' % res1.p[0]
slp1 = '%.3f' % res1.p[1]
inc2 = '%.3f' % res2.p[0]
slp2 = '%.3f' % res2.p[1]
inc3 = '%.3f' % res3.p[0]
slp3 = '%.3f' % res3.p[1]
ax[0].plot(rdi01, lmr0.best_fit, color='k')
# ax[0].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
#            ha='right', va='top', transform=ax[0].transAxes)
ax[1].plot(rdi02, lmr1.best_fit, color='k')
# ax[1].text(0.98, 0.80, inc1+' + '+slp1+'x', fontsize=8,
#            ha='right', va='top', transform=ax[1].transAxes)
ax[2].plot(rdi03, lmr2.best_fit, color='k')
# ax[2].text(0.98, 0.80, inc2+' + '+slp2+'x', fontsize=8,
#            ha='right', va='top', transform=ax[2].transAxes)
ax[3].plot(rdi04, lmr3.best_fit, color='k')
# ax[3].text(0.98, 0.80, inc3+' + '+slp3+'x', fontsize=8,
#            ha='right', va='top', transform=ax[3].transaxes)
plt.savefig('figure5.pdf')
plt.close()
# label=r'min $\pm$ 2yrs', ms=ms)
# ax[1].plot(rdi02, sph02, 'o', c='k', label=r'min + 4yrs', ms=ms)
# ax[2].plot(rdi03, sph03, 'o', c='k', label=r'Max $\pm$ 2yrs', ms=ms)
# ax[3].plot(rdi04, sph04, 'o', c='k', label=r'Max + 4yrs', ms=ms)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.ravel()
ax[0].errorbar(sph01, amp01, fmt='o', yerr=amps01,
               ms=ms, c='k', label=r'min $\pm$ 2yrs')
ax[1].errorbar(sph02, amp02, fmt='o', yerr=amps02,
               ms=ms, c='k', label=r'min + 4yrs')
ax[2].errorbar(sph03, amp03, fmt='o', yerr=amps03,
               ms=ms, c='k', label=r'Max $\pm$ 2yrs')
ax[3].errorbar(sph04, amp04, fmt='o', yerr=amps04,
               ms=ms, c='k', label=r'Max + 4yrs')
ax[0].set_xlabel(r'$S_{\rm ph}$ (ppm)')
ax[0].set_ylabel(r'$A$ (arbitrary units)')
ax[1].set_xlabel(r'$S_{\rm ph}$ (ppm)')
ax[1].set_ylabel(r'$A$ (arbitrary units)')
ax[2].set_xlabel(r'$S_{\rm ph}$ (ppm)')
ax[2].set_ylabel(r'$A$ (arbitrary units)')
ax[3].set_xlabel(r'$S_{\rm ph}$ (ppm)')
ax[3].set_ylabel(r'$A$ (arbitrary units)')

# ax[0].set_yticklabels(['', '1.6', '', '1.8', '', '2.0', ''])
# ax[0].set_xticklabels(['', '55', '', '65', '', '75'])
ax[0].set_yticks(np.arange(1.6, 2.4, 0.2))
ax[0].set_xticks(np.arange(55, 125, 10))
ax[0].set_ylim(1.55, 2.19)
# ax[1].set_yticklabels(['', '1.4', '', '1.6', '', '1.8', '', '2.0', ''])
# ax[1].set_xticklabels(['', '60', '', '80', '', '100', ''])
ax[1].set_yticks(np.arange(1.4, 2.4, 0.2))
ax[1].set_xticks(np.arange(65, 125, 20))
# ax[2].set_ylim(1.31, 1.89)
# ax[2].set_yticklabels(['', '1.2', '', '1.4', '', '1.6', '', '1.8'])
# ax[2].set_xticklabels(['', '80', '', '120', '', '160', ''])
ax[2].set_yticks(np.arange(1.2, 2.4, 0.2))
ax[2].set_xticks(np.arange(80, 200, 40))
# ax[3].set_ylim(1.31, 2.09)
# ax[3].set_yticklabels(['', '1.2', '', '1.6', '', '2.0'])
# ax[3].set_xticklabels(['', '60', '', '100', '', '140', ''])
ax[3].set_yticks(np.arange(1.2, 2.4, 0.4))
ax[3].set_xticks(np.arange(70, 200, 40))
handles, labels = ax[0].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[0].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
handles, labels = ax[1].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[1].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
handles, labels = ax[2].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[2].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
handles, labels = ax[3].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[3].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
col1 = stats.pearsonr(sph01, amp01)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]
col2 = stats.pearsonr(sph02, amp02)
coef2 = 'C. : ' + '%.4f' % col2[0]
pval2 = ' P. : ' + '%.4f' % col2[1]
col3 = stats.pearsonr(sph03, amp03)
coef3 = 'C. : ' + '%.4f' % col3[0]
pval3 = ' P. : ' + '%.4f' % col3[1]
col4 = stats.pearsonr(sph04, amp04)
coef4 = 'C. : ' + '%.4f' % col4[0]
pval4 = ' P. : ' + '%.4f' % col4[1]
# ax[0].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
#            transform=ax[0].transAxes, fontsize=8)
# ax[1].text(0.98, 0.85, coef2 + pval2, ha='right', va='top',
#            transform=ax[1].transAxes, fontsize=8)
# ax[2].text(0.98, 0.85, coef3 + pval3, ha='right', va='top',
#            transform=ax[2].transAxes, fontsize=8)
# ax[3].text(0.98, 0.85, coef4 + pval4, ha='right', va='top',
#            transform=ax[3].transAxes, fontsize=8)
p0 = [0.0, 1.0]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(amp01, x=sph01,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps01)))
res0 = param_value(lmr0)
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(amp02, x=sph02,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps02)))
res1 = param_value(lmr1)
mod2 = Model(xyfitmodel)
lmr2 = mod2.fit(amp03, x=sph03,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps03)))
res2 = param_value(lmr2)
mod3 = Model(xyfitmodel)
lmr3 = mod3.fit(amp04, x=sph04,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps04)))
res3 = param_value(lmr3)
inc0 = '%.3f' % res0.p[0]
slp0 = '%.3f' % res0.p[1]
inc1 = '%.3f' % res1.p[0]
slp1 = '%.3f' % res1.p[1]
inc2 = '%.3f' % res2.p[0]
slp2 = '%.3f' % res2.p[1]
inc3 = '%.3f' % res3.p[0]
slp3 = '%.3f' % res3.p[1]
ax[0].plot(sph01, lmr0.best_fit, color='k')
# ax[0].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
#            ha='right', va='top', transform=ax[0].transAxes)
ax[1].plot(sph02, lmr1.best_fit, color='k')
# ax[1].text(0.98, 0.80, inc0+' + '+slp1+'x', fontsize=8,
#            ha='right', va='top', transform=ax[1].transAxes)
ax[2].plot(sph03, lmr2.best_fit, color='k')
# ax[2].text(0.98, 0.80, inc2+' + '+slp2+'x', fontsize=8,
#            ha='right', va='top', transform=ax[2].transAxes)
ax[3].plot(sph04, lmr3.best_fit, color='k')
# ax[3].text(0.98, 0.80, inc3+' + '+slp3+'x', fontsize=8,
#            ha='right', va='top', transform=ax[3].transAxes)
plt.savefig("figure6.pdf")
plt.close()

# ax[0].plot(rdi01, sph01, 'o', c='k', label=r'min $\pm$ 2yrs', ms=ms)
# ax[1].plot(rdi02, sph02, 'o', c='k', label=r'min + 4yrs', ms=ms)
# ax[2].plot(rdi03, sph03, 'o', c='k', label=r'Max $\pm$ 2yrs', ms=ms)
# ax[3].plot(rdi04, sph04, 'o', c='k', label=r'Max + 4yrs', ms=ms)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.ravel()
ax[0].errorbar(numax01, width01, fmt='o', xerr=numaxs01, ms=ms,
               yerr=widths01, c='k', label=r'min $\pm$ 2yrs')
ax[1].errorbar(numax02, width02, fmt='o', xerr=numaxs02, ms=ms,
               yerr=widths02, c='k', label=r'min + 4yrs')
ax[2].errorbar(numax03, width03, fmt='o', xerr=numaxs03, ms=ms,
               yerr=widths03, c='k', label=r'Max $\pm$ 2yrs')
ax[3].errorbar(numax04, width04, fmt='o', xerr=numaxs04, ms=ms,
               yerr=widths04, c='k', label=r'Max + 4yrs')
ax[0].set_xlabel(r'$\nu_{\rm max}$ $(\mu$Hz)')
ax[0].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)')
ax[1].set_xlabel(r'$\nu_{\rm max}$ $(\mu$Hz)')
ax[1].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)')
ax[2].set_xlabel(r'$\nu_{\rm max}$ $(\mu$Hz)')
ax[2].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)')
ax[3].set_xlabel(r'$\nu_{\rm max}$ $(\mu$Hz)')
ax[3].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)')
ax[0].set_yticks(np.arange(700, 1000, 50))
ax[0].set_xticks(np.arange(3140, 3220, 20))
ax[1].set_yticks(np.arange(700, 1000, 50))
ax[1].set_xticks(np.arange(3140, 3220, 20))
ax[2].set_yticks(np.arange(650, 1000, 50))
ax[2].set_xticks(np.arange(3120, 3300, 40))
ax[2].set_ylim(655, 865)
ax[2].set_xlim(3101, 3210)
ax[3].set_yticks(np.arange(650, 1000, 50))
ax[3].set_xticks(np.arange(3120, 3300, 40))
ax[3].set_ylim(655, 865)
ax[3].set_xlim(3101, 3210)

# ax[0].set_ylim(676, 890)
# ax[0].set_yticklabels(['', '700', '', '750', '', '800', '', '850', ''])
# ax[0].set_xticklabels(['', '3140', '', '3160', '', '3180', ''])
# ax[1].set_ylim(676, 890)
# ax[1].set_yticklabels(['', '700', '', '750', '', '800', '', '850', ''])
# ax[1].set_xticklabels(['', '3140', '', '3160', '', '3180', ''])
# ax[2].set_ylim(651, 870)
# ax[2].set_xlim(3101, 3210)
# ax[2].set_yticklabels(['', '675', '', '725', '', '775', '', '825', ''])
# ax[2].set_xticklabels(['', '3120', '', '3160', '', '3200'])
# ax[3].set_ylim(651, 870)
# ax[3].set_xlim(3101, 3210)
# ax[3].set_yticklabels(['', '675', '', '725', '', '775', '', '825', ''])
# ax[3].set_xticklabels(['', '3120', '', '3160', '', '3200'])
# ax[1].set_yticklabels(['', '650', '', '750', '', '850'])

# ax[2].set_ylim(645, 900)
# ax[2].set_yticklabels(['', '650', '', '750', '', '850', ''])
# ax[2].set_xticklabels(['', '3120', '', '3160', '', '3200'])

# ax[3].set_yticklabels(['', '650', '', '750', '', '850'])
# ax[3].set_xticklabels(['', '3120', '', '3160', '', '3200'])


handles, labels = ax[0].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[0].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
handles, labels = ax[1].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[1].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
handles, labels = ax[2].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[2].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
handles, labels = ax[3].get_legend_handles_labels()
handles = [h[0] for h in handles]
ax[3].legend(handles, labels, frameon=False, markerscale=0, fontsize=12)
col1 = stats.pearsonr(numax01, width01)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]
col2 = stats.pearsonr(numax02, width02)
coef2 = 'C. : ' + '%.4f' % col2[0]
pval2 = ' P. : ' + '%.4f' % col2[1]
col3 = stats.pearsonr(numax03, width03)
coef3 = 'C. : ' + '%.4f' % col3[0]
pval3 = ' P. : ' + '%.4f' % col3[1]
col4 = stats.pearsonr(numax04, width04)
coef4 = 'C. : ' + '%.4f' % col4[0]
pval4 = ' P. : ' + '%.4f' % col4[1]
# ax[0].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
#            transform=ax[0].transAxes, fontsize=8)
# ax[1].text(0.98, 0.85, coef2 + pval2, ha='right', va='top',
#            transform=ax[1].transAxes, fontsize=8)
# ax[2].text(0.98, 0.85, coef3 + pval3, ha='right', va='top',
#            transform=ax[2].transAxes, fontsize=8)
# ax[3].text(0.98, 0.85, coef4 + pval4, ha='right', va='top',
#            transform=ax[3].transAxes, fontsize=8)
p0 = [0.0, 1.0]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(width01, x=numax01,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(widths01)))
res0 = param_value(lmr0)
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(width02, x=numax02,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(widths02)))
res1 = param_value(lmr1)
mod2 = Model(xyfitmodel)
lmr2 = mod2.fit(width03, x=numax03,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(widths03)))
res2 = param_value(lmr2)
mod3 = Model(xyfitmodel)
lmr3 = mod3.fit(width04, x=numax04,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(widths04)))
res3 = param_value(lmr3)
inc0 = '%.3f' % res0.p[0]
slp0 = '%.3f' % res0.p[1]
inc1 = '%.3f' % res1.p[0]
slp1 = '%.3f' % res1.p[1]
inc2 = '%.3f' % res2.p[0]
slp2 = '%.3f' % res2.p[1]
inc3 = '%.3f' % res3.p[0]
slp3 = '%.3f' % res3.p[1]
ax[0].plot(numax01, lmr0.best_fit, color='k')
# ax[0].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
#            ha='right', va='top', transform=ax[0].transAxes)
ax[1].plot(numax02, lmr1.best_fit, color='k')
# ax[1].text(0.98, 0.80, inc0+' + '+slp1+'x', fontsize=8,
#            ha='right', va='top', transform=ax[1].transAxes)
ax[2].plot(numax03, lmr2.best_fit, color='k')
# ax[2].text(0.98, 0.80, inc2+' + '+slp2+'x', fontsize=8,
#            ha='right', va='top', transform=ax[2].transAxes)
ax[3].plot(numax04, lmr3.best_fit, color='k')
# ax[3].text(0.98, 0.80, inc3+' + '+slp3+'x', fontsize=8,
#            ha='right', va='top', transform=ax[3].transAxes)
plt.savefig("figure7.pdf")
plt.close()


end = time.time()
print('\033[01;31;43m', format(end - start, ".3f"), 's\033[00m')
