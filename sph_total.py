# for analysis calculated data
#

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import matplotlib.lines as mlines

import time
import math
from scipy.stats import ks_2samp
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
plt.rcParams["legend.frameon"] = "False"

pth = '/home/kim/research/amplitude_fs/'
pth_r = pth+'main10/results/'

dfp = pd.read_csv('/media/kim/datafold/data/santos2018/santos1.dat')
dfp1 = dfp.drop(['KIC'], axis=1)
dfp1 = dfp1.drop(['KOI'], axis=1)
dfp1.rename(columns={'numax': 'numax0'}, inplace=True)
dfc = pd.read_csv(pth+'main10/correl.csv')
san = pd.read_csv(pth+'santoscsv.csv')
san = san.drop(['KIC'], axis=1)
sun = pd.DataFrame({'logg': [4.43], 'Teff': [5778], 'FeH': [
    0.0], 'Radius': [1], 'prot': [25.3], 'ampv': [0.77], 'ampr': [0.36]})
# for i in range(22):
#     print(dfp.KIC[i]-dfc.Ki])3.

file_list = os.listdir(pth_r)
EXT = '.csv'
filen = [file for file in file_list if file.endswith(EXT)]
filen.sort()
df = pd.DataFrame()
for f in filen:
    df0 = pd.read_csv(pth_r+f, comment='#')
    df = pd.concat([df, df0], ignore_index=True)
# df = df.drop(['n'], axis='columns')
df.width = df.width * 2 * np.sqrt(2*np.log(2))
df.widths = df.widths * 2 * np.sqrt(2*np.log(2))

res = pd.concat([dfc, dfp1, df, san], axis=1)
res['sphr'] = (res.sphmax-res.sphmin)/res.sphmax
slope = []
for ss in range(len(res)):
    if res.slope[ss] >= 0:
        slope = np.append(slope, 1)
    else:
        slope = np.append(slope, -1)
res['slope'] = slope
fname = pth+'main10/analysis/fig_analysis.pdf'
with PdfPages(fname) as pdf:
    """ group plot """
    grp = res.groupby('sn').get_group(1).reset_index(drop=True)
    # groups = grp.groupby('asresult')

    asym = ['o', 'x', '*']
    symc = ['r', 'b', 'b']

    groups = grp.groupby('slope')

    asym = ['o', 'x']
    symc = ['r', 'b']

    fig, ax = plt.subplots(5, 2, figsize=(11, 16))
    ax = ax.ravel()
    i = 0
    hist_FeH = [[], []]
    hist_Teff = [[], []]
    hist_prot = [[], []]
    hist_Radius = [[], []]
    hist_logg = [[], []]
    hist_Kp = [[], []]
    hist_sph = [[], []]
    hist_ampv = [[], []]
    hist_ampr = [[], []]
    hist_sphr = [[], []]

    for name, group in groups:
        param = grp.FeH
        param2 = group.FeH
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_FeH[i] = hist
        ax[0].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.Teff
        param2 = group.Teff
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_Teff[i] = hist
        ax[1].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.prot
        param2 = group.prot
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_prot[i] = hist
        ax[2].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.Radius
        param2 = group.Radius
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_Radius[i] = hist
        ax[3].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.logg
        param2 = group.logg
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_logg[i] = hist
        ax[4].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.Kp
        param2 = group.Kp
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_Kp[i] = hist
        ax[5].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.sph.where(grp.sph < 700)
        param2 = group.sph.where(grp.sph < 700)
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_sph[i] = hist
        ax[6].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))
        param = grp.sphr
        param2 = group.sphr
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_sphr[i] = hist
        ax[7].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.ampv
        param2 = group.ampv
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_ampv[i] = hist
        ax[8].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        param = grp.ampr
        param2 = group.ampr
        num = 10
        maxr = param.max()
        minr = param.min()
        bins0 = np.linspace(minr, maxr, num)
        hist, bins = np.histogram(param2, bins=bins0)
        hist_ampr[i] = hist
        ax[9].hist(param2, bins=num, density=False,
                   alpha=0.6, histtype='bar', color=symc[i], edgecolor='k',
                   linewidth=1.2, range=(minr, maxr))

        i += 1

    ax[0].set_xlabel(r'$[Fe/H]$')
    ax[1].set_xlabel(r'$T_{eff}$')
    ax[2].set_xlabel(r'$P_{rot}$')
    ax[3].set_xlabel(r'$Radius$')
    ax[4].set_xlabel(r'$log(g)$')
    ax[5].set_xlabel(r'$Kp$')
    ax[6].set_xlabel(r'$S_{ph}$')
    ax[7].set_xlabel(r'$\delta S_{ph}/S_{ph,max}$')
    ax[8].set_xlabel(r'$(\delta A)$')
    ax[9].set_xlabel(r'$(\delta A)/A_{max}$')

    ks = ks_2samp(hist_FeH[0], hist_FeH[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[0].text(0.65, 0.92, coef+pval, transform=ax[0].transAxes, fontsize=8)

    ks = ks_2samp(hist_Teff[0], hist_Teff[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[1].text(0.65, 0.92, coef+pval, transform=ax[1].transAxes, fontsize=8)

    ks = ks_2samp(hist_prot[0], hist_prot[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[2].text(0.65, 0.92, coef+pval, transform=ax[2].transAxes, fontsize=8)

    ks = ks_2samp(hist_Radius[0], hist_Radius[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[3].text(0.65, 0.92, coef+pval, transform=ax[3].transAxes, fontsize=8)

    ks = ks_2samp(hist_logg[0], hist_logg[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[4].text(0.65, 0.92, coef+pval, transform=ax[4].transAxes, fontsize=8)

    ks = ks_2samp(hist_Kp[0], hist_Kp[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[5].text(0.65, 0.92, coef+pval, transform=ax[5].transAxes, fontsize=8)

    ks = ks_2samp(hist_sph[0], hist_sph[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[6].text(0.65, 0.92, coef+pval, transform=ax[6].transAxes, fontsize=8)

    ks = ks_2samp(hist_sphr[0], hist_sphr[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[7].text(0.65, 0.92, coef+pval, transform=ax[7].transAxes, fontsize=8)

    ks = ks_2samp(hist_ampv[0], hist_ampv[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[8].text(0.65, 0.92, coef+pval, transform=ax[8].transAxes, fontsize=8)

    ks = ks_2samp(hist_ampr[0], hist_ampr[1])
    coef = 'C. : ' + '%.4f' % ks[0]
    pval = ' P. : ' + '%.4f' % ks[1]
    ax[9].text(0.65, 0.92, coef+pval, transform=ax[9].transAxes, fontsize=8)
    # ax[6].set_xlabel(r'$(\delta A)/A_{max}$')
    # ax[6].set_ylabel(r'$S_{ph}$')
    # ax[7].set_xlabel(r'$(\delta A)/A_{max}$')

    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(11, 13))
    ax = ax.ravel()
    i = 0
    lab = ['-1', '+1']
    aaa = np.linspace(1, 5, len(grp))
    ax[0].scatter(grp.asresult*aaa,
                  grp.dvs*aaa, c='k')  # , label=lab[i])
    ax[0].set_xlim(-5.2, 5.2)
    ax[0].set_ylim(-5.2, 5.2)
    ax[0].set_xlabel('Correlation')
    ax[0].set_ylabel('Santos (F.shift vs A)')
    ax[0].axhline(0, 0, 10, c='gray', alpha=0.5)
    ax[0].axvline(0, 0, 10, c='gray', alpha=0.5)
    ax[1].scatter(grp.asresult*aaa,
                  grp.dvsph*aaa, c='k', label=lab[i])
    ax[1].axhline(0, 0, 10, c='gray', alpha=0.5)
    ax[1].axvline(0, 0, 10, c='gray', alpha=0.5)
    ax[1].set_xlim(-5.2, 5.2)
    ax[1].set_ylim(-5.2, 5.2)
    ax[0].set_title('Correlation + Santos (F.shift vs A)')
    ax[1].set_title('Correlation + Santos (F.shift vs $S_{ph}$)')
    ax[1].set_xlabel('Correlation')
    ax[1].set_ylabel('Santos (F.shift vs $S_{ph}$)')
    xa1 = []
    xa2 = []
    xa3 = []
    xa4 = []
    aa = []
    xb1 = []
    xb2 = []
    xb3 = []
    xb4 = []
    bb = []
    for n in range(len(grp)):
        ax[0].text(grp.asresult[n]*aaa[n], grp.dvs[n]*aaa[n],
                   grp.KIC[n], fontsize=8, c='k')
        ax[1].text(grp.asresult[n]*aaa[n], grp.dvsph[n]*aaa[n],
                   grp.KIC[n], fontsize=8, c='k')  # , color=symc[i])
        xx = grp.asresult
        yy0 = grp.dvs
        yy1 = grp.dvsph
        kk = grp.KIC
        if xx[n] > 0:
            if yy0[n] > 0:
                xa1 = np.append(xa1, kk[n])
                aa = np.append(aa, 1)
            elif yy0[n] < 0:
                xa4 = np.append(xa4, kk[n])
                aa = np.append(aa, 4)
            elif yy0[n] == 0:
                aa = np.append(aa, 0)
        elif xx[n] < 0:
            if yy0[n] > 0:
                xa2 = np.append(xa2, kk[n])
                aa = np.append(aa, 2)
            elif yy0[n] < 0:
                xa3 = np.append(xa3, kk[n])
                aa = np.append(aa, 3)
            elif yy0[n] == 0:
                aa = np.append(aa, 0)
        elif xx[n] == 0:
            aa = np.append(aa, 0)
        if xx[n] > 0:
            if yy1[n] > 0:
                xb1 = np.append(xb1, kk[n])
                bb = np.append(bb, 1)
            elif yy1[n] < 0:
                xb4 = np.append(xb4, kk[n])
                bb = np.append(bb, 4)
            elif yy1[n] == 0:
                bb = np.append(bb, 0)
        elif xx[n] < 0:
            if yy1[n] > 0:
                xb2 = np.append(xb2, kk[n])
                bb = np.append(bb, 2)
            elif yy1[n] < 0:
                xb3 = np.append(xb3, kk[n])
                bb = np.append(bb, 3)
            elif yy1[n] == 0:
                bb = np.append(bb, 0)
        elif xx[n] == 0:
            bb = np.append(bb, 0)
    ax[0].text(4.0, 4.5, 'total:29, star:'+str(len(xa1)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[0].text(-4.8, 4.5, 'total:29, star:'+str(len(xa2)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[0].text(-4.8, -4.5, 'total:29, star:'+str(len(xa3)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[0].text(4.0, -4.5, 'total:29, star:'+str(len(xa4)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(4.0, 4.5, 'total:29, star:'+str(len(xb1)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(-4.8, 4.5, 'total:29, star:'+str(len(xb2)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(-4.8, -4.5, 'total:29, star:'+str(len(xb3)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(4.0, -4.5, 'total:29, star:'+str(len(xb4)),
               fontsize=8, c='r')  # , color=symc[i])
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(11, 13))
    ax = ax.ravel()
    i = 0
    lab = ['-1', '+1']

    aaa = np.linspace(1, 5, len(grp))
    ax[0].scatter(grp.slope*aaa,
                  grp.dvsas*5, c='k')  # , c=symc[i], label=lab[i])
    ax[0].set_xlim(-5.2, 5.2)
    ax[0].set_ylim(-5.2, 5.2)
    ax[0].set_xlabel('Slope')
    ax[0].set_ylabel(r'Santos (F.shift vs A)')
    ax[0].axhline(0, 0, 10, c='gray', alpha=0.5)
    ax[0].axvline(0, 0, 10, c='gray', alpha=0.5)
    ax[1].scatter(grp.slope*aaa,
                  grp.dvssphs*5, c='k')  # , c=symc[i], label=lab[i])
    ax[1].axhline(0, 0, 10, c='gray', alpha=0.5)
    ax[1].axvline(0, 0, 10, c='gray', alpha=0.5)
    ax[1].set_xlim(-5.2, 5.2)
    ax[1].set_ylim(-5.2, 5.2)
    ax[0].set_title(r'Slope + Santos (F.shift vs A)')
    ax[1].set_title(r'Slope + Santos (F.shift vs $S_{ph}$)')
    ax[1].set_xlabel('Slope')
    ax[1].set_ylabel(r'Santos (F.shift vs $S_{ph}$)')

    xa1 = []
    xa2 = []
    xa3 = []
    xa4 = []
    cc = []
    xb1 = []
    xb2 = []
    xb3 = []
    xb4 = []
    dd = []
    for n in range(len(grp)):
        ax[0].text(grp.slope[n]*aaa[n], grp.dvsas[n]*5,
                   grp.KIC[n], fontsize=8, c='k')  # , color=symc[i])
        ax[1].text(grp.slope[n]*aaa[n], grp.dvssphs[n]*5,
                   grp.KIC[n], fontsize=8, c='k')  # , color=symc[i])
        xx = grp.slope
        yy0 = grp.dvsas
        yy1 = grp.dvssphs
        kk = grp.KIC
        if xx[n] > 0:
            if yy0[n] > 0:
                xa1 = np.append(xa1, kk[n])
                cc = np.append(cc, 1)
            elif yy0[n] < 0:
                xa4 = np.append(xa4, kk[n])
                cc = np.append(cc, 4)
        elif xx[n] < 0:
            if yy0[n] > 0:
                xa2 = np.append(xa2, kk[n])
                cc = np.append(cc, 2)
            elif yy0[n] < 0:
                xa3 = np.append(xa3, kk[n])
                cc = np.append(cc, 3)
        if xx[n] > 0:
            if yy1[n] > 0:
                xb1 = np.append(xb1, kk[n])
                dd = np.append(dd, 1)
            elif yy1[n] < 0:
                xb4 = np.append(xb4, kk[n])
                dd = np.append(dd, 4)
        elif xx[n] < 0:
            if yy1[n] > 0:
                xb2 = np.append(xb2, kk[n])
                dd = np.append(dd, 2)
            elif yy1[n] < 0:
                xb3 = np.append(xb3, kk[n])
                dd = np.append(dd, 3)
    ax[0].text(4.0, 4.5, 'total:29, star:'+str(len(xa1)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[0].text(-4.8, 4.5, 'total:29, star:'+str(len(xa2)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[0].text(-4.8, -4.5, 'total:29, star:'+str(len(xa3)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[0].text(4.0, -4.5, 'total:29, star:'+str(len(xa4)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(4.0, 4.5, 'total:29, star:'+str(len(xb1)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(-4.8, 4.5, 'total:29, star:'+str(len(xb2)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(-4.8, -4.5, 'total:29, star:'+str(len(xb3)),
               fontsize=8, c='r')  # , color=symc[i])
    ax[1].text(4.0, -4.5, 'total:29, star:'+str(len(xb4)),
               fontsize=8, c='r')  # , color=symc[i])
    pdf.savefig()
    plt.close()

    grp1 = grp
    grp1['aa'] = aa
    grp1['bb'] = bb
    grp1['cc'] = cc
    grp1['dd'] = dd


######################################
######################################
    """ figure figure figure figure """
    """ figure figure figure figure """
    groups = grp1.groupby('cc')
    i = 0
    ii = [1, 0, 2, 3]
    name0 = ['SA_1(++)', 'SA_2(\N{MINUS SIGN}+)',
             'SA_3(\N{MINUS SIGN}\N{MINUS SIGN})', 'SA_4(+\N{MINUS SIGN})']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.numax, group.amp, xerr=group.numaxs,
                           yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax[ii[i]].set_ylabel(r'$A$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.numax, group.amp)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.amp, x=group.numax,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.amps)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.numax, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.numax, group.amp, xerr=group.numaxs,
                    yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax.set_ylabel(r'$A$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.numax, grp1.amp)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, -0.00001]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.amp, x=grp1.numax,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.amps)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    ax.plot(grp1.numax, lmr0.best_fit, color='red')
    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.numax, group.width, xerr=group.numaxs,
                           yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax[ii[i]].set_ylabel(r'$\delta \nu_{env}$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.numax, group.width)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, 1]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.width, x=group.numax,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.widths)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.numax, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.numax, group.width, xerr=group.numaxs,
                    yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax.set_ylabel(r'$\delta \nu_{env}$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.numax, grp1.width)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, 1]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.width, x=grp1.numax,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.widths)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    ax.plot(grp1.numax, lmr0.best_fit, color='red')
    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.width, group.amp, xerr=group.widths,
                           yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$\delta \nu_{env}$')
        ax[ii[i]].set_ylabel(r'$A$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.width, group.amp)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.amp, x=group.width,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.amps)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.width, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.width, group.amp, xerr=group.widths,
                    yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$\delta \nu_{env}$')
        ax.set_ylabel(r'$A$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.width, grp1.amp)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, -0.00001]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.amp, x=grp1.width,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.amps)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    ax.plot(grp1.width, lmr0.best_fit, color='red')
    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.sph, group.amp,
                           yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$S_{ph}$')
        ax[ii[i]].set_ylabel(r'$A$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.sph, group.amp)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.amp, x=group.sph,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.amps)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.sph, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.sph, group.amp,
                    yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$S_{ph}$')
        ax.set_ylabel(r'$A$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.sph, grp1.amp)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, -0.00001]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.amp, x=grp1.sph,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.amps)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]

    sphdel = np.array(grp1.sph)
    ampdel = np.array(grp1.amp)
    ampsdel = np.array(grp1.amps)
    sphdel = np.delete(sphdel, [14, 16])
    ampdel = np.delete(ampdel, [14, 16])
    ampsdel = np.delete(ampsdel, [14, 16])
    col2 = stats.pearsonr(sphdel, ampdel)
    coef2 = 'C. : ' + '%.4f' % col2[0]
    pval2 = ' P. : ' + '%.4f' % col2[1]
    p0 = [1, -0.01]
    mod1 = Model(xyfitmodel)
    lmr1 = mod1.fit(ampdel, x=sphdel,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(ampsdel)))
    res1 = param_value(lmr1)
    inc1 = '%.6f' % res1.p[0]
    slp1 = '%.6f' % res1.p[1]
    ax.plot(grp1.sph, lmr0.best_fit, color='red')
    ax.plot(sphdel, lmr1.best_fit, color='blue')

    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.75, coef2 + pval2, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.70, inc1+' + '+slp1+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.sph, group.width,
                           yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$S_{ph}$')
        ax[ii[i]].set_ylabel(r'$\delta \nu_{env}$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.sph, group.width)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.width, x=group.sph,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.widths)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.sph, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.sph, group.width,
                    yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$S_{ph}$')
        ax.set_ylabel(r'$\delta \nu_{env}$')

        i += 1
    ax.legend(fontsize=8)
    col1 = stats.pearsonr(grp1.sph, grp1.width)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, 0.1]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.width, x=grp1.sph,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.widths)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]

    sphdel = np.array(grp1.sph)
    ampdel = np.array(grp1.width)
    ampsdel = np.array(grp1.widths)
    sphdel = np.delete(sphdel, [14, 16])
    ampdel = np.delete(ampdel, [14, 16])
    ampsdel = np.delete(ampsdel, [14, 16])
    col2 = stats.pearsonr(sphdel, ampdel)
    coef2 = 'C. : ' + '%.4f' % col2[0]
    pval2 = ' P. : ' + '%.4f' % col2[1]
    p0 = [1, -0.01]
    mod1 = Model(xyfitmodel)
    lmr1 = mod1.fit(ampdel, x=sphdel,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(ampsdel)))
    res1 = param_value(lmr1)
    inc1 = '%.6f' % res1.p[0]
    slp1 = '%.6f' % res1.p[1]
    ax.plot(grp1.sph, lmr0.best_fit, color='red')
    ax.plot(sphdel, lmr1.best_fit, color='blue')

    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.75, coef2 + pval2, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.70, inc1+' + '+slp1+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.sph, group.numax,
                           yerr=group.numaxs, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$S_{ph}$')
        ax[ii[i]].set_ylabel(r'$\nu_{max}$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.sph, group.numax)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.numax, x=group.sph,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.numaxs)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.sph, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.sph, group.numax,
                    yerr=group.numaxs, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$S_{ph}$')
        ax.set_ylabel(r'$\nu_{max}$')

        i += 1
    ax.legend(fontsize=8)
    col1 = stats.pearsonr(grp1.sph, grp1.numax)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, 0.1]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.numax, x=grp1.sph,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.numaxs)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    sphdel = np.array(grp1.sph)
    ampdel = np.array(grp1.numax)
    ampsdel = np.array(grp1.numaxs)
    sphdel = np.delete(sphdel, [14, 16])
    ampdel = np.delete(ampdel, [14, 16])
    ampsdel = np.delete(ampsdel, [14, 16])
    col2 = stats.pearsonr(sphdel, ampdel)
    coef2 = 'C. : ' + '%.4f' % col2[0]
    pval2 = ' P. : ' + '%.4f' % col2[1]
    p0 = [1, -0.01]
    mod1 = Model(xyfitmodel)
    lmr1 = mod1.fit(ampdel, x=sphdel,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(ampsdel)))
    res1 = param_value(lmr1)
    inc1 = '%.6f' % res1.p[0]
    slp1 = '%.6f' % res1.p[1]
    ax.plot(grp1.sph, lmr0.best_fit, color='red')
    ax.plot(sphdel, lmr1.best_fit, color='blue')

    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.75, coef2 + pval2, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.70, inc1+' + '+slp1+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)

    pdf.savefig()
    plt.close()

    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(- +)', '3(- -)', '4(+ -)']
    clr = ['r', 'g', 'b', 'y']
    sym = ['o', 'X', '*', 's']
    # sym = [''
    # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    # cmap = plt.cm.Greys
    # cmap = plt.cm.get_cmap('autumn_r', 5)
    # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=1000, vmax=4200)
    fig = plt.figure(figsize=(10, 10))
    i = 0
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[1])
    ax0 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.numax), max(grp1.numax))
    for name, group in groups:
        s0 = ax1.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.numax, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    ax3.hist(grp1.Teff, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.Teff), max(grp1.Teff)))
    ax0.hist(grp1.logg, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.logg), max(grp1.logg)), orientation=u'horizontal')
    cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, cax=cbaxes, label=r'$\nu_{max}$')
    ax1.legend(fontsize=8)
    ax1.set_xlim(6750, 5250)
    ax3.set_xlim(6750, 5250)
    ax3.set_ylim(10, 0)
    ax0.set_ylim(4.6, 3.9)
    ax0.set_xlim(8, 0)
    ax1.set_ylim(4.6, 3.9)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax0.set_ylabel(r'$log(g)$')
    ax0.set_xlabel(r'Number')
    ax3.set_xlabel(r'$T_{eff}$')
    ax3.set_ylabel(r'Number')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2)
    # gs.update(wspace=0.0, hspace=0.0)
    ax0 = plt.subplot(gs[0])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.amp), max(grp1.amp))
    i = 0
    for name, group in groups:
        s0 = ax0.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.amp, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    # ax3.hist(grp1.Teff, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.Teff), max(grp1.Teff)))
    # ax0.hist(grp1.logg, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.logg), max(grp1.logg)), orientation=u'horizontal')
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, label=r'$A$')
    ax0.legend(fontsize=8)
    ax0.set_xlim(6750, 5250)
    # ax3.set_xlim(6750, 5250)
    # ax3.set_ylim(10, 0)
    # ax0.set_ylim(4.6, 3.9)
    # ax0.set_xlim(8, 0)
    ax0.set_ylim(4.6, 3.9)
    # ax1.set_xticklabels([])
    # ax1.set_yticklabels([])
    ax0.set_ylabel(r'$log(g)$')
    # ax0.set_xlabel(r'Number')
    ax0.set_xlabel(r'$T_{eff}$')
    # ax3.set_ylabel(r'Number')
    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(10, 10))
    # gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.35, hspace=0.25)
    ax1 = plt.subplot(gs[1])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.width), 900)
    i = 0
    for name, group in groups:
        s0 = ax1.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.width, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    # ax3.hist(grp1.Teff, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.Teff), max(grp1.Teff)))
    # ax0.hist(grp1.logg, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.logg), max(grp1.logg)), orientation=u'horizontal')
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, label=r'$\delta \nu_{env}$')
    ax1.legend(fontsize=8)
    ax1.set_xlim(6750, 5250)
    # ax3.set_xlim(6750, 5250)
    # ax3.set_ylim(10, 0)
    # ax0.set_ylim(4.6, 3.9)
    # ax0.set_xlim(8, 0)
    ax1.set_ylim(4.6, 3.9)
    # ax1.set_xticklabels([])
    # ax1.set_yticklabels([])
    ax1.set_ylabel(r'$log(g)$')
    # ax0.set_xlabel(r'Number')
    ax1.set_xlabel(r'$T_{eff}$')
    # ax3.set_ylabel(r'Number')
    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(7, 7))
    # gs = gridspec.GridSpec(1, 1)
    # gs.update(wspace=0.0, hspace=0.0)
    ax2 = plt.subplot(gs[2])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.FeH), max(grp1.FeH))
    i = 0
    for name, group in groups:
        s0 = ax2.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.FeH, cmap='autumn_r', norm=norm)
        i += 1
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, label=r'$[Fe/H]$')  # cax=cbaxes,
    ax2.legend(fontsize=8)
    ax2.set_xlim(6750, 5250)
    ax2.set_ylim(4.6, 3.9)
    ax2.set_ylabel(r'$log(g)$')
    ax2.set_xlabel(r'$T_{eff}$')
    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(7, 7))
    # gs = gridspec.GridSpec(1, 1)
    # gs.update(wspace=0.0, hspace=0.0)
    ax3 = plt.subplot(gs[3])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.sph), 400)
    i = 0
    for name, group in groups:
        s3 = ax3.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.sph, cmap='autumn_r', norm=norm)

        i += 1
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s3, label=r'S$_{\rm ph}$')  # cax=cbaxes,
    # eight = mlines.Line2D([0], [0], color='k')

    ax3.legend(fontsize=8)
    ax3.set_xlim(6750, 5250)
    ax3.set_ylim(4.6, 3.9)
    ax3.set_ylabel(r'$log(g)$')
    ax3.set_xlabel(r'$T_{eff}$')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    i = 0
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[1])
    ax0 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.numax), max(grp1.numax))
    for name, group in groups:
        s0 = ax1.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.numax, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    ax3.hist(grp1.prot, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.prot), max(grp1.prot)))
    ax0.hist(grp1.FeH, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.FeH), max(grp1.FeH)), orientation=u'horizontal')
    cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, cax=cbaxes, label=r'$\nu_{max}$')
    ax1.legend(fontsize=8)
    # ax1.set_xlim(6750, 5250)
    ax0.set_ylim(-0.6, 0.4)
    ax1.set_ylim(-0.6, 0.4)
    ax3.set_ylim(10, 0)
    ax3.set_xlim(0, 35)
    ax1.set_xlim(0, 35)
    ax0.set_xlim(8, 0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax0.set_ylabel(r'$[Fe/H]$')
    ax0.set_xlabel(r'Number')
    ax3.set_xlabel(r'$P_{rot}$')
    ax3.set_ylabel(r'Number')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    i = 0
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.35, hspace=0.25)
    ax0 = plt.subplot(gs[0])
    norm = mcolors.Normalize(min(grp1.amp), max(grp1.amp))
    for name, group in groups:
        s0 = ax0.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.amp, cmap='autumn_r', norm=norm)
        i += 1
    fig.colorbar(s0, label=r'$A$')
    ax0.legend(fontsize=8)
    ax0.set_ylim(-0.6, 0.4)
    ax0.set_xlim(0, 35)
    ax0.set_ylabel(r'$[Fe/H]$')
    ax0.set_xlabel(r'$P_{rot}$')

    i = 0
    ax1 = plt.subplot(gs[1])
    norm = mcolors.Normalize(min(grp1.width), 900)
    for name, group in groups:
        s0 = ax1.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.width, cmap='autumn_r', norm=norm)
        i += 1
    fig.colorbar(s0, label=r'$\delta \nu_{env}$')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.6, 0.4)
    ax1.set_xlim(0, 35)
    ax1.set_ylabel(r'$[Fe/H]$')
    ax1.set_xlabel(r'$P_{rot}$')

    i = 0
    ax2 = plt.subplot(gs[2])
    norm = mcolors.Normalize(min(grp1.sph), 400)
    for name, group in groups:
        s0 = ax2.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.sph, cmap='autumn_r', norm=norm)
        i += 1
    fig.colorbar(s0, label=r'$S_{ph}$')
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.6, 0.4)
    ax2.set_xlim(0, 35)
    ax2.set_ylabel(r'$[Fe/H]$')
    ax2.set_xlabel(r'$P_{rot}$')
    pdf.savefig()
    plt.close()
#################################################
#################################################
    groups = grp1.groupby('dd')
    i = 0
    ii = [1, 0, 2, 3]
    name0 = ['SS_1(++)', 'SS_2(\N{MINUS SIGN}+)',
             'SS_3(\N{MINUS SIGN}\N{MINUS SIGN})', 'SS_4(+\N{MINUS SIGN})']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.numax, group.amp, xerr=group.numaxs,
                           yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax[ii[i]].set_ylabel(r'$A$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.numax, group.amp)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.amp, x=group.numax,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.amps)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.numax, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.numax, group.amp, xerr=group.numaxs,
                    yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax.set_ylabel(r'$A$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.numax, grp1.amp)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, -0.00001]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.amp, x=grp1.numax,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.amps)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    ax.plot(grp1.numax, lmr0.best_fit, color='red')
    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.numax, group.width, xerr=group.numaxs,
                           yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax[ii[i]].set_ylabel(r'$\delta \nu_{env}$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.numax, group.width)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, 1]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.width, x=group.numax,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.widths)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.numax, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.numax, group.width, xerr=group.numaxs,
                    yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$\nu_{max} (\mu$Hz)')
        ax.set_ylabel(r'$\delta \nu_{env}$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.numax, grp1.width)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, 1]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.width, x=grp1.numax,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.widths)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    ax.plot(grp1.numax, lmr0.best_fit, color='red')
    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.width, group.amp, xerr=group.widths,
                           yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$\delta \nu_{env}$')
        ax[ii[i]].set_ylabel(r'$A$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.width, group.amp)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.amp, x=group.width,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.amps)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.width, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.width, group.amp, xerr=group.widths,
                    yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$\delta \nu_{env}$')
        ax.set_ylabel(r'$A$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.width, grp1.amp)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, -0.00001]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.amp, x=grp1.width,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.amps)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    ax.plot(grp1.width, lmr0.best_fit, color='red')
    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.sph, group.amp,
                           yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$S_{ph}$')
        ax[ii[i]].set_ylabel(r'$A$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.sph, group.amp)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.amp, x=group.sph,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.amps)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.sph, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.sph, group.amp,
                    yerr=group.amps, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$S_{ph}$')
        ax.set_ylabel(r'$A$')

        i += 1
    ax.legend(fontsize=8)

    col1 = stats.pearsonr(grp1.sph, grp1.amp)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, -0.00001]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.amp, x=grp1.sph,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.amps)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]

    sphdel = np.array(grp1.sph)
    ampdel = np.array(grp1.amp)
    ampsdel = np.array(grp1.amps)
    sphdel = np.delete(sphdel, [14, 16])
    ampdel = np.delete(ampdel, [14, 16])
    ampsdel = np.delete(ampsdel, [14, 16])
    col2 = stats.pearsonr(sphdel, ampdel)
    coef2 = 'C. : ' + '%.4f' % col2[0]
    pval2 = ' P. : ' + '%.4f' % col2[1]
    p0 = [1, -0.01]
    mod1 = Model(xyfitmodel)
    lmr1 = mod1.fit(ampdel, x=sphdel,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(ampsdel)))
    res1 = param_value(lmr1)
    inc1 = '%.6f' % res1.p[0]
    slp1 = '%.6f' % res1.p[1]
    ax.plot(grp1.sph, lmr0.best_fit, color='red')
    ax.plot(sphdel, lmr1.best_fit, color='blue')

    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.75, coef2 + pval2, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.70, inc1+' + '+slp1+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.sph, group.width,
                           yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$S_{ph}$')
        ax[ii[i]].set_ylabel(r'$\delta \nu_{env}$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.sph, group.width)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.width, x=group.sph,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.widths)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.sph, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.sph, group.width,
                    yerr=group.widths, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$S_{ph}$')
        ax.set_ylabel(r'$\delta \nu_{env}$')

        i += 1
    ax.legend(fontsize=8)
    col1 = stats.pearsonr(grp1.sph, grp1.width)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, 0.1]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.width, x=grp1.sph,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.widths)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]

    sphdel = np.array(grp1.sph)
    ampdel = np.array(grp1.width)
    ampsdel = np.array(grp1.widths)
    sphdel = np.delete(sphdel, [14, 16])
    ampdel = np.delete(ampdel, [14, 16])
    ampsdel = np.delete(ampsdel, [14, 16])
    col2 = stats.pearsonr(sphdel, ampdel)
    coef2 = 'C. : ' + '%.4f' % col2[0]
    pval2 = ' P. : ' + '%.4f' % col2[1]
    p0 = [1, -0.01]
    mod1 = Model(xyfitmodel)
    lmr1 = mod1.fit(ampdel, x=sphdel,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(ampsdel)))
    res1 = param_value(lmr1)
    inc1 = '%.6f' % res1.p[0]
    slp1 = '%.6f' % res1.p[1]
    ax.plot(grp1.sph, lmr0.best_fit, color='red')
    ax.plot(sphdel, lmr1.best_fit, color='blue')

    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.75, coef2 + pval2, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.70, inc1+' + '+slp1+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    pdf.savefig()
    plt.close()

    i = 0
    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(-+)', '3(--)', '4(+-)']
    clr = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.ravel()
    for name, group in groups:
        ax[ii[i]].errorbar(group.sph, group.numax,
                           yerr=group.numaxs, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax[ii[i]].set_xlabel(r'$S_{ph}$')
        ax[ii[i]].set_ylabel(r'$\nu_{max}$')
        ax[ii[i]].legend(fontsize=8)
        col1 = stats.pearsonr(group.sph, group.numax)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        p0 = [0.0, -0.00001]
        mod0 = Model(xyfitmodel)
        lmr0 = mod0.fit(group.numax, x=group.sph,
                        a=p0[0], b=p0[1], weights=np.sqrt(1.0/(group.numaxs)))
        res0 = param_value(lmr0)
        inc0 = '%.6f' % res0.p[0]
        slp0 = '%.6f' % res0.p[1]
        ax[ii[i]].plot(group.sph, lmr0.best_fit, color='red')
        ax[ii[i]].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                       transform=ax[ii[i]].transAxes, fontsize=8)
        ax[ii[i]].text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
                       ha='right', va='top', transform=ax[ii[i]].transAxes)
        i += 1
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax = ax.ravel()
    i = 0
    for name, group in groups:
        ax.errorbar(group.sph, group.numax,
                    yerr=group.numaxs, fmt='o', ms=4, label=name0[i], color=clr[i])
        ax.set_xlabel(r'$S_{ph}$')
        ax.set_ylabel(r'$\nu_{max}$')

        i += 1
    ax.legend(fontsize=8)
    col1 = stats.pearsonr(grp1.sph, grp1.numax)
    coef1 = 'C. : ' + '%.4f' % col1[0]
    pval1 = ' P. : ' + '%.4f' % col1[1]
    p0 = [0.0, 0.1]
    mod0 = Model(xyfitmodel)
    lmr0 = mod0.fit(grp1.numax, x=grp1.sph,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(grp1.numaxs)))
    res0 = param_value(lmr0)
    inc0 = '%.6f' % res0.p[0]
    slp0 = '%.6f' % res0.p[1]
    sphdel = np.array(grp1.sph)
    ampdel = np.array(grp1.numax)
    ampsdel = np.array(grp1.numaxs)
    sphdel = np.delete(sphdel, [14, 16])
    ampdel = np.delete(ampdel, [14, 16])
    ampsdel = np.delete(ampsdel, [14, 16])
    col2 = stats.pearsonr(sphdel, ampdel)
    coef2 = 'C. : ' + '%.4f' % col2[0]
    pval2 = ' P. : ' + '%.4f' % col2[1]
    p0 = [1, -0.01]
    mod1 = Model(xyfitmodel)
    lmr1 = mod1.fit(ampdel, x=sphdel,
                    a=p0[0], b=p0[1], weights=np.sqrt(1.0/(ampsdel)))
    res1 = param_value(lmr1)
    inc1 = '%.6f' % res1.p[0]
    slp1 = '%.6f' % res1.p[1]
    ax.plot(grp1.sph, lmr0.best_fit, color='red')
    ax.plot(sphdel, lmr1.best_fit, color='blue')

    ax.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.80, inc0+' + '+slp0+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.75, coef2 + pval2, ha='right', va='top',
            transform=ax.transAxes, fontsize=8)
    ax.text(0.98, 0.70, inc1+' + '+slp1+'x', fontsize=8,
            ha='right', va='top', transform=ax.transAxes)

    pdf.savefig()
    plt.close()

    ii = [1, 0, 2, 3]
    # name0 = ['1(++)', '2(- +)', '3(- -)', '4(+ -)']
    clr = ['r', 'g', 'b', 'y']
    sym = ['o', 'X', '*', 's']
    # sym = [''
    # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    # cmap = plt.cm.Greys
    # cmap = plt.cm.get_cmap('autumn_r', 5)
    # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=1000, vmax=4200)
    fig = plt.figure(figsize=(10, 10))
    i = 0
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[1])
    ax0 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.numax), max(grp1.numax))
    for name, group in groups:
        s0 = ax1.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.numax, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    ax3.hist(grp1.Teff, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.Teff), max(grp1.Teff)))
    ax0.hist(grp1.logg, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.logg), max(grp1.logg)), orientation=u'horizontal')
    cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, cax=cbaxes, label=r'$\nu_{max}$')
    ax1.legend(fontsize=8)
    ax1.set_xlim(6750, 5250)
    ax3.set_xlim(6750, 5250)
    ax3.set_ylim(10, 0)
    ax0.set_ylim(4.6, 3.9)
    ax0.set_xlim(8, 0)
    ax1.set_ylim(4.6, 3.9)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax0.set_ylabel(r'$log(g)$')
    ax0.set_xlabel(r'Number')
    ax3.set_xlabel(r'$T_{eff}$')
    ax3.set_ylabel(r'Number')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2)
    # gs.update(wspace=0.0, hspace=0.0)
    ax0 = plt.subplot(gs[0])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.amp), max(grp1.amp))
    i = 0
    for name, group in groups:
        s0 = ax0.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.amp, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    # ax3.hist(grp1.Teff, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.Teff), max(grp1.Teff)))
    # ax0.hist(grp1.logg, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.logg), max(grp1.logg)), orientation=u'horizontal')
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, label=r'$A$')
    ax0.legend(fontsize=8)
    ax0.set_xlim(6750, 5250)
    # ax3.set_xlim(6750, 5250)
    # ax3.set_ylim(10, 0)
    # ax0.set_ylim(4.6, 3.9)
    # ax0.set_xlim(8, 0)
    ax0.set_ylim(4.6, 3.9)
    # ax1.set_xticklabels([])
    # ax1.set_yticklabels([])
    ax0.set_ylabel(r'$log(g)$')
    # ax0.set_xlabel(r'Number')
    ax0.set_xlabel(r'$T_{eff}$')
    # ax3.set_ylabel(r'Number')
    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(10, 10))
    # gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.35, hspace=0.25)
    ax1 = plt.subplot(gs[1])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.width), 900)
    i = 0
    for name, group in groups:
        s0 = ax1.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.width, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    # ax3.hist(grp1.Teff, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.Teff), max(grp1.Teff)))
    # ax0.hist(grp1.logg, bins=10, density=False,
    #          alpha=0.6, histtype='bar', color='gray', edgecolor='k',
    #          linewidth=1.2, range=(min(grp1.logg), max(grp1.logg)), orientation=u'horizontal')
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, label=r'$\delta \nu_{env}$')
    ax1.legend(fontsize=8)
    ax1.set_xlim(6750, 5250)
    # ax3.set_xlim(6750, 5250)
    # ax3.set_ylim(10, 0)
    # ax0.set_ylim(4.6, 3.9)
    # ax0.set_xlim(8, 0)
    ax1.set_ylim(4.6, 3.9)
    # ax1.set_xticklabels([])
    # ax1.set_yticklabels([])
    ax1.set_ylabel(r'$log(g)$')
    # ax0.set_xlabel(r'Number')
    ax1.set_xlabel(r'$T_{eff}$')
    # ax3.set_ylabel(r'Number')
    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(7, 7))
    # gs = gridspec.GridSpec(1, 1)
    # gs.update(wspace=0.0, hspace=0.0)
    ax2 = plt.subplot(gs[2])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.FeH), max(grp1.FeH))
    i = 0
    for name, group in groups:
        s0 = ax2.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.FeH, cmap='autumn_r', norm=norm)
        i += 1
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, label=r'$[Fe/H]$')  # cax=cbaxes,
    ax2.legend(fontsize=8)
    ax2.set_xlim(6750, 5250)
    ax2.set_ylim(4.6, 3.9)
    ax2.set_ylabel(r'$log(g)$')
    ax2.set_xlabel(r'$T_{eff}$')
    # pdf.savefig()
    # plt.close()

    # fig = plt.figure(figsize=(7, 7))
    # gs = gridspec.GridSpec(1, 1)
    # gs.update(wspace=0.0, hspace=0.0)
    ax3 = plt.subplot(gs[3])
    # ax0 = plt.subplot(gs[0])
    # ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.sph), 400)
    i = 0
    for name, group in groups:
        s3 = ax3.scatter(group.Teff, group.logg,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.sph, cmap='autumn_r', norm=norm)

        i += 1
    # cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s3, label=r'S$_{\rm ph}$')  # cax=cbaxes,
    # eight = mlines.Line2D([0], [0], color='k')

    ax3.legend(fontsize=8)
    ax3.set_xlim(6750, 5250)
    ax3.set_ylim(4.6, 3.9)
    ax3.set_ylabel(r'$log(g)$')
    ax3.set_xlabel(r'$T_{eff}$')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    i = 0
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[1])
    ax0 = plt.subplot(gs[0])
    ax3 = plt.subplot(gs[3])
    norm = mcolors.Normalize(min(grp1.numax), max(grp1.numax))
    for name, group in groups:
        s0 = ax1.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.numax, cmap='autumn_r', norm=norm)
        # facecolor='none', edgecolors='w')  # color=clr[i])
        # s0 = ax.scatter(group.Teff, group.logg, c=group.numax,
        #                 s=50, marker=sym[i], cmap='autumn_r', norm=norm, label=name0[i])
        i += 1
    ax3.hist(grp1.prot, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.prot), max(grp1.prot)))
    ax0.hist(grp1.FeH, bins=10, density=False,
             alpha=0.6, histtype='bar', color='gray', edgecolor='k',
             linewidth=1.2, range=(min(grp1.FeH), max(grp1.FeH)), orientation=u'horizontal')
    cbaxes = fig.add_axes([0.905, 0.504, 0.02, 0.37])
    fig.colorbar(s0, cax=cbaxes, label=r'$\nu_{max}$')
    ax1.legend(fontsize=8)
    # ax1.set_xlim(6750, 5250)
    ax0.set_ylim(-0.6, 0.4)
    ax1.set_ylim(-0.6, 0.4)
    ax3.set_ylim(10, 0)
    ax3.set_xlim(0, 35)
    ax1.set_xlim(0, 35)
    ax0.set_xlim(8, 0)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax0.set_ylabel(r'$[Fe/H]$')
    ax0.set_xlabel(r'Number')
    ax3.set_xlabel(r'$P_{rot}$')
    ax3.set_ylabel(r'Number')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    i = 0
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.35, hspace=0.25)
    ax0 = plt.subplot(gs[0])
    norm = mcolors.Normalize(min(grp1.amp), max(grp1.amp))
    for name, group in groups:
        s0 = ax0.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.amp, cmap='autumn_r', norm=norm)
        i += 1
    fig.colorbar(s0, label=r'$A$')
    ax0.legend(fontsize=8)
    ax0.set_ylim(-0.6, 0.4)
    ax0.set_xlim(0, 35)
    ax0.set_ylabel(r'$[Fe/H]$')
    ax0.set_xlabel(r'$P_{rot}$')

    i = 0
    ax1 = plt.subplot(gs[1])
    norm = mcolors.Normalize(min(grp1.width), 900)
    for name, group in groups:
        s0 = ax1.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.width, cmap='autumn_r', norm=norm)
        i += 1
    fig.colorbar(s0, label=r'$\delta \nu_{env}$')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.6, 0.4)
    ax1.set_xlim(0, 35)
    ax1.set_ylabel(r'$[Fe/H]$')
    ax1.set_xlabel(r'$P_{rot}$')

    i = 0
    ax2 = plt.subplot(gs[2])
    norm = mcolors.Normalize(min(grp1.sph), 400)
    for name, group in groups:
        s0 = ax2.scatter(group.prot, group.FeH,
                         s=50, label=name0[i], marker=sym[i],
                         c=group.sph, cmap='autumn_r', norm=norm)
        i += 1
    fig.colorbar(s0, label=r'$S_{ph}$')
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.6, 0.4)
    ax2.set_xlim(0, 35)
    ax2.set_ylabel(r'$[Fe/H]$')
    ax2.set_xlabel(r'$P_{rot}$')
    pdf.savefig()
    plt.close()
end = time.time()
print('\033[01;31;43m', format(end - start, ".3f"), 's\033[00m')
