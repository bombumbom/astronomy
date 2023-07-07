# 1) calculated amplitude variations
# 2) Full & overlap time series
# t = 200315
# rivsed version = 200315
# for calculated Sph + KEPSEISMIC data (KADACS)
# 01435467
import os
import pandas as pd

import time
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from astropy.timeseries import LombScargle
from lmfit import Minimizer, Parameters, report_fit, Model
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from kbseism import estimate_background, nu_max
from kb import kbreduct  # , binspec
from background.bgm import kbgauss, xyfitglog, kbparameters, param_value, harvey1, xyfitmodel

start = time.time()
# physical paramters
dfp = pd.read_csv('/home/kim/datafold/data/santos2018/santos1.dat')

uHz_conv = 1e-6 * 24. * 60. * 60.

# pth0 = '/media/kim/datafold/data/'
pth0 = '/media/kim/datafold/data/'
pth_s = pth0+'Kepseismic/'
for (path, dir, files) in os.walk(pth_s):
    dir.sort()
    # print(dir)
    break

pth_ls = pth0+'KIC_lightcurve/'

pth_ = '/home/kim/research/amplitude_fs/main10/'
pth_pdf = pth_+'figures/'
pth_rst = pth_+'results/'
nm = dfp.numax

# for k in range(0, len(dir)):
# for k in range(0, 1):
for k in range(1):
    k = 0
    stq = 1

    # print('Directory Number : ', k)
    amp = []
    amps = []
    numax = []
    width = []
    t_sei = []
    df = pd.DataFrame()
    time_l = pd.DataFrame()
    time_s = pd.DataFrame()
    time_d = pd.DataFrame()
    quaa = []
    sph = []
    sphq = []
    t_sph = []
    t_sphq = []

    ##################
    """ Read Files """
    ##################
    print('\n \033[1;31m Staring analysis of ' +
          dir[k]+' // Number (k) : ', k, '\033[1;00m')
    file_list_s = os.listdir(pth_s+dir[k])
    EXTs = '55d_kepler_v1_cor-filt-inp.fits'
    file_s = [file for file in file_list_s if file.endswith(EXTs)]
    file_s.sort()
    print('\n Read KEPSEISMIC files : \n' + file_s[0])
    hdus = fits.open(pth_s+dir[k]+'/'+file_s[0])
    tsph = hdus[1].data.TIME + 2400000 - 2454833
    fsph = hdus[1].data.FLUX

    file_list_ll = os.listdir(pth_ls+dir[k])
    file_list_ll.sort()
    EXTll = 'long.fits'
    file_ll = [file for file in file_list_ll if file.endswith(EXTll)]
    file_ll.sort()
    for l in range(len(file_ll)):
        qua = file_ll[l].split('_')
        hdul = fits.open(pth_ls + dir[k]+'/' + file_ll[l])
        tl = hdul[1].data.TIME
        tse = pd.DataFrame({'start': [min(tl)], 'end': [
                           max(tl)], 'quarter': [qua[1]]})
        # for quarter time (start and end)
        time_l = pd.concat([time_l, tse], ignore_index=True)

    ts = tsph[0]
    te = ts + 5*dfp.prot[k]
    td = 5*dfp.prot[k]/8

    while te <= tsph[-1]:
        whe = np.where((tsph > ts) & (tsph <= te))
        sph = np.append(sph, np.nanstd(fsph[whe]))
        t_sph = np.append(t_sph, np.median(tsph[whe]))
        ts = ts + td
        te = ts + 5 * dfp.prot[k]

    file_list_ls = os.listdir(pth_ls+dir[k])
    EXTls = 'short.fits'
    file_ls = [file for file in file_list_ls if file.endswith(EXTls)]
    file_ls.sort()
    print('\n Read KIC lightcurve files : ', len(file_ls))
    ########################################
    """Read Fits files & make lightcurves"""
    ########################################
    fname = pth_pdf+'kic' + dir[k] + '_tot.pdf'
    print('pdf file : ' + fname)
    for q in range(len(file_ls)):
        qua = file_ls[q].split('_')
        quaa = np.append(quaa, qua[1])
        hdu = fits.open(pth_ls + dir[k] + '/' + file_ls[q])
        t0 = hdu[1].data.TIME
        f0 = hdu[1].data.FLUX
        quarter = [qua[1] for i in range(len(t0))]
        df1 = pd.DataFrame({'time': t0, 'flux': f0, 'quarter': quarter})
        df = pd.concat([df, df1], ignore_index=True)  # Time series of quarters
        tse = pd.DataFrame({'start': [min(t0)], 'end': [
                           max(t0)], 'quarter': [qua[1]]})
        # for quarter time (start and end)
        time_s = pd.concat([time_s, tse], ignore_index=True)
    ################################################
    """Analysis Quarter"""
    ################################################
    df.flux = df.flux * 1e6

    file_list_lt = os.listdir(pth_ls + dir[k])
    EXTlt = 'tot.fits'
    file_lt = [file for file in file_list_lt if file.endswith(EXTlt)]
    file_lt.sort()
    print('\n Read KIC total lightcurve files')
    with PdfPages(fname) as pdf:
        hdu = fits.open(pth_ls + dir[k] + '/' + file_lt[0])
        t0 = hdu[1].data.TIME
        f0 = hdu[1].data.FLUX * 1e6
        step = 0

        # starting the continue quarters
        print(f'\n \033[1;31mStarting quarters : {quaa[stq]}\033[1:00m')
        whe_q = np.where((t0 >= time_s.start[stq]))
        t1 = t0[whe_q]
        f1 = f0[whe_q]
        # length 90 days overlap 45 days
        ov = 45
        length = 90
        lc_s = t1[0]
        lc_e = lc_s+length
        """Start the Analysis"""
        while lc_e <= max(t1):
            step += 1
            print(f'\n \033[1;32m Step : {step} \n '
                  f'day: {lc_s:.3f}d ~ {lc_e:.3f}d \n '
                  f'length & overlap: {lc_e-lc_s} & {ov} day \033[00m')

            whe_l = np.where((t1 >= lc_s) & (t1 <= lc_e))
            tse = pd.DataFrame({'start': [lc_s], 'end': [lc_e]})
            time_d = pd.concat([time_d, tse], ignore_index=True)

            fig, ax = plt.subplots(3, 1, figsize=(10, 10))
            plt.suptitle(f'day: {lc_s:.3f}d ~ {lc_e:.3f}d // {lc_e-lc_s} day')
            lc_s = lc_s + ov
            lc_e = lc_s + length

            time_ls = t1[whe_l]
            flux_ls = f1[whe_l]
            num = len(time_ls)
            t_sei = np.append(t_sei, np.median(time_ls))
            freq = np.linspace(1, 8000, 50000)
            freq0 = freq * uHz_conv
            model = LombScargle(time_ls, flux_ls)
            power = model.power(freq0, method='fast', normalization='psd')
            power *= uHz_conv / num
            bin0 = freq[10] - freq[9]
            wid = int(5 / bin0)
            power_s = kbreduct.smoothing(power, wid)
            bg = estimate_background(freq, power_s, width=0.04)

            ax[0].plot(time_ls, flux_ls)
            ax[0].set_ylabel('Flux')
            ax[0].set_xlabel('Time (days)')

            ax[1].plot(freq, power_s)
            ax[1].plot(freq, bg, c='red', linewidth=3, alpha=0.7)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            ax[1].set_xlim(500, 5000)
            ax[1].set_xlabel(r'freqeuncy ($\mu Hz$)')
            ax[1].set_ylabel(r'PSD')

            power_b = power_s / bg
            whef = np.where((freq > 500.) & (freq < 6000.))
            freq_l = freq[whef]
            power_l = power_b[whef]
            pg0 = [1, nm[k], 0.26*nm[k]**0.77, 1.]
            pg = kbparameters(pg0)

            minner = Minimizer(kbgauss, pg, fcn_args=(freq_l, power_l))
            bg_res = minner.minimize()
            report_fit(bg_res)
            res_g = kbgauss(bg_res.params.valuesdict(),
                            freq_l, power_l, value=1)
            resg = param_value(bg_res)
            ax[2].plot(freq, power_b)
            ax[2].plot(freq_l, res_g.fv, 'grey',
                       label='total model', linewidth=3)
            ax[2].plot(freq_l, res_g.pex, ':', c='grey',
                       label='p-mode excess', linewidth=2)
            ax[2].plot(freq_l, res_g.whn, '--', c='grey',
                       label='white noise', linewidth=2)
            ax[2].set_xscale('log')
            ax[2].set_yscale('log')
            ax[2].set_xlim(500, 5000)
            ax[2].set_ylim(1e-1, 8)
            ax[2].set_xlabel(r'freqeuncy ($\mu Hz$)')
            ax[2].set_ylabel('Power (arbitary unit)')
            ax[2].legend()
            pdf.savefig()
            plt.close()
            amp = np.append(amp, resg.p[0])
            amps = np.append(amps, resg.st[0])
            numax = np.append(numax, resg.p[1])
            width = np.append(width, resg.p[2])

        for s in range(len(time_d)):
            whe_s = np.where((t_sph >= time_d.start[s]) & (
                t_sph <= time_d.end[s]))
            tss = np.mean(t_sph[whe_s])
            sphh = np.mean(sph[whe_s])
            t_sphq = np.append(t_sphq, tss)
            sphq = np.append(sphq, sphh)

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10))
        plt.suptitle('Sph vs p-mode hump')
        ax0.plot(t_sph, sph, color='blue')
        ax0a = ax0.twinx()
        ax0a.plot(t_sei, amp, '-o', color='red', label='amp')
        ax0a.legend()

        ax1.plot(t_sph, sph, color='blue')
        ax1a = ax1.twinx()
        ax1a.plot(t_sei, numax, '-o', color='red', label='nu_max')
        ax1a.legend()

        ax2.plot(t_sph, sph, color='blue')
        ax2a = ax2.twinx()
        ax2a.plot(t_sei, width, '-o', color='red', label='width')
        ax2a.legend()
        pdf.savefig()
        plt.close()

        ##################
        ##################
        col0 = stats.pearsonr(sphq, amp)
        coef0 = 'C. : ' + '%.4f' % col0[0]
        pval0 = ' P. : ' + '%.4f' % col0[1]
        col1 = stats.pearsonr(sphq, numax)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        col2 = stats.pearsonr(sphq, width)
        coef2 = 'C. : ' + '%.4f' % col2[0]
        pval2 = ' P. : ' + '%.4f' % col2[1]
        #
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10))
        plt.suptitle('Sph vs p-mode hump')
        ax0.plot(t_sphq, sphq, '-o', color='blue')
        ax0a = ax0.twinx()
        ax0a.errorbar(t_sei, amp, fmt='-o', yerr=amps,
                      color='red', label='amp')
        ax0a.legend()
        ax0a.text(0.98, 0.85, coef0 + pval0, ha='right', va='top',
                  transform=ax0a.transAxes, fontsize=8)

        ax1.plot(t_sphq, sphq, '-o', color='blue')
        ax1a = ax1.twinx()
        ax1a.plot(t_sei, numax, '-o', color='red', label='nu_max')
        ax1a.legend()
        ax1a.text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                  transform=ax1a.transAxes, fontsize=8)

        ax2.plot(t_sphq, sphq, '-o', color='blue')
        ax2a = ax2.twinx()
        ax2a.plot(t_sei, width, '-o', color='red', label='width')
        ax2a.legend()
        ax2.text(0.98, 0.85, coef2 + pval2, ha='right', va='top',
                 transform=ax2a.transAxes, fontsize=8)
        pdf.savefig()
        plt.close()
        #

        amod = Model(xyfitmodel)
        p0 = [0.5, -0.01]
        lmresuta = amod.fit(amp, x=sphq,
                            a=p0[0], b=p0[1], weights=np.sqrt(1.0/(amps)))
        modela = param_value(lmresuta)

        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(sphq, amp, 'o', label='A')
        ax[0].plot(sphq, lmresuta.best_fit, color='red')
        ax[0].set_xlabel(r'$S_{ph}$')
        ax[0].set_ylabel('ampitude')
        ax[0].text(0.98, 0.85, coef0 + pval0, ha='right', va='top',
                   transform=ax[0].transAxes, fontsize=8)

        ax[1].plot(sphq, numax, 'o', label=r'$\nu_{max}$')
        ax[1].set_xlabel(r'$S_{ph}$')
        ax[1].set_ylabel(r'$\nu_{max}$')
        ax[1].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                   transform=ax[1].transAxes, fontsize=8)

        ax[2].plot(sphq, width, 'o', label='width')
        ax[2].set_xlabel(r'$S_{ph}$')
        ax[2].set_ylabel('width')
        ax[2].text(0.98, 0.85, coef2 + pval2, ha='right', va='top',
                   transform=ax[2].transAxes, fontsize=8)
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        col0 = stats.pearsonr(width, amp)
        coef0 = 'C. : ' + '%.4f' % col0[0]
        pval0 = ' P. : ' + '%.4f' % col0[1]
        col1 = stats.pearsonr(width, numax)
        coef1 = 'C. : ' + '%.4f' % col1[0]
        pval1 = ' P. : ' + '%.4f' % col1[1]
        col2 = stats.pearsonr(numax, amp)
        coef2 = 'C. : ' + '%.4f' % col2[0]
        pval2 = ' P. : ' + '%.4f' % col2[1]

        ax[0].plot(width, amp, 'o', label='A')
        ax[0].set_xlabel('width')
        ax[0].set_ylabel('ampitude')
        ax[0].text(0.98, 0.85, coef0 + pval0, ha='right', va='top',
                   transform=ax[0].transAxes, fontsize=8)

        ax[1].plot(width, numax, 'o', label=r'$\nu_{max}$')
        ax[1].set_xlabel('width')
        ax[1].set_ylabel(r'$\nu_{max}$')
        ax[1].text(0.98, 0.85, coef1 + pval1, ha='right', va='top',
                   transform=ax[1].transAxes, fontsize=8)

        ax[2].plot(numax, amp, 'o', label='width')
        ax[2].set_xlabel(r'$\nu_{max}$')
        ax[2].set_ylabel('amplitude')
        ax[2].text(0.98, 0.85, coef2 + pval2, ha='right', va='top',
                   transform=ax[2].transAxes, fontsize=8)
        pdf.savefig()
        plt.close()

        numt = len(t1)
        freq_t0 = np.linspace(1, 8000, 100000)
        freqt = freq_t0 * uHz_conv
        model = LombScargle(t1, f1)
        power_t0 = model.power(freqt, method='fast', normalization='psd')
        power_t0 *= uHz_conv / numt
        bin0 = freq_t0[10]-freq_t0[9]
        wid = int(5/bin0)
        power_ts = kbreduct.smoothing(power_t0, wid)
        bgt = estimate_background(freq_t0, power_ts, width=0.04)
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        plt.suptitle('Total light curves')
        ax[0].plot(t1[::10], f1[::10])
        ax[1].plot(freq_t0, power_ts)
        ax[1].plot(freq_t0, bgt, c='red', linewidth=3, alpha=0.7)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlim(500, 5000)

        power_tb = power_ts / bgt
        whef = np.where((freq_t0 > 500.) & (freq_t0 < 6000.))
        freq_tl = freq_t0[whef]
        power_tl = power_tb[whef]
        pg0 = [1, nm[k], 0.26 * nm[k] ** 0.77, 1.]
        pg = kbparameters(pg0)
        minner = Minimizer(kbgauss, pg, fcn_args=(freq_tl, power_tl))
        bg_res = minner.minimize()
        report_fit(bg_res)
        res_g = kbgauss(bg_res.params.valuesdict(),
                        freq_tl, power_tl, value=1)
        resg = param_value(bg_res)
        ax[2].plot(freq_t0, power_tb)
        ax[2].plot(freq_tl, res_g.fv, 'grey', label='total model', linewidth=3)
        ax[2].plot(freq_tl, res_g.pex, ':', c='grey',
                   label='p-mode excess', linewidth=2)
        ax[2].plot(freq_tl, res_g.whn, '--', c='grey',
                   label='white noise', linewidth=2)
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].set_xlim(500, 5000)
        ax[2].set_ylim(1e-1, 8)
        ax[2].legend()
        pdf.savefig()
        plt.close()

    ampv = max(amp) - min(amp)
    ampr = ampv/max(amp)

    result = pd.DataFrame({'amp': [resg.p[0]], 'amps': [resg.st[0]],
                           'numax': [resg.p[1]], 'numaxs': [resg.st[1]],
                           'width': [resg.p[2]], 'widths': [resg.st[2]],
                           'sph': np.mean(sph), 'ampv': [ampv], 'ampr': [ampr],
                           'amax': [max(amp)], 'amin': [min(amp)],
                           'sphmax': [max(sph)], 'sphmin': [min(sph)],
                           'slope': [modela.p[1]], 'slopes': [modela.st[1]]})
    f = open(pth_rst+dir[k]+'.csv', 'w')
    f.write('# KIC '+dir[k]+'\n')
    f.write('# quarters :' + ','.join(quaa[stq:])+'\n')
    f.write(f'# length : {length}\n')
    f.write(f'# overlap : {ov}\n')
    # result.to_csv(pth_rst+dir[k]+'.csv',float_format='%.5f')
    result.to_csv(f, float_format='%.5f', index=False)
    f.close()

end = time.time()
print('\033[01;31;43m', format(end - start, ".3f"), 's\033[00m')
