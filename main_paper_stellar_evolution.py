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
import mesa_reader as mr


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
sdel = 135.1
steff = 5770
pthm = '/home/kim/my_mesa/premstoms/z001/1M_z0014/LOGS/history.data'
h1m = mr.MesaData(pthm)
etem1 = 10**h1m.log_Teff
lum1 = 10**h1m.log_L
Ds1 = ((etem1/steff)**3)/((lum1)**0.75)*sdel

pthm = '/home/kim/my_mesa/z001/1.1M_z001/LOGS/history.data'
h2m = mr.MesaData(pthm)
etem2 = 10**h2m.log_Teff
lum2 = 10**h2m.log_L
Ds2 = ((etem2/steff)**3)/((lum2)**0.75)*sdel

pthm = '/home/kim/my_mesa/z001/1.2M_z001/LOGS/history.data'
h3m = mr.MesaData(pthm)
etem3 = 10**h3m.log_Teff
lum3 = 10**h3m.log_L
Ds3 = ((etem3/steff)**3)/((lum3)**0.75)*sdel

pthm = '/home/kim/my_mesa/z001/1.3M_z001/LOGS/history.data'
h4m = mr.MesaData(pthm)
etem4 = 10**h4m.log_Teff
lum4 = 10**h4m.log_L
Ds4 = ((etem4/steff)**3)/((lum4)**0.75)*sdel

pthm = '/home/kim/my_mesa/z001/0.9M_z001/LOGS/history.data'
h5m = mr.MesaData(pthm)
etem5 = 10**h5m.log_Teff
lum5 = 10**h5m.log_L
Ds5 = ((etem5/steff)**3)/((lum5)**0.75)*sdel

pthm = '/home/kim/my_mesa/z001/0.8M_z001/LOGS/history.data'
h6m = mr.MesaData(pthm)
etem6 = 10**h6m.log_Teff
lum6 = 10**h6m.log_L
Ds6 = ((etem6/steff)**3)/((lum6)**0.75)*sdel

pthm = '/home/kim/my_mesa/premstoms/z001/1.4M_z0014/LOGS/history.data'
h7m = mr.MesaData(pthm)
etem7 = 10**h7m.log_Teff
lum7 = 10**h7m.log_L
Ds7 = ((etem7/steff)**3)/((lum7)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0015/1M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem30 = 10**hm.log_Teff
lum30 = 10**hm.log_L
Ds30 = ((etem30/steff)**3)/((lum30)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0015/0.9M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem31 = 10**hm.log_Teff
lum31 = 10**hm.log_L
Ds31 = ((etem31/steff)**3)/((lum31)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0015/1.1M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem32 = 10**hm.log_Teff
lum32 = 10**hm.log_L
Ds32 = ((etem32/steff)**3)/((lum32)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0015/1.2M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem33 = 10**hm.log_Teff
lum33 = 10**hm.log_L
Ds33 = ((etem33/steff)**3)/((lum33)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0015/1.3M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem34 = 10**hm.log_Teff
lum34 = 10**hm.log_L
Ds34 = ((etem34/steff)**3)/((lum34)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0015/1.4M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem35 = 10**hm.log_Teff
lum35 = 10**hm.log_L
Ds35 = ((etem35/steff)**3)/((lum35)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0005/0.9M_z001/LOGS/history.data'
hm = mr.MesaData(pthm)
etem41 = 10**hm.log_Teff
lum41 = 10**hm.log_L
Ds41 = ((etem41/steff)**3)/((lum41)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0005/1M_z001/LOGS/history.data'
hm = mr.MesaData(pthm)
etem40 = 10**hm.log_Teff
lum40 = 10**hm.log_L
Ds40 = ((etem40/steff)**3)/((lum40)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0005/1.1M_z001/LOGS/history.data'
hm = mr.MesaData(pthm)
etem42 = 10**hm.log_Teff
lum42 = 10**hm.log_L
Ds42 = ((etem42/steff)**3)/((lum42)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0005/0.8M_z001/LOGS/history.data'
hm = mr.MesaData(pthm)
etem43 = 10**hm.log_Teff
lum43 = 10**hm.log_L
Ds43 = ((etem43/steff)**3)/((lum43)**0.75)*sdel

pthm = '/home/kim/my_mesa/z0005/0.7M_z001/LOGS/history.data'
hm = mr.MesaData(pthm)
etem44 = 10**hm.log_Teff
lum44 = 10**hm.log_L
Ds44 = ((etem44/steff)**3)/((lum44)**0.75)*sdel


pthm = '/home/kim/my_mesa/premstoms/z0015/1M_z0015/LOGS/history.data'
hm = mr.MesaData(pthm)
etem555 = 10**hm.log_Teff
lum555 = 10**hm.log_L
Ds555 = ((etem555/steff)**3)/((lum555)**0.75)*sdel

pthm = '/home/kim/history.data'
hm133 = mr.MesaData(pthm)
etem133 = 10**hm133.log_Teff
lum133 = 10**hm133.log_L
Ds133 = ((etem133/steff)**3)/((lum133)**0.75)*sdel

pthm = '/home/kim/history2.data'
hm1332 = mr.MesaData(pthm)
etem1332 = 10**hm1332.log_Teff
lum1332 = 10**hm1332.log_L
Ds1332 = ((etem1332/steff)**3)/((lum1332)**0.75)*sdel

pthm = '/home/kim/my_mesa/premstoms/z0014/1M_z0014/LOGS/history.data'
hm = mr.MesaData(pthm)
etems1 = 10**hm.log_Teff
lums1 = 10**hm.log_L
Dss1 = ((etems1/steff)**3)/((lums1)**0.75)*sdel
age1 = hm.star_age

pthm = '/home/kim/my_mesa/premstoms/z0014/1.2M_z0014/LOGS/history.data'
hm = mr.MesaData(pthm)
etems12 = 10**hm.log_Teff
lums12 = 10**hm.log_L
Dss12 = ((etems12/steff)**3)/((lums12)**0.75)*sdel
age12 = hm.star_age

pthm = '/home/kim/my_mesa/premstoms/z0014/1.4M_z0014/LOGS/history.data'
hm = mr.MesaData(pthm)
etems14 = 10**hm.log_Teff
lums14 = 10**hm.log_L
Dss14 = ((etems14/steff)**3)/((lums14)**0.75)*sdel
age14 = hm.star_age

pthm = '/home/kim/my_mesa/premstoms/z0008/1M_z0008/LOGS/history.data'
hm = mr.MesaData(pthm)
etems081 = 10**hm.log_Teff
lums081 = 10**hm.log_L
Dss081 = ((etems081/steff)**3)/((lums081)**0.75)*sdel
age081 = hm.star_age

pthm = '/home/kim/my_mesa/premstoms/z0008/1.2M_z0008/LOGS/history.data'
hm = mr.MesaData(pthm)
etems082 = 10**hm.log_Teff
lums082 = 10**hm.log_L
Dss082 = ((etems082/steff)**3)/((lums082)**0.75)*sdel
age082 = hm.star_age

pthm = '/home/kim/my_mesa/premstoms/z0008/1.4M_z0008/LOGS/history.data'
hm = mr.MesaData(pthm)
etems083 = 10**hm.log_Teff
lums083 = 10**hm.log_L
Dss083 = ((etems083/steff)**3)/((lums083)**0.75)*sdel
age083 = hm.star_age

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
fshifts = []
tsei = []
sph = []
sphq = []
t_sph = []
t_sphq = []
hdu = fits.open(pth+file_list[0])
t0 = hdu[3].data.DATEBARREGTT
f0 = hdu[3].data.FLUXBARREG
t_min = min(t0)
fname = 'etc.pdf'
c = 0
res_xx = pd.DataFrame()
psx = pd.DataFrame()
ps0 = pd.DataFrame()
ps = pd.DataFrame()
ps1 = pd.DataFrame()
ps2 = pd.DataFrame()
df = pd.DataFrame()
tt = pd.DataFrame()
power_00 = []
with PdfPages(fname) as pdf:
    file_list.sort()
    for f in range(1, 2):
        hdu = fits.open(pth+file_list[f])
        t0 = hdu[3].data.DATEBARREGTT
        f0 = hdu[3].data.FLUXBARREG
        ttmin = t0[0]
        t0 = t0 - ttmin
        ts = t0*1.0
        fs = f0*1.0
        fs = kbreduct.detrend(ts, fs, 2, 3)
        fs = kbreduct.kboutlier(fs, 3)

        hrot = 3.45
        ovs = 5*hrot/8
        lcs_sph = min(ts)
        lce_sph = lcs_sph + 5*hrot
        print(r'calculating the Sph')
        while lce_sph <= ts[-1]:
            whsph = np.where((ts >= lcs_sph) & (ts <= lce_sph))
            sph = np.append(sph, np.nanstd(fs[whsph]))
            t_sph = np.append(t_sph, np.mean(ts[whsph]))
            lcs_sph = lcs_sph + ovs
            lce_sph = lcs_sph + 5*hrot

        lc_s = min(t0)
        ov = 10
        length = 40
        lc_e = lc_s + length
        lcee = lc_e+1
        while lc_e <= max(t0):
            whe = np.where((t0 >= lc_s) & (t0 <= lc_e))
            whs = np.where((t_sph >= lc_s) & (t_sph <= lc_e))
            t_sphq = np.append(t_sphq, np.mean(t_sph[whs])-t_min)
            sphq = np.append(sphq, np.mean(sph[whs]))
            t1 = t0[whe]
            f1 = f0[whe]

            f1 = kbreduct.detrend(t1, f1, 2, 3)
            f1 = kbreduct.kboutlier(f1, 3)
            t1 = t1  # -t_min
            tsei = np.append(tsei, np.mean(t1))

            # fig, ax = plt.subplots(2, 1, figsize=(10, 10))

            # ax[0].plot(t1, f1, c='k')
            # ax[0].set_xlabel('date')
            # ax[0].set_ylabel('flux')

            num = len(f1)
            freq = np.linspace(1, 8000, 100000)  # np.int64(num/2))
            freq0 = freq * uhz_conv
            model = LombScargle(t1, f1)
            power = model.power(freq0, method='fast', normalization='psd')
            power *= uhz_conv/num
            bin0 = freq[10]-freq[9]
            wid = int(5/bin0)
            power_f = kbreduct.smoothing(power, wid)
            whe = np.where(freq >= 200)
            freq_f = freq[whe]
            power_f = power_f[whe]
            powerxx = power[whe]

            # power_00 += power
            p = [50, 50, 100, 100,  1, 1800, 100, 0.001]
            mini = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            p = kbparameters(p)
            minner = Minimizer(bgm2, p, fcn_args=(freq_f, power_f))
            m_res = minner.minimize()
            final = power_f + m_res.residual
            report_fit(m_res)
            p00 = m_res.params.valuesdict()
            resb = param_value(m_res)
            res_yy = bgm2(m_res.params.valuesdict(),
                          freq_f, power_f, value=1)

            # ax[1].plot(freq_f, power_f, c='black')
            # ax[1].plot(freq_f, res_yy.fv, c='red', label='total')
            # ax[1].plot(freq_f, res_yy.bg0, c='lightcoral', label='facular')
            # ax[1].plot(freq_f, res_yy.bg1, c='coral', label='granulaton')
            # # ax[1].plot(freq_f, res_yy.bg2, c='coral', label='granulaton2')
            # ax[1].plot(freq_f, res_yy.pex, c='blue', label='p-mode excess')
            # ax[1].plot(freq_f, res_yy.whn, '--',
            #            c='gray', label='white noise')
            # ax[1].set_xlabel(r"frequency [$\mu$hz]")
            # ax[1].set_ylabel(r"power $(ppm^2 / \mu hz)$")
            # ax[1].set_yscale('log')
            # ax[1].set_xlim(100, 5000)
            # ax[1].set_ylim(3e-2, 10)
            # ax[1].legend()
            # pdf.savefig()
            # plt.close()

            res_p = np.array(resb.p).reshape(1, 8)
            res_st = np.array(resb.st).reshape(1, 8)

            result1 = pd.DataFrame(res_p, columns=['p0', 'p1', 'p2', 'p3',
                                                   'amp', 'numax', 'width', 'whn'])
            result2 = pd.DataFrame(res_st, columns=['p0s', 'p1s', 'p2s', 'p3s',
                                                    'amps', 'numaxs', 'widths', 'whns'])
            result0 = pd.concat([result1, result2], axis=1)

            df = df.append(result0, ignore_index=True)
            amp = np.append(amp, resb.p[4])
            numax = np.append(numax, resb.p[5])
            width = np.append(width, resb.p[6])
            amps = np.append(amps, resb.st[4])
            lc_s = lc_s + ov
            lc_e = lc_s + length

            wid = int(1/bin0)
            # power_ff = np.array(power_f - res_yy.bg0 - res_yy.bg1 - res_yy.whn)
            power_ff = kbreduct.smoothing(power, wid)
            whc0 = np.where((freq_f >= 0) & (freq_f <= 8000))
            # whcx = np.where((freq >= 0) & (freq <= 8000))
            whc = np.where((freq >= 1837) & (freq <= 1847))
            whc1 = np.where((freq >= 1750) & (freq <= 1760))
            whc2 = np.where((freq >= 2096) & (freq <= 2110))

            # whc = np.where((freq_f >= 1850-400) & (freq_f <= 1850+400))
            freq_0 = freq_f[whc0]
            freq_x = freq_f
            power_0 = power_f[whc0]
            power_x = power_f
            freq_c = freq[whc]
            power_c = power_ff[whc]
            freq_c1 = freq[whc1]
            power_c1 = power_ff[whc1]
            freq_c2 = freq[whc2]
            power_c2 = power_ff[whc2]
            cs = str(c)
            if c == 0:
                psx['freq'] = freq_x
                ps0['freq'] = freq_0
                ps['freq'+cs] = freq_c
                ps1['freq'+cs] = freq_c1
                ps2['freq'+cs] = freq_c2
            psx['power'+cs] = power_x
            ps0['power'+cs] = power_0
            ps['power'+cs] = power_c
            ps1['power'+cs] = power_c1
            ps2['power'+cs] = power_c2
            res_xx['bg0'+cs] = res_yy.bg0
            res_xx['bg1'+cs] = res_yy.bg1
            res_xx['pex'+cs] = res_yy.pex
            res_xx['whn'+cs] = res_yy.whn
            c += 1

            # whc = np.where((freq_f >= 1400) & (freq_f <= 2100))
            # freq_c = freq_f[whc]
            # power_c = power_f[whc]
            # p2 = pd.DataFrame({'p': power_c})
            # cs = str(c)
            # ps['freq'+cs] = freq_c
            # ps['power'+cs] = power_c
            # c += 1
    power_00 = 0
    power_c0 = 0
    power_cc0 = 0
    power_ccc0 = 0
    psps0 = np.array(ps0)
    psps = np.array(ps)
    psps1 = np.array(ps1)
    psps2 = np.array(ps2)
    for cc in range(c):
        # print(cc)
        power_00 += psps0[:, cc+1]
        power_c0 += psps[:, cc+1]
        power_cc0 += psps1[:, cc+1]
        power_ccc0 += psps2[:, cc+1]
    power_00 = power_00/c
    power_c0 = power_c0/c
    power_cc0 = power_cc0/c
    power_ccc0 = power_ccc0/c
    gmodel = Model(gaussian)

    cad = ps.freq0[1]-ps.freq0[0]
    cad1 = ps1.freq0[1]-ps1.freq0[0]
    cad2 = ps2.freq0[1]-ps2.freq0[0]
    lags = signal.correlation_lags(
        len(power_c0), len(power_c0))*cad
    lags1 = signal.correlation_lags(
        len(power_cc0), len(power_cc0))*cad1
    lags2 = signal.correlation_lags(
        len(power_ccc0), len(power_ccc0))*cad2
    for cc in range(c):
        corr0 = signal.correlate(
            power_c0, psps[:, cc+1], mode='full')
        corr0 /= np.max(corr0)
        result = gmodel.fit(corr0, x=lags, amp=1, cen=0, wid=2)
        res = param_value(result)
        # print(result.fit_report())
        cen = '%.4f' % res.p[1]
        fshift = np.append(fshift, res.p[1])
        fshifts = np.append(fshifts, res.st[1])
        # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        # plt.suptitle('Frequency Shift Seqeunce (1845uHz) : ' +
        #              str(cc), size=20)
        # ax[0].plot(ps.freq0, power_c0)
        # ax[1].plot(ps.freq0, psps[:, cc+1])
        # ax[2].plot(lags, corr0)
        # ax[2].plot(lags, result.best_fit, 'r-', label='center = '+cen)
        # ax[2].axvline(x=0, linestyle=':')
        # ax[2].axvline(x=res.p[1], linestyle=':', color='r')
        # ax[2].set_xlim(-10, 10)
        # ax[2].legend()
        # pdf.savefig()
        # plt.close()
    for cc in range(c):
        corr0 = signal.correlate(
            power_cc0, psps1[:, cc+1], mode='full')
        corr0 /= np.max(corr0)
        result = gmodel.fit(corr0, x=lags1, amp=1, cen=0, wid=2)
        res = param_value(result)
        # print(result.fit_report())
        cen = '%.4f' % res.p[1]
        fshift1 = np.append(fshift1, res.p[1])
        # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        # plt.suptitle('Frequency Shift Seqeunce (1760uHz) : ' +
        #              str(cc), size=20)
        # ax[0].plot(ps1.freq0, power_cc0)
        # ax[1].plot(ps1.freq0, psps1[:, cc+1])
        # ax[2].plot(lags1, corr0)
        # ax[2].plot(lags1, result.best_fit, 'r-', label='center = '+cen)
        # ax[2].axvline(x=0, linestyle=':')
        # ax[2].axvline(x=res.p[1], linestyle=':', color='r')
        # ax[2].set_xlim(-10, 10)
        # ax[2].legend()
        # pdf.savefig()
        # plt.close()
    for cc in range(c):
        corr0 = signal.correlate(
            power_ccc0, psps2[:, cc+1], mode='full')
        corr0 /= np.max(corr0)
        result = gmodel.fit(corr0, x=lags2, amp=1, cen=0, wid=2)
        res = param_value(result)
        # print(result.fit_report())
        cen = '%.4f' % res.p[1]
        fshift2 = np.append(fshift2, res.p[1])
        # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        # plt.suptitle('Frequency Shift Seqeunce (2010uHz) : ' +
        #              str(cc), size=20)
        # ax[0].plot(ps2.freq0, power_ccc0)
        # ax[1].plot(ps2.freq0, psps2[:, cc+1])
        # ax[2].plot(lags2, corr0)
        # ax[2].plot(lags2, result.best_fit, 'r-', label='center = '+cen)
        # ax[2].axvline(x=0, linestyle=':')
        # ax[2].axvline(x=res.p[1], linestyle=':', color='r')
        # ax[2].set_xlim(-10, 10)
        # ax[2].legend()
        # pdf.savefig()
        # plt.close()
# df.amp = np.sqrt(df.amp)
# df.amps = 1/2*df.amps/df.amp
df.width = df.width * 2 * np.sqrt(2*np.log(2))
df.widths = df.widths * 2 * np.sqrt(2*np.log(2))
df['fshift'] = fshift
df['fshifts'] = fshifts
df['tsei'] = tsei
df['sphq'] = sphq
dff = df.drop([9])
fshifta = (fshift+fshift1+fshift2)/3
strt = '%.1f' % (ttmin+2400000)


delx0 = 1756.2
delx1 = 1840.7
dely0 = 2.3  # 2.42
dely1 = 2.3  # 2.251

delix0 = [delx0, delx0]
deliy0 = [dely0, 2.7]

delix1 = [delx1, delx1]
deliy1 = [dely1, 2.7]

ms = 5
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85,
                    top=0.8)
ax.plot(psx.freq, psx.power0-(res_xx.bg00+res_xx.bg10),
        c='k', linewidth=0.7, alpha=0.6)
#ax.plot(psx.freq, res_xx.pex0, c='k', linewidth=2)
#ax.plot(psx.freq, res_xx.whn0, c='k', linestyle='--')
ax.plot(psx.freq, res_xx.whn0+res_xx.pex0, c='k', linewidth=2)
# ax.arrow(delx0, 2.7, 86.1, 0, width=0.05)

yy = 2.5
ax.annotate('', xy=(1735, yy), xytext=(1865, yy),
            arrowprops={'arrowstyle': '<->'}, va='center')
# ax.annotate('', xy=(1850, yy), xytext=(1950, yy),
#             arrowprops={'arrowstyle': '->'}, va='center')
# ax.annotate('', xy=(1650, yy), xytext=(1750, yy),
#             arrowprops={'arrowstyle': '<-'}, va='center')
# ax.text(0.481, 0.85, r'$\Delta \nu$', transform=ax.transAxes, fontsize=17)
ax.text(0.340, 0.845, r'$\Delta \nu$', transform=ax.transAxes, fontsize=16)

# ax.plot(delix0, deliy0, 'k:')
# ax.plot(delix1, deliy1, 'k:')
ax.set_ylabel(r'PSD (ppm$^2$/$\mu $Hz)', size=18)
ax.set_xlabel(r'Frequency ($\mu $Hz)', size=18)
ax.tick_params(axis='both', direction='in',
               top=True, right=True, labelsize=18)
ax.set_xticks(np.arange(1000, 4400, 1000))
ax.set_yticks(np.arange(0, 4, 1))
ax.set_ylim(0, 3)
ax.set_xlim(600, 4000)

plt.savefig('figure2.pdf')
plt.close()
# ax.plot(ps0.freq, ps0.power9)
# ax.plot(ps0.freq, res_yy.fv)
# ax.plot(ps0.freq, res_yy.bg0)
# ax.plot(ps0.freq, res_yy.bg1)
# ax.plot(ps0.freq, res_yy.pex)

ms = 4
fig, ax = plt.subplots(4, 1, figsize=(9, 12))
plt.subplots_adjust(hspace=0.45)
ax[0].errorbar(df.tsei, df.fshift, yerr=df.fshifts, fmt='o', c='k', ms=ms)
ax[0].set_xlabel(r'Time (JD-'+strt+')', size=13)
ax[0].set_ylabel(r'$\delta \nu$ $(\mu$Hz)', size=13)
ax[0].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=13)
ax[0].set_xticks(np.arange(20, 120, 20))
ax[0].set_xlim(18, 112)


ax[1].errorbar(df.tsei, df.amp, yerr=df.amps,
               fmt='o', c='k', ms=ms)
ax[1].set_xlabel(r'Time (JD-'+strt+')', size=13)
ax[1].set_ylabel(r'$A$ (ppm$^{2}$)', size=13)
ax[1].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=13)
ax[1].set_xticks(np.arange(20, 120, 20))
ax[1].set_xlim(18, 112)

ax[2].errorbar(df.tsei, df.numax, yerr=df.numaxs,
               fmt='o', c='k', ms=ms)
ax[2].set_xlabel(r'Time (JD-'+strt+')', size=13)
ax[2].set_ylabel(r'$\nu_{\rm max}$ $(\mu$Hz)', size=13)
ax[2].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=13)
ax[2].set_xticks(np.arange(20, 120, 20))
ax[2].set_xlim(18, 112)

ax[3].errorbar(df.tsei, df.width, yerr=df.widths, fmt='o', c='k', ms=ms)
ax[3].set_xlabel('Time (JD-'+strt+')', size=13)
ax[3].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', size=13)
ax[3].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=13)
ax[3].set_yticks(np.arange(900, 1101, 200))
ax[3].set_xticks(np.arange(20, 120, 20))
ax[3].set_xlim(18, 112)

plt.savefig('figure3.pdf')
plt.close()

# plt.subplots_adjust(left=0.09, bottom=0.1, right=0.95,
#                     top=0.9, wspace=0.32, hspace=0.2)


ms = 5
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85,
                    top=0.8)

ax.errorbar(df.tsei, df.fshift, fmt='-o', yerr=df.fshifts, c='k', ms=ms)
ax.set_xlabel(r'Time (JD-'+strt+')', size=18)
ax.set_ylabel(r'$\delta \nu$ $(\mu $Hz)', size=18)
ax.tick_params(axis='both', direction='in',
               top=True, right=True, labelsize=18)
ax.set_xticks(np.arange(20, 120, 20))
ax.set_yticks(np.arange(-0.5, 0.51, 0.2))

ax.set_xlim(15, 115)

axa = ax.twinx()
axa.errorbar(df.tsei, df.sphq, fmt='--o', c='k', ms=ms)
axa.set_xlabel(r'Time (JD-'+strt+')', size=18)
axa.set_ylabel(r'$S_{\rm ph}$ (ppm)', size=18)
axa.tick_params(axis='both', direction='in',
                top=True, right=True, labelsize=18)
axa.set_xticks(np.arange(20, 120, 20))
axa.set_xlim(15, 115)
plt.savefig('figure5.pdf')
plt.close()


fig, ax = plt.subplots(3, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.35)
col0 = stats.pearsonr(df.sphq, df.amp)
coef0 = 'C. : ' + '%.4f' % col0[0]
pval0 = ' P. : ' + '%.4f' % col0[1]

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(df.amp, x=df.sphq,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(df.amps)))
res0 = param_value(lmr0)
inc0 = '%.6f' % res0.p[0]
slp0 = '+ %.6f x' % res0.p[1]

col1 = stats.pearsonr(dff.sphq, dff.amp)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]

p1 = [10.0, -0.001]
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(dff.amp, x=dff.sphq,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(dff.amps)))
res1 = param_value(lmr1)
inc1 = '%.6f' % res1.p[0]
slp1 = '+ %.6f x' % res1.p[1]

linex = np.linspace(955, 1050, 5)
liney0 = xyfitmodel(linex, a=res0.p[0], b=res0.p[1])
liney1 = xyfitmodel(linex, a=res1.p[0], b=res1.p[1])

ax[0].errorbar(df.sphq, df.amp, c='k', yerr=df.amps,
               fmt='o', label=coef0+pval0)
ax[0].plot(linex, liney0, color='k', label=inc0+slp0)
ax[0].plot(linex, liney1, '--', c='k')
ax[0].set_xlabel(r'$S_{\rm ph}$ (ppm)', size=15)
ax[0].set_ylabel(r'$A$ (ppm$^{2}$)', size=15)
# ax[0].legend(fontsize=7)
ax[0].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
ax[0].set_yticks(np.arange(0.35, 0.60, 0.05))
ax[0].set_ylim(0.41, 0.54)

col0 = stats.pearsonr(df.sphq, df.numax)
coef0 = 'C. : ' + '%.4f' % col0[0]
pval0 = ' P. : ' + '%.4f' % col0[1]

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(df.numax, x=df.sphq,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(df.numaxs)))
res0 = param_value(lmr0)
inc0 = '%.6f' % res0.p[0]
slp0 = '+ %.6f x' % res0.p[1]

col1 = stats.pearsonr(dff.sphq, dff.numax)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]

p1 = [10.0, -0.001]
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(dff.numax, x=dff.sphq,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(dff.numaxs)))
res1 = param_value(lmr1)
inc1 = '%.6f' % res1.p[0]
slp1 = '+ %.6f x' % res1.p[1]

linex = np.linspace(955, 1050, 5)
liney0 = xyfitmodel(linex, a=res0.p[0], b=res0.p[1])
liney1 = xyfitmodel(linex, a=res1.p[0], b=res1.p[1])

ax[1].errorbar(df.sphq, df.numax,  c='k', yerr=df.numaxs,
               fmt='o', label=coef0+pval0)
ax[1].plot(linex, liney0, color='k', label=inc0+slp0)
ax[1].plot(linex, liney1, '--', color='k', label=inc0+slp0)
ax[1].set_xlabel(r'$S_{\rm ph}$ (ppm)', size=15)
ax[1].set_ylabel(r'$\nu_{\rm max}$ $(\mu$Hz)', size=15)
ax[1].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
# ax[1].legend(fontsize=7)
col0 = stats.pearsonr(df.sphq, df.width)
coef0 = 'C. : ' + '%.4f' % col0[0]
pval0 = ' P. : ' + '%.4f' % col0[1]

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(df.width, x=df.sphq,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(df.widths)))
res0 = param_value(lmr0)
inc0 = '%.6f' % res0.p[0]
slp0 = '+ %.6f x' % res0.p[1]

col1 = stats.pearsonr(dff.sphq, dff.width)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]

p1 = [10.0, -0.001]
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(dff.width, x=dff.sphq,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(dff.widths)))
res1 = param_value(lmr1)
inc1 = '%.6f' % res1.p[0]
slp1 = '+ %.6f x' % res1.p[1]

linex = np.linspace(955, 1050, 5)
liney0 = xyfitmodel(linex, a=res0.p[0], b=res0.p[1])
liney1 = xyfitmodel(linex, a=res1.p[0], b=res1.p[1])

ax[2].errorbar(df.sphq, df.width,  c='k', yerr=df.widths,
               fmt='o', label=coef0+pval0)
ax[2].plot(linex, liney0, color='k', label=inc0+slp0)
ax[2].plot(linex, liney1, '--', color='k', label=inc0+slp0)
ax[2].set_xlabel(r'$S_{\rm ph}$ (ppm)', size=15)
ax[2].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', size=15)
# ax[2].legend(fontsize=7)
ax[2].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
ax[2].set_yticks(np.arange(800, 1200, 150))

plt.savefig('figure6.pdf')
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(10, 12))
# plt.subplots_adjust(hspace=0.35)

col0 = stats.pearsonr(df.amp, df.numax)
coef0 = 'C. : ' + '%.4f' % col0[0]
pval0 = ' P. : ' + '%.4f' % col0[1]

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(df.numax, x=df.amp,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(df.numaxs)))
res0 = param_value(lmr0)
inc0 = '%.6f' % res0.p[0]
slp0 = '+ %.6f x' % res0.p[1]

col1 = stats.pearsonr(dff.amp, dff.numax)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]

p1 = [10.0, -0.001]
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(dff.numax, x=dff.amp,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(dff.numaxs)))
res1 = param_value(lmr1)
inc1 = '%.6f' % res1.p[0]
slp1 = '+ %.6f x' % res1.p[1]

linex = np.linspace(0.41, 0.545, 5)
liney0 = xyfitmodel(linex, a=res0.p[0], b=res0.p[1])
liney1 = xyfitmodel(linex, a=res1.p[0], b=res1.p[1])

ax[0].errorbar(df.amp, df.numax,  c='k', yerr=df.numaxs,
               fmt='o', label=coef0+pval0)
ax[0].plot(linex, liney0, color='k', label=inc0+slp0)
ax[0].plot(linex, liney1, '--', color='k', label=inc0+slp0)
ax[0].set_xlabel(r'$A$ (ppm$^{2}$)', size=15)
ax[0].set_ylabel(r'$\nu_{\rm max}$ $(\mu$Hz)', size=15)
ax[0].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
ax[0].set_yticks(np.arange(1720, 1810, 40))
ax[0].set_xticks(np.arange(0.41, 0.55, 0.03))

ax[0].set_xlim(0.40, 0.55)
# ax[1].legend(fontsize=7)
col0 = stats.pearsonr(df.amp, df.width)
coef0 = 'C. : ' + '%.4f' % col0[0]
pval0 = ' P. : ' + '%.4f' % col0[1]

p0 = [10.0, -0.001]
mod0 = Model(xyfitmodel)
lmr0 = mod0.fit(df.width, x=df.amp,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(df.widths)))
res0 = param_value(lmr0)
inc0 = '%.6f' % res0.p[0]
slp0 = '+ %.6f x' % res0.p[1]

col1 = stats.pearsonr(dff.amp, dff.width)
coef1 = 'C. : ' + '%.4f' % col1[0]
pval1 = ' P. : ' + '%.4f' % col1[1]

p1 = [10.0, -0.001]
mod1 = Model(xyfitmodel)
lmr1 = mod0.fit(dff.width, x=dff.amp,
                a=p0[0], b=p0[1], weights=np.sqrt(1.0/(dff.widths)))
res1 = param_value(lmr1)
inc1 = '%.6f' % res1.p[0]
slp1 = '+ %.6f x' % res1.p[1]

linex = np.linspace(0.41, 0.545, 5)
liney0 = xyfitmodel(linex, a=res0.p[0], b=res0.p[1])
liney1 = xyfitmodel(linex, a=res1.p[0], b=res1.p[1])

ax[1].errorbar(df.amp, df.width,  c='k', yerr=df.widths,
               fmt='o', label=coef0+pval0)
ax[1].plot(linex, liney0, color='k', label=inc0+slp0)
ax[1].plot(linex, liney1, '--', color='k', label=inc0+slp0)
ax[1].set_xlabel(r'$A$ (ppm$^{2}$)', size=15)
ax[1].set_ylabel(r'$\delta \nu_{\rm env}$ $(\mu$Hz)', size=15)
# ax[2].legend(fontsize=7)
ax[1].tick_params(axis='both', direction='in',
                  top=True, right=True, labelsize=15)
ax[1].set_yticks(np.arange(800, 1200, 150))
ax[1].set_xticks(np.arange(0.41, 0.55, 0.03))
ax[1].set_xlim(0.40, 0.55)

plt.savefig('figure4.pdf')
plt.close()

delnu = delta_nu_ps2(freq, power_00, numax=1850, fwhm=250)
teff = 6590


pth = '/home/kim//research/amplitude_fs/HD49933/tracks/'
pth1 = '/home/kim//research/amplitude_fs/HD49933/KIC/'
file_list = os.listdir(pth)
file_list.sort()
file_list1 = os.listdir(pth1)
dt = pd.read_csv(pth1+file_list1[0], delimiter=',')
ds = pd.read_csv(pth1+file_list1[1], delimiter=',')

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(teff, delnu[0], marker='*', c='k', s=90)
ax.scatter(steff, sdel, marker='$\\bigodot$', c='k', s=90)
ax.set_yticks(np.arange(30, 161, 40))
w = 1
# ax.plot(etem1[15:], Ds1[15:], c='k',
#         label='z=0.010', linewidth=w)
# # ax.plot(etem2[:], Ds2[:], ':', c='k', linewidth=w)
# ax.plot(etem3[23:], Ds3[23:], c='k', linewidth=w)
# ax.plot(etem7[23:], Ds7[23:], c='k', linewidth=w)

ax.plot(etems081[7:], Dss081[7:], c='k',
        label='Z=0.008', linewidth=w)
ax.plot(etems082[5:], Dss082[5:], c='k', linewidth=w)
ax.plot(etems083[5:], Dss083[5:], c='k', linewidth=w)


ax.plot(etems1[8:], Dss1[8:], ':', c='k',
        linewidth=w, label='Z=0.014')
ax.plot(etems12[18:], Dss12[18:], ':', c='k', linewidth=w)
ax.plot(etems14[4:], Dss14[4:],  ':', c='k', linewidth=w)

ax.legend(frameon=False, fontsize=15)

ax.set_xlim(7500, 5000)
ax.set_ylim(175, 10)
ax.set_xlabel(r'$T_{\rm eff}$ (K)', size=18)
ax.set_ylabel(r'$\Delta \nu$ $(\mu$Hz)', size=18)
ax.tick_params(axis='both', direction='in',
               top=True, right=True, labelsize=18)
ax.text(0.73, 0.04, '1.0M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.60, 0.08, '1.0M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
# ax.text(0.66, 0.08, '1.0M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.50, 0.35, '1.2M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.36, 0.375, '1.2M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
# ax.text(0.43, 0.375, '1.2M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.27, 0.52, '1.4M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.01, 0.47, '1.4M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
# ax.text(0.13, 0.53, '1.4M$_{\\odot}$', transform=ax.transAxes, fontsize=12)

plt.savefig('figure6x.pdf')
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(teff, delnu[0], marker='*', c='k', s=90)
ax.scatter(steff, sdel, marker='$\\bigodot$', c='k', s=90)
ax.set_yticks(np.arange(30, 161, 40))
w = 1
ax.plot(etems081[7:], Dss081[7:], c='k',
        label='Z=0.008', linewidth=w)
ax.plot(etems082[5:], Dss082[5:], c='k', linewidth=w)
ax.plot(etems083[5:], Dss083[5:], c='k', linewidth=w)

ax.plot(etems1[8:], Dss1[8:], '--', c='k',
        linewidth=w, label='Z=0.014', dashes=(5, 8))
ax.plot(etems12[18:], Dss12[18:], '--', c='k', linewidth=w, dashes=(5, 8))
ax.plot(etems14[4:], Dss14[4:],  '--', c='k', linewidth=w, dashes=(5, 8))

# age083 - 80 - 2e9
# age082 - 49 - 2e9
# age081 - 43 - 2e9
# age14  - 58 - 2e9
# age12  - 55 - 2e9
# age1   - 37 - 2e9
agex0 = [etems083[80], etems082[49], etems081[43]]
agex1 = [etems14[58], etems12[55], etems1[37]]
agey0 = [Dss083[80], Dss082[49], Dss081[43]]
agey1 = [Dss14[58], Dss12[55], Dss1[37]]

#age083  -105 - 2.25
#age082 - 54 - 2.25
#age081 - 45 - 2.205
#age14  - 68 -2.26
#age12  - 59 -2.25
#age1   - 39 -2.28
agex2 = [etems083[105], etems082[54], etems081[45]]
agey2 = [Dss083[105], Dss082[54], Dss081[45]]

agex3 = [etems14[68], etems12[59], etems1[39]]
agey3 = [Dss14[68], Dss12[59], Dss1[39]]


#ax.plot(agex0, agey0, 'k:')
#ax.plot(agex1, agey1, 'k:')
ax.plot(agex2, agey2, 'k:', alpha=0.3)
ax.plot(agex3, agey3, 'k:', alpha=0.3)
#ax.scatter(etems082[54],  Dss082[54], marker='o', s=100)
ax.legend(frameon=False, fontsize=15)

ax.set_xlim(7800, 5000)
ax.set_ylim(175, 10)
ax.set_xlabel(r'$T_{\rm eff}$ (K)', size=18)
ax.set_ylabel(r'$\Delta \nu$ $(\mu$Hz)', size=18)
ax.tick_params(axis='both', direction='in',
               top=True, right=True, labelsize=18)
ax.text(0.75, 0.04, '1.0M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.63, 0.08, '1.0M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.54, 0.36, '1.2M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.42, 0.375, '1.2M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.34, 0.52, '1.4M$_{\\odot}$', transform=ax.transAxes, fontsize=12)
ax.text(0.12, 0.47, '1.4M$_{\\odot}$', transform=ax.transAxes, fontsize=12)

plt.savefig('figure7.pdf')
plt.close()

col0 = stats.pearsonr(dff.sphq, dff.fshift)
coef0 = 'C. : ' + '%.4f' % col0[0]
pval0 = ' P. : ' + '%.4f' % col0[1]

end = time.time()
print('\033[01;31;43m', format(end-start, ".3f"), 's\033[00m')
