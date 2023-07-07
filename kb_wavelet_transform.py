import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import astropy.units as u
from tqdm import tqdm
import numpy as np
import os
import time
from astropy.io import fits
import pycwt as wavelet
from pycwt.helpers import find
import math
start = time.time()


miss = "KIC"
cdc = 'long'
if cdc == 'short':
    icdc = 'sc'
elif cdc == 'long':
    icdc = 'lc'

number1 = "10904491"
# number2 = "10355856"
# number3 = "4166395"
# number4 = "3446150"



number = number1

target = miss+" "+number
print(target, cdc, 'cadence')

###########################
## parameters setting ##
thrsh = 50  # aperture threshold small=50, large=2
ft_wnleng = 401  # flatten width length default = 401
sm_width = 10  # smooth width default = 10
###########################

pth0 = "/home/kim/.lightkurve-cache/mastDownload/Kepler/"
pth1 = []
ext = "kplr0"+number
for (path, dir, files) in os.walk(pth0):
    for dirname in dir:
        ext0 = dirname.split('_')
        cadc = dirname.split('_')
        if ext0[0] == ext and cadc[1] == icdc:
            pth1.append(pth0+dirname+'/')
pth1.sort()
dup = len(pth1)
if dup > 0:
  print('Using already having Pixelfiles')
  tpf = []
  for i in range(dup):
    for (path, dir, files) in os.walk(pth1[i]):
      files.sort()
#      print(files)
      for fname in files:
        print(fname)
        tpf.append(lk.KeplerTargetPixelFile(pth1[i]+fname))
elif dup == 0:
  print('Download Target Pixelfiles')
  tpf = lk.search_targetpixelfile(
    target, cadence=cdc).download_all()


fname = "figure_"+miss+"_"+number+".pdf"

num = len(tpf)
print("Figure : ", fname)
with PdfPages(fname) as pdf:

    ltc = []
    for n in range(num):
        lc_large = tpf[n].to_lightcurve().normalize()#.remove_nans()
        ltc.append(lc_large)
        num2 = len(ltc[n].time)
        for n2 in range(num2):
            if ltc[n].time[n2] != ltc[n].time[n2]:
                print(ltc[n].time[n2])
            elif math.isinf(ltc[n].time[n2]):
                print(n2, ltc[n].time[n2])
        if n == 0:
            ax = ltc[n].plot(
                label=str(tpf[n].quarter))
        else:
            ltc[n].plot(
                ax=ax, label=str(tpf[n].quarter))
        if lc_large.quarter < 10:
            lb = '0'+str(lc_large.quarter)
        else:
            lb = str(lc_large.quarter)
        lc_large.to_fits(path='KIC'+number+'/'+'KIC'+number+'_Q'+lb+'.fits', overwrite=True)
    plt.title('Lightcurve Figures KIC'+number)
    pdf.savefig()
    plt.close()

end = time.time()
print('\033[01;31;43m', format(end-start, ".3f"), 's\033[00m')
