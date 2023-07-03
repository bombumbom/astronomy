import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline


class kbreduct:
    # def __init__(self):
    #   self.result = 0

    @staticmethod
    def kbnan(flux):
        num = 0
        minf = float("-inf")
        inf = float("inf")
        n = len(flux)
        for fl in range(n):
            if flux[fl] == inf or flux[fl] == minf or flux[fl] != flux[fl]:
                num += 1
                if fl != 0:
                    flux[fl] = flux[fl-1]
                j = 0
                while True and fl == 0:
                    # print('fl = 0', j)
                    j += 1
                    if flux[fl+j] != minf and flux[fl+j] != inf and flux[fl+j] == flux[fl+j]:
                        flux[fl] = flux[fl + j]
                        break
                continue
            continue
  #      print('')
  #      print('complete : kbnan  &  the number of (-inf +inf) = ', num, '\n')
        result = flux
        return result

    @staticmethod
    def kboutlier(y, s):

        N = len(y)
        std = np.std(y)
        men = np.mean(y)
        if y[0] > men+s*std or y[0] < men-s*std:
            n = 0
            while y[n] > men+s*std or y[n] < men-s*std:
                n += 1
                if y[n] <= men+s*std and y[n] >= men-s*std:
                    y[0] = y[n]

        for i in range(1, N):
            if y[i] > men+s*std or y[i] < men-s*std:
                y[i] = y[i-1]

        print('')
        print('Complete : remove outliers  &  sigma  = %.1f' % s + '\n')
        return y

    @staticmethod
    def kbdisx(x, **kwargs):
        dic = kwargs.get('dic', 0)
        n0 = 0
        N = len(x)
        dt = np.mean(x[1:-1]-x[0:-2])
        if dic > 0:
            dt = dic
        for i in range(1, N):
            dis = x[i]-x[i-1]
            if dis >= dt*1.3:
                print('discrete length = {0:2.5f} days'.format(dis), i)
                x[i:] = x[i:]-dis+dt
                n0 += 1

        print('')
        print('Complete : discrete > continue  &  number = ', n0, '\n')
        return x

    @staticmethod
    def kbdisx_int(x, y, **kwargs):
        # discrete data > interpolation
        kind = kwargs.get('kind', 'linear')
        kind = kind
        value = kwargs.get('value', 'none')
        n0 = 0
        N = len(x)
        dt = np.mean(x[1:-1]-x[0:-2])
        i0 = 1
        for i in range(i0, N):
            dis = x[i]-x[i-1]
            if dis >= dt*1.3:
                print('discrete length = {0:2.5f} days'.format(dis), i)
                dis = x[1]-x[0]
                num = math.ceil((x[i]-x[i-1])/dis)-1
                xnew = (np.arange(num)+1)*dis+x[i-1]
#                f = interp1d(x, y, kind=kind)
                f = Rbf(x, y)
#               f = InterpolatedUnivariateSpline(x, y)
                x = np.insert(x, i, xnew)
                # y = np.insert(y, i, xnew*0)
                y = np.insert(y, i, f(xnew))
                n0 += 1
                i0 = i
                N = len(x)
        print('')
        print('complete : discrete > continue interpolation &  number = ', n0, '\n')
        if value == 1:
            return xnew, rbf(xnew)
        return x, y

    @ staticmethod
    def detrend(x, y, b, deg, **kwargs):
        fits = kwargs.get('fits', 'None')
        print(fits)
        x = np.array(x)
        y = np.array(y)
        mx0 = -1
        cf = []
        x = x-min(x)
        dn = math.ceil(max(x)/b)
        if b >= max(x):
            coeffs = np.polynomial.legendre.legfit(x, y, deg)
            fit = np.polynomial.legendre.legval(x, coeffs)
            cf = y - fit
            print('')
            print(
                'Complete : kbdetrend // time bin : {}, degree : {} \n'.format(
                    b, deg))
            if fits == 1:
                return cf, fit
            else:
                return cf
        else:
            for n in range(dn):
                mx1 = (n+1)*b
                whe = np.where((x > mx0) & (x <= mx1))
                xi = x[whe]
                yi = y[whe]
                coeffs = np.polynomial.legendre.legfit(xi, yi, deg)
                fit = np.polynomial.legendre.legval(xi, coeffs)
                correct_flux = yi - fit
                cf = np.append(cf, correct_flux)
                mx0 = mx1
            print('')
            print('Complete : kbdetrend  // time bin : {}, degree : {} \n'.format(b, deg))
            if fits == 1:
                return cf, fit
            else:
                return cf

    @ staticmethod
    def binspec(x, y, **kwargs):
        print('\n \033[1;34m ### Starting kbreduct.binspec ###', '\033[00m')
        freq = kwargs.get('freq', 'None')
        bins = kwargs.get('bins', 'None')
        names = kwargs.get('names', 'None')
        x = np.array(x)
        y = np.array(y)
        if names == 'None':
            names = ['freq', 'power']
#        print(names, names[0], names[1])

        # bins setting
        if bins == 'None':
            bins = 3
            print('bin is None & default = ', bins)
        # bins setting

        if freq == 'freq':
            xmax = max(x)
            xmin = min(x)
            f = np.array(x)
            d = f[1:] - f[0:-1]
            print('frequency setting')
            if bins < np.mean(d):
                bins = min(d)
                print('\033[1;31;1m bin is very small < ',
                      format(bins, ".4f"), '\033[00m \n')
                raise NotImplementedError
            print('binsize =', bins)

        else:
            xmax = len(x)
            xmin = 0
            print('point setting')
            print('binsize =', bins)

        z = 0
        zz = 0
        rest = 0
        fmn = int((xmax-xmin)/bins)
        xm = np.array([])
        ym = np.array([])
        if fmn == (xmax-xmin)/bins:
            rest = -1
        fmn = fmn + 1
        if freq == 'freq':
            for i in range(fmn):
                fb = np.where(np.logical_and(x >= xmin + bins * i,
                                             x < xmin + bins + bins * i))
                fb0 = fb[0]
                if len(fb0) == 0:
                    xm = np.append(xm, xmin+bins*i*1./2.*bins)
                    ym = np.append(ym, 0.)
                    z = 1
                if z != 1:
                    if i == fmn-1 and rest == -1:
                        fb = np.where(np.logical_and(x >= xmin + bins * i,
                                                     x <= xmin + bins + bins * i))
                    xb = x[fb]
                    xm = np.append(xm, np.mean(xb))
                    yb = y[fb]
                    ym = np.append(ym, np.mean(yb))
                    z = 0
        else:
            for i in range(fmn):
                if i == fmn-1:
                    xm = np.append(xm, np.mean(x[i*bins:]))
                    ym = np.append(ym, np.mean(y[i*bins:]))
                    zz = 1
                if zz != 1:
                    zz = 0
                    xm = np.append(xm, np.mean(x[i*bins:i*bins+bins]))
                    ym = np.append(ym, np.mean(y[i*bins:i*bins+bins]))
        print('complete : binspec & the number of data = ', len(xm), '\n')
        df = pd.DataFrame({names[0]: xm, names[1]: ym})
        return df

    @ staticmethod
    def kbbinning(x, y, n):  # , deg):
        num = math.ceil(len(y)/n)
        print('\n'+'The number of bined data: ', num)

        y2 = np.zeros(num)
        x2 = np.zeros(num)
        # n : the number of points

        win = np.zeros(n)
        for i in range(n):
            win[i] = np.exp(-1./2. * ((i-(n-1.)/2.)/(0.4*(n-1.)/2.)) ** 2)

        i0 = 0
        for i in range(num):
            yn = len(y[i0:i0+n])
            if yn == n:
                y2[i] = np.mean(y[i0:i0+n]*win)
                x2[i] = np.median(x[i0:i0+n])
                i0 += n
            else:
                win.sort()
                y2[i] = np.mean(y[i0:i0+yn]*win[0:yn])
                x2[i] = np.median(x[i0:i0+yn])
        return x2, y2

    @ staticmethod
    def smoothing(y, n=10, std=1, amp=1):  # , deg):
        print(
            '\n \033[1;34m ### Starting kbreduct.smoothing ###', '\033[00m')
        num = len(y)
        n = int(n)
        if (n % 2) == 0:
            n += 1
        tn = int(n/2)
        y1 = np.zeros(len(y))
        win = np.zeros(n)
        for i in range(n):
            win[i] = amp*np.exp(-1./2.*((i-(n-1)/2.)/(std*(n-1.)/2.))**2)

        for i in range(num):
            binn = n
            y0 = np.array(y[i-tn:i+tn+1])
            if i-tn < 0:
                y0 = np.zeros(n)
                for j in range(n):
                    if i-tn+j >= 0:
                        y0[j] = y[i-tn+j]
                        binn = (n+(i-tn))
            if i+tn > num-1:
                y0 = np.zeros(n)
                for j in range(n):
                    if i-tn+j <= num-1:
                        y0[j] = y[i-tn+j]
                        binn = (n+(num-(i+tn)-1))
            #print('i : ', i, binn, len(y0))
            y1[i] = sum(win*y0)/binn
        return y1
