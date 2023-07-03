import pandas as pd
import numpy as np
from lmfit import Parameters
# for lmfit


def karof(p, x, y, **kwargs):

    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**3.5))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**6.2))
    pex = abs(p['p4'])*np.exp((-(x-abs(p['p5']))**2.) /
                              (2.*(abs(p['p6']))**2.))
    whn = abs(p['p7'])*x/x

    fv = bg0+bg1
    fv += pex
    fv += whn
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'pex': pex,
             'whn': whn
             })
        return result
    return fv - y


# def karof2(p, x, y, **kwargs):

#     value = kwargs.get('value', 'none')

#     value = kwargs.get('value', 'none')

#     bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
#            (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**abs(p['p7'])))
#     bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
#            (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**abs(p['p8'])))
#     pex = abs(p['p4'])*np.exp((-(x-abs(p['p5']))**2.) /
#                               (2.*(abs(p['p6']))**2.))
#     # whn = abs(p['p7'])*x/x

#     fv = bg0+bg1
#     fv += pex
#     # fv += whn
#     if value == 1:
#         result = pd.DataFrame(
#             {'fv': fv,
#              'bg0': bg0,
#              'bg1': bg1,
#              'pex': pex
#              # 'whn': whn
#              })
#         return result
#     return fv - y


def bgmsu3(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**2.+(2.*np.pi*x*abs(p['p1'])/1e6)**4.))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**2.+(2.*np.pi*x*abs(p['p3'])/1e6)**4.))
    bg2 = ((4.*abs(p['p4'])**2.*abs(p['p5'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p5'])/1e6)**2.+(2.*np.pi*x*abs(p['p5'])/1e6)**4.))

    pex = abs(p['p6'])*np.exp(-(((x-abs(p['p7']))**2.) /
                                (2.*(abs(p['p8']))**2.))**p['p9'])
    whn = 0  # abs(p['p9'])

    fv = bg0 + bg1
    fv += bg2
    fv += pex
    # fv += whn

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'bg2': bg2,
             'pex': pex,
             'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y


def bgm2(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')
    # white = kwargs.get('white', 1)
#    print(white)
    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**2.+(2.*np.pi*x*abs(p['p1'])/1e6)**4.))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**2.+(2.*np.pi*x*abs(p['p3'])/1e6)**4.))

    pex = abs(p['p4'])*np.exp((-(x-abs(p['p5']))**2.) /
                              (2.*(abs(p['p6']))**2.))

    whn = abs(p['p7'])*x/x

    fv = bg0 + 0.
    fv += bg1
    fv += pex
    fv += whn
    # if white != 0:
    #     whn = abs(p['p7'])
    #     fv += whn
    # else:
    #     whn = np.float64(np.zeros(len(x)))
    #     whn = whn + np.float('NaN')

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'pex': pex,
             'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y


def bgm3(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**2.+(2.*np.pi*x*abs(p['p1'])/1e6)**4.))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**2.+(2.*np.pi*x*abs(p['p3'])/1e6)**4.))
    bg2 = ((4.*abs(p['p4'])**2.*abs(p['p5'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p5'])/1e6)**2.+(2.*np.pi*x*abs(p['p5'])/1e6)**4.))

    pex = abs(p['p6'])*np.exp((-(x-abs(p['p7']))**2.) /
                              (2.*(abs(p['p8']))**2.))
    whn = abs(p['p9'])

    fv = bg0+bg1+bg2
    fv += pex
    fv += whn

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'bg2': bg2,
             'pex': pex,
             'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y


def bgm0w(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**3.9))
    pex = abs(p['p2'])*np.exp((-(x-abs(p['p3']))**2.) /
                              (2.*(abs(p['p4']))**2.))
    whn = abs(p['p5'])*x/x
    fv = bg0+pex
    fv += whn
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'pex': pex,
             'whn': whn
             })
        return result
    return fv - y


def bgm0n(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**3.7))
    pex = abs(p['p2'])*np.exp((-(x-abs(p['p3']))**2.) /
                              (2.*(abs(p['p4']))**2.))
    # whn = abs(p['p5'])*x/x
    fv = bg0+pex
    # fv += whn
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'pex': pex  # ,
             # 'whn': whn
             })
        return result
    return fv - y


def kbgauss(p, x, y, **kwargs):

    value = kwargs.get('value', 'none')

    pex = abs(p['p0'])*np.exp((-(x-abs(p['p1']))**2.) /
                              (2.*(abs(p['p2']))**2.))
    whn = abs(p['p3'])  # - abs(p[4])

    fv = (pex+whn)
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv, 'pex': pex, 'whn': whn})
        return result
    return fv - y


def kbgausslog(p, x, y, **kwargs):

    value = kwargs.get('value', 'none')

    # pex = np.log10(abs(p['p0']))+np.exp((-(x-abs(p['p1']))**2.) /
    #                                    (2.*(abs(p['p2']))**2.)))
    pex = np.log10(abs(p['p0']))+((-(x-abs(p['p1']))**2.) /
                                  (2.*(abs(p['p2']))**2.))*np.log10(np.exp(1))
#    whn = abs(p['p3'])  # - abs(p[4])

    fv = (pex)  # + whn)
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv, 'pex': pex})  # , 'whn': whn})
        return result
    return fv - y


def param_value(p):
    # Parameter type to genernal array
    p0 = []
    p_std = []
    for name, param in p.params.items():
        p0 = np.append(p0, param.value)
        p_std = np.append(p_std, param.stderr)
    result = pd.DataFrame(
        {'p': p0, 'st': p_std})
    return result


def kbparameters(p, **kwargs):
    mini = kwargs.get('mini', np.zeros_like(p))
    maxi = kwargs.get('maxi', np.zeros_like(p)+np.float('Inf'))
    maxi = np.float64(maxi)
    n = len(p)

    p0 = Parameters()
    for i in range(n):
        ps = 'p'+str(i)
        p0.add(ps, p[i], min=mini[i], max=maxi[i])

    return p0


def xyfit(p, x, y, *args, **kwargs):  #
    value = kwargs.get('value', 'none')
    x = np.array(x)
    y = np.array(y)
    # print(len(args))

    if len(args) == 0:
        sig = np.zeros_like(x)+1
    else:
        sig = np.array(args)
    # print(sig)
    # print(len(sig))
    # fv = p['p1'] * x ** p['p0']
    fv = p['p0'] + p['p1'] * x
    # print(x, y)
    # print(fv)
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv})
        return result
    # print('fv-y ', (fv-y) - (fv-y)/sig)
    # print(fv-y)/args
    return (fv - y) * sig


def xyfitmodel(x, a, b, **kwargs):  # *args
    value = kwargs.get('value', 'none')
    x = np.array(x)

    # print(len(args))

    # if len(args) == 0:
    #     sig = np.zeros_like(x)+1
    # else:
    #     sig = np.array(args)
    # print(sig)
    # print(len(sig))
    fv = a + b * x
    # print(x, y)
    # print(fv)
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv})
        return result
    # print('fv-y ', (fv-y) - (fv-y)/sig)
    # print(fv-y)/args
    return fv  # *sig


def xyfitg(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    # fv = p['p1'] * x ** p['p0']
    bg0 = 10**(p['p0'])*10**(p['p1'] * x)
    # bg0 = p['p0']*x**p['p1']
    pex = abs(p['p2'])*np.exp((-(x-abs(p['p3']))**2.) /
                              (2.*(abs(p['p4']))**2.))  # - p['p5']*x/x
    fv = bg0 + pex  # + whn
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv, 'bg0': bg0, 'pex': pex})
        return result
    return fv - y


def xyfitglog(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    # bg0 = p['p1'] * x ** p['p0']
    # bg0 = 10**(p['p0'] + p['p1']*x)
    # pex = abs(p['p2'])*np.exp((-(x-abs(p['p3']))**2.) /
    #                           (2.*(abs(p['p4']))**2.))  # - p['p5']*x/x
    # print(10**bg0)
    # bg0 = 10**(p['p0']) * 10**(p['p1']*x)
    # bg0 = np.log10(((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
    #                 (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**4)))
    # print(p)

    # pex = np.log10(pex)
    bg0 = p['p0'] + p['p1'] * x
    # pex = np.log10(abs(p['p2']))+np.exp((-(x-abs(p['p3']))**2.) /
    #                                    (2.*(abs(p['p4']))**2.)))
    # print(pex, ',', np.exp((-(x-abs(p['p3']))**2.) /
    #                        (2.*(abs(p['p4']))**2.)), abs(p['p2']))
    pex = np.log10(abs(p['p2']))+((-(x-abs(p['p3']))**2.) /
                                  (2.*(abs(p['p4']))**2.))*np.log10(np.exp(1))  #
    # whn = p['p5']  # *x/x
    fv = bg0 + pex  # + whn
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv, 'bg0': bg0, 'pex': pex})  # , 'whn': whn})
        return result
    return fv - y


def xyfitp(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    # fv = p['p1'] * x ** p['p0']
    bg0 = p['p0'] * x ** p['p1']
    fv = bg0
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv, 'bg0': bg0})
        return result
    return fv - y


def xyfit2(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    # fv = p['p1'] * x ** p['p0']
    bg0 = p['p0'] * x ** p['p1']
    pex = abs(p['p2'])*np.exp((-(x-abs(p['p3']))**2.) /
                              (2.*(abs(p['p4']))**2.))
    fv = bg0 + pex
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv, 'bg0': bg0, 'pex': pex})
        return result
    return fv - y


def harvey2_4(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**4))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**4))

    pex = abs(p['p4'])*np.exp((-(x-abs(p['p5']))**2.) /
                              (2.*(abs(p['p6']))**2.))
    whn = abs(p['p7'])

    fv = bg0 + bg1
    fv += pex
    fv += whn

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'pex': pex,
             'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y


def harvey3_4(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**4.))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**4.))
    bg2 = ((4.*abs(p['p4'])**2.*abs(p['p5'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p5'])/1e6)**4.))

    pex = abs(p['p6'])*np.exp(-(((x-abs(p['p7']))**2.) /
                                (2.*(abs(p['p8']))**2.)))
    whn = abs(p['p9'])

    fv = bg0 + bg1
    fv += bg2
    fv += pex
    fv += whn

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'bg2': bg2,
             'pex': pex,
             'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y


def harvey1(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**p['p2']))
    pex = abs(p['p3'])*np.exp((-(x-abs(p['p4']))**2.) /
                              (2.*(abs(p['p5']))**2.))
    # whn = abs(p['p6'])*x/x
    fv = bg0+pex
    # fv += whn
    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'pex': pex
             # 'whn': whn
             })
        return result
    return fv - y


def harvey2(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**p['p7']))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**p['p8']))

    pex = abs(p['p4'])*np.exp((-(x-abs(p['p5']))**2.) /
                              (2.*(abs(p['p6']))**2.))
   # whn = abs(p['p7'])

    fv = bg0 + bg1
    fv += pex
    # fv += whn

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'pex': pex  # ,
             # 'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y


def harvey3(p, x, y, **kwargs):
    value = kwargs.get('value', 'none')

    bg0 = ((4.*abs(p['p0'])**2.*abs(p['p1'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p1'])/1e6)**p['p9']))
    bg1 = ((4.*abs(p['p2'])**2.*abs(p['p3'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p3'])/1e6)**p['p10']))
    bg2 = ((4.*abs(p['p4'])**2.*abs(p['p5'])/1e6) /
           (1.+(2. * np.pi*x*abs(p['p5'])/1e6)**p['p11']))

    pex = abs(p['p6'])*np.exp(-(((x-abs(p['p7']))**2.) /
                                (2.*(abs(p['p8']))**2.)))
    # whn = abs(p['p9'])

    fv = bg0 + bg1
    fv += bg2
    fv += pex
    # fv += whn

    if value == 1:
        result = pd.DataFrame(
            {'fv': fv,
             'bg0': bg0,
             'bg1': bg1,
             'bg2': bg2,
             'pex': pex
             # 'whn': whn
             })
        # {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
        return result
    return fv - y
