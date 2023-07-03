import numpy as np
import pandas as pd


def kbgm(p, x, **kwargs):
    model = kwargs.get('model', 'None')
    value = kwargs.get('value', 'None')
    if model == 'None':
        model = 'bgmhu3'

    ### backgournd bgmhu3 ###
    if model == 'bgmhu3':

        fv1 = ((4.*abs(p[0])**2.*abs(p[1])/1e6) /
               (1.+(2. * np.pi*x*abs(p[1])/1e6)**2.+(2.*np.pi*x*abs(p[1])/1e6)**4.))
        fv2 = ((4.*abs(p[2])**2.*abs(p[3])/1e6) /
               (1.+(2. * np.pi*x*abs(p[3])/1e6)**2.+(2.*np.pi*x*abs(p[3])/1e6)**4.))
        fv3 = ((4.*abs(p[4])**2.*abs(p[5])/1e6) /
               (1.+(2. * np.pi*x*abs(p[5])/1e6)**2.+(2.*np.pi*x*abs(p[5])/1e6)**4.))
        fv4 = abs(p[6])
        fv5 = abs(p[7])*np.exp((-(x-abs(p[8]))**2.) /
                               (2.*(abs(p[9]))**2.))
        fv = (fv1+fv2+fv3+fv4+fv5)
        if value == 1:
            result = pd.DataFrame(
                {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv5': fv5, 'fv': fv})
            return result
        return fv

    ### backgournd bgmhu2 ###
    if model == 'bgmhu2':

        fv1 = ((4.*abs(p[0])**2.*abs(p[1])/1e6) /
               (1.+(2. * np.pi*x*abs(p[1])/1e6)**2.+(2.*np.pi*x*abs(p[1])/1e6)**4.))
        fv2 = ((4.*abs(p[2])**2.*abs(p[3])/1e6) /
               (1.+(2. * np.pi*x*abs(p[3])/1e6)**2.+(2.*np.pi*x*abs(p[3])/1e6)**4.))
        fv3 = abs(p[4])  # -abs(p[4])
        fv4 = abs(p[5])*np.exp((-(x-abs(p[6]))**2.) /
                               (2.*(abs(p[7]))**2.))
        fv = (fv1+fv2+fv3+fv4)
        if value == 1:
            result = pd.DataFrame(
                {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
            return result
        return fv

    ### backgournd bgmcorg ###
    if model == 'bgmcorg':

        fv1 = ((4.*p[0]**2.*abs(p[1])/1e6) /
               (1.+(2. * np.pi*x*abs(p[1])/1e6)**abs(p[2])))
        fv2 = abs(p[3])
        fv3 = abs(p[4])*np.exp((-(x-abs(p[5]))**2.) /
                               (2.*(abs(p[6]))**2.))
        fv = (fv1+fv2+fv3)
        if value == 1:
            result = pd.DataFrame(
                {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv': fv})
            return result
        return fv

    if model == 'bgmkarof':
        # karoff (2012) mnras
        fv1 = ((4.*abs(p[0])**2.*abs(p[1])/1e6) /
               (1.+(2. * np.pi*x*abs(p[1])/1e6)**3.5+(2.*np.pi*x*abs(p[1])/1e6)**4.))
        fv2 = ((4.*abs(p[2])**2.*abs(p[3])/1e6) /
               (1.+(2. * np.pi*x*abs(p[3])/1e6)**6.2+(2.*np.pi*x*abs(p[3])/1e6)**4.))
        fv3 = abs(p[4])  # - abs(p[4])
        fv4 = abs(p[5])*np.exp((-(x-abs(p[6]))**2.) /
                               (2.*(abs(p[7]))**2.))
        fv = (fv1+fv2+fv3+fv4)
        if value == 1:
            result = pd.DataFrame(
                {'fv1': fv1, 'fv2': fv2, 'fv3': fv3, 'fv4': fv4, 'fv': fv})
            return result
        return fv

    if model == 'gauss':
        fv1 = abs(p[3])
        fv2 = abs(p[0])*np.exp((-(x-abs(p[1]))**2.) /
                               (2.*(abs(p[2]))**2.))
        fv = fv1 + fv2

        return fv


def kfunc(p, x, y, model1):
    #    print(model1)
    return kbgm(p, x, model=model1) - y


# def generate_data(t, A, sigma, omega, noise=0, n_outliers=0, random_state=0):
#     y = A * np.exp(-sigma * t) * np.sin(omega * t)
#     rnd = np.random.RandomState(random_state)
#     error = noise * rnd.randn(t.size)
#     outliers = rnd.randint(0, t.size, n_outliers)
#     error[outliers] *= 35
#     return y + error


# def model2(x, u):
#     return x[0] * np.exp(-x[1] * u) * np.sin(x[2] * u)


# def fun2(x, u, y):
#     return model2(x, u) - y


# # def model(x, u):
# #    return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])


# def fun(x, u, y):
#     return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3]) - y
# #    return model(x, u) - y
