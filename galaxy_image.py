import os
import subprocess
from tkinter import W

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii, fits
from astropy.wcs import WCS
from marvin import config
from marvin.tools.cube import Cube
from marvin.tools.image import Image
from marvin.tools.maps import Maps

from fit_kinematic_pa_kb2 import fit_kinematic_pa_kb

import multiprocessing as mp
from functools import partial
import warnings

warnings.filterwarnings("ignore")


plt.style.use("seaborn-darkgrid")

matplotlib.rcdefaults()
config.setRelease("DR17")


def kb_multi(df, ty, fold, i):
    print(ty, fold)
    filename = "8.1.8.corrected_pafit_1re_nonzero_frac_marvin_list.star.gas.SFE_220719"

    mname = df.plateifu[i]

    print(i, df.name[i], df.plateifu[i], "\n")
    rows = 2
    cols = 4
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(50, 25))

    image = Image(plateifu=mname)

    #! [0, 0] 이미지 그리기
    image_kb = image.data
    image_kb = np.array(image_kb)
    axes[0, 0].imshow(image_kb)

    maps = Maps(plateifu=mname)
    map_header = maps.header

    ha = maps.emline_gflux_ha_6564
    ha_map = ha.value

    # --- stellar_vel map
    velocity_flux = maps.stellar_vel
    velocity_map = velocity_flux.value

    velocity_flux.plot(fig=fig, ax=axes[0, 1])

    # --- gas_vel map -----
    velocity_ha = maps.emline_gvel_ha_6564
    velocity_hamap = velocity_ha.value

    velocity_ha.plot(fig=fig, ax=axes[1, 1])

    rbin_map = maps.spx_ellcoo_elliptical_radius
    rbin_value = rbin_map.value
    rbin_map.plot(fig=fig, ax=axes[1, 0])

    r_eff = map_header["reff"]

    ind_re = np.where(rbin_value == np.min(rbin_value))
    xc = ind_re[1]
    yc = ind_re[0]

    #! ========================================
    error = ""
    for j in range(2):
        if j == 0:
            vel_map = velocity_map
        else:
            vel_map = velocity_hamap

        ind = np.where(
            (np.array(rbin_value) < float(r_eff)) & (vel_map != 0)
        )

        xind = ind[1]
        yind = ind[0]
        xbin = xind - xc
        ybin = yind - yc

        vel_median = np.median(vel_map[yind, xind])
        velocity_remain = vel_map - vel_median

        ax0 = 243 + 4 * j
        ax1 = 244 + 4 * j

        try:
            (
                angBest,
                angErr,
                vSyst,
                velSym,
                angBest_kb,
            ) = fit_kinematic_pa_kb(
                xbin,
                ybin,
                velocity_remain[yind, xind],
                ax0,
                ax1,
                quiet="True",
                plot=True,
            )
            if error == "error":
                plt.savefig(
                    "figures_failed/8.1.8.corrected_pafit_1re_nonzero.SFE_kinematic_PA.220720_%s_res.pdf"
                    % i
                )
                continue
        except:
            print("kinematic calculation error")
            error = "error"
            #! 이름 알맞게 변경
            if j != 0:
                plt.savefig(
                    "figures_failed/8.1.8.corrected_pafit_1re_nonzero.SFE_kinematic_PA.220720_%s_res.pdf"
                    % i
                )
            continue

        if j == 0:
            data = [df.name[i], mname]
            result0 = pd.DataFrame(
                np.array(data).reshape(1, 2)
            )  # , columns = ["i", "name", "s_PA", "s_e", "s_offset"])
            result0["xc"] = xc
            result0["yc"] = yc
            result0["star_pa"] = angBest
            result0["star_err"] = angErr
            result0["star_offset"] = vSyst
            result0["star_median"] = vel_median
            result0["star_velSym"] = velSym[0]
        else:
            result0["gas_pa"] = angBest
            result0["gas_err"] = angErr
            result0["gas_offset"] = vSyst
            result0["gas_median"] = vel_median
            result0["gas_velSym"] = velSym[0]
            result0["r_eff"] = r_eff

            result0.to_csv(
                "results_csv/" + filename + "_%s.csv" % i,
                float_format="%.2f",
            )
        axes[j, 2].set_xlabel("Pa : %s" % angBest, fontsize=20)

        axes[j, 3].set_xlabel("kb_Pa : %s" % angBest_kb, fontsize=20)

    plt.suptitle("Galaxy Number : %s" % i, fontsize=25)
    if error != "error":
        plt.savefig(
            "figures/8.1.8.corrected_pafit_1re_nonzero.SFE_kinematic_PA.220720_%s_res.pdf"
            % i
        )


if __name__ == "__main__":
    dir = ""
    dr17 = dir + "1.data/8.data_dr17/"
    type0 = "SFE"
    fold = " fofofo"

    list = "1.5.6.SFR_M.id." + type0 + ".220621.txt"

    df = pd.read_fwf(list)

    procn = 20
    pool0 = mp.Pool(processes=procn)

    a = [i for i in range(len(df))]
    a = [0, 1, 2, 3, 4]
    #! 멀티프로세싱 처음 사용 할 때 주석 제거

    ret = pool0.map(partial(kb_multi, df, type0, fold), a)

    subprocess.call(
        ["pdfunite ./figures/*_res.pdf ./figures/total.pdf"],
        shell=True,
    )
