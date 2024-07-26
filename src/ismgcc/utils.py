from tqdm import tqdm
import pandas as pd
import numpy as np

def get_gauss_profiles(vaxis, amp, vlsr, vel_disp):
    return amp * np.exp(-((vaxis - vlsr) / vel_disp) ** 2 / 2)

def get_synthetic_spectra_batch(vcdf, vaxis):
    n_vc = len(vcdf)
    vaxis_2d = np.tile(vaxis, (n_vc, 1))
    amp = vcdf["amp"].values.reshape(n_vc, 1)
    vlsr = vcdf["VLSR"].values.reshape(n_vc, 1)
    vel_disp = vcdf["vel_disp"].values.reshape(n_vc, 1)
    spec_2d = get_gauss_profiles(vaxis_2d, amp, vlsr, vel_disp)
    return spec_2d

def get_synthetic_spectra_of_single_pixels(vcdf, vaxis):
    spec_2d = get_synthetic_spectra_batch(vcdf, vaxis)
    outdf = vcdf[["x_pos", "y_pos"]].copy()
    spec_df = pd.DataFrame(spec_2d, columns=vaxis, index=outdf.index)
    spec_df = pd.concat([outdf, spec_df], axis=1)
    return spec_df.groupby(["x_pos", "y_pos"]).sum()

def get_synthetic_average_spectrum_of_one_struct(vcdf, vaxis):
    n_vc_each_xy = vcdf[["x_pos", "y_pos"]].value_counts()
    n_pix = len(n_vc_each_xy)
    spec_2d = get_synthetic_spectra_batch(vcdf, vaxis)
    syn_avg_spec = spec_2d.sum(axis=0) / n_pix
    return syn_avg_spec

def get_v_lim_by_percentage_of_peak(vaxis, spec, percent=0.1):
    threshold = np.max(spec) * percent
    indices = np.where(spec >= threshold)
    i1, i2 = np.min(indices), np.max(indices)
    v1, v2 = vaxis[i1], vaxis[i2]
    if v1 <= v2:
        return v1, v2, i1, i2
    else:
        return v2, v1, i2, i1

def get_synthetic_single_spectra_for_all_struct(vcdf, vaxis, id_col="serial_id1"):
    g = vcdf.groupby(id_col)
    output = []
    for i, cdf in tqdm(g):
        r = get_synthetic_spectra_of_single_pixels(cdf, vaxis)
        r["serial_id1"] = i
        output.append(r)
    outdf = pd.concat(output).reset_index(drop=False).set_index(["serial_id1", "x_pos", "y_pos"]).sort_index()
    return outdf


from spectral_cube import SpectralCube, Projection
def goodlooking(cube: SpectralCube, rms: Projection):
    # https://github.com/shbzhang/mwispy/blob/main/src/mwispy/cubemoment.py
    rmsdata = rms.value
    cubedata = cube.filled_data[:].value.copy()

    #only keep those pixels with at least 3 "consecutive" channels above 3 sigma
    mskcube = np.ndarray(cube.shape, dtype=bool)

    for i in range(cube.shape[0]):
        mskcube[i,:,:] = cubedata[i,:,:] > rmsdata * 3
    mskcube = mskcube & np.roll(mskcube, 1, axis=0) & np.roll(mskcube, 2, axis=0)
    mskcube = mskcube | np.roll(mskcube, -1, axis=0) | np.roll(mskcube, -2, axis=0)
    cubedata *= mskcube 

    return SpectralCube(cubedata, cube.wcs)

import numpy as np
import pandas as pd
from spectral_cube import Projection
from astropy.io import fits
from astropy.wcs import WCS

def read_projection_from_file(file, hdu_i=0):
    try:
        with fits.open(file) as f:
            proj = Projection.from_hdu(f[hdu_i])
    except ValueError as e:
        hdr = fits.getheader(file,)
        data = fits.getdata(file,)
        wcs=WCS(hdr)
        while len(wcs.get_axis_types()) > 2:
            wcs = wcs.dropaxis(-1)

        proj = Projection(data, wcs=wcs)

    return proj


def get_map_2d_from_pix_table(table, wcs2d, shape_out, value_col, unit, default_value=np.nan, x_col="x_pos", y_col="y_pos"):
    x = np.array(np.round(table[x_col].values), dtype=int)
    y = np.array(np.round(table[y_col].values), dtype=int)
    values = table[value_col].values
    arr2d = np.zeros(shape_out, dtype=values.dtype) * default_value
    arr2d[y, x] = values
    return Projection(arr2d, unit=unit, wcs=wcs2d)
