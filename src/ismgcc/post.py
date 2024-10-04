import numpy as np
import pandas as pd
from astropy import units as u
from tqdm import tqdm
from pathlib import Path
from spectral_cube import SpectralCube
from skimage.morphology import flood_fill
from skimage.measure import find_contours
from regions import PixCoord, PolygonPixelRegion, PolygonSkyRegion, Regions
from .utils import tqdm_joblib
from joblib import Parallel, delayed
from typing import Optional
import os

def get_boundary_by_flood(xy_list: np.ndarray):
    """Flood fill the boundary of a cloud structure

    Args:
        xy_list (np.ndarray): (n_points, 2)

    Returns:
        returns a PolygonPixelRegion object
    """
    xmin, ymin = xy_list.min(axis=0)
    xmax, ymax = xy_list.max(axis=0)
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    x = xy_list[:, 0] - xmin
    y = xy_list[:, 1] - ymin

    bw0 = np.zeros(shape=shape)
    bw0[y, x] = 1
    ii = np.zeros((bw0.shape[0] + 4, bw0.shape[1] + 4))
    ii[2:-2, 2:-2] = bw0
    iii = flood_fill(ii, (0, 0), new_value=-1, tolerance=0.1, connectivity=1)
    bw1 = (iii > -1)
    css = find_contours(bw1, level=0.5,fully_connected="high")
    contour = sorted(css, key=len, reverse=True)[0]
    contour = contour - 2
    contour[:, 0] += ymin
    contour[:, 1] += xmin
    pixcoords = PixCoord(x=[x for y, x in contour], y=[y for y, x in contour])
    pixreg = PolygonPixelRegion(pixcoords)
    return pixreg

def get_pixel_wise_table(cdf):
    g = cdf.groupby(["x_pos", "y_pos"])
    vlsr_by_pix = g.apply(get_vlsr_of_pixel)
    vel_disp_by_pix = g.apply(get_vel_disp_of_pixel)
    vlsr_by_pix.name = "VLSR"
    vel_disp_by_pix.name = "vel_disp"
    W = g.apply(get_int_tot_of_pixel)
    W.name = "int_tot"

    outdf = pd.DataFrame([vlsr_by_pix, vel_disp_by_pix, W], ).T
    outdf["Teq"] = W / np.sqrt(2 * np.pi) / outdf["vel_disp"]
    return outdf

def get_vlsr_of_pixel(vc_df_in_one_pix):
    t = vc_df_in_one_pix
    v = t["VLSR"]
    w = t["int_tot"]
    W = np.sum(w)
    return np.sum(w * v) / W

def get_vel_disp_of_pixel(vc_df_in_one_pix):
    t = vc_df_in_one_pix
    sigma = t["vel_disp"]
    v = t["VLSR"]
    w = t["int_tot"]
    W = np.sum(w)

    v_mean = get_vlsr_of_pixel(t)

    t = (np.sum(w * (sigma ** 2 + (v - v_mean) ** 2 ) ) )  / W
    return np.sqrt(t)

def get_int_tot_of_pixel(vc_df_in_one_pix):
    t = vc_df_in_one_pix
    w = t["int_tot"]
    W = np.sum(w)
    return W

from scipy.interpolate import LinearNDInterpolator
def get_pixel_wise_vlo_vhi(pix_wise_df, n_sigma=1.5):
    x = pix_wise_df["x_pos"].values
    y = pix_wise_df["y_pos"].values
    xy = np.vstack([x, y]).T
    z = pix_wise_df["VLSR"].values
    vel_disp = pix_wise_df["vel_disp"].values
    mid_surf = LinearNDInterpolator(xy, z,)
    vhi_surf = LinearNDInterpolator(xy, z + n_sigma * vel_disp,)
    vlo_surf = LinearNDInterpolator(xy, z - n_sigma * vel_disp,)

    X, Y = np.meshgrid(np.arange(x.min(), x.max() + 1), np.arange(y.min(), y.max() + 1))
    Z0 = mid_surf(X, Y)
    Z1 = vlo_surf(X, Y)
    Z2 = vhi_surf(X, Y)

    v_lo_hi_surf_df = pd.DataFrame(
        {
            "x_pos": X.flatten(),
            "y_pos": Y.flatten(),
            "vmid": Z0.flatten(),
            "vlo": Z1.flatten(),
            "vhi": Z2.flatten(),
        }
        ).dropna().set_index(["x_pos", "y_pos"])

    return v_lo_hi_surf_df

def add_Tpeak_vpeak(pixdf, cube):
    pixdf = pixdf.copy()
    vaxis = cube.spectral_axis
    x = pixdf["x_pos"].values
    y = pixdf["y_pos"].values
    vlo = pixdf["vlo"].values * u.km / u.s
    vhi = pixdf["vhi"].values * u.km / u.s
    zlo = np.array(np.round(cube.wcs.spectral.world_to_pixel(vlo)), int)
    zhi = np.array(np.round(cube.wcs.spectral.world_to_pixel(vhi)), int)
    n_channels = len(vaxis)
    zlo[zlo < 0] = 0
    zhi[zhi >= n_channels] = n_channels - 1

    T_peaks = []
    v_peaks = []
    z_peaks = []
    for i in range(len(pixdf)):
        if pixdf.iloc[i, :].isna().any():
            # if ther is any NaN in current row
            T_peaks.append(np.nan)
            v_peaks.append(np.nan * u.km / u.s)
            z_peaks.append(-1)
        else:
            z1, z2 = zlo[i], zhi[i]
            z1, z2 = min(z1, z2), max(z1, z2)
            spec = cube.filled_data[z1: z2+1, y[i], x[i] ]
            if len(spec) > 0:
                T_peak = spec.max().value 
                idx_peak = np.argmax(spec)
                v_peak = vaxis[zlo[i]: zhi[i] + 1][idx_peak]

                T_peaks.append(T_peak)
                v_peaks.append(v_peak)
                z_peaks.append(zlo[i] + idx_peak)
            else:
                T_peaks.append(np.nan)
                v_peaks.append(np.nan * u.km / u.s)
                z_peaks.append(-1)

    pixdf["T_peak"] = T_peaks
    pixdf["v_peak"] = u.Quantity(v_peaks).to(u.km / u.s).value
    pixdf["z_peak"] = z_peaks

    return pixdf

class PostProcess:
    def _load_default_params(self):
        self.params = {
            "minimal_number_of_pixels": 16,
        }
        
    def __init__(self, 
                 cube: SpectralCube, 
                 table: pd.DataFrame, 
                 cluster_col: str,
                 cluster_col_out: str,
                 n_jobs: Optional[int]=None
                 ):
        """Initialize the PostProcess object

        Args:
            cube (SpectralCube): the data cube before gaussian decomposition
            table (pd.DataFrame): the output table given by a DecomposedPPVStructureFinder
            cluster_col (str): name of the column in `table` considered as the cluster id
            n_jobs (int): Number of cpus used 
        """
        self._load_default_params()
        self.table = table
        self.cube = cube
        self.cluster_col = cluster_col
        self.cluster_col_out = cluster_col_out
        self.n_jobs = n_jobs if n_jobs else (int(os.cpu_count() * 0.5) - 1) 

    @staticmethod
    def _post_process_one_structure(df, sid, cluster_col_in="serial_id", cluster_col_out="serial_id1"):
        idx = df[cluster_col_in] == sid
        cdf = df.loc[idx, :].copy()
        boundary = get_boundary_by_flood(cdf[["x_pos", "y_pos"]].values)

        pix_wise_df = get_pixel_wise_table(cdf)
        pix_wise_df = pix_wise_df.reset_index()
        v_lo_hi_surf_df = get_pixel_wise_vlo_vhi(pix_wise_df)
        pix_wise_df = pix_wise_df.join(v_lo_hi_surf_df, on=["x_pos", "y_pos"], how="left")
        idx = boundary.contains(PixCoord(x=df["x_pos"], y=df["y_pos"]))

        col_name = [c for c in cdf.columns.tolist() if c.startswith("has") and c.endswith("pix")][0]
        ccdf = df.loc[idx, :].join(v_lo_hi_surf_df, on=["x_pos", "y_pos"], how="inner")
        ccdf = ccdf.query("VLSR < vhi and VLSR > vlo and serial_id != @sid")
        ccdf = ccdf.loc[~ccdf[col_name] , :] # weak and broad VCs 
        cdf["is_added_by_vlo_vhi"] = False
        ccdf["is_added_by_vlo_vhi"] = True

        new_cdf = pd.concat([cdf, ccdf]).sort_index()
        new_cdf[cluster_col_out] = sid
        new_pix_wise_df = get_pixel_wise_table(new_cdf)
        new_pix_wise_df[cluster_col_out] = sid
        new_pix_wise_df = new_pix_wise_df.reset_index()
        new_v_lo_hi_surf_df = get_pixel_wise_vlo_vhi(new_pix_wise_df)
        new_pix_wise_df = new_pix_wise_df.join(new_v_lo_hi_surf_df, on=["x_pos", "y_pos"], how="left")

        v_mean = np.average(new_pix_wise_df["VLSR"].values, weights=new_pix_wise_df["int_tot"].values)
        boundary.meta["text"] = f"serial_id:{sid:d} v:{v_mean:.1f}"

        return boundary, pix_wise_df, ccdf, new_pix_wise_df, new_cdf

    def process(self, output_prefix, output_dir="./"):
        df = self.table
        n_pix_min = self.params["minimal_number_of_pixels"],
        filtered_df = df.query("pix_count >= @n_pix_min")

        path_out = Path(output_dir)
        sids = filtered_df[self.cluster_col].unique()
        if len(sids) == 0:
            output = path_out / (output_prefix +  "-NO-VALID-STRUCTS.txt")
            with open(output, "w") as f:
                f.write(f"No valid structures with Npix >= {n_pix_min}")
            return 
            
        # results = []
        # for sid in tqdm(sids):
        #     r = self._post_process_one_structure(sid)
        #     results.append(r)
        # parallel version 
        with tqdm_joblib(tqdm(desc="Post Process...", total=len(sids))) as progress_bar:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._post_process_one_structure)(
                    self.table, sid, self.cluster_col, self.cluster_col_out
                )  for sid in sids)

        boundaries = Regions([r[0] for r in results ])
        boundaries_wcs = Regions([r[0].to_sky(self.cube.wcs.celestial) for r in results ])
        all_new_cdf = pd.concat([r[4] for r in results]).reset_index(drop=True)
        all_pix_wise_df_before_vlo_hi = pd.concat([r[1] for r in results]).reset_index(drop=True)
        all_pix_wise_df = pd.concat([r[3] for r in results]).reset_index(drop=True)

        cube = self.cube.with_spectral_unit(u.km / u.s)
        all_pix_wise_df = add_Tpeak_vpeak(all_pix_wise_df, cube)
        all_pix_wise_df_before_vlo_hi = add_Tpeak_vpeak(all_pix_wise_df_before_vlo_hi, cube)

        output = path_out / (output_prefix +  "-pixel-boundaries.reg")
        boundaries.write(output, format="ds9", overwrite=True)
        output = path_out / (output_prefix +  "-pixel-boundaries-wcs.reg")
        boundaries_wcs.write(output, format="ds9", overwrite=True)
        output = path_out / (output_prefix +  "-VCs-after-vlohi.csv")
        all_new_cdf.to_csv(output, index=False)
        output = path_out / (output_prefix +  "-pixels-before-vlohi.csv")
        all_pix_wise_df_before_vlo_hi.to_csv(output, index=False)
        output = path_out / (output_prefix +  "-pixels-after-vlohi.csv")
        all_pix_wise_df.to_csv(output, index=False)

        