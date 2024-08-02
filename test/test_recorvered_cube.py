# %%
from pathlib import Path
import os
os.chdir(Path(__file__).parent)
# %%


from ismgcc import get_recorvered_cube, get_rawcube_cutout_as_recovred_cube
import pandas as pd
from spectral_cube import SpectralCube
from astropy import units as u

rawcube = SpectralCube.read("../example/demo-data/GL14.fits.gz")
vcdf = pd.read_csv("../example/demo-results/GL14L-bwc=0.50-r=3.00-db=0.50-snrth0=0.0-snrth1=5.0-resolution=0.01-VCs-after-vlohi.csv")
vcdf = vcdf.query("serial_id1 == 4")
rcube = get_recorvered_cube(rawcube, vcdf, recovred_value_threshold=0.1)
rcube.write("recorvered_cube_sid2.fits")

cutout_cube = get_rawcube_cutout_as_recovred_cube(rawcube, rcube, recovered_threshold=0.3)
cutout_cube.write("cutout_rawcube_sid2.fits")

cutout_cube = get_rawcube_cutout_as_recovred_cube(rawcube, rcube, recovered_threshold=-1)
cutout_cube.write("cutout_rawcube_sid2_nomask.fits")
