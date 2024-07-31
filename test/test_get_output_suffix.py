from ismgcc import DecomposedPPVStructureFinder, PostProcess
import pandas as pd
from spectral_cube import SpectralCube
from astropy import units as u

bwc = 0.5
r = 3
snrth0 = 0
snrth1 = 5
db = 0.5
resolution = 0.01
n_process = 4

input_file = "../example/demo-data/GL14_fit_fin_sf-p2_finalize.csv"
df = pd.read_csv(input_file)

finder = DecomposedPPVStructureFinder(df, params={
    "bandwidth_coef": bwc, 
    "r": r, 
    "snr_th0": snrth0, 
    "snr_th1": snrth1, 
    "decision_boundary": db, 
    "community_resolution": resolution
    },
    n_jobs=n_process
    )

output_suffix = finder.get_output_suffix()
print(output_suffix)