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

input_file = "./demo-data/GL14_fit_fin_sf-p2_finalize.csv"
df = pd.read_csv(input_file)

finder = DecomposedPPVStructureFinder(df, params={
    "bandwidth_coef": bwc, 
    "r": r, 
    "snr_th0": snrth0, 
    "snr_th1": snrth1, 
    "decision_boundary": db, 
    "community_resolution": resolution,
    },
    n_jobs=n_process
    )
output_suffix = finder.get_output_suffix()
# outdf = finder.find_structures()
output = f"./demo-results/GL14-{output_suffix}.csv"
# outdf.to_csv(output, index=False)

outdf = pd.read_csv(output)
file = "./demo-data/GL14.fits.gz"
u.add_enabled_units(u.def_unit(['K (T_MB)'], represents=u.K)) 
cube = SpectralCube.read(file)
pp = PostProcess(cube, outdf, "serial_id", 'serial_id1', n_jobs=4)
pp.process("GL14-" + output_suffix, "./demo-results/")
