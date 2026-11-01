import json
import os
import numpy as np
import xtrack as xt

# Load the line from the JSON file
file = '/mnt/c/muco/code/latex_table_hyb_qd/RCS4_GF/normal_RCS4_gf.json'
line = xt.Line.from_json(file)
tw = line.twiss4d()
tw.plot()

# Extract the parameters
params = {
    'qx': tw.qx,
    'qy' : tw.qy,
    'betx_min': np.min(tw.betx),
    'betx_max': np.max(tw.betx),
    'bety_min': np.min(tw.bety),
    'bety_max': np.max(tw.bety),
    'dx_min': np.min(tw.dx),
    'dx_max': np.max(tw.dx),
    'mcf': tw.momentum_compaction_factor,
    'chroma_x': tw.dqx,
    'chroma_y': tw.dqy
}

# Extract the base name of the input file
base_name = os.path.splitext(os.path.basename(file))[0]

# Define the output file path
output_file = os.path.join(os.path.dirname(file), f'{base_name}_optics_parameters.json')

# Write the parameters to a JSON file
with open(output_file, 'w') as f:
    json.dump(params, f, indent=4)

print(f'Parameters have been written to {output_file}')