import json    

def save_para(rcs, file_path):
    """Save geometry parameters from RingGeometry object. 

    Args:
        rcs (RingGeometry): RingGeometry object.
        file_path (str): File path to save at.
    """
    para_rcs = {
    "type": rcs.type,
    "circumference": rcs.circumference,
    "injection_energy": rcs.injection_energy,
    "extraction_energy": rcs.extraction_energy,
    "length_short_insertion": rcs.length_short_insertion,
    "dipole_spacing": rcs.dipole_spacing,
    "nb_RF_section": rcs.nb_RF_section,
    "nb_arc": rcs.nb_arc,
    "nb_cell_arc": rcs.nb_cell_arc,
    "dipoles": rcs.dipoles,
    "filling_ratio": rcs.filling_ratio,
    "filling_ratio_RF": rcs.filling_ratio_RF,
    "pattern": rcs.pattern
    }

    with open(file_path, "w") as outfile: 
        json.dump(para_rcs, outfile, indent=4)