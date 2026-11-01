from cpymad.madx import Madx
import matplotlib.pyplot as plt

madx = Madx()

particle_type = "posmuon"  # Example particle type
energy = RCS.E_inj*1e-9  # Example energy in MeV

# Load the sequence from the file
sequence_file = '/mnt/c/muco/1arc_RCS2_for_FLUKA.seq'
with open(sequence_file, 'r') as file:
    sequence_content = file.read()

# Define the beam and load the sequence into MAD-X
madx.input(f'''
    beam, particle={particle_type}, energy={energy};
''')
madx.call(sequence_file)
madx.use(sequence="s1arc_RCS2")
madx.input("select, flag=twiss, column=NAME, KEYWORD, S, L, TILT, KICK, HKICK, VKICK, ANGLE, K0L, K0SL, K1L, K1SL, K2L, K2SL, K3L, K3SL, BETX, BETY, X, PX, Y, PY, DX, DPX, DY, DPY, ALFX, ALFY;")
madx.twiss(file="toto.tfs")

twiss_table = madx.twiss()
plt.plot(twiss_table.s, twiss_table.betx)