# USed for version 2.

from v2.model import MU_EARTH
from v2.simulation import run_two_satellite_simulation
from v2.visualisation import plot_distance_vs_time, plot_orbits
import math, numpy as np

# Define satellites (orbital elements)
sat1 = dict(a_km=7000.0, e=0.001, inc_deg=51.6, raan_deg=40.0, argp_deg=0.0, M0_deg=0.0)
sat2 = dict(a_km=7000.0, e=0.001, inc_deg=51.8, raan_deg=40.2, argp_deg=5.0, M0_deg=10.0)

# Time horizon: 2 orbital periods
n1 = math.sqrt(MU_EARTH / sat1["a_km"]**3)
T1 = 2*np.pi/n1

df, result = run_two_satellite_simulation(sat1, sat2, t_end=2*T1, dt=5.0)

csv_path = "./two_satellite_sim.csv"
df.to_csv(csv_path, index=False)

print("Simulation result:", result)
plot_distance_vs_time(df)
plot_orbits(df)
