# Orbital collision-check demo: two satellites in Keplerian orbits
# Assumptions: two-body (Earth + satellite), no perturbations (no J2, drag, SRP), ECI frame.
# Units: km, s, rad.

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

MU_EARTH = 398600.4418  # km^3/s^2

def solve_kepler_elliptic(M, e, tol=1e-12, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for 0 <= e < 1 using Newton-Raphson."""
    # Normalize M to [-pi, pi] for faster convergence
    M = (M + np.pi) % (2*np.pi) - np.pi
    # Initial guess (good for most e): E0 = M if e < 0.8 else pi
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(max_iter):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def coe_to_r_eci(a_km, e, inc_deg, raan_deg, argp_deg, M0_deg, t_s):
    """Return ECI position vector (km) at time t_s from epoch using Kepler propagation."""
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    M0 = np.deg2rad(M0_deg)
    n = math.sqrt(MU_EARTH / a_km**3)  # rad/s
    M = M0 + n * t_s
    E = solve_kepler_elliptic(M, e)
    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    # Radius
    r_p = a_km * (1 - e * np.cos(E))
    # Perifocal coordinates
    r_pqw = np.vstack((r_p * np.cos(nu), r_p * np.sin(nu), np.zeros_like(nu)))
    # Rotation matrices
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(argp), np.sin(argp)
    R3_O = np.array([[cO, -sO, 0],
                     [sO,  cO, 0],
                     [ 0,   0, 1]])
    R1_i = np.array([[1, 0, 0],
                     [0, ci, si],
                     [0, -si, ci]])
    R3_o = np.array([[co, -so, 0],
                     [so,  co, 0],
                     [ 0,   0, 1]])
    Q = R3_O @ R1_i @ R3_o
    r_eci = Q @ r_pqw
    return r_eci

# Example satellites (edit these to try your own cases)
sat1 = dict(a_km=7000.0, e=0.001, inc_deg=51.6, raan_deg=40.0, argp_deg=0.0, M0_deg=0.0)
sat2 = dict(a_km=7000.0, e=0.001, inc_deg=51.8, raan_deg=40.2, argp_deg=5.0, M0_deg=10.0)

# Simulation horizon: 2 orbital periods of sat1
n1 = math.sqrt(MU_EARTH / sat1["a_km"]**3)
T1 = 2*np.pi/n1
t_end = 2*T1
dt = 5.0  # seconds
t = np.arange(0.0, t_end+dt, dt)

r1 = coe_to_r_eci(t_s=t, **sat1)  # shape (3, N)
r2 = coe_to_r_eci(t_s=t, **sat2)

# Distances between satellites (km)
dist = np.linalg.norm(r1 - r2, axis=0)

# Find minimum distance and time
idx_min = int(np.argmin(dist))
t_min = t[idx_min]
d_min_km = float(dist[idx_min])

# Decide "collision" if below this threshold (e.g., 0.1 km = 100 m)
collision_threshold_km = 0.1
collision = d_min_km <= collision_threshold_km

# Prepare dataframe and save
df = pd.DataFrame({
    "time_s": t,
    "r1_x_km": r1[0], "r1_y_km": r1[1], "r1_z_km": r1[2],
    "r2_x_km": r2[0], "r2_y_km": r2[1], "r2_z_km": r2[2],
    "distance_km": dist
})
csv_path = "./two_satellite_sim.csv"
df.to_csv(csv_path, index=False)

# Plot distance vs time
plt.figure()
plt.plot(t/60.0, dist)  # minutes vs km
plt.xlabel("Time (minutes)")
plt.ylabel("Separation distance (km)")
plt.title("Two-satellite separation over time (two-body Keplerian model)")
plt.grid(True)
plt.show()

# 3D orbit plot (optional, simple)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(r1[0], r1[1], r1[2], label="Sat 1")
ax.plot(r2[0], r2[1], r2[2], label="Sat 2")
# Draw Earth sphere (approximate)
Re = 6378.137
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x = Re*np.cos(u)*np.sin(v)
y = Re*np.sin(u)*np.sin(v)
z = Re*np.cos(v)
ax.plot_wireframe(x, y, z, linewidth=0.2)
ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
ax.set_title("Orbits in ECI (two periods)")
ax.legend()
plt.show()

result = {
    "collision_threshold_km": collision_threshold_km,
    "t_min_s": t_min,
    "d_min_km": d_min_km,
    "collision": collision,
    "csv_path": csv_path,
}

result
