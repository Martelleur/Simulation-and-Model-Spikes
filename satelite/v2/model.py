import numpy as np
import math

MU_EARTH = 398600.4418  # km^3/s^2

def solve_kepler_elliptic(M, e, tol=1e-12, max_iter=50):
    M = (M + np.pi) % (2*np.pi) - np.pi
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
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    M0 = np.deg2rad(M0_deg)
    n = math.sqrt(MU_EARTH / a_km**3)
    M = M0 + n * t_s
    E = solve_kepler_elliptic(M, e)
    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r_p = a_km * (1 - e * np.cos(E))
    r_pqw = np.vstack((r_p * np.cos(nu), r_p * np.sin(nu), np.zeros_like(nu)))
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(argp), np.sin(argp)
    R3_O = np.array([[cO, -sO, 0],[sO, cO, 0],[0, 0, 1]])
    R1_i = np.array([[1, 0, 0],[0, ci, si],[0, -si, ci]])
    R3_o = np.array([[co, -so, 0],[so, co, 0],[0, 0, 1]])
    Q = R3_O @ R1_i @ R3_o
    return Q @ r_pqw
