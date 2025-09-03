import numpy as np
import pandas as pd
from v2.model import coe_to_r_eci

def run_two_satellite_simulation(sat1, sat2, t_end, dt, collision_threshold_km=0.1):
    t = np.arange(0.0, t_end+dt, dt)
    r1 = coe_to_r_eci(t_s=t, **sat1)
    r2 = coe_to_r_eci(t_s=t, **sat2)
    dist = np.linalg.norm(r1 - r2, axis=0)
    idx_min = int(np.argmin(dist))
    result = {
        "collision_threshold_km": collision_threshold_km,
        "t_min_s": float(t[idx_min]),
        "d_min_km": float(dist[idx_min]),
        "collision": dist[idx_min] <= collision_threshold_km,
    }
    df = pd.DataFrame({
        "time_s": t,
        "r1_x_km": r1[0], "r1_y_km": r1[1], "r1_z_km": r1[2],
        "r2_x_km": r2[0], "r2_y_km": r2[1], "r2_z_km": r2[2],
        "distance_km": dist
    })
    return df, result
