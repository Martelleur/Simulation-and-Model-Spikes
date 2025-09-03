from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MU_EARTH = 3.986004418e14  # m^3/s^2

@dataclass
class State:
    t: float
    r: np.ndarray  # shape (3,), meters
    v: np.ndarray  # shape (3,), m/s

def two_body_acc(r: np.ndarray) -> np.ndarray:
    # a = -mu * r / |r|^3
    norm = np.linalg.norm(r)
    return -MU_EARTH * r / (norm**3)

def rk4_step(state: State, dt: float) -> State:
    # y = [r, v], y' = [v, a(r)]
    r0, v0 = state.r, state.v

    def f_rv(r, v):
        return v, two_body_acc(r)

    k1_r, k1_v = f_rv(r0, v0)
    k2_r, k2_v = f_rv(r0 + 0.5*dt*k1_r, v0 + 0.5*dt*k1_v)
    k3_r, k3_v = f_rv(r0 + 0.5*dt*k2_r, v0 + 0.5*dt*k2_v)
    k4_r, k4_v = f_rv(r0 + dt*k3_r,   v0 + dt*k3_v)

    r1 = r0 + (dt/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v1 = v0 + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return State(t=state.t + dt, r=r1, v=v1)

def altitude(r: np.ndarray, r_earth: float = 6378.137e3) -> float:
    return np.linalg.norm(r) - r_earth
