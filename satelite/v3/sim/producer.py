from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from .orbit import State, rk4_step, altitude, MU_EARTH
from . import sensor

@dataclass
class SimParams:
    a: float                 # semi-major axis (m)
    e: float                 # eccentricity (0..1)
    inc: float               # inclination (rad) (unused in simple init)
    dt: float                # integrator step (s)
    t_total: float           # total sim time (s)
    sample_state_hz: float   # how often to emit state events
    sample_frame_hz: float   # how often to emit frame events
    warmup: float = 0.0      # seconds to ignore in summaries

def kepler_init(a: float, e: float) -> State:
    # Simple perigee start in equatorial plane for demo (not general)
    r_p = a * (1 - e)
    v_p = (MU_EARTH * (2.0/r_p - 1.0/a)) ** 0.5
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, v_p, 0.0])
    return State(t=0.0, r=r0, v=v0)

def run_producer(q, params: SimParams):
    state = kepler_init(params.a, params.e)
    t_next_state = 0.0
    t_next_frame = 0.0
    dt_state = 1.0 / params.sample_state_hz
    dt_frame = 1.0 / params.sample_frame_hz

    frames_emitted = 0
    while state.t <= params.t_total:
        # Emit state at cadence
        if state.t >= t_next_state:
            alt = float(altitude(state.r))
            q.put({ "type": "state",
                    "t": float(state.t),
                    "r": state.r.tolist(),
                    "v": state.v.tolist(),
                    "alt": alt })
            t_next_state += dt_state

        # Emit frame at cadence
        if state.t >= t_next_frame:
            img = sensor.render_frame_from_state(state.r)  # float32 [0,1]
            # to keep queue light, compress to uint8 here
            frame_u8 = (img * 255.0).astype('uint8')
            q.put({ "type": "frame",
                    "t": float(state.t),
                    "id": int(frames_emitted),
                    "img": frame_u8 })
            frames_emitted += 1
            t_next_frame += dt_frame

        # Integrate one step
        state = rk4_step(state, params.dt)

    q.put({ "type": "summary",
            "t_end": float(state.t),
            "frames": int(frames_emitted) })
    q.put({ "type": "END" })
