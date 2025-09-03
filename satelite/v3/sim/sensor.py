from __future__ import annotations
import numpy as np

def gaussian_spot(h: int, w: int, cx: float, cy: float, sigma: float = 5.0, amp: float = 1.0) -> np.ndarray:
    y = np.arange(h).reshape(-1,1)
    x = np.arange(w).reshape(1,-1)
    return amp * np.exp(-(((x - cx)**2) + ((y - cy)**2)) / (2*sigma*sigma))

def render_frame_from_state(r_vec, frame_h=128, frame_w=128):
    # Map subpoint proxy to image coords: use longitude-like angle from r_vec
    # (Toy: not real camera model.)
    x, y, z = r_vec
    angle_x = np.arctan2(y, x)  # [-pi, pi]
    angle_y = np.arctan2(z, np.hypot(x,y))  # [-pi/2, pi/2]

    # Normalize angles to [0, 1] then to pixel coords
    u = (angle_x + np.pi) / (2*np.pi)
    v = (angle_y + (np.pi/2)) / (np.pi)
    cx = u * (frame_w - 1)
    cy = (1.0 - v) * (frame_h - 1)

    img = gaussian_spot(frame_h, frame_w, cx, cy, sigma=6.0, amp=1.0)
    # Add a radial vignette to look nicer
    yy, xx = np.indices((frame_h, frame_w))
    rr = np.hypot((xx - frame_w/2)/(frame_w/2), (yy - frame_h/2)/(frame_h/2))
    vignette = np.clip(1.2 - 0.7*rr, 0.0, 1.0)
    img = img * vignette
    # Normalize to [0,1]
    mx = img.max() if img.max() > 0 else 1.0
    return (img / mx).astype(np.float32)
