# Satellite Sim Stream (parallel streaming demo)

This is a minimal, educational satellite simulation demonstrating a **parallel streaming** pattern:

- A **producer** process propagates a satellite in a two-body Earth orbit and generates synthetic **sensor frames**.
- A **consumer** process receives events through a multiprocessing **Queue**, writing **CSV** (state history) and **PNG** (frames).
- A **visualizer** script loads outputs and draws simple charts (each in its own figure).

> This mirrors the queueing example but applied to an orbital/sensor scenario.

## How to run

```bash
python run_stream.py
python visualize.py
```

Outputs are saved under `./out/`:
- `states.csv` — time, position (m), velocity (m/s), altitude (m)
- `frames/frame_000000.png` — synthetic grayscale sensor frames
- `summary.json` — simple aggregates about the run
- Plots created by `visualize.py`

## Physics & simplifications

- Two-body dynamics with Earth's gravitational parameter μ = 3.986004418e14 m³/s².
- Cartesian ECI frame; no Earth rotation / atmosphere / J2 / SRP — **educational only**.
- RK4 fixed step integrator.
- Sensor frames: 128×128 grayscale with a Gaussian spot whose image-plane position is tied to the current subpoint; this is a mock to demonstrate image streaming, *not* a rendering pipeline.

## Parallel streaming

- The **producer** emits events: `state` (dict) and `frame` (numpy array) at chosen cadences.
- The **consumer** writes to disk incrementally and maintains a tiny running summary.
- After the run, `visualize.py` plots altitude-over-time and a simple ground-track proxy.

## References (for the model, not the codebase)

- Bate, Mueller, White. *Fundamentals of Astrodynamics*. Dover.
- Vallado. *Fundamentals of Astrodynamics and Applications*.
- ECSS–E–ST–40–07C Rev.1 (SMP Level 1): separation of model/simulation, outputs, services.

