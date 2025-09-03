# Simulation, model and visualisation

---

## **1. The Models**

### a) Central-Body (Two-Body) Gravity Model

* We modeled each satellite’s motion under the **two-body assumption**:

  * Only Earth’s gravity acts.
  * No drag, no Earth oblateness (J2), no third-body effects (Sun/Moon), no solar radiation pressure.
* Earth’s gravitational parameter used:

  $$
  \mu = 398600.4418 \, \text{km}^3/\text{s}^2
  $$

This is the simplest accurate-enough model for basic orbital mechanics.

---

### b) Satellite State Representation

Each satellite’s orbit is defined using **classical orbital elements (COEs)**:

1. **Semi-major axis (a)** → size of the orbit.
2. **Eccentricity (e)** → shape (0 = circle, <1 = ellipse).
3. **Inclination (i)** → tilt relative to Earth’s equator.
4. **RAAN (Ω)** → orientation of the ascending node.
5. **Argument of perigee (ω)** → orientation of closest approach.
6. **Mean anomaly at epoch (M₀)** → where the satellite is along its orbit at time zero.

From these, we can compute the satellite’s position at any time.

---

### c) Orbit Propagation

* The satellite’s **mean anomaly** evolves as:

  $$
  M(t) = M_0 + n t, \quad n = \sqrt{\mu / a^3}
  $$

  where $n$ = mean motion (rad/s).

* To get the true position:

  * Solve **Kepler’s Equation**: $M = E - e\sin E$, for eccentric anomaly $E$.
  * Compute true anomaly $\nu$ and radius $r$.
  * Convert from **perifocal coordinates (PQW)** to **Earth-Centered Inertial (ECI)** using rotation matrices.

This gives the 3D position of each satellite as a function of time.

---

## **2. The Simulation**

### Setup

* Two satellites were initialized with slightly different orbital elements (different inclinations, RAAN, and anomalies).
* Simulation duration = about **two orbits of Satellite 1**.
* Time step = **5 seconds**.

### Process

1. Propagate both satellites through time using the two-body model.
2. At each timestep, compute their **position vectors** in ECI (X, Y, Z in km).
3. Compute their **distance**:

   $$
   d(t) = \| \mathbf{r}_1(t) - \mathbf{r}_2(t) \|
   $$
4. Find the **minimum distance** and check if it falls below a chosen **collision threshold** (e.g., 0.1 km).

---

## **3. The Results**

* The closest approach distance between the two demo satellites was **≈ 1841.8 km**.
* This is **far larger** than the 0.1 km (100 m) collision threshold.
* Therefore: **no collision occurred** in this simulation.
* Plots were generated:

  * Separation distance vs. time.
  * 3D view of the two orbital paths around Earth.

---

## **4. What This Shows**

* The **model** is a simplified Keplerian (two-body) orbit model, good for conceptual demonstrations and initial analysis.
* The **simulation** tested relative motion of two satellites and whether their orbits intersect closely enough to risk collision.
* This kind of setup is the basis for **conjunction analysis** in space operations — though real-world analysis uses higher fidelity models (perturbations, drag, precise ephemerides, SGP4 with TLEs).

---

👉 So in summary:
We built a **mathematical model of orbital motion** using Kepler’s laws, simulated two satellites in slightly different orbits, tracked their positions over time, and measured the **minimum separation** to check for a collision.

---
