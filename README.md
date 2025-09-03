# Simulation-and-Model-Spikes

## Model vs. Simulation (conceptual distinction)

* **Model**
  A model is the **abstract representation** of a system or process. It encodes structure, rules, parameters, and behaviors of the system of interest.

  * Examples:

    * A set of differential equations modeling orbital mechanics.
    * A queuing model for customers arriving at a service desk.
    * A UML state machine describing a subsystem’s modes.

* **Simulation**
  A simulation is the **execution or enactment** of a model over time (or over some experimental conditions). It’s the process of *running the model*, producing a sequence of states, outputs, or trajectories.

  * Examples:

    * Numerically integrating orbital equations to produce spacecraft position vs. time.
    * Running discrete-event scheduling to produce queue wait times and lengths.
    * Stepping through the state machine with test stimuli to see transitions.

**Key point:**
Models are *static representations* (rules, equations, logic). Simulations are *dynamic experiments* — the unfolding of those rules under given conditions.

This matches the distinction in your Wikipedia citation: the **model** encodes *what the system is*; the **simulation** shows *what happens when it runs*.

---

## Role of Outputs and Visualisations

When we *simulate* a model, the immediate product is **data** — state trajectories, events, metrics. But raw data is rarely the end goal. We need **outputs** (structured results) and **visualisations** (human-understandable forms) to enable *interpretation, experimentation, and decision-making*.

### 1. **Outputs (numerical, structured)**

Outputs are what the simulation engine produces:

* Time series (e.g., temperature vs. time).
* Event logs (e.g., “message received at t=10 ms”).
* Aggregate measures (e.g., average queue wait time).

These outputs are what make experimentation possible: you can rerun the simulation with different parameters and compare output sets.

### 2. **Visualisations (representations for understanding)**

Visualisation is the *mapping from outputs to perception*. It can be:

* **Static plots:** line charts, histograms of outcomes.
* **Animated traces:** spacecraft trajectory evolving in a 3D viewer.
* **Immersive/VR:** digital twins where engineers “see” the system behavior.
* **Dashboards:** KPIs updated in real-time during simulation.

Good visualisation provides **intuition and pattern recognition**, letting humans detect trends, anomalies, or emergent behaviors that may not be obvious in raw data.

---

## Why this distinction matters in experimentation

When we talk about **simulation-based experimentation**, we really have three layers:

1. **Model (assumptions):** Defines *what you believe the system is*. If the model is wrong, the simulation can be beautifully visualized but misleading.
2. **Simulation (execution):** Generates *behaviors under scenarios*. Here, fidelity of algorithms (e.g., DES kernel accuracy, ODE solver stability) matters.
3. **Output/visualisation (interpretation):** Enables *human or automated learning from the run*. This is where experimentation yields knowledge.

These layers create a pipeline:

```
   Model  ->  Simulation run  ->  Outputs  ->  Visualisation/Analysis
```

Experimentation then loops back:

* Change **model parameters** (e.g., spacecraft mass, customer arrival rate).
* Re-run **simulation**.
* Collect new **outputs**.
* Compare visualisations/metrics.

This is the essence of *scientific method via simulation*: models are hypotheses, simulations are experiments, outputs/visualisations are observations.

---

## Example: Queueing system (DES)

* **Model:**
  A single-server queue with exponential arrivals and service times (M/M/1).
* **Simulation:**
  Discrete-event kernel advances from one arrival/service completion to the next, updating the state.
* **Output:**
  List of waiting times for 10,000 customers, average queue length, max delay.
* **Visualisation:**

  * Histogram of wait times.
  * Time series of queue length.
  * Dashboard: “Utilisation = ρ = 0.85”.

Here the **model** is just the abstract rules. The **simulation** is the DES run. The **outputs** are the numerical metrics, and the **visualisations** are how humans grasp them.

---

## Example: Space sensor digital twin

* **Model:**
  Equations for orbital dynamics + camera sensor noise models.
* **Simulation:**
  Time-stepped integration producing satellite positions, plus synthetic images from the sensor model.
* **Outputs:**

  * CSV trajectory of positions.
  * Image sequence (mock sensor data).
* **Visualisation:**

  * 3D orbit plot.
  * Side-by-side “true” vs. “simulated camera” images.
  * Interactive timeline for anomaly exploration.

---

**So, in summary:**

* **Models** = representation (structure, assumptions).
* **Simulations** = enactment of models over time.
* **Outputs** = raw results of simulation runs.
* **Visualisations** = interpretations for understanding.

The distinction matters because **experimentation relies on outputs** to test hypotheses, while **visualisation makes outputs accessible** to human decision-makers.

