# Simulation of a M/M/1 Model

Perfect — let’s clarify step by step.

---

## What is **M/M/1**?

“M/M/1” is classic queueing notation from **Kendall’s notation**, widely used in operations research and computer science to classify queues.

It means:

* **First M (“Markovian arrivals”):**
  Interarrival times are exponentially distributed (i.e., arrivals follow a **Poisson process** with rate λ).
  → Customers arrive randomly but with an average rate.

* **Second M (“Markovian service”):**
  Service times are also exponentially distributed, with mean rate μ.
  → Each service duration is random but memoryless.

* **The “1”:**
  There is **one server**.

Other assumptions in the basic M/M/1 model:

* Queue discipline is usually **FIFO** (first in, first out).
* The waiting room (queue) is infinite.
* Arrivals and services are independent.

So, **M/M/1** is not the simulation itself — it is the **mathematical model** of a very specific queueing system.

---

## Is M/M/1 the Model?

* The **model** is the **M/M/1 assumptions**:

  * arrivals ∼ Exp(λ)
  * service times ∼ Exp(μ)
  * single server, FIFO
* That’s a *conceptual description of the system* we want to study.

The **simulation** then *enacted this model over time* by generating random arrival and service times, moving customers through the queue, and recording metrics.

---

## Why is this important?

* **Model (M/M/1):** Abstract rules → “What we think the system is.”
* **Simulation:** Running those rules → “What happens when the system operates.”
* **Outputs:** Wait times, queue length, utilisation.
* **Visualisation:** Graphs (histograms, traces) that help humans interpret the results.

---

### Example Analogy

Imagine you want to study a coffee shop with one barista.

* The **M/M/1 model** is your abstraction:

  * Customers arrive randomly (Poisson arrivals).
  * Barista takes a random time to serve each customer (exponential service).
  * One barista only.

* The **simulation** is like building a digital coffee shop where you “play out” these arrivals and services minute by minute (or event by event).

* The **outputs** are the statistics you collect: how long people wait, how long the queue gets, how busy the barista is.


---

## The **Model**

The model is the **abstract description** of the system you are studying — in this case, a **queueing system**.

Specifically, the **M/M/1 queue model** represents:

* **Entities:** “customers” (or jobs, packets, tasks) arriving to be served.
* **Arrival process:** arrivals follow a **Poisson process** with rate **λ** (so interarrival times are exponentially distributed).
* **Service process:** a single server, with service times exponentially distributed with rate **μ**.
* **Queue discipline:** **FIFO** (first in, first out).
* **Server capacity:** one server, one customer served at a time, unlimited waiting room.
* **Key metrics of interest:** waiting time in queue, total system time (wait + service), server utilisation, average queue length.

This **model encodes assumptions about reality** (e.g., arrivals are memoryless, service times are i.i.d exponential). It’s a mathematical abstraction, not yet “run.”

---

## The **Simulation**

The simulation is the **execution of that model over time** under specific conditions (parameters, random seeds, stop criteria).

In this example:

* The **Discrete-Event Simulation (DES) kernel** advances the simulation clock from one *event* to the next:

  * **Arrival events** add customers to the queue (and possibly trigger service if the server is idle).
  * **Departure events** remove customers from the server, record waiting/service times, and potentially start the next service.
* No time steps are used — the clock jumps to the next scheduled event.
* The simulation continues until a stopping condition is met (e.g., 8,000 customers processed).

The simulation’s job is to **generate trajectories of the system state** (queue lengths, wait times, utilisation), given the abstract model rules.

---

## Why do we do this?

Because:

1. **Analytical models are limited.**
   The M/M/1 has known closed-form solutions, but many real systems are too complex (multiple servers, priorities, network of queues, time-varying rates). Simulation is a general method for experimenting with such models.

2. **Experimentation with “what if?” scenarios.**
   By simulating, we can change λ (arrival intensity) or μ (service speed) and see the impact on waiting times, congestion, or utilisation — without disrupting a real system.

3. **Output and visualisation.**
   The simulation produces *outputs* (wait times, queue traces) that can be aggregated, analysed statistically, and visualised (histograms, time series). These outputs let us *understand the behavior of the model* and make decisions.

---

✅ **So in one sentence:**
The **model** defines the *rules of the queueing system*, while the **simulation** enacts those rules over time to produce *observable outcomes* (outputs), which we then analyse and visualise to learn about performance and behaviour.

---

Would you like me to also contrast this example with a case where **analytical results exist** (like M/M/1 formulas for $E[W]$, $E[L]$) — and show how simulation outputs line up with the theory? That would highlight *why* we simulate even when math is possible.
