# M/M/1 queue via a tiny DES kernel + outputs & visualizations
# - Discrete-event simulation (DES) core
# - M/M/1 model (Poisson arrivals, exponential service)
# - Outputs: raw event/metric logs
# - Visualizations: histogram of wait times, time series of queue length
#
# You can re-run this cell to regenerate with different parameters.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict
import heapq
import itertools
import math
import random
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------
# DES kernel
# ----------------------
SimTime = float  # seconds for convenience

@dataclass(order=True)
class _Event:
    time: SimTime
    priority: int
    _seq: int
    fn: Callable[[], None]
    cancelled: bool = False

class DES:
    def __init__(self, start_time: SimTime = 0.0):
        self._now: SimTime = start_time
        self._heap: List[_Event] = []
        self._seq_counter = itertools.count()
        self._handles: Dict[int, _Event] = {}

    @property
    def now(self) -> SimTime:
        return self._now

    def schedule_at(self, t: SimTime, fn: Callable[[], None], priority: int = 0) -> int:
        if t < self._now:
            raise ValueError("Cannot schedule in the past")
        ev = _Event(time=t, priority=priority, _seq=next(self._seq_counter), fn=fn)
        heapq.heappush(self._heap, ev)
        handle = ev._seq
        self._handles[handle] = ev
        return handle

    def schedule_in(self, dt: SimTime, fn: Callable[[], None], priority: int = 0) -> int:
        return self.schedule_at(self._now + dt, fn, priority)

    def cancel(self, handle: int) -> bool:
        ev = self._handles.pop(handle, None)
        if not ev or ev.cancelled:
            return False
        ev.cancelled = True
        return True

    def run_until(self, predicate: Callable[[], bool]) -> None:
        while self._heap and not predicate():
            ev = heapq.heappop(self._heap)
            if ev.cancelled:
                continue
            self._now = ev.time
            ev.fn()

# ----------------------
# M/M/1 model
# ----------------------
class MM1:
    def __init__(self, sim: DES, lam: float, mu: float, n_customers: int, seed: Optional[int] = 42, warmup: int = 500):
        if lam <= 0 or mu <= 0:
            raise ValueError("λ and μ must be positive")
        if lam >= mu:
            # Theoretically unstable (ρ >= 1) but we allow it; just warn in stats.
            pass
        self.sim = sim
        self.lam = lam
        self.mu = mu
        self.n_customers = n_customers
        self.warmup = warmup

        self.rng = random.Random(seed)
        self.arrivals: int = 0
        self.departures: int = 0
        self.server_busy: bool = False
        self.queue: List[Tuple[int, float]] = []  # (customer_id, arrival_time)
        self.next_customer_id: int = 0

        # Metrics
        self.wait_times: List[float] = []          # per-customer waiting time in queue
        self.system_times: List[float] = []        # per-customer total time (wait + service)
        self.queue_trace: List[Tuple[float, int]] = []  # (time, queue_length)
        self.utilization_time: float = 0.0
        self.last_busy_change: float = sim.now
        self.last_queue_len: int = 0

    def exp(self, rate: float) -> float:
        return self.rng.expovariate(rate)

    def record_queue_len(self, t: float, q_len: int):
        # record only on change to keep the trace compact
        if q_len != self.last_queue_len:
            self.queue_trace.append((t, q_len))
            self.last_queue_len = q_len

    def start(self):
        # seed first arrival
        self.sim.schedule_in(self.exp(self.lam), self._on_arrival)
        # initial queue length
        self.record_queue_len(self.sim.now, 0)

    def _on_arrival(self):
        cid = self.next_customer_id
        self.next_customer_id += 1
        t = self.sim.now
        self.arrivals += 1

        # enqueue
        self.queue.append((cid, t))
        self.record_queue_len(t, len(self.queue))

        # if server idle, start service immediately
        if not self.server_busy:
            self._begin_service()

        # schedule next arrival if we still need more customers
        if self.arrivals < self.n_customers:
            self.sim.schedule_in(self.exp(self.lam), self._on_arrival)

    def _begin_service(self):
        if not self.queue:
            self.server_busy = False
            # utilization update if transitioning to idle
            self._update_utilization(busy=False)
            return

        # Pop next customer (FIFO)
        cid, arrival_time = self.queue.pop(0)
        self.record_queue_len(self.sim.now, len(self.queue))

        # Start service
        self._update_utilization(busy=True)
        self.server_busy = True
        service_time = self.exp(self.mu)
        start_service_time = self.sim.now

        def _complete():
            t = self.sim.now
            self.departures += 1
            # metrics
            wait = start_service_time - arrival_time
            total = t - arrival_time
            if self.departures > self.warmup:  # apply warmup discard
                self.wait_times.append(wait)
                self.system_times.append(total)

            # next
            self._begin_service()

        self.sim.schedule_in(service_time, _complete)

    def _update_utilization(self, busy: bool):
        # update area-under-curve only when busy/idle changes
        now = self.sim.now
        duration = now - self.last_busy_change
        if self.server_busy:  # was busy since last change
            self.utilization_time += duration
        self.last_busy_change = now

    def done(self) -> bool:
        # End when we've served n_customers
        return self.departures >= self.n_customers

    def results(self) -> dict:
        rho = self.lam / self.mu
        sim_time = self.sim.now
        util = self.utilization_time / sim_time if sim_time > 0 else float('nan')
        df_summary = {
            "lambda": self.lam,
            "mu": self.mu,
            "rho (λ/μ)": rho,
            "customers (after warmup)": len(self.wait_times),
            "sim_time": sim_time,
            "avg_wait": statistics.fmean(self.wait_times) if self.wait_times else float('nan'),
            "p95_wait": float(np.percentile(self.wait_times, 95)) if self.wait_times else float('nan'),
            "avg_system_time": statistics.fmean(self.system_times) if self.system_times else float('nan'),
            "server_utilization (time-avg)": util,
            "Little's law check E[Lq]≈λ*E[W]": self.lam * (statistics.fmean(self.wait_times) if self.wait_times else float('nan'))
        }
        return df_summary

# ----------------------
# Run an experiment
# ----------------------
lam = 0.9     # arrival rate (1/s)
mu = 1.0      # service rate (1/s)
N  = 8000     # total customers to complete
warmup = 500  # drop first 500 to reduce initialization bias

sim = DES()
model = MM1(sim, lam, mu, n_customers=N, warmup=warmup, seed=123)
model.start()
sim.run_until(model.done)

summary = model.results()
summary_df = pd.DataFrame([summary])

# Save raw outputs
outdir = Path("/mnt/data")
outdir.mkdir(parents=True, exist_ok=True)
waits_csv = outdir / "mm1_wait_times.csv"
queue_csv = outdir / "mm1_queue_trace.csv"
summary_csv = outdir / "mm1_summary.csv"

pd.Series(model.wait_times, name="wait_time_sec").to_csv(waits_csv, index=False)
pd.DataFrame(model.queue_trace, columns=["time_sec", "queue_len"]).to_csv(queue_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

# ----------------------
# Visualisations
# ----------------------

# 1) Histogram of waiting times
plt.figure()
plt.hist(model.wait_times, bins=60)
plt.title("M/M/1 waiting time distribution (after warmup)")
plt.xlabel("Wait time (s)")
plt.ylabel("Frequency")
hist_path = outdir / "mm1_wait_hist.png"
plt.savefig(hist_path, bbox_inches="tight")
plt.show()

# 2) Queue length over time (step-like plot)
# Construct a step series
times = [t for t, _ in model.queue_trace]
qlens = [q for _, q in model.queue_trace]
if times and qlens:
    plt.figure()
    plt.step(times, qlens, where="post")
    plt.title("M/M/1 queue length over time (entire run)")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue length")
    qlen_path = outdir / "mm1_queue_length_step.png"
    plt.savefig(qlen_path, bbox_inches="tight")
    plt.show()

# Display summary as an interactive table
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("M/M/1 Summary (DES run)", summary_df)

# Return file locations for download
str(waits_csv), str(queue_csv), str(summary_csv), str(hist_path), str(qlen_path)
