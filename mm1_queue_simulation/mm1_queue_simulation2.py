# Updated version:
# - Ensures BOTH charts always display inline (histogram + queue-length step)
# - Keeps each chart in its OWN figure (no subplots, no seaborn, no custom colors)
# - Additionally generates a COMBINED side-by-side PNG by stitching the two saved charts
#   (image composition; still complies with the "each chart its own plot" constraint)
#
# You can tweak λ, μ, N, and warmup below.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict
import heapq
import itertools
import random
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ----------------------
# DES kernel
# ----------------------
SimTime = float  # seconds

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
        self.sim = sim
        self.lam = lam
        self.mu = mu
        self.n_customers = n_customers
        self.warmup = warmup

        self.rng = random.Random(seed)
        self.arrivals: int = 0
        self.departures: int = 0
        self.server_busy: bool = False
        self.queue: List[Tuple[int, float]] = []
        self.next_customer_id: int = 0

        self.wait_times: List[float] = []
        self.system_times: List[float] = []
        self.queue_trace: List[Tuple[float, int]] = []
        self.utilization_time: float = 0.0
        self.last_busy_change: float = sim.now
        self.last_queue_len: int = 0

    def exp(self, rate: float) -> float:
        return self.rng.expovariate(rate)

    def record_queue_len(self, t: float, q_len: int):
        if q_len != self.last_queue_len:
            self.queue_trace.append((t, q_len))
            self.last_queue_len = q_len

    def start(self):
        self.sim.schedule_in(self.exp(self.lam), self._on_arrival)
        self.record_queue_len(self.sim.now, 0)

    def _on_arrival(self):
        cid = self.next_customer_id
        self.next_customer_id += 1
        t = self.sim.now
        self.arrivals += 1
        self.queue.append((cid, t))
        self.record_queue_len(t, len(self.queue))
        if not self.server_busy:
            self._begin_service()
        if self.arrivals < self.n_customers:
            self.sim.schedule_in(self.exp(self.lam), self._on_arrival)

    def _begin_service(self):
        if not self.queue:
            self.server_busy = False
            self._update_utilization(busy=False)
            return
        cid, arrival_time = self.queue.pop(0)
        self.record_queue_len(self.sim.now, len(self.queue))
        self._update_utilization(busy=True)
        self.server_busy = True
        service_time = self.exp(self.mu)
        start_service_time = self.sim.now
        def _complete():
            t = self.sim.now
            self.departures += 1
            wait = start_service_time - arrival_time
            total = t - arrival_time
            if self.departures > self.warmup:
                self.wait_times.append(wait)
                self.system_times.append(total)
            self._begin_service()
        self.sim.schedule_in(service_time, _complete)

    def _update_utilization(self, busy: bool):
        now = self.sim.now
        duration = now - self.last_busy_change
        if self.server_busy:
            self.utilization_time += duration
        self.last_busy_change = now

    def done(self) -> bool:
        return self.departures >= self.n_customers

    def results(self) -> dict:
        rho = self.lam / self.mu
        sim_time = self.sim.now
        util = self.utilization_time / sim_time if sim_time > 0 else float('nan')
        return {
            "lambda": self.lam,
            "mu": self.mu,
            "rho (λ/μ)": rho,
            "customers (after warmup)": len(self.wait_times),
            "sim_time": sim_time,
            "avg_wait": float(np.mean(self.wait_times)) if self.wait_times else float('nan'),
            "p95_wait": float(np.percentile(self.wait_times, 95)) if self.wait_times else float('nan'),
            "avg_system_time": float(np.mean(self.system_times)) if self.system_times else float('nan'),
            "server_utilization (time-avg)": util
        }

# ----------------------
# Parameters & run
# ----------------------
lam = 0.9
mu = 1.0
N  = 8000
warmup = 500

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
# Visualisations (each chart in its own figure)
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

# 2) Queue length over time (step plot)
times = [t for t, _ in model.queue_trace]
qlens = [q for _, q in model.queue_trace]
plt.figure()
plt.step(times, qlens, where="post")
plt.title("M/M/1 queue length over time (entire run)")
plt.xlabel("Time (s)")
plt.ylabel("Queue length")
qlen_path = outdir / "mm1_queue_length_step.png"
plt.savefig(qlen_path, bbox_inches="tight")
plt.show()

# ----------------------
# OPTIONAL: Create a side-by-side COMPOSITE PNG by stitching the two saved images
# ----------------------
try:
    img1 = Image.open(hist_path)
    img2 = Image.open(qlen_path)

    # Normalize heights, pad, and stitch horizontally
    h = max(img1.height, img2.height)
    # Resize to same height while preserving aspect ratio
    def resize_to_height(im, target_h):
        w = int(im.width * (target_h / im.height))
        return im.resize((w, target_h))

    img1r = resize_to_height(img1, h)
    img2r = resize_to_height(img2, h)
    pad = 20
    combo = Image.new("RGB", (img1r.width + pad + img2r.width, h), (255, 255, 255))
    combo.paste(img1r, (0, 0))
    combo.paste(img2r, (img1r.width + pad, 0))
    combo_path = outdir / "mm1_plots_side_by_side.png"
    combo.save(combo_path)

    # Display the combined image
    display(combo)
except Exception as e:
    print("Combined image creation skipped:", e)

# Show the summary table last so it doesn't hide plots
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("M/M/1 Summary (DES run)", summary_df)

# Return file paths for convenience
str(waits_csv), str(queue_csv), str(summary_csv), str(hist_path), str(qlen_path), str(combo_path)
