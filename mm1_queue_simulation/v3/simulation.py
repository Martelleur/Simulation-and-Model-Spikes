# Parallel streaming example:
# - A producer process runs the DES M/M/1 simulation and STREAMS outputs (events) to a multiprocessing.Queue.
# - A consumer process reads events concurrently and writes them to CSV incrementally (and keeps running aggregates).
# - After both finish, we load the stored outputs and produce the two plots (each in its own figure).
#
# Notes:
# - This demonstrates decoupling "simulation execution" from "output handling/visualisation/storage".
# - The visualisations still happen at the end for reproducibility, but the storage pipeline ran in parallel.
# - You can adapt the consumer to push to a DB, message bus, or a live dashboard instead of CSV files.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict, Any
import heapq
import itertools
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
import time
import os

# ----------------------
# DES kernel (identical semantics as before)
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
# Streaming M/M/1 model
# ----------------------
class MM1Streaming:
    """
    Same logic as MM1, but emits events into an mp.Queue:
      - {"type": "queue_len", "t": time, "q": length}
      - {"type": "wait", "t": departure_time, "w": wait_time, "tot": system_time}
      - {"type": "summary", ...} (final)
    """
    def __init__(self, sim: DES, lam: float, mu: float, n_customers: int, out_q: mp.Queue,
                 seed: Optional[int] = 42, warmup: int = 500):
        self.sim = sim
        self.lam = lam
        self.mu = mu
        self.n_customers = n_customers
        self.warmup = warmup
        self.out_q = out_q

        self.rng = random.Random(seed)
        self.arrivals = 0
        self.departures = 0
        self.server_busy = False
        self.queue: List[Tuple[int, float]] = []
        self.next_customer_id = 0

        self.utilization_time = 0.0
        self.last_busy_change = sim.now
        self.last_queue_len = 0

    def exp(self, rate: float) -> float:
        return self.rng.expovariate(rate)

    def record_queue_len(self, t: float, q_len: int):
        if q_len != self.last_queue_len:
            self.out_q.put({"type": "queue_len", "t": t, "q": q_len})
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
                self.out_q.put({"type": "wait", "t": t, "w": wait, "tot": total})
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

# ----------------------
# Producer / Consumer processes
# ----------------------
SENTINEL = {"type": "END"}

def producer_proc(outdir: str, params: Dict[str, Any], q: mp.Queue):
    sim = DES()
    model = MM1Streaming(sim, params["lam"], params["mu"], params["N"], out_q=q, seed=123, warmup=params["warmup"])
    model.start()
    sim.run_until(model.done)
    # send a lightweight summary (utilization approximation)
    q.put({"type": "summary", "sim_time": sim.now, "approx_util": model.utilization_time / sim.now if sim.now else float('nan')})
    q.put(SENTINEL)

def consumer_proc(outdir: str, q: mp.Queue):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    waits_path = outdir / "stream_waits.csv"
    qlen_path  = outdir / "stream_queue_len.csv"
    # Write headers
    with open(waits_path, "w") as f1, open(qlen_path, "w") as f2:
        f1.write("t,w,tot\n")
        f2.write("t,q\n")

    # running aggregates
    n_wait = 0
    sum_wait = 0.0
    p95_buffer: List[float] = []
    t0 = time.time()
    while True:
        msg = q.get()
        if msg == SENTINEL:
            break
        typ = msg.get("type")
        if typ == "wait":
            with open(waits_path, "a") as f1:
                f1.write(f"{msg['t']},{msg['w']},{msg['tot']}\n")
            n_wait += 1
            sum_wait += msg["w"]
            p95_buffer.append(msg["w"])
        elif typ == "queue_len":
            with open(qlen_path, "a") as f2:
                f2.write(f"{msg['t']},{msg['q']}\n")
        elif typ == "summary":
            # could persist elsewhere; for demo just print
            print("Producer summary:", msg)

        # Throttle a bit to simulate streaming work (optional)
        if (time.time() - t0) > 0.5:
            t0 = time.time()

    # final aggregates
    if n_wait > 0:
        avg_wait = sum_wait / n_wait
        p95 = float(np.percentile(p95_buffer, 95))
    else:
        avg_wait, p95 = float('nan'), float('nan')
    with open(outdir / "stream_summary.txt", "w") as f:
        f.write(f"avg_wait={avg_wait}\np95_wait={p95}\n")

# ----------------------
# Orchestrate the parallel run
# ----------------------
outdir = Path("./stream_demo")
outdir.mkdir(parents=True, exist_ok=True)
params = {"lam": 0.9, "mu": 1.0, "N": 8000, "warmup": 500}

q = mp.Queue(maxsize=1000)
p_prod = mp.Process(target=producer_proc, args=(str(outdir), params, q))
p_cons = mp.Process(target=consumer_proc, args=(str(outdir), q))

p_cons.start()
p_prod.start()
p_prod.join()
p_cons.join()

# ----------------------
# Load outputs and visualize (post-run)
# ----------------------
waits_df = pd.read_csv(outdir / "stream_waits.csv")
qlen_df  = pd.read_csv(outdir / "stream_queue_len.csv")

# Plots: each chart in its own figure, no custom colors
plt.figure()
plt.hist(waits_df["w"], bins=60)
plt.title("STREAMED M/M/1 waiting time distribution (after warmup)")
plt.xlabel("Wait time (s)")
plt.ylabel("Frequency")
hist_path = outdir / "stream_wait_hist.png"
plt.savefig(hist_path, bbox_inches="tight")
plt.show()

plt.figure()
plt.step(qlen_df["t"], qlen_df["q"], where="post")
plt.title("STREAMED M/M/1 queue length over time")
plt.xlabel("Time (s)")
plt.ylabel("Queue length")
qlen_path = outdir / "stream_queue_length_step.png"
plt.savefig(qlen_path, bbox_inches="tight")
plt.show()

# Show quick summary table
summary_df = pd.read_csv(outdir / "stream_summary.txt", sep="=", header=None, names=["metric","value"])
# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("Streaming Aggregates", summary_df)

str(hist_path), str(qlen_path), str(outdir / "stream_waits.csv"), str(outdir / "stream_queue_len.csv"), str(outdir / "stream_summary.txt")
