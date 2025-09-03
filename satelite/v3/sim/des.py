from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict
import heapq, itertools

SimTime = float

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
        self._seq = itertools.count()
        self._handles: Dict[int, _Event] = {}

    @property
    def now(self) -> SimTime: return self._now

    def schedule_at(self, t: SimTime, fn: Callable[[], None], priority: int = 0) -> int:
        if t < self._now:
            raise ValueError("Cannot schedule in the past")
        ev = _Event(time=t, priority=priority, _seq=next(self._seq), fn=fn)
        heapq.heappush(self._heap, ev)
        self._handles[ev._seq] = ev
        return ev._seq

    def schedule_in(self, dt: SimTime, fn: Callable[[], None], priority: int = 0) -> int:
        return self.schedule_at(self._now + dt, fn, priority)

    def cancel(self, h: int) -> bool:
        ev = self._handles.pop(h, None)
        if not ev or ev.cancelled: return False
        ev.cancelled = True
        return True

    def run_until_empty(self):
        while self._heap:
            ev = heapq.heappop(self._heap)
            if ev.cancelled: continue
            self._now = ev.time
            ev.fn()
