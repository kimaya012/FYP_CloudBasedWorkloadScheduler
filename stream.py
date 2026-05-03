"""
stream.py — Real-Time Task Stream Generator
============================================
Simulates continuous cloud task arrivals for dynamic scheduling evaluation.
"""

import threading
import random
import time


class StreamGenerator:
    """Generates a stochastic stream of cloud tasks at a configurable rate."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.task_count: int = 0
        self.is_running: bool = False

    def _generate(self, interval: float):
        while not self._stop_event.is_set():
            # CPU demand drawn from a bimodal distribution (low + high demand tasks)
            if random.random() < 0.65:
                demand = random.gauss(mu=0.25, sigma=0.1)   # Low-demand tasks
            else:
                demand = random.gauss(mu=0.75, sigma=0.12)  # High-demand tasks
            demand = max(0.0, min(1.0, demand))
            self.task_count += 1
            time.sleep(interval)

    def start(self, interval: float = 1.0):
        if self.is_running:
            return
        self._stop_event.clear()
        self.task_count = 0
        self.is_running = True
        self._thread = threading.Thread(target=self._generate, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.is_running = False
        return self.task_count
