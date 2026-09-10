"""Compiled all-edge LIF simulation for FlyScroll.

Same membrane/synapse constants as the DOOMFLY/Shiu-like probe. Analytic
subthreshold integration, threshold check each 0.1 ms, 1.8 ms delay.
Retina and lamina use a DECLARED coarse spiking approximation to graded cells.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def advance(
    ptr,
    post,
    weight,
    v,
    g,
    refractory,
    drive,
    queue,
    queue_count,
    cursor,
    steps,
    dt,
    counts,
    active,
    active_flag,
    nactive,
):
    av = math.exp(-dt / 20)
    ag = math.exp(-dt / 5)
    coupling = (av - ag) / 3
    delay_slots = queue.shape[0]
    for step in range(steps):
        slot = cursor % delay_slots
        for k in range(nactive[0]):
            i = active[k]
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                v[i] = -52 + (v[i] + 52) * av + drive[i] * (1 - av) + g[i] * coupling
                g[i] *= ag
                if v[i] > -45:
                    counts[i] += 1
                    future = (cursor + int(round(1.8 / dt))) % delay_slots
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                g[j] += weight[e]
                if active_flag[j] == 0:
                    active_flag[j] = 1
                    active[nactive[0]] = j
                    nactive[0] += 1
        queue_count[slot] = 0
        future = (cursor + int(round(1.8 / dt))) % delay_slots
        for q in range(queue_count[future]):
            i = queue[future, q]
            v[i] = -52
            g[i] = 0
            refractory[i] = int(round(2.2 / dt))
        cursor += 1
    return cursor


class Brain:
    """Whole-graph (or labelled crop) LIF network driven by retinal channels."""

    def __init__(self, path: str | Path, dt: float = 0.1):
        if dt != 0.1:
            raise ValueError("This kernel supports only dt=0.1 ms.")
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        required = [
            "ptr",
            "post",
            "weight",
            "ids",
            "retina",
            "uv",
            "lamina",
            "chromatic",
            "chromatic_uv",
            "chromatic_channel",
            "plastic_edges",
            "novelty_mbon",
            "novelty_dan",
            "reverse_dn",
            "kenyon",
            "superclass",
        ]
        for key in required:
            if key not in data:
                raise ValueError(f"Graph archive missing {key}")
            setattr(self, key, data[key])
        self.cell_type = data["cell_type"] if "cell_type" in data else None
        self.hexes = data["hexes"] if "hexes" in data else None
        self.xyz = data["xyz"] if "xyz" in data else None
        self.soma_mask = data["soma_mask"].astype(bool) if "soma_mask" in data else None
        self.cloud_index = data["cloud_index"] if "cloud_index" in data else None
        self.baseline_weight = self.weight.copy()
        n = len(self.ids)
        for key, dtype in [
            ("ptr", np.int64),
            ("post", np.int32),
            ("weight", np.float32),
            ("ids", np.int64),
            ("retina", np.int32),
            ("lamina", np.int32),
            ("chromatic", np.int32),
        ]:
            x = getattr(self, key)
            if x.ndim != 1 or x.dtype != dtype or not x.flags.c_contiguous:
                raise ValueError(f"Invalid native graph array: {key}")
        if (
            n < 1
            or self.ptr.shape != (n + 1,)
            or self.ptr[0] != 0
            or self.ptr[-1] != len(self.post)
            or np.any(np.diff(self.ptr) < 0)
            or len(self.weight) != len(self.post)
        ):
            raise ValueError("Invalid CSR graph")
        if not np.isfinite(self.weight).all():
            raise ValueError("Nonfinite synaptic weight")
        for x in [self.post, self.retina, self.lamina, self.chromatic]:
            if len(x) and (np.any(x < 0) or np.any(x >= n)):
                raise ValueError("Graph index out of bounds")
        if self.uv.shape != (len(self.retina), 2):
            raise ValueError("Invalid receptor UV coordinates")
        if self.hexes is None or self.hexes.shape != (len(self.retina), 2):
            raise ValueError("Invalid receptor ommatidial coordinates")
        self.dt = dt
        self.n = n
        self.cursor = 0
        self.v = np.full(self.n, -52, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.queue = np.zeros((int(round(1.8 / dt)) + 1, self.n), dtype=np.int32)
        self.queue_count = np.zeros(self.queue.shape[0], dtype=np.int32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.luminance = np.zeros(len(self.retina), dtype=np.float32)
        self.chromatic_level = np.zeros(len(self.chromatic), dtype=np.float32)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        initial = np.unique(np.r_[self.retina, self.lamina, self.chromatic])
        self.active[: len(initial)] = initial
        self.active_flag[initial] = 1
        self.nactive = np.asarray([len(initial)], dtype=np.int32)
        self.total_spikes = 0
        self.sim_ms = 0.0

    def step(
        self,
        luminance: np.ndarray,
        chromatic: np.ndarray | None = None,
        duration_ms: float = 1.0,
        lamina_bias: float = 12.0,
        enrich_drive=None,
    ):
        if len(luminance) != len(self.retina) or not np.all(np.isfinite(luminance)):
            raise ValueError("A finite luminance sample is required for every mapped receptor")
        if chromatic is None:
            chromatic = np.zeros(len(self.chromatic), dtype=np.float32)
        if len(chromatic) != len(self.chromatic) or not np.all(np.isfinite(chromatic)):
            raise ValueError("Chromatic samples must match mapped R7/R8 receptors")
        if not math.isfinite(duration_ms) or not math.isfinite(lamina_bias):
            raise ValueError("Finite duration and current required")
        steps = int(round(duration_ms / self.dt))
        if steps < 1:
            raise ValueError("Duration too short")
        alpha = 1 - math.exp(-steps * self.dt / 10)
        self.luminance += alpha * (np.clip(luminance, 0, 1) - self.luminance)
        self.chromatic_level += alpha * (np.clip(chromatic, 0, 1) - self.chromatic_level)
        self.drive.fill(0)
        self.drive[self.lamina] = lamina_bias
        self.drive[self.retina] = 30 * self.luminance / (0.02 + self.luminance)
        if len(self.chromatic):
            self.drive[self.chromatic] = 30 * self.chromatic_level / (0.02 + self.chromatic_level)
        proxy_info = enrich_drive(self) if enrich_drive is not None else None
        # Ensure externally driven cells stay in the active set.
        driven = np.flatnonzero(self.drive > 0)
        for i in driven:
            if self.active_flag[i] == 0:
                self.active_flag[i] = 1
                self.active[self.nactive[0]] = i
                self.nactive[0] += 1
        self.counts.fill(0)
        start = time.perf_counter()
        self.cursor = advance(
            self.ptr,
            self.post,
            self.weight,
            self.v,
            self.g,
            self.refractory,
            self.drive,
            self.queue,
            self.queue_count,
            self.cursor,
            steps,
            self.dt,
            self.counts,
            self.active,
            self.active_flag,
            self.nactive,
        )
        elapsed = time.perf_counter() - start
        self.total_spikes += int(self.counts.sum())
        self.sim_ms += steps * self.dt
        self._compact_active()
        return self.counts.copy(), elapsed, proxy_info

    def _compact_active(self) -> None:
        """Drop quiescent cells so a long session does not become all-neuron dense."""
        n = int(self.nactive[0])
        if n == 0:
            return
        indices = self.active[:n]
        keep = (
            (self.drive[indices] != 0)
            | (np.abs(self.v[indices] + 52.0) > 0.5)
            | (np.abs(self.g[indices]) > 0.1)
            | (self.refractory[indices] > 0)
        )
        dropped = indices[~keep]
        if len(dropped):
            self.active_flag[dropped] = 0
        retained = indices[keep]
        self.active[: len(retained)] = retained
        self.nactive[0] = len(retained)

    def mean_rate(self, indices: np.ndarray, seconds: float) -> float:
        if seconds <= 0 or len(indices) == 0:
            return 0.0
        return float(self.counts[indices].sum() / (len(indices) * seconds))
