"""Population-novelty interest readout and KC→MBON habituation.

Scroll when fast population prediction error stays low for a dwell after a
short refractory. Existing KC→novelty-MBON edges retain anti-Hebbian depression
and continuous homeostatic recovery; reel changes do not force a novelty pulse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class HabituationRule:
    """Anti-Hebbian depression on existing plastic edges + dishabituation."""

    eta: float = 0.002
    min_fraction: float = 0.25  # floor: never silence the novelty channel entirely
    max_fraction: float = 1.0
    kc_trace_seconds: float = 0.5
    # Max fractional drop from reel-start weight toward the floor, per reel.
    per_reel_drop_budget: float = 0.40
    # Slow pull toward baseline (homeostasis), seconds time-constant.
    homeostasis_tau_seconds: float = 18.0
    depressed_edges: int = 0

    def __post_init__(self):
        self._y_kc = None
        self._reel_start_w = None

    def bind(self, n_edges: int):
        self._y_kc = np.zeros(n_edges, dtype=np.float32)
        self._reel_start_w = None

    def begin_reel(self, brain) -> None:
        edges = brain.plastic_edges
        if len(edges) == 0:
            self._reel_start_w = np.zeros(0, dtype=np.float32)
            return
        self._reel_start_w = np.asarray(brain.weight[edges], dtype=np.float32).copy()

    def dishabituate(self, brain, surprise: float) -> dict:
        """Surprise-gated recovery toward baseline (classic dishabituation)."""
        edges = brain.plastic_edges
        if len(edges) == 0:
            return {"recovered_edges": 0, "recover_frac": 0.0, "surprise": float(surprise)}
        # Map surprise → how far to walk back toward baseline this transition.
        strength = float(np.clip((float(surprise) - 0.12) / 0.55, 0.0, 1.0))
        recover = 0.30 + 0.60 * strength  # 30–90% of the gap to baseline
        base = brain.baseline_weight[edges]
        cur = brain.weight[edges]
        brain.weight[edges] = cur + recover * (base - cur)
        self.begin_reel(brain)
        return {
            "recovered_edges": int(len(edges)),
            "recover_frac": round(recover, 3),
            "surprise": round(float(surprise), 4),
        }

    def step(self, brain, seconds: float, learning: bool = True) -> dict:
        edges = brain.plastic_edges
        if self._y_kc is None or len(self._y_kc) != len(edges):
            self.bind(len(edges))
        if self._reel_start_w is None or len(self._reel_start_w) != len(edges):
            self.begin_reel(brain)
        if len(edges) == 0 or seconds <= 0:
            return {"plastic_edges": 0, "mean_weight_fraction": 1.0, "depressed_edges": 0}

        pre = np.empty(len(edges), dtype=np.int32)
        for i, e in enumerate(edges):
            pre[i] = int(np.searchsorted(brain.ptr, e, side="right") - 1)
        kc_hz = brain.counts[pre].astype(np.float32) / seconds
        ak = math.exp(-seconds / self.kc_trace_seconds)
        self._y_kc[:] = self._y_kc * ak + kc_hz * (1 - ak)

        if learning:
            drive = self.eta * self._y_kc * seconds
            brain.weight[edges] -= drive * np.abs(brain.baseline_weight[edges])

            # Homeostatic drift toward baseline (slow unlearning of depression).
            homeo = 1.0 - math.exp(-seconds / max(self.homeostasis_tau_seconds, 1e-3))
            base = brain.baseline_weight[edges]
            brain.weight[edges] = brain.weight[edges] + homeo * (base - brain.weight[edges])

            lo = self.min_fraction * brain.baseline_weight[edges]
            hi = self.max_fraction * brain.baseline_weight[edges]
            start = self._reel_start_w
            for i, e in enumerate(edges):
                base_e = float(brain.baseline_weight[e])
                w = float(brain.weight[e])
                lo_i, hi_i = float(lo[i]), float(hi[i])
                # Per-reel budget: cannot fall more than budget of the way from start → floor.
                start_e = float(start[i]) if start is not None and i < len(start) else base_e
                if base_e >= 0:
                    floor_budget = start_e - self.per_reel_drop_budget * (start_e - lo_i)
                    w = min(hi_i, max(w, lo_i, floor_budget))
                else:
                    # Negative weights: "floor" is more negative magnitude bound.
                    ceil_budget = start_e + self.per_reel_drop_budget * (lo_i - start_e)
                    w = max(hi_i, min(w, lo_i, ceil_budget))
                brain.weight[e] = w

        frac = brain.weight[edges] / np.where(
            brain.baseline_weight[edges] == 0, 1.0, brain.baseline_weight[edges]
        )
        self.depressed_edges = int(np.count_nonzero(np.abs(frac) < 0.99))
        return {
            "plastic_edges": int(len(edges)),
            "mean_weight_fraction": float(np.mean(np.abs(frac))) if len(frac) else 1.0,
            "depressed_edges": self.depressed_edges,
            "mean_kc_hz": float(kc_hz.mean()) if len(kc_hz) else 0.0,
            "per_reel_drop_budget": self.per_reel_drop_budget,
        }


@dataclass
class InterestDecoder:
    """Phasic novelty + tonic interest → watch-vs-scroll.

    Boredom is *adaptive*: phasic novelty must stay below a fraction of this
    reel's own peak (and/or a low absolute floor) for ``boredom_dwell_seconds``
    after a short refractory. Absolute-only thresholds fail when ongoing motion
    holds a mid-level novelty floor above a fixed Hz cutoff (→ always timeout).
    """

    phasic_tau_seconds: float = 0.12
    tonic_tau_seconds: float = 0.85
    scroll_threshold: float = 14.0  # absolute phasic floor (Hz-equivalent)
    peak_fraction: float = 0.40  # bored when phasic < fraction of reel peak
    boredom_dwell_seconds: float = 1.6
    minimum_low_samples: int = 5
    refractory_seconds: float = 0.35
    max_watch_seconds: float = 5.0
    mdn_boost: float = 0.35
    min_watch_seconds: float = 0.35

    phasic_novelty: float = 0.0
    tonic_interest: float = 0.0
    reel_peak_phasic: float = 0.0
    low_novelty_seconds: float = 0.0
    low_novelty_samples: int = 0
    watch_seconds: float = 0.0
    scrolls: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self):
        if self.min_watch_seconds != 0.35 and self.refractory_seconds == 0.35:
            self.refractory_seconds = float(self.min_watch_seconds)

    @property
    def interest(self) -> float:
        return self.tonic_interest

    def _boredom_level(self) -> float:
        """Adaptive low-novelty bar for this reel."""
        rel = self.peak_fraction * max(self.reel_peak_phasic, self.scroll_threshold)
        return max(self.scroll_threshold, rel)

    def observe(
        self,
        brain,
        seconds: float,
        surprise: float | None = None,
        sensory_novelty: float | None = None,
    ) -> dict:
        if seconds <= 0:
            raise ValueError("Positive interval required")
        novelty_hz = brain.mean_rate(brain.novelty_mbon, seconds)
        mdn_hz = brain.mean_rate(brain.reverse_dn, seconds)
        dan_hz = brain.mean_rate(brain.novelty_dan, seconds)
        kc_hz = brain.mean_rate(brain.kenyon, seconds)

        if sensory_novelty is not None:
            # Prediction error across the fly-eye + visual populations is the
            # primary signal. MBON/DAN/MDN rates remain biological readouts.
            gated = max(0.0, float(sensory_novelty))
        elif surprise is None:
            gated = novelty_hz
        else:
            # Surprise is a multiplier, but saturates so mid-motion can't pin interest forever.
            gated = novelty_hz * float(np.clip(0.15 + 0.9 * surprise, 0.15, 1.4))

        a_fast = 1 - math.exp(-seconds / self.phasic_tau_seconds)
        a_slow = 1 - math.exp(-seconds / self.tonic_tau_seconds)
        self.phasic_novelty = self.phasic_novelty * (1 - a_fast) + gated * a_fast
        self.tonic_interest = self.tonic_interest * (1 - a_slow) + gated * a_slow
        self.watch_seconds += seconds

        # Track peak after a brief open so the opening spike defines "new".
        if self.watch_seconds >= 0.15:
            self.reel_peak_phasic = max(self.reel_peak_phasic, self.phasic_novelty)

        phasic_eff = self.phasic_novelty - self.mdn_boost * mdn_hz
        bar = self._boredom_level()
        if phasic_eff < bar:
            accel = 1.0 + 0.015 * max(mdn_hz, 0.0)
            self.low_novelty_seconds += seconds * accel
            self.low_novelty_samples += 1
        else:
            self.low_novelty_seconds = 0.0
            self.low_novelty_samples = 0

        past_refractory = self.watch_seconds >= self.refractory_seconds
        # Need a real peak before relative boredom can fire (else first frames look "low").
        peaked = self.reel_peak_phasic >= max(self.scroll_threshold * 1.2, 8.0)
        bored = (
            past_refractory
            and peaked
            and self.low_novelty_seconds >= self.boredom_dwell_seconds
            and self.low_novelty_samples >= self.minimum_low_samples
        )
        timed_out = self.watch_seconds >= self.max_watch_seconds
        scroll = bool(bored or timed_out)
        reason = (
            "timeout"
            if timed_out and not bored
            else ("bored" if scroll else "watching")
        )

        state = {
            "interest": round(self.tonic_interest, 3),
            "phasic_novelty": round(self.phasic_novelty, 3),
            "reel_peak_phasic": round(self.reel_peak_phasic, 3),
            "boredom_bar": round(bar, 3),
            "effective_interest": round(float(phasic_eff), 3),
            "novelty_mbon_hz": round(novelty_hz, 3),
            "novelty_dan_hz": round(dan_hz, 3),
            "mdn_hz": round(mdn_hz, 3),
            "kenyon_hz": round(kc_hz, 3),
            "surprise_gate": None if surprise is None else round(float(surprise), 4),
            "sensory_novelty": (
                None if sensory_novelty is None else round(float(sensory_novelty), 3)
            ),
            "low_novelty_seconds": round(self.low_novelty_seconds, 3),
            "low_novelty_samples": self.low_novelty_samples,
            "minimum_low_samples": self.minimum_low_samples,
            "boredom_dwell_seconds": self.boredom_dwell_seconds,
            "refractory_seconds": self.refractory_seconds,
            "watch_seconds": round(self.watch_seconds, 3),
            "scroll": scroll,
            "reason": reason,
            "scrolls": self.scrolls,
        }
        self.history.append(state)
        if len(self.history) > 240:
            self.history = self.history[-240:]
        return state

    def on_scroll(self):
        self.scrolls += 1
        self.watch_seconds = 0.0
        self.low_novelty_seconds = 0.0
        self.low_novelty_samples = 0
        self.reel_peak_phasic = 0.0
        # Preserve continuous sensory state. A genuinely different next reel
        # creates its own prediction error; no synthetic reset pulse is added.
        self.phasic_novelty *= 0.70
        self.tonic_interest *= 0.85
