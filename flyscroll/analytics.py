"""Session-side watch analytics for the spectator dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _envelope_downsample(series: list[dict], limit: int) -> list[dict]:
    """Preserve the full time span and local extrema in a bounded plot payload."""
    if len(series) <= limit or limit < 4:
        return series
    interior = len(series) - 2
    buckets = max(1, (limit - 2) // 2)
    edges = np.linspace(0, interior, buckets + 1, dtype=np.int64)
    out = [series[0]]
    for start, stop in zip(edges[:-1], edges[1:]):
        chunk_start = int(start) + 1
        chunk_stop = int(stop) + 1
        if chunk_stop <= chunk_start:
            continue
        indices = range(chunk_start, chunk_stop)
        lo = min(indices, key=lambda i: series[i]["v"])
        hi = max(indices, key=lambda i: series[i]["v"])
        for i in sorted({lo, hi}):
            out.append(series[i])
    out.append(series[-1])
    return out[:limit]


@dataclass
class WatchAnalytics:
    """Tracks interest/novelty traces, per-reel summaries, and avg watch length."""

    max_watch_seconds: float = 5.0
    live_window: int = 1200  # maximum points sent; stored history spans the session
    max_reels: int = 40

    session_t: float = 0.0
    live_interest: list[dict] = field(default_factory=list)
    live_novelty: list[dict] = field(default_factory=list)
    scroll_events: list[dict] = field(default_factory=list)

    current_trace: list[dict] = field(default_factory=list)
    reel_id: str = ""
    reel_title: str = ""

    completed: list[dict] = field(default_factory=list)
    watch_lengths: list[float] = field(default_factory=list)
    avg_watch_history: list[dict] = field(default_factory=list)
    outcome_counts: dict[str, int] = field(
        default_factory=lambda: {"bored": 0, "full_watch": 0}
    )

    def start_reel(self, reel_id: str, title: str) -> None:
        self.reel_id = reel_id
        self.reel_title = title
        self.current_trace = []

    def observe(
        self,
        *,
        interest: float,
        novelty: float,
        watch_seconds: float,
        surprise: float | None,
        tick: int,
        dt_seconds: float,
    ) -> None:
        self.session_t += max(float(dt_seconds), 0.0)
        t = round(self.session_t, 3)
        self.current_trace.append(
            {
                "t": round(watch_seconds, 3),
                "interest": round(float(interest), 3),
                "novelty": round(float(novelty), 3),
                "surprise": None if surprise is None else round(float(surprise), 4),
                "tick": int(tick),
            }
        )
        self.live_interest.append({"t": t, "v": round(float(interest), 3), "tick": int(tick)})
        self.live_novelty.append({"t": t, "v": round(float(novelty), 3), "tick": int(tick)})

    def on_scroll(
        self,
        *,
        reason: str,
        watch_seconds: float,
        tick: int,
        media_watch_seconds: float | None = None,
        reel_duration_seconds: float | None = None,
    ) -> dict:
        duration = float(reel_duration_seconds or 0.0)
        measured_watch = (
            max(0.0, float(media_watch_seconds or 0.0))
            if duration > 0
            else float(watch_seconds)
        )
        denominator = duration if duration > 0 else max(self.max_watch_seconds, 1e-6)
        watch_pct = _clip(100.0 * measured_watch / denominator, 0.0, 100.0)
        event = {
            "t": round(self.session_t, 3),
            "watch_seconds": round(measured_watch, 3),
            "neural_watch_seconds": round(watch_seconds, 3),
            "duration_seconds": round(duration, 3) if duration > 0 else None,
            "watch_pct": round(watch_pct, 1),
            "reason": reason,
            "tick": int(tick),
            "reel_id": self.reel_id,
            "title": self.reel_title,
        }
        self.scroll_events.append(event)
        outcome = "full_watch" if watch_pct >= 95.0 else "bored"
        self.outcome_counts[outcome] += 1

        trace = list(self.current_trace)
        entry = {
            "reel_id": self.reel_id,
            "title": self.reel_title,
            "watch_seconds": round(measured_watch, 3),
            "neural_watch_seconds": round(watch_seconds, 3),
            "duration_seconds": round(duration, 3) if duration > 0 else None,
            "watch_pct": round(watch_pct, 1),
            "reason": reason,
            "tick": int(tick),
            "interest": [{"t": p["t"], "v": p["interest"]} for p in trace],
            "novelty": [{"t": p["t"], "v": p["novelty"]} for p in trace],
            "skip_t": round(watch_seconds, 3),
        }
        self.completed.insert(0, entry)
        if len(self.completed) > self.max_reels:
            self.completed = self.completed[: self.max_reels]

        self.watch_lengths.append(float(measured_watch))
        avg = sum(self.watch_lengths) / len(self.watch_lengths)
        self.avg_watch_history.append(
            {
                "n": len(self.watch_lengths),
                "avg": round(avg, 3),
                "watch": round(measured_watch, 3),
                "t": round(self.session_t, 3),
            }
        )
        self.current_trace = []
        return entry

    @property
    def avg_watch_seconds(self) -> float | None:
        if not self.watch_lengths:
            return None
        return round(sum(self.watch_lengths) / len(self.watch_lengths), 3)

    def snapshot(self) -> dict:
        lengths = list(self.watch_lengths)
        dist = None
        if lengths:
            arr = np.asarray(lengths, dtype=np.float64)
            dist = {
                "n": int(len(arr)),
                "mean": round(float(arr.mean()), 3),
                "std": round(float(arr.std()), 3),
                "p10": round(float(np.percentile(arr, 10)), 3),
                "p50": round(float(np.percentile(arr, 50)), 3),
                "p90": round(float(np.percentile(arr, 90)), 3),
                "min": round(float(arr.min()), 3),
                "max": round(float(arr.max()), 3),
            }
        return {
            "live_interest": _envelope_downsample(self.live_interest, self.live_window),
            "live_novelty": _envelope_downsample(self.live_novelty, self.live_window),
            "scroll_events": self.scroll_events,
            "reels": self.completed,
            "avg_watch_seconds": self.avg_watch_seconds,
            "avg_watch_history": self.avg_watch_history,
            "watches": len(self.watch_lengths),
            "max_watch_seconds": self.max_watch_seconds,
            "watch_time_distribution": dist,
            "outcome_counts": dict(self.outcome_counts),
        }
