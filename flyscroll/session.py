"""Live doomscroll session: frames → retina → LIF → interest → scroll."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from flyscroll.engine import Brain
from flyscroll.feed import Feed
from flyscroll.flyvision import FlyEyeProcessor, PopulationNovelty
from flyscroll.interest import HabituationRule, InterestDecoder

ROOT = Path(__file__).resolve().parents[1]


def encode_jpeg(rgb: np.ndarray, quality: int = 70) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


@dataclass
class SessionConfig:
    scale: str = "full"
    dataset: str = "malecns_v1"
    frame_ms: float = 50.0  # neural time advanced per displayed frame
    learning: bool = True
    interest_threshold: float = 14.0
    min_watch_seconds: float = 0.35  # refractory (legacy CLI name)
    max_watch_seconds: float = 5.0
    boredom_dwell_seconds: float = 1.6
    peak_fraction: float = 0.40
    seed: int = 0
    shorts: bool = False


class DoomscrollSession:
    def __init__(self, feed: Feed, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        graph = (
            ROOT
            / "outputs"
            / "flyscroll"
            / self.config.dataset
            / self.config.scale
            / "graph.npz"
        )
        manifest_path = graph.with_name("manifest.json")
        if not graph.exists():
            raise FileNotFoundError(
                f"Missing prepared graph at {graph}. Run: "
                f"python -m flyscroll.prepare --scale {self.config.scale}"
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.brain = Brain(graph)
        self.feed = feed
        decoder_max_watch = (
            math.inf if self.config.shorts else self.config.max_watch_seconds
        )
        self.decoder = InterestDecoder(
            scroll_threshold=self.config.interest_threshold,
            refractory_seconds=self.config.min_watch_seconds,
            min_watch_seconds=self.config.min_watch_seconds,
            max_watch_seconds=decoder_max_watch,
            boredom_dwell_seconds=self.config.boredom_dwell_seconds,
            peak_fraction=self.config.peak_fraction,
        )
        self.habituation = HabituationRule()
        self.habituation.bind(len(self.brain.plastic_edges))
        self.habituation.begin_reel(self.brain)
        self.eye = FlyEyeProcessor(self.brain)
        self.population_novelty = PopulationNovelty(
            n_features=len(self.eye.FEATURE_NAMES)
        )
        self.population_rate_ema = np.zeros(11, dtype=np.float32)
        self.run_id = str(uuid.uuid4())
        self.tick = 0
        self.started = time.time()
        self.latest: dict = {"status": "starting"}
        from flyscroll.analytics import WatchAnalytics

        self.analytics = WatchAnalytics(max_watch_seconds=self.config.max_watch_seconds)
        cur = self.feed.current
        self.analytics.start_reel(getattr(cur, "reel_id", ""), getattr(cur, "title", ""))
        self._last_dishabituation: dict = {}
        self._last_wall_step: float | None = None
        self._init_cloud()
        self.anatomy.update(self.eye.anatomy_payload())

    def _init_cloud(self):
        brain = self.brain
        if brain.xyz is None or brain.soma_mask is None:
            self.cloud_index = np.zeros(0, dtype=np.int32)
            self.cloud_xyz = np.zeros((0, 3), dtype=np.float32)
            self.cloud_activity = np.zeros(0, dtype=np.float32)
            self.cloud_baseline = np.zeros(0, dtype=np.float32)
            self.anatomy = {"points": 0, "xyz_b64": "", "dtype": "float32", "coverage": 0}
            return
        from flyscroll.anatomy import build_spatial_edges, pick_display_indices, refit_display_xyz

        # Subsample lamina: tonic bias makes a full lamina include look "always on".
        lamina = np.asarray(brain.lamina, dtype=np.int32)
        if len(lamina) > 400:
            lamina = lamina[np.linspace(0, len(lamina) - 1, 400, dtype=int)]
        must = np.unique(
            np.concatenate(
                [
                    np.asarray(brain.retina, dtype=np.int32),
                    np.asarray(brain.chromatic, dtype=np.int32),
                    lamina,
                ]
            )
        )
        self.cloud_index = pick_display_indices(
            brain.soma_mask.astype(bool),
            brain.superclass,
            max_points=10000,
            must_include=must,
            cell_type=brain.cell_type,
            xyz=np.asarray(brain.xyz, dtype=np.float32),
        )
        self.cloud_xyz = refit_display_xyz(
            np.asarray(brain.xyz[self.cloud_index], dtype=np.float32)
        )
        self.cloud_activity = np.zeros(len(self.cloud_index), dtype=np.float32)
        self.cloud_baseline = np.zeros(len(self.cloud_index), dtype=np.float32)
        pairs, edge_colors, node_colors = build_spatial_edges(
            brain.ptr,
            brain.post,
            self.cloud_index,
            brain.superclass,
            self.cloud_xyz,
            max_edges=48_000,
            seed=self.config.seed + 17,
        )
        located = int(brain.soma_mask.sum())
        self.anatomy = {
            "points": int(len(self.cloud_index)),
            "edges": int(len(pairs)),
            "neurons": int(brain.n),
            "soma_located": located,
            "xyz_b64": _b64(self.cloud_xyz),
            "edges_b64": _b64(pairs),
            "edge_rgb_b64": _b64(edge_colors),
            "node_rgb_b64": _b64(node_colors),
            "dtype": "float32",
            "edge_dtype": "int32",
            "color_dtype": "uint8",
            "layout": "x=lateral, y=-raw_y (up), z=depth",
            "source": "MaleCNS somaLocation + subsampled spatial edges",
            "note": "Rainbow filaments by lateral position; spikes only as sparse sparks",
            "style": "schlegel_filaments",
        }

    def _update_cloud(self, counts: np.ndarray, seconds: float) -> dict:
        if len(self.cloud_index) == 0 or seconds <= 0:
            return {"points": 0}
        spikes = counts[self.cloud_index].astype(np.float32)
        # Slow baseline tracks tonic firing (lamina bias, steady drive).
        b_alpha = 1 - math.exp(-seconds / 0.6)
        self.cloud_baseline *= 1 - b_alpha
        self.cloud_baseline += b_alpha * spikes
        phasic = np.maximum(0.0, spikes - self.cloud_baseline)
        decay = math.exp(-seconds / 0.08)
        self.cloud_activity *= decay
        self.cloud_activity += phasic * 3.0
        glow = 1.0 - np.exp(-self.cloud_activity / 2.5)
        packed = np.rint(np.clip(glow, 0, 1) * 255).astype(np.uint8)
        return {
            "points": int(len(packed)),
            "activity_b64": _b64(packed),
            "dtype": "uint8",
            "active_points": int(np.count_nonzero(packed > 20)),
            "mode": "phasic_spike",
        }

    def step(self) -> dict:
        now = time.monotonic()
        neural_seconds = self.config.frame_ms / 1000.0
        if self._last_wall_step is None:
            wall_seconds = neural_seconds
        else:
            wall_seconds = float(np.clip(now - self._last_wall_step, 0.005, 1.0))
        self._last_wall_step = now
        reel = self.feed.current
        frame = reel.frame
        reel.advance()
        eye_frame = self.eye.process(frame, self.brain, dt_seconds=wall_seconds)

        def enrich(brain):
            return self.eye.apply_feature_drive(brain, eye_frame)

        counts, wall, proxy = self.brain.step(
            eye_frame.luminance,
            eye_frame.chromatic,
            duration_ms=self.config.frame_ms,
            enrich_drive=enrich,
        )
        plastic = self.habituation.step(
            self.brain, neural_seconds, learning=self.config.learning
        )
        group_rates = {}
        for name, inds in self.manifest.get("readout_groups", {}).items():
            if inds:
                group_rates[name] = round(
                    float(
                        counts[np.asarray(inds, dtype=np.int32)].sum()
                        / (len(inds) * neural_seconds)
                    ),
                    3,
                )
        visual_group_names = (
            "motion_up",
            "motion_down",
            "motion_horizontal",
            "widefield_vertical",
            "widefield_horizontal",
            "looming",
            "small_object",
            "bar_edge",
            "novelty_mbon",
            "novelty_dan",
            "visual_dn",
        )
        rate_vector = np.asarray(
            [math.log1p(group_rates.get(name, 0.0)) for name in visual_group_names],
            dtype=np.float32,
        )
        rate_alpha = 1.0 - math.exp(-wall_seconds / 0.45)
        self.population_rate_ema += rate_alpha * (
            rate_vector - self.population_rate_ema
        )
        population_vector = np.concatenate(
            [
                eye_frame.feature_vector,
                # Spikes counted in a 50 ms bin are noisy, especially for tiny
                # groups such as VS. Smooth and down-weight them so analog
                # content, not Poisson-like bin variance, controls prediction.
                0.05 * self.population_rate_ema,
            ]
        )
        novelty_state = self.population_novelty.update(population_vector, wall_seconds)
        surprise = novelty_state["population_novelty"] / 100.0
        interest = self.decoder.observe(
            self.brain,
            wall_seconds,
            sensory_novelty=novelty_state["population_novelty"],
        )
        media_now = getattr(self.feed, "current_watch_metrics", {}) or {}
        media_watch = float(media_now.get("watch_seconds", 0.0) or 0.0)
        media_duration = float(media_now.get("duration_seconds", 0.0) or 0.0)
        media_completed = (
            self.config.shorts
            and media_duration > 0
            and media_watch >= 0.98 * media_duration
        )
        if media_completed:
            interest["scroll"] = True
            interest["reason"] = "completed"
        self.analytics.observe(
            interest=interest["interest"],
            novelty=novelty_state["population_novelty"],
            watch_seconds=interest["watch_seconds"],
            surprise=surprise,
            tick=self.tick + 1,
            dt_seconds=wall_seconds,
        )
        scrolled = False
        scroll_attempted = False
        if interest["scroll"]:
            scroll_attempted = True
            previous_reel_id = str(reel.reel_id)
            nxt = self.feed.scroll()
            if str(nxt.reel_id) != previous_reel_id:
                media = getattr(self.feed, "last_watch_metrics", {}) or {}
                self.analytics.on_scroll(
                    reason=interest["reason"],
                    watch_seconds=interest["watch_seconds"],
                    tick=self.tick + 1,
                    media_watch_seconds=media.get("watch_seconds"),
                    reel_duration_seconds=media.get("duration_seconds"),
                )
                self.decoder.on_scroll()
                # Keep sensory prediction continuous across reels. The actual visual
                # transition, not a hard-coded reset, supplies dishabituation.
                self._last_dishabituation = {
                    "recovered_edges": 0,
                    "policy": "continuous_stimulus_specific_homeostasis",
                }
                self.habituation.begin_reel(self.brain)
                self.analytics.start_reel(nxt.reel_id, nxt.title)
                scrolled = True
                # Navigation latency belongs to neither reel's watch duration.
                self._last_wall_step = time.monotonic()
            else:
                interest["scroll"] = False
                interest["reason"] = "scroll_failed"
        self.tick += 1
        cloud = self._update_cloud(counts, wall_seconds)
        frame_jpeg = "" if self.config.shorts else encode_jpeg(frame)
        state = {
            "status": "running",
            "run_id": self.run_id,
            "tick": self.tick,
            "neural_ms": round(self.brain.sim_ms, 3),
            "wall_ms": round(wall * 1000, 3),
            "realtime_factor": round((self.config.frame_ms / 1000.0) / max(wall, 1e-9), 4),
            "total_spikes": self.brain.total_spikes,
            "active_neurons": int(self.brain.nactive[0]),
            "shorts": bool(self.config.shorts),
            "capture_backend": getattr(self.feed, "capture_backend", "local_frames"),
            "reel": {
                "id": reel.reel_id,
                "title": reel.title,
                "index": self.feed.index,
                "frame": reel.index,
                "n_reels": len(self.feed.reels),
            },
            "interest": interest,
            "habituation": {**plastic, "dishabituation": self._last_dishabituation},
            "proxy": proxy,
            "population_novelty": novelty_state,
            "fly_eye": self.eye.frame_payload(eye_frame),
            "group_rates": group_rates,
            "cloud": cloud,
            "analytics": self.analytics.snapshot(),
            "scrolled": scrolled,
            "scroll_attempted": scroll_attempted,
            "scroll_error": getattr(self.feed, "last_scroll_error", ""),
            "media_completed": media_completed,
            "frame_jpeg": frame_jpeg,
            "input_sha256": hashlib.sha256(
                eye_frame.luminance.tobytes()
            ).hexdigest()[:16],
            "scale": self.config.scale,
            "cropped": bool(self.manifest.get("cropped")),
            "scientifically_validated": False,
            "decoder": {
                "phasic_novelty": interest.get("phasic_novelty"),
                "tonic_interest": interest.get("interest"),
                "low_novelty_seconds": interest.get("low_novelty_seconds"),
                "boredom_dwell_seconds": interest.get("boredom_dwell_seconds"),
                "refractory_seconds": interest.get("refractory_seconds"),
            },
            "generated_at_ms": int(time.time() * 1000),
        }
        self.latest = state
        return state
