"""Unit tests that do not require the full MaleCNS graph."""

from __future__ import annotations

import numpy as np

from flyscroll.feed import synthesize_brainrot, Feed
from flyscroll.flyvision import FlyEyeProcessor, PopulationNovelty
from flyscroll.interest import InterestDecoder, HabituationRule
from flyscroll.analytics import WatchAnalytics
from flyscroll.retina import luminance_samples, chromatic_samples
from flyscroll.transmitters import transmitter_signs


class _EyeBrain:
    def __init__(self, rows=8, cols=8):
        yy, xx = np.mgrid[0:rows, 0:cols]
        self.hexes = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        self.uv = np.column_stack(
            [
                (xx.ravel() + 0.5) / cols,
                (yy.ravel() + 0.5) / rows,
            ]
        ).astype(np.float32)
        self.cell_type = np.asarray(["other"] * (rows * cols), dtype="U16")
        self.chromatic_uv = np.zeros((0, 2), dtype=np.float32)
        self.chromatic_channel = np.zeros(0, dtype="U16")
        self.drive = np.zeros(len(self.cell_type), dtype=np.float32)


def test_transmitter_signs():
    signs, uncertain = transmitter_signs(["acetylcholine", "gaba", "unknown", "acetylcholine,gaba"])
    assert list(signs) == [1, -1, 1, 1]
    assert list(uncertain) == [False, False, True, True]


def test_retina_sampling_shape():
    rgb = np.zeros((64, 36, 3), dtype=np.uint8)
    rgb[:, :, 1] = 255
    uv = np.asarray([[0.2, 0.3], [0.8, 0.7]], dtype=np.float32)
    luma = luminance_samples(rgb, uv)
    assert luma.shape == (2,)
    assert float(luma.mean()) > 0.5
    chroma = chromatic_samples(rgb, uv, np.asarray(["green", "blue"]))
    assert chroma.shape == (2,)
    assert chroma[0] > chroma[1]


def test_fly_eye_separates_on_and_off_transients():
    brain = _EyeBrain()
    eye = FlyEyeProcessor(brain)
    dark = np.zeros((96, 96, 3), dtype=np.uint8)
    bright_patch = dark.copy()
    bright_patch[24:72, 24:72] = 255
    eye.process(dark, brain, 0.05)
    on_frame = eye.process(bright_patch, brain, 0.05)
    off_frame = eye.process(dark, brain, 0.05)
    assert on_frame.features["on"] > on_frame.features["off"]
    assert off_frame.features["off"] > off_frame.features["on"]


def test_fly_eye_distinguishes_opposite_motion():
    brain = _EyeBrain()

    def motion_energy(positions):
        eye = FlyEyeProcessor(brain)
        total = np.zeros(2, dtype=np.float64)
        for x in positions:
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            frame[:, x : x + 14] = 255
            out = eye.process(frame, brain, 0.05)
            total += [out.features["motion_left"], out.features["motion_right"]]
        return total

    forward = motion_energy([8, 22, 36, 50, 64])
    backward = motion_energy([64, 50, 36, 22, 8])
    assert forward.sum() > 0
    assert backward.sum() > 0
    assert int(np.argmax(forward)) != int(np.argmax(backward))


def test_population_novelty_decays_for_repetition_and_recovers_for_change():
    novelty = PopulationNovelty(n_features=4)
    a = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    b = np.asarray([1.0, 0.0, 0.8, 0.1], dtype=np.float32)
    novelty.update(a, 0.05)
    for _ in range(60):
        repeated = novelty.update(a, 0.05)
    changed = novelty.update(b, 0.05)
    assert repeated["population_novelty"] < 1.0
    assert changed["population_novelty"] > repeated["population_novelty"] + 5.0


def test_full_session_plot_history_is_downsampled_without_losing_span():
    analytics = WatchAnalytics(live_window=120)
    analytics.start_reel("r1", "test")
    for i in range(1000):
        analytics.observe(
            interest=float(i % 17),
            novelty=1000.0 if i == 500 else float(i % 23),
            watch_seconds=i * 0.05,
            surprise=None,
            tick=i,
            dt_seconds=0.05,
        )
    snap = analytics.snapshot()
    interest = snap["live_interest"]
    novelty = snap["live_novelty"]
    assert len(interest) <= 120
    assert len(novelty) <= 120
    assert interest[0]["t"] == 0.05
    assert interest[-1]["t"] == 50.0
    assert max(p["v"] for p in novelty) == 1000.0


def test_watch_outcome_histogram_counts_bored_and_full_watches():
    analytics = WatchAnalytics(max_watch_seconds=5.0)
    analytics.start_reel("a", "bored reel")
    analytics.on_scroll(reason="bored", watch_seconds=2.0, tick=1)
    analytics.start_reel("b", "full reel")
    analytics.on_scroll(reason="timeout", watch_seconds=5.0, tick=2)
    assert analytics.snapshot()["outcome_counts"] == {"bored": 1, "full_watch": 1}


def test_watch_percentage_uses_actual_media_progress_and_duration():
    analytics = WatchAnalytics(max_watch_seconds=5.0)
    analytics.start_reel("a", "long reel")
    entry = analytics.on_scroll(
        reason="timeout",
        watch_seconds=5.0,
        media_watch_seconds=6.0,
        reel_duration_seconds=24.0,
        tick=1,
    )
    assert entry["watch_pct"] == 25.0
    assert entry["watch_seconds"] == 6.0
    assert entry["duration_seconds"] == 24.0
    assert analytics.snapshot()["outcome_counts"] == {"bored": 1, "full_watch": 0}


def test_feed_scrolls():
    reels = synthesize_brainrot(seed=1, n_reels=3, w=72, h=128, fps=10)
    feed = Feed(reels)
    first = feed.current.reel_id
    feed.scroll()
    assert feed.current.reel_id != first


def test_interest_scrolls_when_bored():
    class FakeBrain:
        novelty_mbon = np.asarray([0])
        reverse_dn = np.asarray([0])
        novelty_dan = np.asarray([0])
        kenyon = np.asarray([0])
        counts = np.zeros(1, dtype=np.int32)
        _hz = 80.0

        def mean_rate(self, indices, seconds):
            return self._hz

    dec = InterestDecoder(
        scroll_threshold=10.0,
        peak_fraction=0.4,
        refractory_seconds=0.2,
        boredom_dwell_seconds=0.25,
        max_watch_seconds=5.0,
    )
    brain = FakeBrain()
    # Opening spike to set reel peak.
    for _ in range(5):
        brain._hz = 80.0
        state = dec.observe(brain, 0.1)
    # Then collapse novelty so relative boredom can fire.
    brain._hz = 5.0
    state = None
    for _ in range(40):
        state = dec.observe(brain, 0.1)
        if state["scroll"]:
            break
    assert state["scroll"] is True
    assert state["reason"] == "bored"


def test_dishabituation_recovers_weights():
    class FakeBrain:
        plastic_edges = np.asarray([0], dtype=np.int32)
        ptr = np.asarray([0, 1], dtype=np.int64)
        weight = np.asarray([0.3], dtype=np.float32)
        baseline_weight = np.asarray([1.0], dtype=np.float32)
        counts = np.asarray([1], dtype=np.int32)

    rule = HabituationRule()
    brain = FakeBrain()
    info = rule.dishabituate(brain, surprise=0.8)
    assert brain.weight[0] > 0.3
    assert info["recover_frac"] > 0.3


def test_per_reel_budget_limits_drop():
    class FakeBrain:
        plastic_edges = np.asarray([0], dtype=np.int32)
        ptr = np.asarray([0, 1], dtype=np.int64)
        weight = np.asarray([1.0], dtype=np.float32)
        baseline_weight = np.asarray([1.0], dtype=np.float32)
        counts = np.asarray([200], dtype=np.int32)

    rule = HabituationRule(eta=0.5, per_reel_drop_budget=0.2, min_fraction=0.1, homeostasis_tau_seconds=1e9)
    brain = FakeBrain()
    rule.begin_reel(brain)
    for _ in range(30):
        rule.step(brain, 0.05, learning=True)
    # Cannot fall more than ~20% of (1.0 → 0.1) = 0.18 below start → floor at 0.82
    assert brain.weight[0] >= 0.80


def test_spatial_edge_colors():
    from flyscroll.anatomy import build_spatial_edges, rainbow_rgb
    import numpy as np

    assert rainbow_rgb(0.0) != rainbow_rgb(1.0)
    ptr = np.asarray([0, 1, 2], dtype=np.int64)
    post = np.asarray([1, 0], dtype=np.int32)
    cloud = np.asarray([0, 1], dtype=np.int32)
    sc = np.asarray(["ol_intrinsic", "visual_projection"], dtype="U64")
    xyz = np.asarray([[-1, 0, 0], [1, 0, 0]], dtype=np.float32)
    pairs, colors, node_colors = build_spatial_edges(
        ptr, post, cloud, sc, xyz, max_edges=10
    )
    assert pairs.shape == (2, 2)
    assert colors.shape == (2, 3)
    assert node_colors.shape == (2, 3)
    # Left and right nodes should sit in different hue bands.
    assert not np.array_equal(node_colors[0], node_colors[1])

    from flyscroll.anatomy import normalize_cloud
    import numpy as np

    xyz = np.asarray([[0, 0, 0], [100, 50, 25], [np.nan, 0, 0]], dtype=np.float32)
    mask = np.asarray([True, True, False])
    out = normalize_cloud(xyz, mask)
    assert out.shape == (3, 3)
    assert np.isfinite(out[mask]).all()
    assert (out[~mask] == 0).all()


def test_habituation_depresses_weights():
    class FakeBrain:
        plastic_edges = np.asarray([0], dtype=np.int32)
        ptr = np.asarray([0, 1], dtype=np.int64)
        weight = np.asarray([1.0], dtype=np.float32)
        baseline_weight = np.asarray([1.0], dtype=np.float32)
        counts = np.asarray([50], dtype=np.int32)

    rule = HabituationRule(eta=0.05)
    brain = FakeBrain()
    before = float(brain.weight[0])
    info = rule.step(brain, 0.05, learning=True)
    assert brain.weight[0] < before
    assert info["plastic_edges"] == 1


def test_shorts_portrait_crop():
    from PIL import Image
    import numpy as np

    # Mimic WindowStreamer crop/resize path without ScreenCaptureKit.
    rgb = np.zeros((400, 800, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    img = Image.fromarray(rgb, mode="RGB")
    target_w, target_h = 360, 640
    iw, ih = img.size
    target_aspect = target_w / target_h
    src_aspect = iw / ih
    if src_aspect > target_aspect:
        new_w = int(ih * target_aspect)
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    out = np.asarray(img, dtype=np.uint8)
    assert out.shape == (640, 360, 3)
