"""Brainrot reel feed: local videos or procedural synthetic shorts.

By default this module never hits commercial platforms: drop MP4/WebM files into
a directory, or use the built-in generative "brainrot" patterns. Live YouTube
Shorts are opt-in only via ``flyscroll serve --shorts`` (see ``flyscroll.shorts``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class Reel:
    def __init__(self, reel_id: str, title: str, frames: list[np.ndarray], fps: float = 20.0):
        if not frames:
            raise ValueError("Reel needs at least one frame")
        self.reel_id = reel_id
        self.title = title
        self.frames = frames
        self.fps = fps
        self.index = 0

    @property
    def frame(self) -> np.ndarray:
        return self.frames[self.index % len(self.frames)]

    def advance(self) -> np.ndarray:
        self.index = (self.index + 1) % len(self.frames)
        return self.frame

    @property
    def duration_seconds(self) -> float:
        return len(self.frames) / self.fps


def _solid(w, h, rgb) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = rgb
    return frame


def _pil_frame(w, h, paint) -> np.ndarray:
    img = Image.new("RGB", (w, h), (12, 10, 18))
    draw = ImageDraw.Draw(img)
    paint(draw, w, h)
    return np.asarray(img, dtype=np.uint8)


def synthesize_brainrot(seed: int = 0, n_reels: int = 8, w: int = 360, h: int = 640, fps: float = 20.0) -> list[Reel]:
    """Procedural short-form patterns with high motion / contrast / color pop."""
    rng = np.random.default_rng(seed)
    reels: list[Reel] = []
    recipes = [
        ("checker_flicker", "Checker Flicker"),
        ("looming_orb", "Looming Orb"),
        ("scroll_bars", "Doom Bars"),
        ("rgb_strobe", "RGB Strobe"),
        ("swarm_dots", "Swarm Dots"),
        ("face_proxy", "Face Proxy"),
        ("text_flash", "Caption Flash"),
        ("tunnel", "Tunnel Zoom"),
    ]
    for i in range(n_reels):
        key, title = recipes[i % len(recipes)]
        n_frames = int(fps * float(rng.uniform(3.0, 7.0)))
        frames = []
        for t in range(n_frames):
            phase = t / max(n_frames - 1, 1)

            def paint(draw, width, height, key=key, phase=phase, t=t, i=i):
                if key == "checker_flicker":
                    cell = 40
                    on = (t // 2) % 2
                    for y in range(0, height, cell):
                        for x in range(0, width, cell):
                            if ((x // cell) + (y // cell) + on) % 2 == 0:
                                draw.rectangle([x, y, x + cell, y + cell], fill=(240, 240, 40))
                elif key == "looming_orb":
                    r = int(20 + phase * min(width, height) * 0.55)
                    cx, cy = width // 2, height // 2
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 40, 40))
                elif key == "scroll_bars":
                    offset = int(phase * height)
                    for y in range(-40, height + 40, 48):
                        yy = (y + offset) % (height + 48) - 24
                        draw.rectangle([0, yy, width, yy + 24], fill=(40, 200, 255))
                elif key == "rgb_strobe":
                    colors = [(255, 0, 80), (0, 255, 120), (40, 80, 255)]
                    draw.rectangle([0, 0, width, height], fill=colors[t % 3])
                elif key == "swarm_dots":
                    rng_local = np.random.default_rng(i * 1000 + t)
                    for _ in range(80):
                        x = int(rng_local.integers(0, width))
                        y = int(rng_local.integers(0, height))
                        r = int(rng_local.integers(3, 12))
                        draw.ellipse(
                            [x - r, y - r, x + r, y + r],
                            fill=(int(rng_local.integers(80, 255)), 220, 40),
                        )
                elif key == "face_proxy":
                    cx, cy = width // 2, int(height * 0.42)
                    draw.ellipse([cx - 90, cy - 110, cx + 90, cy + 110], fill=(240, 200, 170))
                    eye_y = cy - 20
                    for ex in (cx - 35, cx + 35):
                        draw.ellipse([ex - 12, eye_y - 12, ex + 12, eye_y + 12], fill=(20, 20, 20))
                    smile = 10 + int(8 * math.sin(phase * math.pi * 4))
                    draw.arc([cx - 40, cy + 20, cx + 40, cy + 60 + smile], 20, 160, fill=(120, 40, 40), width=4)
                elif key == "text_flash":
                    draw.rectangle([0, 0, width, height], fill=(10, 10, 10))
                    if t % 6 < 3:
                        draw.rectangle(
                            [40, height // 2 - 40, width - 40, height // 2 + 40],
                            fill=(255, 255, 0),
                        )
                else:  # tunnel
                    for k in range(12, 0, -1):
                        scale = (k / 12) * (1 - 0.5 * phase)
                        rw, rh = int(width * scale / 2), int(height * scale / 2)
                        color = (20 * k, 10 * k, 40 + 15 * k)
                        draw.rectangle(
                            [width // 2 - rw, height // 2 - rh, width // 2 + rw, height // 2 + rh],
                            outline=color,
                            width=6,
                        )

            frames.append(_pil_frame(w, h, paint))
        reels.append(Reel(f"synth-{i:02d}-{key}", f"{title} #{i+1}", frames, fps=fps))
    return reels


def load_video_reel(path: Path, max_frames: int = 180, target_w: int = 360) -> Reel:
    """Load a local video file as a reel (requires imageio)."""
    import imageio.v3 as iio

    frames = []
    for i, frame in enumerate(iio.imiter(path)):
        if i >= max_frames:
            break
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        frame = frame[..., :3]
        h, w = frame.shape[:2]
        if w != target_w:
            new_h = max(1, int(round(h * (target_w / w))))
            frame = np.asarray(
                Image.fromarray(frame).resize((target_w, new_h), Image.Resampling.BILINEAR)
            )
        frames.append(frame.astype(np.uint8))
    if not frames:
        raise ValueError(f"No frames read from {path}")
    meta = iio.immeta(path) or {}
    fps = float(meta.get("fps") or 20.0)
    return Reel(path.stem, path.name, frames, fps=fps)


def load_feed(directory: Path | None = None, seed: int = 0, n_reels: int = 8) -> list[Reel]:
    """Prefer local videos in ``directory``; otherwise synthesize brainrot reels."""
    if directory is not None and directory.exists():
        videos = sorted(
            p
            for p in directory.iterdir()
            if p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi"}
        )
        if videos:
            return [load_video_reel(p) for p in videos]
    return synthesize_brainrot(seed=seed, n_reels=n_reels)


class Feed:
    """Circular doomscroll queue."""

    def __init__(self, reels: list[Reel]):
        if not reels:
            raise ValueError("Feed is empty")
        self.reels = reels
        self.index = 0

    @property
    def current(self) -> Reel:
        return self.reels[self.index % len(self.reels)]

    def scroll(self) -> Reel:
        self.index = (self.index + 1) % len(self.reels)
        self.current.index = 0
        return self.current
