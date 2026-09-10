"""Video frame → photoreceptor drive.

Maps RGB frames onto inferred ommatidial coordinates. Chromatic R7/R8 channels
are declared sRGB proxies (no ultraviolet in typical reel footage).
"""

from __future__ import annotations

import numpy as np

# Linear sRGB → relative luminance (Rec. 709).
_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

# UV proxy: mix of blue + a little green. Declared engineering stand-in.
_UV_PROXY = np.asarray([0.15, 0.25, 0.60], dtype=np.float32)


def _linearize(rgb: np.ndarray) -> np.ndarray:
    p = rgb.astype(np.float32) / 255.0
    return np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)


def _sample(rgb_lin: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of HxWxC linear RGB at normalised UV coordinates."""
    h, w = rgb_lin.shape[:2]
    x = uv[:, 0] * (w - 1)
    y = uv[:, 1] * (h - 1)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (x - x0).astype(np.float32)[:, None]
    dy = (y - y0).astype(np.float32)[:, None]
    return (
        (1 - dx) * (1 - dy) * rgb_lin[y0, x0]
        + dx * (1 - dy) * rgb_lin[y0, x1]
        + (1 - dx) * dy * rgb_lin[y1, x0]
        + dx * dy * rgb_lin[y1, x1]
    )


def luminance_samples(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Achromatic drive for every mapped R1–R6 receptor."""
    if len(uv) == 0:
        return np.zeros(0, dtype=np.float32)
    lin = _linearize(rgb)
    return (_sample(lin, uv) @ _LUMA).astype(np.float32)


def chromatic_samples(rgb: np.ndarray, uv: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """Per-receptor chromatic drive for R7/R8 using declared channel proxies."""
    if len(uv) == 0:
        return np.zeros(0, dtype=np.float32)
    lin = _linearize(rgb)
    sampled = _sample(lin, uv)
    out = np.zeros(len(channels), dtype=np.float32)
    for i, channel in enumerate(channels):
        name = str(channel)
        if name == "blue":
            out[i] = float(sampled[i, 2])
        elif name == "green":
            out[i] = float(sampled[i, 1])
        elif name == "uv_proxy":
            out[i] = float(sampled[i] @ _UV_PROXY)
        else:
            out[i] = float(sampled[i] @ _LUMA)
    return np.clip(out, 0, 1).astype(np.float32)


def retinal_drive(rgb: np.ndarray, brain) -> tuple[np.ndarray, np.ndarray]:
    """Return (luminance, chromatic) arrays matching ``brain`` receptor lists."""
    luma = luminance_samples(rgb, brain.uv)
    chroma = chromatic_samples(rgb, brain.chromatic_uv, brain.chromatic_channel)
    return luma, chroma
