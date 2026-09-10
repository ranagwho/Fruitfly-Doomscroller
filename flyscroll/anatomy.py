"""Soma-coordinate anatomy for activation point clouds.

Uses MaleCNS ``somaLocation`` (nm). Missing somata are omitted from the cloud,
never invented — the viewer labels coverage.
"""

from __future__ import annotations

import numpy as np
import pyarrow.feather as feather

_MB_PREFIXES = ("KC", "MBON", "PAM", "PPL")

# Neon palette inspired by Schlegel-style connectome plates (sRGB 0–255).
SUPERCLASS_RGB = {
    "ol_sensory": (255, 230, 70),
    "ol_intrinsic": (80, 255, 120),
    "visual_projection": (255, 55, 210),
    "visual_projection_tbc": (255, 90, 230),
    "visual_centrifugal": (255, 140, 40),
    "cb_intrinsic": (130, 100, 255),
    "cb_sensory": (180, 140, 255),
    "descending_neuron": (255, 80, 120),
    "ascending_neuron": (255, 120, 180),
    "vnc_intrinsic": (40, 220, 255),
    "vnc_sensory": (60, 200, 255),
    "vnc_motor": (255, 200, 90),
}
DEFAULT_RGB = (140, 170, 255)


def _is_mb_type(name: str) -> bool:
    return str(name).startswith(_MB_PREFIXES)


def soma_xyz_for_ids(body_ids: np.ndarray, annotations_path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz float32 Nx3, mask bool N) aligned to ``body_ids``."""
    ann = feather.read_table(annotations_path).to_pandas().set_index("bodyId")
    xyz = np.full((len(body_ids), 3), np.nan, dtype=np.float32)
    for i, bid in enumerate(body_ids):
        if bid not in ann.index:
            continue
        loc = ann.at[bid, "somaLocation"]
        if loc is None:
            continue
        try:
            if isinstance(loc, float) and np.isnan(loc):
                continue
            coords = [float(c) for c in loc]
        except (TypeError, ValueError):
            continue
        if len(coords) != 3 or not np.isfinite(coords).all():
            continue
        xyz[i] = coords
    mask = np.isfinite(xyz).all(axis=1)
    return xyz, mask


def normalize_cloud(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Center/scale finite points. Axes: X lateral, -Y vertical (anterior/up-ish), Z depth."""
    out = np.zeros_like(xyz, dtype=np.float32)
    pts = xyz[mask]
    if len(pts) == 0:
        return out
    center = pts.mean(axis=0)
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    span = float(np.max(hi - lo)) or 1.0
    scaled = (xyz - center) / (0.5 * span)
    # Lateral squeeze: optic lobes sit far apart in raw coords; pull them in.
    out[:, 0] = scaled[:, 0] * 0.62
    out[:, 1] = -scaled[:, 1]
    out[:, 2] = scaled[:, 2] * 0.88
    out[~mask] = 0
    return out


def refit_display_xyz(xyz: np.ndarray) -> np.ndarray:
    """Re-center/scale the *sampled* cloud so lobes sit as one plate, not two islands."""
    if len(xyz) == 0:
        return xyz
    out = np.asarray(xyz, dtype=np.float32).copy()
    lo = np.percentile(out, 1, axis=0)
    hi = np.percentile(out, 99, axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo)) or 1.0
    out = (out - center) / (0.48 * span)

    # Collapse the hollow between optic lobes: shift each hemisphere toward ±target.
    x = out[:, 0]
    left = x < 0
    right = x >= 0
    target = 0.28
    if np.any(left):
        med_l = float(np.median(x[left]))
        if med_l < -1e-4:
            x[left] = x[left] - (med_l + target)
    if np.any(right):
        med_r = float(np.median(x[right]))
        if med_r > 1e-4:
            x[right] = x[right] - (med_r - target)
    # Soft outer knee so extremes don't clip the frame.
    knee = 0.5 + 0.5 / (1.0 + np.abs(x) * 2.2)
    out[:, 0] = x * knee

    lo2 = np.percentile(out, 1, axis=0)
    hi2 = np.percentile(out, 99, axis=0)
    mid = 0.5 * (lo2 + hi2)
    span2 = float(np.max(hi2 - lo2)) or 1.0
    out = (out - mid) / (0.5 * span2)
    # Flatten depth so the plate reads face-on and fills XY, not a 3D blob.
    out[:, 2] *= 0.55
    return out.astype(np.float32)


def pick_display_indices(
    mask: np.ndarray,
    superclass: np.ndarray,
    max_points: int = 12000,
    must_include: np.ndarray | None = None,
    cell_type: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Subsample biased toward optic-lobe sensory cells.

    KC/MBON/PAM/PPL are excluded: under the visual-crop novelty proxy they fire
    every frame and paint false permanent blobs between the optic lobes.
    """
    have = np.flatnonzero(mask)
    if cell_type is not None:
        mb = np.fromiter((_is_mb_type(t) for t in cell_type), dtype=bool, count=len(cell_type))
        have = have[~mb[have]]

    if must_include is not None and len(must_include):
        must = np.unique(must_include[mask[must_include]])
        if cell_type is not None:
            must = np.asarray([i for i in must if not _is_mb_type(cell_type[i])], dtype=np.int32)
    else:
        must = np.zeros(0, dtype=np.int32)

    if len(have) <= max_points:
        return np.unique(np.concatenate([must, have])).astype(np.int32)[:max_points]

    priority = np.ones(len(mask), dtype=np.float32)
    for i, sc in enumerate(superclass):
        name = str(sc)
        if name.startswith("ol_sensory"):
            priority[i] = 6.0
        elif name == "ol_intrinsic":
            priority[i] = 2.2  # enough for lobes, not so many the midline vanishes
        elif "visual_projection" in name:
            priority[i] = 7.0  # fill the bridge between optic lobes
        elif "visual" in name:
            priority[i] = 5.0
        elif "descending" in name or "vnc" in name:
            priority[i] = 2.5
        elif name.startswith("cb_"):
            priority[i] = 3.5  # central mass for a continuous plate
        else:
            priority[i] = 1.0

    remaining_slots = max(0, max_points - len(must))
    pool = have[~np.isin(have, must)]
    rng = np.random.default_rng(41027)

    def take(pool_idx: np.ndarray, n: int, scores: np.ndarray | None = None) -> np.ndarray:
        if len(pool_idx) == 0 or n <= 0:
            return np.zeros(0, dtype=np.int32)
        n = min(n, len(pool_idx))
        if scores is None:
            scores = priority[pool_idx] * rng.random(len(pool_idx))
        return pool_idx[np.argsort(scores)[::-1][:n]].astype(np.int32)

    # Spatial midline first (soma X near 0) — class tags alone sit in the lobes.
    mid_pool = pool
    mid_scores = None
    if xyz is not None and len(pool):
        xs = np.asarray(xyz, dtype=np.float32)[pool, 0]
        x_abs = np.abs(xs - float(np.median(xs)))
        x_thr = float(np.percentile(x_abs, 28))  # central ~28% of lateral span
        mid_mask = x_abs <= max(x_thr, 1e-3)
        mid_pool = pool[mid_mask]
        # Prefer true center + high class priority.
        mid_scores = (priority[mid_pool] * (1.4 - 0.9 * (x_abs[mid_mask] / (x_thr + 1e-6)))) * (
            0.35 + 0.65 * rng.random(len(mid_pool))
        )
    else:
        mid_pool = np.asarray(
            [
                i
                for i in pool
                if "visual" in str(superclass[i]) or str(superclass[i]).startswith("cb_")
            ],
            dtype=np.int32,
        )

    mid_slots = min(len(mid_pool), max(remaining_slots * 45 // 100, remaining_slots // 3))
    mid_chosen = take(mid_pool, mid_slots, mid_scores)

    rest_pool = pool[~np.isin(pool, mid_chosen)]
    rest_slots = max(0, remaining_slots - len(mid_chosen))
    chosen = take(rest_pool, rest_slots)
    out = np.unique(
        np.concatenate(
            [must.astype(np.int32), mid_chosen.astype(np.int32), chosen.astype(np.int32)]
        )
    )
    if len(out) > max_points:
        keep_first = np.unique(np.concatenate([must, mid_chosen])).astype(np.int32)
        extra = out[~np.isin(out, keep_first)]
        budget = max(0, max_points - len(keep_first))
        out = np.concatenate([keep_first[:max_points], extra[:budget]])[:max_points]
    return out.astype(np.int32)


def rainbow_rgb(t: float) -> tuple[int, int, int]:
    """Schlegel-plate-like hue: left warm → center cyan → right magenta."""
    t = float(np.clip(t, 0.0, 1.0))
    # Hue wheel positions chosen to match the reference plate.
    h = 0.10 + 0.78 * t  # ~yellow/lime → cyan → violet/magenta
    s, v = 0.92, 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    u = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, u, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, u
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = u, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def build_spatial_edges(
    ptr: np.ndarray,
    post: np.ndarray,
    cloud_index: np.ndarray,
    superclass: np.ndarray,
    xyz: np.ndarray,
    max_edges: int = 140_000,
    seed: int = 41027,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample directed edges; color by lateral (X) rainbow, not cell class blobs."""
    cloud_index = np.asarray(cloud_index, dtype=np.int32)
    xyz = np.asarray(xyz, dtype=np.float32)
    n_cloud = len(cloud_index)

    if n_cloud == 0:
        empty_i = np.zeros((0, 2), dtype=np.int32)
        empty_c = np.zeros((0, 3), dtype=np.uint8)
        return empty_i, empty_c, empty_c

    xs = xyz[:, 0]
    x_lo, x_hi = np.percentile(xs, 2), np.percentile(xs, 98)
    span = float(x_hi - x_lo) or 1.0
    node_t = np.clip((xs - x_lo) / span, 0, 1)
    node_colors = np.asarray([rainbow_rgb(float(t)) for t in node_t], dtype=np.uint8)

    global_to_local = -np.ones(int(ptr.shape[0] - 1), dtype=np.int32)
    global_to_local[cloud_index] = np.arange(n_cloud, dtype=np.int32)

    pre_list: list[np.ndarray] = []
    post_list: list[np.ndarray] = []
    for local, global_i in enumerate(cloud_index):
        start = int(ptr[global_i])
        end = int(ptr[global_i + 1])
        if end <= start:
            continue
        targets = post[start:end]
        local_post = global_to_local[targets]
        keep = local_post >= 0
        if not np.any(keep):
            continue
        kept = local_post[keep]
        pre_list.append(np.full(len(kept), local, dtype=np.int32))
        post_list.append(kept.astype(np.int32, copy=False))

    if not pre_list:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.uint8),
            node_colors,
        )

    pre_arr = np.concatenate(pre_list)
    post_arr = np.concatenate(post_list)
    keep = pre_arr != post_arr
    pre_arr, post_arr = pre_arr[keep], post_arr[keep]

    # Prefer mid-range + midline-crossing fibers so the center isn't empty.
    delta = xyz[pre_arr] - xyz[post_arr]
    dist = np.linalg.norm(delta, axis=1)
    mid_x = 0.5 * (xyz[pre_arr, 0] + xyz[post_arr, 0])
    crosses = (xyz[pre_arr, 0] * xyz[post_arr, 0]) < 0
    near_mid = np.abs(mid_x) < 0.35
    weights = np.clip(dist, 0.04, 0.9) ** 0.55
    weights = weights * (1.0 + 0.7 * crosses.astype(np.float32) + 0.45 * near_mid.astype(np.float32))

    rng = np.random.default_rng(seed)
    if len(pre_arr) > max_edges:
        probs = weights / weights.sum()
        pick = rng.choice(len(pre_arr), size=max_edges, replace=False, p=probs)
        pre_arr = pre_arr[pick]
        post_arr = post_arr[pick]

    pairs = np.stack([pre_arr, post_arr], axis=1).astype(np.int32)
    mid_t = 0.5 * (node_t[pre_arr] + node_t[post_arr])
    colors = np.asarray([rainbow_rgb(float(t)) for t in mid_t], dtype=np.uint8)
    return pairs, colors, node_colors
