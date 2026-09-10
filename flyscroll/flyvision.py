"""Fly-inspired graded visual front end and population novelty.

The connectome supplies wiring, but not the analog computations performed before
spiking. This module adds explicit, declared physiology-inspired stages:

* ommatidial sampling and luminance adaptation
* local divisive contrast normalization
* parallel ON/OFF transients
* delayed-neighbour motion (four cardinal directions)
* looming, small-object, edge, flicker, and chromatic channels

No fixed "new reel" pulse is used. Novelty is prediction error in the resulting
multidimensional population response.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from flyscroll.retina import _linearize, _sample, chromatic_samples


_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


@dataclass
class FlyEyeFrame:
    luminance: np.ndarray
    chromatic: np.ndarray
    features: dict[str, float]
    feature_vector: np.ndarray
    column_rgb: np.ndarray
    column_activity: np.ndarray


class FlyEyeProcessor:
    """Stateful analog front end on the inferred ommatidial lattice."""

    FEATURE_NAMES = (
        "on",
        "off",
        "flicker",
        "motion_left",
        "motion_right",
        "motion_up",
        "motion_down",
        "expansion",
        "contraction",
        "small_object",
        "bar_edge",
        "chromatic",
        "local_contrast",
    )

    def __init__(self, brain):
        hexes = np.asarray(brain.hexes, dtype=np.float32)
        unique_hex, first, inverse = np.unique(
            hexes, axis=0, return_index=True, return_inverse=True
        )
        self.hexes = unique_hex
        self.receptor_to_column = inverse.astype(np.int32)
        self.n_columns = len(unique_hex)

        uv = np.asarray(brain.uv, dtype=np.float32)
        sums = np.zeros((self.n_columns, 2), dtype=np.float32)
        counts = np.bincount(inverse, minlength=self.n_columns).astype(np.float32)
        np.add.at(sums, inverse, uv)
        self.uv = sums / np.maximum(counts[:, None], 1.0)

        k = min(7, self.n_columns)
        _, neighbours = cKDTree(self.uv).query(self.uv, k=k)
        if neighbours.ndim == 1:
            neighbours = neighbours[:, None]
        # First result is the point itself.
        self.neighbours = neighbours[:, 1:].astype(np.int32)
        delta = self.uv[self.neighbours] - self.uv[:, None, :]
        norm = np.linalg.norm(delta, axis=2, keepdims=True)
        self.neighbour_dir = delta / np.maximum(norm, 1e-6)

        center = np.mean(self.uv, axis=0)
        radial = self.uv - center
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
        self.radial = radial.astype(np.float32)

        self.slow_luminance: np.ndarray | None = None
        self.fast_contrast: np.ndarray | None = None
        self.prev_luminance: np.ndarray | None = None
        self.prev_signal: np.ndarray | None = None
        self.motion_baseline = np.zeros(4, dtype=np.float32)
        self.channel_maps: dict[str, np.ndarray] = {}

    def _motion_channels(
        self, signal: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        if self.prev_signal is None or self.neighbours.shape[1] == 0:
            pair = np.zeros_like(self.neighbour_dir[:, :, 0])
        else:
            now_neighbour = signal[self.neighbours]
            prev_neighbour = self.prev_signal[self.neighbours]
            pair = (
                self.prev_signal[:, None] * now_neighbour
                - signal[:, None] * prev_neighbour
            )

        dx = self.neighbour_dir[:, :, 0]
        # Image Y points down.
        dy = self.neighbour_dir[:, :, 1]
        horizontal_map = np.mean(pair * dx, axis=1)
        vertical_map = np.mean(pair * dy, axis=1)
        horizontal = float(np.mean(horizontal_map))
        vertical = float(np.mean(vertical_map))
        magnitude = np.mean(np.abs(pair), axis=1)

        # Radial alignment approximates expansion/contraction around visual centre.
        radial_alignment = (
            self.neighbour_dir[:, :, 0] * self.radial[:, None, 0]
            + self.neighbour_dir[:, :, 1] * self.radial[:, None, 1]
        )
        radial_map = np.mean(pair * radial_alignment, axis=1)
        radial_flow = float(np.mean(radial_map))

        scale = 28.0
        channels = {
            "motion_left": max(0.0, -horizontal * scale),
            "motion_right": max(0.0, horizontal * scale),
            "motion_up": max(0.0, -vertical * scale),
            "motion_down": max(0.0, vertical * scale),
            "expansion": max(0.0, radial_flow * scale),
            "contraction": max(0.0, -radial_flow * scale),
        }
        maps = {
            "motion_left": np.maximum(-horizontal_map, 0.0),
            "motion_right": np.maximum(horizontal_map, 0.0),
            "motion_up": np.maximum(-vertical_map, 0.0),
            "motion_down": np.maximum(vertical_map, 0.0),
            "expansion": np.maximum(radial_map, 0.0),
        }
        return magnitude.astype(np.float32), channels, maps

    def process(self, rgb: np.ndarray, brain, dt_seconds: float = 0.05) -> FlyEyeFrame:
        rgb_lin = _linearize(rgb)
        column_rgb_lin = _sample(rgb_lin, self.uv)
        luminance = (column_rgb_lin @ _LUMA).astype(np.float32)

        if self.slow_luminance is None:
            self.slow_luminance = luminance.copy()
            self.fast_contrast = np.zeros_like(luminance)
            self.prev_luminance = luminance.copy()
            self.prev_signal = np.zeros_like(luminance)

        # Photoreceptor light adaptation: slow local operating point.
        a_slow = 1.0 - math.exp(-dt_seconds / 0.65)
        self.slow_luminance += a_slow * (luminance - self.slow_luminance)
        adapted = (luminance - self.slow_luminance) / (
            0.08 + self.slow_luminance
        )

        # Lamina/medulla-like local divisive normalization.
        if self.neighbours.shape[1]:
            surround = adapted[self.neighbours]
            local_mean = surround.mean(axis=1)
            local_scale = np.sqrt(
                np.mean((surround - local_mean[:, None]) ** 2, axis=1)
            )
        else:
            local_mean = np.zeros_like(adapted)
            local_scale = np.full_like(adapted, 0.1)
        normalized = np.clip(
            (adapted - local_mean) / (0.12 + local_scale), -3.0, 3.0
        )

        a_fast = 1.0 - math.exp(-dt_seconds / 0.045)
        self.fast_contrast += a_fast * (normalized - self.fast_contrast)
        assert self.prev_luminance is not None
        transient = np.clip(
            (luminance - self.prev_luminance) / (0.08 + self.slow_luminance),
            -3.0,
            3.0,
        )
        on = np.maximum(transient, 0.0)
        off = np.maximum(-transient, 0.0)
        signal = on - off

        motion_map, motion, motion_channel_maps = self._motion_channels(signal)
        spatial_edge = (
            np.mean(np.abs(normalized[:, None] - normalized[self.neighbours]), axis=1)
            if self.neighbours.shape[1]
            else np.abs(normalized)
        )
        widefield = sum(motion.values())
        local_transient = on + off
        small_map = np.maximum(
            local_transient - np.mean(local_transient) - 0.45 * motion_map, 0.0
        )
        self.channel_maps = {
            **motion_channel_maps,
            "small_object": small_map.astype(np.float32),
            "bar_edge": spatial_edge.astype(np.float32),
        }

        opponent_rg = column_rgb_lin[:, 0] - column_rgb_lin[:, 1]
        opponent_by = column_rgb_lin[:, 2] - 0.5 * (
            column_rgb_lin[:, 0] + column_rgb_lin[:, 1]
        )
        chromatic_contrast = float(np.std(opponent_rg) + np.std(opponent_by))

        features = {
            "on": float(np.mean(on)),
            "off": float(np.mean(off)),
            "flicker": float(np.mean(local_transient)),
            **motion,
            "small_object": float(np.mean(small_map)),
            "bar_edge": float(np.mean(spatial_edge)),
            "chromatic": chromatic_contrast,
            "local_contrast": float(np.mean(np.abs(normalized))),
        }

        # Graded receptor output centred around the adapted local contrast.
        receptor_signal = np.clip(
            0.32 + 0.18 * normalized + 0.38 * signal, 0.0, 1.0
        )
        receptor_luminance = receptor_signal[self.receptor_to_column].astype(
            np.float32
        )
        chromatic = chromatic_samples(
            rgb, brain.chromatic_uv, brain.chromatic_channel
        )

        self.prev_signal = signal.copy()
        self.prev_luminance = luminance.copy()
        vector = np.asarray(
            [features[name] for name in self.FEATURE_NAMES], dtype=np.float32
        )
        column_rgb = np.rint(np.clip(column_rgb_lin, 0, 1) * 255).astype(np.uint8)
        activity = np.rint(
            np.clip(
                0.5 * local_transient
                + 0.35 * motion_map
                + 0.3 * spatial_edge,
                0,
                1,
            )
            * 255
        ).astype(np.uint8)

        return FlyEyeFrame(
            luminance=receptor_luminance,
            chromatic=chromatic,
            features=features,
            feature_vector=vector,
            column_rgb=column_rgb,
            column_activity=activity,
        )

    def apply_feature_drive(self, brain, frame: FlyEyeFrame) -> dict:
        """Describe the front end; downstream populations remain synaptic readouts."""
        return {
            "proxy": "fly_eye_analog_frontend",
            "features": {k: round(v, 5) for k, v in frame.features.items()},
            "direct_kc_drive": False,
            "feature_driven_cells": 0,
            "drive_policy": "mapped_photoreceptors_only",
        }

    def anatomy_payload(self) -> dict:
        return {
            "eye_columns": self.n_columns,
            "eye_uv_b64": _b64(self.uv.astype(np.float32)),
            "eye_hex_b64": _b64(self.hexes.astype(np.float32)),
        }

    @staticmethod
    def frame_payload(frame: FlyEyeFrame) -> dict:
        return {
            "columns": int(len(frame.column_activity)),
            "rgb_b64": _b64(frame.column_rgb),
            "activity_b64": _b64(frame.column_activity),
            "features": {k: round(v, 5) for k, v in frame.features.items()},
        }


class PopulationNovelty:
    """Multiscale prediction error over fly-eye and connectome populations."""

    def __init__(self, n_features: int, tau_predict: float = 0.55, tau_scale: float = 5.0):
        self.n_features = n_features
        self.tau_predict = tau_predict
        self.tau_scale = tau_scale
        self.predictor: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.fast_score = 0.0
        self.tonic_score = 0.0
        self.samples = 0

    def update(self, vector: np.ndarray, dt_seconds: float) -> dict:
        z = np.asarray(vector, dtype=np.float32)
        if self.predictor is None or len(self.predictor) != len(z):
            self.predictor = z.copy()
            self.scale = np.maximum(np.abs(z) * 0.1, 0.03).astype(np.float32)
            residual = np.zeros_like(z)
        else:
            residual = z - self.predictor

        assert self.scale is not None
        normalized = np.abs(residual) / (0.04 + self.scale)
        # Top-quartile pooling: rare strong pathway changes matter without one scalar
        # pixel statistic dominating all content.
        take = max(1, len(normalized) // 4)
        salient = np.partition(normalized, -take)[-take:]
        raw = float(np.clip(np.mean(salient) * 28.0, 0.0, 180.0))

        a_fast = 1.0 - math.exp(-dt_seconds / 0.12)
        a_tonic = 1.0 - math.exp(-dt_seconds / 0.9)
        self.fast_score += a_fast * (raw - self.fast_score)
        self.tonic_score += a_tonic * (self.fast_score - self.tonic_score)

        a_pred = 1.0 - math.exp(-dt_seconds / self.tau_predict)
        a_scale = 1.0 - math.exp(-dt_seconds / self.tau_scale)
        self.predictor += a_pred * residual
        self.scale += a_scale * (np.abs(residual) - self.scale)
        self.samples += 1

        return {
            "population_novelty": round(self.fast_score, 3),
            "population_interest": round(self.tonic_score, 3),
            "raw_prediction_error": round(raw, 3),
            "active_dimensions": int(np.count_nonzero(normalized > 1.0)),
            "dimensions": int(len(z)),
        }
