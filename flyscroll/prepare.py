"""Compile retained edges, retinal projection, and doomscroll readouts.

Memory-conscious: streams edge batches, releases intermediates, and uses a
tight labelled crop for ``--scale visual`` (not the entire optic lobe).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from flyscroll import populations as pop
from flyscroll.anatomy import normalize_cloud, pick_display_indices, soma_xyz_for_ids
from flyscroll.transmitters import transmitter_signs

ROOT = Path(__file__).resolve().parents[1]
CONTACT_GAIN_MV = 0.275

# Named optic-lobe cell types kept in the visual crop (instead of all ~89k
# ol_intrinsic cells, which OOM a 16 GB laptop during CSR build).
OL_INTRINSIC_KEEP = (
    pop.LAMINA
    + pop.MOTION
    + pop.WIDEFIELD_HORIZONTAL
    + pop.WIDEFIELD_VERTICAL
    + pop.FEATURE_DETECTORS
    + (
        "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "TmY15",
        "C2", "C3", "T1", "T2", "T2a", "T3", "Tlp", "Lawf1", "Lawf2",
        "Dm1", "Dm2", "Dm3", "Dm4", "Dm8", "Dm9", "Pm1", "Pm2", "Pm3", "Pm4",
    )
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _type_mask(nodes, names: tuple[str, ...]) -> np.ndarray:
    return nodes.cell_type.isin(names).to_numpy()


def _kenyon_mask(nodes) -> np.ndarray:
    types = nodes.cell_type.fillna("").astype(str)
    return types.str.startswith(pop.KENYON_PREFIX).to_numpy()


def _visual_keep_mask(nodes) -> np.ndarray:
    sc = nodes.superclass.fillna("").astype(str)
    ct = nodes.cell_type.fillna("").astype(str)
    keep = sc.isin(
        (
            "ol_sensory",
            "visual_projection",
            "visual_centrifugal",
            "visual_projection_tbc",
        )
    )
    keep |= _type_mask(
        nodes,
        pop.RECEPTORS
        + pop.LAMINA
        + pop.NOVELTY_MBON
        + pop.NOVELTY_DAN
        + pop.REVERSE_DN
        + pop.STEERING_DN
        + pop.FORWARD_DN
        + pop.VISUAL_DN
        + pop.CONTEXT_MBON
        + pop.REWARD_DAN
        + pop.AVERSIVE_DAN
        + OL_INTRINSIC_KEEP,
    )
    keep |= _kenyon_mask(nodes)
    keep |= ct.str.startswith(("MBON", "PAM", "PPL", "MBON"))
    return np.asarray(keep, dtype=bool)


def _stream_edges(path: Path):
    reader = ipc.open_file(pa.memory_map(str(path), "r"))
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        yield (
            batch.column(0).to_numpy(),
            batch.column(1).to_numpy(),
            batch.column(2).to_numpy(),
        )


def _collect_cropped_edges(edge_path: Path, keep: np.ndarray, remap: np.ndarray):
    pre_chunks, post_chunks, count_chunks = [], [], []
    retained = 0
    for pre, post, count in _stream_edges(edge_path):
        mask = keep[pre] & keep[post]
        if not np.any(mask):
            continue
        pre_chunks.append(remap[pre[mask]])
        post_chunks.append(remap[post[mask]])
        count_chunks.append(count[mask])
        retained += int(mask.sum())
    if not pre_chunks:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
        )
    return (
        np.concatenate(pre_chunks).astype(np.uint32, copy=False),
        np.concatenate(post_chunks).astype(np.uint32, copy=False),
        np.concatenate(count_chunks).astype(np.uint32, copy=False),
    )


def _load_all_edges(edge_path: Path):
    edges = ipc.open_file(pa.memory_map(str(edge_path), "r")).read_all()
    return (
        edges.column("pre_index").to_numpy(),
        edges.column("post_index").to_numpy(),
        edges.column("synapse_count").to_numpy(),
    )


def _build_csr(pre, post, count, n_nodes, signs):
    order = np.argsort(pre, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=n_nodes))].astype(np.int64)
    post_csr = post[order].astype(np.int32, copy=False)
    weight = (count[order].astype(np.float32) * signs[pre[order]] * CONTACT_GAIN_MV).astype(
        np.float32
    )
    return ptr, post_csr, weight, order


def _build_retina(annotations, pre, post, count):
    """Vectorized R1-R6 column inference from contacts onto hex-annotated L1/L2/L3."""
    types = annotations.type.to_numpy()
    receptor = types == "R1-R6"
    hex1 = annotations.assignedOlHex1.to_numpy()
    hex2 = annotations.assignedOlHex2.to_numpy()
    anchors = np.isin(types, ["L1", "L2", "L3"]) & ~np.isnan(hex1.astype(float))
    selected = receptor[pre] & anchors[post]
    if not np.any(selected):
        empty = np.zeros(0, dtype=np.int32)
        return empty, np.zeros((0, 2), dtype=np.float32), np.zeros(0), np.zeros((0, 2)), int(receptor.sum())

    sel_pre = pre[selected].astype(np.int64)
    sel_post = post[selected]
    sel_w = count[selected].astype(np.int64)
    h1 = hex1[sel_post].astype(np.float64)
    h2 = hex2[sel_post].astype(np.float64)

    # Pack (pre, hex1, hex2) into a structured key and aggregate synapse counts.
    # Round hex coords to milli-units so float keys group stably.
    keys = np.stack(
        [sel_pre, np.round(h1 * 1000).astype(np.int64), np.round(h2 * 1000).astype(np.int64)],
        axis=1,
    )
    # Lexicographic sort then reduce consecutive duplicates.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    weights = sel_w[order]
    change = np.ones(len(keys), dtype=bool)
    change[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.flatnonzero(change)
    group_w = np.add.reduceat(weights, starts)
    group_keys = keys[starts]

    # For each receptor, pick the hex with max weight.
    receptor_ids = group_keys[:, 0]
    rec_change = np.ones(len(receptor_ids), dtype=bool)
    rec_change[1:] = receptor_ids[1:] != receptor_ids[:-1]
    rec_starts = np.flatnonzero(rec_change)
    # Within each receptor's slice, argmax weight.
    indices = []
    hexes = []
    confidence = []
    for s, e in zip(rec_starts, np.r_[rec_starts[1:], len(group_w)]):
        sl = slice(s, e)
        local = group_w[sl]
        best = int(np.argmax(local))
        total = int(local.sum())
        row = group_keys[s + best]
        indices.append(int(row[0]))
        hexes.append((row[1] / 1000.0, row[2] / 1000.0))
        confidence.append(local[best] / total if total else 0.0)

    indices_arr = np.asarray(indices, dtype=np.int32)
    hexes_arr = np.asarray(hexes, dtype=np.float64)
    xy = np.stack(
        [hexes_arr[:, 0] - 0.5 * hexes_arr[:, 1], np.sqrt(3) / 2 * hexes_arr[:, 1]],
        axis=1,
    )
    uv = np.empty_like(xy, dtype=np.float32)
    sides = annotations.rootSide.to_numpy()[indices_arr]
    for side in ("L", "R"):
        mask = sides == side
        if not np.any(mask):
            continue
        z = xy[mask]
        span = z.max(axis=0) - z.min(axis=0)
        span[span == 0] = 1.0
        z = (z - z.min(axis=0)) / span
        uv[mask, 0] = 0.60 * z[:, 0] if side == "L" else 0.40 + 0.60 * (1 - z[:, 0])
        uv[mask, 1] = 1 - z[:, 1]
    return indices_arr, uv, np.asarray(confidence), hexes_arr, int(receptor.sum())


def _build_chromatic(annotations, indices_luma, hexes, luma_uv):
    if len(indices_luma) == 0:
        empty = np.zeros(0, dtype=np.int32)
        return {
            "chromatic": empty,
            "chromatic_uv": np.zeros((0, 2), dtype=np.float32),
            "chromatic_channel": np.zeros(0, dtype="U16"),
        }
    hex_to_uv = {(float(h[0]), float(h[1])): luma_uv[i] for i, h in enumerate(hexes)}
    hex_keys = (
        np.asarray(list(hex_to_uv.keys()), dtype=np.float64)
        if hex_to_uv
        else np.zeros((0, 2))
    )
    sides_luma = annotations.rootSide.to_numpy()[indices_luma]
    chromatic_types = set(pop.CHROMATIC_RECEPTORS)
    types = annotations.type.to_numpy()
    sides = annotations.rootSide.to_numpy()
    h1 = annotations.assignedOlHex1.to_numpy()
    h2 = annotations.assignedOlHex2.to_numpy()
    idxs, uvs, channels = [], [], []
    for i, ctype in enumerate(types):
        if ctype not in chromatic_types:
            continue
        hh1, hh2 = h1[i], h2[i]
        if hh1 == hh1 and hh2 == hh2 and len(hex_keys):
            key = (float(hh1), float(hh2))
            if key in hex_to_uv:
                uv = hex_to_uv[key]
            else:
                d = np.sum((hex_keys - np.asarray(key)) ** 2, axis=1)
                nearest = tuple(float(x) for x in hex_keys[int(np.argmin(d))])
                uv = hex_to_uv[nearest]
        else:
            side_mask = sides_luma == sides[i]
            uv = (
                luma_uv[side_mask].mean(axis=0)
                if np.any(side_mask)
                else np.asarray([0.5, 0.5], dtype=np.float32)
            )
        idxs.append(i)
        uvs.append(np.asarray(uv, dtype=np.float32))
        channels.append(pop.RECEPTOR_CHANNEL.get(str(ctype), "green"))
    return {
        "chromatic": np.asarray(idxs, dtype=np.int32),
        "chromatic_uv": np.asarray(uvs, dtype=np.float32).reshape(-1, 2),
        "chromatic_channel": np.asarray(channels, dtype="U16"),
    }


def _plastic_edges(pre_csr, post_csr, nodes):
    kc = _kenyon_mask(nodes)
    mbon = _type_mask(nodes, pop.NOVELTY_MBON)
    return np.flatnonzero(kc[pre_csr] & mbon[post_csr]).astype(np.int32)


def _readouts(nodes, annotations) -> list[dict]:
    wanted = set(pop.all_named_types())
    rows = []
    types = nodes.cell_type.fillna("").astype(str)
    for i, ctype in enumerate(types):
        if ctype not in wanted:
            continue
        side = str(annotations.somaSide.iloc[i]) if "somaSide" in annotations else "?"
        rows.append(
            {
                "index": int(i),
                "id": str(nodes.source_id.iloc[i]),
                "type": ctype,
                "side": side,
            }
        )
    kc_inds = np.flatnonzero(_kenyon_mask(nodes))
    if len(kc_inds):
        n_sample = min(32, len(kc_inds))
        sample = kc_inds[np.linspace(0, len(kc_inds) - 1, n_sample, dtype=int)]
        for i in sample:
            rows.append(
                {
                    "index": int(i),
                    "id": str(nodes.source_id.iloc[i]),
                    "type": str(nodes.cell_type.iloc[i]),
                    "side": str(annotations.somaSide.iloc[i]),
                    "group": "kenyon_sample",
                }
            )
    return rows


def prepare(dataset: str = "malecns_v1", scale: str = "full") -> dict:
    if scale not in ("full", "visual"):
        raise ValueError("scale must be 'full' or 'visual'")
    if dataset != "malecns_v1":
        raise ValueError("Female retinal projection is unresolved; do not copy the male map.")

    root = ROOT / "connectome_data" / dataset
    edge_path = root / "normalized/edges.arrow"
    nodes = feather.read_table(root / "normalized/neurons.feather").to_pandas()
    report = json.loads((root / "normalized/report.json").read_text())
    _log(f"loaded {len(nodes)} neurons")

    if scale == "visual":
        keep = _visual_keep_mask(nodes)
        kept_idx = np.flatnonzero(keep)
        remap = -np.ones(len(nodes), dtype=np.int32)
        remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)
        _log(f"visual crop: keeping {len(kept_idx)} / {len(nodes)} neurons; streaming edges…")
        pre, post, count = _collect_cropped_edges(edge_path, keep, remap)
        nodes = nodes.iloc[kept_idx].reset_index(drop=True)
        _log(f"retained {len(pre)} edges in crop")
    else:
        _log("loading full edge list (needs ~2–4 GB free RAM)…")
        pre, post, count = _load_all_edges(edge_path)
        _log(f"full graph edges: {len(pre)}")

    signs, uncertain = transmitter_signs(nodes.neurotransmitter)
    _log("building CSR…")
    ptr, post_csr, weight, order = _build_csr(pre, post, count, len(nodes), signs)
    pre_csr = pre[order].astype(np.int32, copy=False)
    del order

    annotations = (
        feather.read_table(root / "annotations.feather")
        .to_pandas()
        .set_index("bodyId")
        .loc[nodes.source_id]
        .reset_index(drop=True)
    )
    _log("inferring retina…")
    indices, uv, confidence, hexes, retina_total = _build_retina(annotations, pre, post, count)
    chromatic = _build_chromatic(annotations, indices, hexes, uv)
    plastic = _plastic_edges(pre_csr, post_csr, nodes)
    readouts = _readouts(nodes, annotations)

    group_index = {name: [] for name in pop.READOUT_GROUPS}
    for r in readouts:
        for name, types in pop.READOUT_GROUPS.items():
            if r["type"] in types:
                group_index[name].append(r["index"])

    manifest = {
        "dataset": dataset,
        "scale": scale,
        "neurons": len(nodes),
        "edges": int(ptr[-1]),
        "synaptic_contacts": int(np.asarray(count, dtype=np.uint64).sum()),
        "retina_total": retina_total,
        "retina_mapped": int(len(indices)),
        "retina_unmapped": int(retina_total - len(indices)),
        "chromatic_mapped": int(len(chromatic["chromatic"])),
        "plastic_kc_to_novelty_mbon": int(len(plastic)),
        "projection_confidence_median": float(np.median(confidence)) if len(confidence) else None,
        "uncertain_sign_neurons": int(uncertain.sum()),
        "readouts": readouts,
        "readout_groups": {k: v for k, v in group_index.items()},
        "source_hashes": report["source_hashes"],
        "retina_model": (
            "R1-R6 luminance plus R7/R8 chromatic proxies. Column inferred from "
            "contacts onto annotated L1/L2/L3; overlapping viewport projection."
        ),
        "visual_dynamics": (
            "LIF proxy for graded photoreceptors/lamina; tonic lamina current."
        ),
        "interest_interface": (
            "Interest = EMA of novelty MBON (MBON16/17/28) rate; scroll when "
            "interest falls below threshold after min watch (MDN optional gate)."
        ),
        "habituation": (
            f"Anti-Hebbian depression on {len(plastic)} existing KC→novelty-MBON edges."
        ),
        "cropped": scale == "visual",
        "crop_note": (
            "Labelled sensory/projection/MB/DN crop; excludes bulk ol_intrinsic."
            if scale == "visual"
            else None
        ),
    }

    out = ROOT / "outputs" / "flyscroll" / dataset / scale
    out.mkdir(parents=True, exist_ok=True)
    _log("resolving soma coordinates…")
    raw_xyz, soma_mask = soma_xyz_for_ids(
        nodes.source_id.to_numpy(dtype=np.int64),
        root / "annotations.feather",
    )
    xyz = normalize_cloud(raw_xyz, soma_mask)
    display = pick_display_indices(
        soma_mask,
        nodes.superclass.fillna("unassigned").to_numpy(),
        12000,
        must_include=np.unique(
            np.concatenate(
                [
                    indices,
                    chromatic["chromatic"],
                    np.flatnonzero(_type_mask(nodes, pop.LAMINA)).astype(np.int32),
                ]
            )
        ),
        cell_type=nodes.cell_type.fillna("unknown").to_numpy(),
        xyz=xyz,
    )
    manifest["soma_located"] = int(soma_mask.sum())
    manifest["cloud_points"] = int(len(display))
    _log(f"writing {out / 'graph.npz'}…")
    np.savez_compressed(
        out / "graph.npz",
        ptr=ptr,
        post=post_csr,
        weight=weight,
        ids=nodes.source_id.to_numpy(dtype=np.int64),
        retina=indices,
        uv=uv.astype(np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        hexes=np.asarray(hexes, dtype=np.float32),
        lamina=np.flatnonzero(_type_mask(nodes, pop.LAMINA)).astype(np.int32),
        chromatic=chromatic["chromatic"],
        chromatic_uv=chromatic["chromatic_uv"],
        chromatic_channel=chromatic["chromatic_channel"],
        plastic_edges=plastic,
        novelty_mbon=np.flatnonzero(_type_mask(nodes, pop.NOVELTY_MBON)).astype(np.int32),
        novelty_dan=np.flatnonzero(_type_mask(nodes, pop.NOVELTY_DAN)).astype(np.int32),
        reverse_dn=np.flatnonzero(_type_mask(nodes, pop.REVERSE_DN)).astype(np.int32),
        kenyon=np.flatnonzero(_kenyon_mask(nodes)).astype(np.int32),
        superclass=np.asarray(nodes.superclass.fillna("unassigned").astype(str), dtype="U64"),
        cell_type=np.asarray(nodes.cell_type.fillna("unknown").astype(str), dtype="U64"),
        xyz=xyz,
        soma_mask=soma_mask.astype(np.uint8),
        cloud_index=display,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        k: v
        for k, v in manifest.items()
        if k not in ("source_hashes", "readouts", "readout_groups")
    }
    print(json.dumps(summary))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="malecns_v1")
    parser.add_argument("--scale", choices=["full", "visual"], default="visual")
    args = parser.parse_args(argv)
    prepare(args.dataset, scale=args.scale)


if __name__ == "__main__":
    main()
