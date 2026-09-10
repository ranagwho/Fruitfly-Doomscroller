# FlyScroll

A fruit-fly-inspired **doomscroller** built on the MaleCNS controller. The frut-fly watches short-form brainrot reels through modeled photoreceptors, stays on a clip until novelty-compartment MBON activity stays high, and scrolls when that interest habituates.

Based on the recent fruit fly connectome published by [Google](https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/).

## How it works

1. Frames are coarsely sampled into ommatidial columns and normalized.
2. Video is fed into the mapped R1–R8 receptors.
3. Compute novelty as prediction error across the analog channels. Compute Interest as is its slower tonic component.
5. Sustained Kenyon activity depresses existing KC→novelty-MBON synapses, with slow homeostasis. The feed advances after a sustained content-sensitive novelty drop or max-watch timeout.

## Setup

Python **3.11+** (3.12 works). Several GB of RAM for the full graph.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[video,dev]"
```

Connectome feathers should live in `connectome_data/malecns_v1/` (already present if you downloaded them). Then:

```sh
flyscroll import
flyscroll prepare --scale visual          # labelled crop for 16GB laptops (~70k cells)
# or:
flyscroll prepare --scale full            # intact retained CNS (~166k neurons; needs more RAM)
```

The visual crop omits many `ol_intrinsic` cells. FlyScroll does **not** inject a scalar frame-difference current into Kenyon cells or directly stimulate T4/T5/LC feature detectors. Prefer `--scale full` when you have the memory.

## Run

```sh
flyscroll serve --scale visual --port 8767
# open http://127.0.0.1:8767/
```

The spectator shows the reel beside a **neon MaleCNS wiring plate**: subsampled spatial edges colored by lateral rainbow, with phasic spike sparks on somata (drag to orbit, scroll to zoom, double-click to reset). Anatomy loads once from `/anatomy`; glow updates on `/state`.

Drop your own MP4/WebM files in `reels/` and pass `--reels reels`, or use the built-in procedural brainrot patterns.

### Live YouTube Shorts (no login required)

Playwright opens public Shorts in headed Chromium. **ScreenCaptureKit** streams the selected window into the fly; Chromium compositor screenshots are used as a fallback when macOS returns blank window frames.

```sh
pip install -e ".[shorts]"
PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium
# Grant Screen Recording to the app that runs flyscroll:
#   System Settings → Privacy & Security → Screen & System Audio Recording
# If you use Cursor's integrated terminal, enable **Cursor** (Terminal.app alone is not enough).
# Then fully quit/reopen that app.
flyscroll serve --scale visual --shorts
```

Headless smoke:

```sh
flyscroll demo --scale visual --steps 40
```

## Layout

| Path | Role |
| --- | --- |
| `flyscroll/` | Import, preparation, simulation, visual decoder, Shorts integration, and spectator |
| `connectome_data/malecns_v1/` | MaleCNS v1.0 source feathers |
| `outputs/flyscroll/` | Prepared `graph.npz` + manifests |

## License / attribution

FlyScroll code is licensed under the [MIT License](LICENSE). MaleCNS data is
**CC BY 4.0**; see [NOTICE](NOTICE) for attribution.
