# FlyScroll

A fruit-fly-inspired **doomscroller** built around the MaleCNS connectome.
Video passes through an analog compound-eye model and mapped photoreceptors;
population prediction error estimates novelty and advances the feed after
sustained boredom or media completion.

Based on the recent fruit fly connectome published by [Google](https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/).

## How it works

1. Frames are sampled into graded ommatidial columns with adaptation and local normalization.
2. ON/OFF, motion, looming, edge, flicker, chromatic, and small-object channels describe changes in the retinal signal.
3. Adapted input is injected only into mapped R1–R8 photoreceptors.
4. Population prediction error supplies content-sensitive novelty; interest is its slower tonic component.
5. The feed advances after sustained low novelty or after 98% of the current video's duration.

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

The visual crop omits many `ol_intrinsic` cells. FlyScroll does **not** inject a
scalar frame-difference current into Kenyon cells or directly stimulate
T4/T5/LC feature detectors. Adapted graded input is applied only to mapped
photoreceptors; analog feature channels and synaptic readout populations
jointly inform the decoder. Prefer `--scale full` when you have the memory.

The present behavioral decision is primarily an engineered population-novelty
decoder. Kenyon, novelty-MBON, and MDN activity can remain at zero, so this is
not evidence that the reconstructed mushroom-body pathway itself learned to
doomscroll.

## Run

```sh
flyscroll serve --scale visual --port 8767
# open http://127.0.0.1:8767/
```

The spectator shows the reel beside a **neon MaleCNS wiring plate**: subsampled spatial edges colored by lateral rainbow, with phasic spike sparks on somata (drag to orbit, scroll to zoom, double-click to reset). Anatomy loads once from `/anatomy`; glow updates on `/state`.

Drop your own MP4/WebM files in `reels/` and pass `--reels reels`, or use the built-in procedural brainrot patterns.

### Live YouTube Shorts (no login required)

Opt-in: Playwright opens public Shorts in headed Chromium. **ScreenCaptureKit**
streams the selected window into the fly; Chromium compositor screenshots are
used as a fallback when macOS returns blank window frames. Navigation tries a
wheel gesture, the Shorts next button, and keyboard input, then verifies that
the YouTube video ID changed.

```sh
pip install -e ".[shorts]"
PLAYWRIGHT_BROWSERS_PATH=0 playwright install chromium
# Grant Screen Recording to the app that runs flyscroll:
#   System Settings → Privacy & Security → Screen & System Audio Recording
# If you use Cursor's integrated terminal, enable **Cursor** (Terminal.app alone is not enough).
# Then fully quit/reopen that app.
flyscroll serve --scale visual --shorts
```

A headed Chromium window is the Shorts viewport. In Shorts mode the spectator
shows the ommatidial samples, analog feature channels, connectome activity,
full-session novelty/interest history, and per-reel watch statistics. Video is
unmuted when playback starts. Consent controls are handled when detected; if a
regional dialog remains, accept it once in Chromium. Watch percentages use the
video's media duration rather than a request timeout. The ScreenCaptureKit path
is macOS-only.

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
