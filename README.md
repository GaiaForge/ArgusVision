# ArgusVision

Live camera capture and LiZAD zero-shot anomaly detection for **production line assembly inspection**, built for Tesla Automation Prüm. PCB inspection (solder quality, discoloration, unknown surface defects) is the first concrete use case being tested, but the tool itself is general-purpose — the same capture/label/zero-shot-detect workflow applies to any assembly inspection task (missing components, wrong parts, incorrect placement) where you can't enumerate every possible defect in advance. Complements Mantis/Halcon's deterministic geometric checks (connector present, component seated, fiducial alignment) — the two are meant to run in series on the same inspection point, not replace each other.

## Hardware

- NVIDIA Jetson AGX Orin (JetPack 7.2 / L4T R39.2)
- Lucid Vision Labs TRIO64S-C (GigE Vision camera, via Arena SDK)

## Architecture

```
Lucid camera (GigE) --Arena SDK--> app.py (Gradio UI) --HTTP--> lizad_server.py --loads--> lizad_engine.py
                                                        (LiZADClient)          (FastAPI, port 8000)   (DINOv3 + MobileCLIP2 + checkpoint)
```

The LiZAD model runs in its own long-lived server process (`lizad_server.py`), kept separate from the UI process so the (slow-to-load) models don't need reloading every time the UI restarts. `app.py` itself has no torch/ML dependency — it talks to the inference server over plain HTTP via `lizad_client.py`.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Gradio UI — camera control, live preview, capture/labeling workflow, inspection tab |
| `lizad_engine.py` | Loads DINOv3 + MobileCLIP2 + the LiZAD checkpoint, runs inference. Used only by `lizad_server.py`. |
| `lizad_server.py` | FastAPI service wrapping `lizad_engine.py`, exposes `/infer` on port 8000. Run as its own process. |
| `lizad_client.py` | Lightweight HTTP client matching `LiZADEngine`'s `.run()` interface — no torch dependency. |

## UI tabs

- **Live Capture** — live camera feed, focus sharpness readout, capture button with quick-select labels (auto-discovers folders under `inspection_data/images/`), undo last capture, recent-captures gallery.
- **Camera Settings** — Exposure/Gain/White Balance, each with an Auto/Manual toggle (manual reveals a slider).
- **Inspection** — toggles live LiZAD inference on/off, shows the anomaly heatmap overlay, an adjustable anomaly threshold, and a pass/fail verdict.

Styled to Tesla's internal UI Design Standards (confluence.teslamotors.com/spaces/CONHUB/pages/4179687050) — light theme, `#3e6be2` primary blue, tab-based layout, large high-contrast controls.

## Adapting to a different assembly type

`lizad_server.py` currently loads the engine with `class_name="pcb"` — this feeds LiZAD's text-prompt templates (`"flawless {}"`, `"damaged {}"`, etc.), so it directly affects detection quality. When pointing this at a different assembly (wire harness, connector, whatever), change `class_name` to match what's actually in frame, and reconsider whether `trained_on_visa` or `trained_on_mvtec` is the better-matched checkpoint for that object type (see the checkpoint's original training categories before assuming).

## Setup on the Jetson

Both the camera SDK (`arena_api`) and LiZAD's stack (torch/transformers/open_clip_torch) need to be in the **same** conda environment for `lizad_server.py` to work (camera access happens in `app.py`, but both processes need to run from inside the `LiZAD` project's `conda activate LiZAD` environment for imports to resolve).

**Automated:** clone this repo on the Jetson, then run the setup script from inside it:

```bash
git clone https://github.com/GaiaForge/ArgusVision.git
cd ArgusVision
bash setup.sh
```

This copies `app.py`, `lizad_engine.py`, `lizad_server.py`, and `lizad_client.py` into `~/LiZAD/` (they need to live there specifically — `lizad_engine.py` imports `backbones` and `model` as local packages relative to that directory), activates the `LiZAD` conda environment, and installs `gradio`, `fastapi`, `uvicorn`, `requests`, and the Arena SDK Python wheel (auto-detected from `~/Downloads/ARENA_API-*.whl`).

**Manual equivalent**, if you'd rather do it by hand or the script hits something environment-specific:

```bash
cp app.py lizad_engine.py lizad_server.py lizad_client.py ~/LiZAD/
conda activate LiZAD
pip install gradio fastapi uvicorn requests
pip install <path-to-arena-api-wheel>.whl   # if not already installed in this env
```

## Running

Start the inference server first, as its own long-running process:

```bash
cd ~/LiZAD
conda activate LiZAD
python lizad_server.py
```

Then, in a separate terminal, start the UI:

```bash
cd ~/LiZAD
conda activate LiZAD
python app.py
```

Open `http://<jetson-ip>:7860` in a browser.

## Known unknowns / things to verify on first real run

This was built and UI-tested on a Windows machine without real hardware (camera and inference imports are wrapped defensively so the UI previews without either). Not yet run end-to-end on the Jetson. Specific things worth checking on first real deployment:

- The exact `ZSADModel`/`ImageEncoder`/`TextEncoder` call signatures in `lizad_engine.py` were reconstructed from reading LiZAD's source on GitHub, not verified against a real checkpoint execution — expect possible debugging here.
- Gradio's generator-based live-update pattern (`demo.load(...)` with a `yield`-ing function) for multiple simultaneous outputs.
- Theme CSS property names in `app.py` (`block_background_fill`, etc.) against whatever Gradio version ends up installed on the Jetson.
