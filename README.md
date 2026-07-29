# ArgusVision

**A vision inspection comparison platform for production line assembly inspection.**

Capture one labeled dataset from a real part, then evaluate it against *multiple* detection approaches — zero-shot anomaly detection, reference-image anomaly detection, and deterministic classical CV — to find out empirically which one actually fits that specific process. Different defect types suit fundamentally different detection strategies, and the only reliable way to find out which is to test them side by side on real images from the real line.

Originally built as a LiZAD-only inspection UI; it grew into a comparison platform once testing showed that no single detector handles every defect class well.

## Which approach for which process?

This is the core question the tool exists to answer. Current guidance, based on measurements taken on this hardware:

| Approach | Answers | Needs upfront | Best for | Measured result |
|---|---|---|---|---|
| **LiZAD** (zero-shot) | "Does this look different from a generic *flawless* part?" | Nothing | Open-ended **surface** defects too varied to enumerate — scratches, cracks, discoloration | Weak on missing components (~0.25 vs ~0.0 — a real but fragile margin). Not yet tested on the surface defects it's actually designed for. |
| **PatchCore** (reference-image) | "Does this look different from *my own* known-good parts?" | ~15+ captured normal images | Presence/absence and structural defects where you have good examples but don't want a hand-coded rule | Clean separation on 7 missing MOSFETs: normal `0.000` vs defect `0.521` |
| **EfficientAD** (reference-image) | "Is anything *logically* wrong — missing, misplaced, wrong quantity?" | More normal images than PatchCore (it trains a network, not a memory bank) | **Logical** anomalies specifically — the class where LiZAD proved weak | Only run on handheld data so far, where nothing separated. Effectively unmeasured. |
| **Channel Check** (classical CV) | "Is the expected color present at this exact position?" | A one-time calibration | Fully-specifiable checks at **fixed** positions — "channel 3 must hold the blue wire" | Fastest, fully explainable, names *which* region failed |
| **YOLO** (supervised) | "Which known object classes are present, and where?" | A labeled dataset with bounding boxes + retraining per change | Enumerable object types worth the labeling investment | Not yet tested in this project |

The pattern that keeps holding: **match the tool to how specifiable the question is.** If you can state the rule exactly, a deterministic check beats an anomaly model — faster, explainable, and it can name the specific failure. Reserve anomaly detection for the genuine long tail you *can't* specify in advance.

**See [FINDINGS.md](FINDINGS.md)** for what has actually been measured, the reasoning behind each decision, and the open questions. This README covers how to *use* the tool; FINDINGS covers what it has *learned*.

Intended to complement — not replace — a dedicated deterministic machine-vision runtime doing geometric checks (connector present, component seated, fiducial alignment). The two are meant to run in series on the same inspection point.

## Hardware

- NVIDIA Jetson AGX Orin (JetPack 7.2 / L4T R39.2, compute capability `sm_87`)
- Lucid Vision Labs TRIO64S-C (GigE Vision camera, via Arena SDK)

## Architecture

```
                                        ┌── HTTP ──> lizad_server.py       (8000) ──> DINOv3 + MobileCLIP2 + checkpoint
Lucid camera --Arena SDK--> app.py ─────┼── HTTP ──> patchcore_server.py   (8001) ──> anomalib PatchCore memory bank
                            (Gradio UI) ├── HTTP ──> efficientad_server.py (8002) ──> anomalib EfficientAD network
                                        └── in-process ──> Channel Check (OpenCV, no model)
```

Each model runs in its own long-lived server process, kept separate from the UI so the slow-to-load models don't reload every time the UI restarts, and so one crashing doesn't take down the others. `app.py` itself has **no torch/ML dependency** — it talks to the inference servers over plain HTTP.

No server is required for Live Capture or Channel Check. Start only the detector(s) you actually want to compare.

PatchCore and EfficientAD share almost all of their implementation (`detector_server.py`, `detector_client.py`, `detector_data.py`, `experiment.py`, `train_service.py`) and differ only in model class, port, and a couple of fit parameters. That's deliberate: one live path to debug instead of two, and it guarantees the two detectors are compared on genuinely equal terms rather than through subtly different plumbing.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Gradio UI — camera control, capture/labeling, and one tab per detection approach |
| `lizad_engine.py` | Loads DINOv3 + MobileCLIP2 + LiZAD checkpoint. Used only by `lizad_server.py`. |
| `lizad_server.py` | FastAPI service wrapping LiZAD, `/infer` + `/infer_multi` on port 8000 |
| `lizad_client.py` | Lightweight HTTP client for LiZAD — no torch dependency |
| `detector_server.py` | **Shared** FastAPI implementation for both trainable detectors — `/infer`, `/health`, `/train`, `/train_status` |
| `detector_client.py` | **Shared** HTTP client base for both trainable detectors |
| `detector_data.py` | **Shared** capture-directory layout and anomalib datamodule construction |
| `train_service.py` | **Shared** background training job + checkpoint metadata sidecar |
| `experiment.py` | **Shared** offline evaluation runner — guarantees both detectors report in the same format |
| `patchcore_server.py` / `_client.py` / `_experiment.py` | PatchCore config on top of the shared modules (port 8001) |
| `efficientad_server.py` / `_client.py` / `_experiment.py` | EfficientAD config on top of the shared modules (port 8002) |
| `full_setup.sh` | Canonical conda environment rebuild (**must be sourced, not executed**) |
| `install_anomalib.sh` | Installs anomalib without disturbing the Jetson-specific torch build |
| `create_launchers.sh` | Creates self-updating desktop launcher icons |
| `setup.sh` | Copies app files into `~/LiZAD/` and installs UI dependencies |

## UI tabs

- **Live Capture** — live feed, focus sharpness, and the dataset-building workflow. Hierarchical labeling: pick a **Category** (`normal` / `defect`), then a **Subtype** for defects (`missing_component`, `wrong_placement`, `wrong_color`, or type a new one). Click thumbnails to multi-select for deletion, or **Open Image Folder** for bulk work in the file manager.
- **Comparison** — runs every reachable detector over the same dataset, ranks by gap, and records the run with its setup conditions. Exports a Markdown report. See "The Comparison tab" below.
- **LiZAD (Zero-Shot)** — live inference, anomaly heatmap with bounding boxes around each flagged region, threshold slider with a "suggest from captured images" helper, and a prompt-set comparison tool.
- **PatchCore (Reference-Based)** — live scoring against a memory bank built from your captured `normal` images, plus a **Train** button.
- **EfficientAD (Logical Anomalies)** — live scoring from a network trained on your `normal` images, plus a **Train** button with live progress.
- **Channel Check (Classical CV)** — click-to-calibrate colored regions, then live per-region pass/fail. No model or GPU involved.
- **Camera Settings** — Exposure/Gain/White Balance with Auto/Manual toggles, live sharpness + brightness readouts, and a **Lock Camera Settings** switch.

Each detector tab opens with a plain-language explanation of what that approach is good at, what it's weak at, and what it needs — so the tool teaches the selection logic rather than assuming the operator already knows it. Tabs with nothing to train (LiZAD is zero-shot; Channel Check is calibrated, not learned) say so explicitly, so a missing Train button never reads as a missing feature.

## Training, from the browser

Both trainable detectors are trained from their tab — no SSH, no terminal:

1. Capture good parts in **Live Capture** under category `normal`.
2. Open the detector's tab and click **Train**. PatchCore takes seconds; EfficientAD takes minutes and shows epoch progress. Training runs server-side, so you can navigate away.
3. The **Model Status** line always says what the running model was actually trained on:

   > Trained on 18 normal images at 14:32.   12 new images captured since — retrain recommended.

That staleness warning is the point: you can tell at a glance whether the model you're testing reflects the data you've collected. Checkpoints persist to `checkpoints/<detector>/`, so restarting a server reloads the trained model rather than retraining or starting empty.

Styled to an internal UI design standard for operator-facing tooling — light theme, `#3e6be2` primary blue, large high-contrast controls.

## The Comparison tab

This is what the platform is for. One button scores the same captured dataset with every detector whose server is running, ranks them by **gap** (worst-normal to best-defect distance), and writes the run to `comparison_history.json`.

Before running, fill in **Setup conditions** — part, defect class, mounting, working distance, lighting. Those persist between sessions, and exposure and gain are read from the camera automatically. This is not bureaucracy: a gap number without its conditions can't be acted on, and a handheld desk measurement can otherwise be quoted months later as though it were controlled.

**Export Report** writes Markdown to `reports/` containing the conditions, the ranked results, threshold guidance, and **caveats derived from the run itself** — not fixtured, thin evaluation split, marginal gap, partial comparison. The limitations travel with the numbers.

Verdicts are graded, because a positive gap can still be worthless: `separates` (gap ≥ 0.10), `marginal`, or `OVERLAP`. Threshold advice is withheld below the marginal band rather than stated confidently and then contradicted by the caveats beneath it.

**Delete Last Run** drops a botched run. **Clear History** archives to `reports/` rather than deleting — the table is meant to be an auditable record.

## Calibrate before you compare

Two hard-won lessons from real testing — skip these and your scores will be noise:

1. **Lock camera settings.** With Auto Exposure/Gain running, the camera re-converges every time the scene changes, so consecutive frames of the *same* board score differently. An early missing-component reading drifted between 0.25 and 0.007 purely from exposure hunting. Let Auto settle, then lock it.
2. **Mount the camera rigidly.** A hand-held or loosely-mounted camera introduces enough frame-to-frame variation to flip a marginal verdict back and forth on its own.

Then capture a proper dataset — 15-30 normal and several defect images, varying position slightly the way the part will actually appear — before trusting any threshold. Single-sample comparisons are unreliable: PatchCore initially showed a dramatic "clean separation" on one image pair that vanished entirely once more images were evaluated.

## Adapting to a different assembly type

`lizad_server.py` loads the engine with `class_name="pcb"`, which feeds LiZAD's text-prompt templates and directly affects detection quality. Point it at a different assembly by changing `class_name` to match what's in frame, and reconsider whether `trained_on_visa` or `trained_on_mvtec` suits that object type. Override the checkpoint without editing code:

```bash
LIZAD_CHECKPOINT=checkpoints/trained_on_mvtec/model.pth python lizad_server.py
```

`lizad_engine.py` also defines targeted per-failure-mode prompt sets (`missing_component`, `wrong_placement`, `wrong_color`) alongside the generic pair; the **Compare Prompt Sets** button scores your captured images against all of them and reports which separates best.

## Setup on the Jetson

The camera SDK (`arena_api`) and the model stacks must live in the **same** conda environment.

```bash
git clone https://github.com/GaiaForge/ArgusVision.git
cd ArgusVision
bash setup.sh                # copies files into ~/LiZAD/, installs UI dependencies
bash install_anomalib.sh     # optional — needed for the PatchCore and EfficientAD tabs
bash create_launchers.sh     # creates the desktop launcher icons
```

Files must live in `~/LiZAD/` specifically — `lizad_engine.py` imports `backbones` and `model` as local packages relative to that directory.

If the conda environment itself needs rebuilding (new Jetson, corrupted env, JetPack upgrade), **start from `full_setup.sh`** rather than reinventing the dependency list — getting the right Jetson-specific PyTorch build is genuinely difficult (see below).

## Running

Easiest: double-click the desktop icons created by `create_launchers.sh`. Each one pulls the latest code, kills any prior instance on its port, starts the service, and opens a browser once it's actually up.

- **ArgusVision LiZAD Server** (port 8000)
- **ArgusVision PatchCore Server** (port 8001)
- **ArgusVision EfficientAD Server** (port 8002)
- **ArgusVision** (port 7860) — the UI

Start whichever detector server(s) you want to compare, then the UI. Manual equivalent:

```bash
cd ~/LiZAD && conda activate LiZAD && python lizad_server.py        # terminal 1
cd ~/LiZAD && conda activate LiZAD && python patchcore_server.py    # terminal 2
cd ~/LiZAD && conda activate LiZAD && python efficientad_server.py  # terminal 3
cd ~/LiZAD && conda activate LiZAD && python app.py                 # terminal 4
```

Then open `http://<jetson-ip>:7860`.

A detector server with no trained checkpoint starts fine and says so in the UI — click **Train** on its tab. Retraining after capturing more images is also a button; no restart required.

## Environment gotchas (hard-won)

**PyTorch on Jetson AGX Orin is a minefield.** Orin is `sm_87` — completely different from NVIDIA's SBSA/server ARM chips (`sm_110`/`sm_121`) despite both being "ARM64 Linux CUDA", so a generic ARM64 torch wheel silently installs the wrong architecture. Use the Jetson-specific index `https://pypi.jetson-ai-lab.io/jp6/cu129`, which only publishes **Python 3.10** builds — so the conda env must be created with `python=3.10`. A full set of `nvidia-*-cu12` runtime libraries must also be installed separately (including `nvidia-cudss-cu12`, easy to miss). All captured in `full_setup.sh`.

**anomalib will try to replace your torch.** Its `[cu126]`/`[cu130]` extras pull generic CUDA wheels that would re-trigger the architecture mismatch above. `install_anomalib.sh` installs it with `--no-deps` plus a hand-curated dependency list, and verifies CUDA is still intact afterward.

**The GigE camera's IP config does not survive reboots by default.** The interface is `end0` (not `eth0`). NetworkManager's default DHCP profile will fight any manual `ip addr add`, since there's no DHCP server on a direct camera link. Fix it durably:

```bash
sudo nmcli connection modify "Wired connection 1" connection.autoconnect no
sudo nmcli connection add type ethernet ifname end0 con-name end0-camera ip4 169.254.1.100/16
sudo nmcli connection up end0-camera
```

Diagnose with `nmcli device status` — `end0` stuck on "connecting (getting IP configuration)" is the tell. **Check this before assuming an Arena SDK or camera hardware failure.**

## Known unknowns

- **The live serving path has been validated only offline.** `detector_server.py`'s `PredictDataset`/`engine.predict()`-per-request path, and the checkpoint save/load via `Engine.trainer.save_checkpoint()` / `Model.load_from_checkpoint()`, are a best-effort reading of anomalib 2.x's API. Only the offline experiment path has actually run. Validate PatchCore's server before trusting EfficientAD's — they share the implementation, so a fix in one fixes both.
- **EfficientAD is unmeasured.** Its `MIN_TRAIN_IMAGES = 20` and `MAX_EPOCHS = 20` are starting points, not tuned values. With a small dataset expect "inconclusive" rather than a verdict; `experiment.py` tries to say which case applies rather than reporting a false negative.
- PatchCore and EfficientAD expose a **score only** — no anomaly map, so no heatmap or bounding boxes on those tabs yet.
- **Scores are not comparable across detectors or across runs** (anomalib normalizes per fit). Compare the *gap* between normal and defect within a single run — never raw values.
- Both LiZAD and the anomalib models resize input to a square (518×518 for LiZAD), so a small defect in a wide frame loses detail. Framing tightly on the region of interest likely matters more than it currently gets credit for.
- The `Folder` datamodule's default splits hold back a surprising fraction of captured images from evaluation; `val_split_ratio=0.0` reduces this, but the exact train/test/val accounting still isn't fully pinned down.
- Staleness is detected by **image count only**. Replacing images without changing the count, or recapturing under different lighting, won't be flagged.
