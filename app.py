import gradio as gr
import numpy as np
import cv2
import threading
import time
import os
import json
import types
import subprocess
from datetime import datetime

try:
    from arena_api.system import system
    ARENA_AVAILABLE = True
except ImportError:
    system = None
    ARENA_AVAILABLE = False

try:
    from lizad_client import LiZADClient
    LIZAD_AVAILABLE = True
except ImportError:
    LiZADClient = None
    LIZAD_AVAILABLE = False

try:
    from patchcore_client import PatchCoreClient
    PATCHCORE_AVAILABLE = True
except ImportError:
    PatchCoreClient = None
    PATCHCORE_AVAILABLE = False

try:
    from efficientad_client import EfficientAdClient
    EFFICIENTAD_AVAILABLE = True
except ImportError:
    EfficientAdClient = None
    EFFICIENTAD_AVAILABLE = False


def overlay_heatmap(frame_rgb, anomaly_map, alpha=0.45):
    h, w = frame_rgb.shape[:2]
    resized = cv2.resize(anomaly_map, (w, h))
    normalized = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_rgb, alpha, 0)


# A region below this fraction of the frame area is treated as noise rather
# than a real anomaly blob - keeps single stray pixels from drawing boxes.
MIN_BOX_AREA_FRACTION = 0.001


def boxes_from_anomaly_map(anomaly_map, frame_shape, threshold):
    h, w = frame_shape[:2]
    resized = cv2.resize(anomaly_map, (w, h))
    mask = (resized >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_BOX_AREA_FRACTION * h * w

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        region_score = float(resized[y:y + bh, x:x + bw].max())
        boxes.append((x, y, bw, bh, region_score))
    return boxes


def draw_boxes(frame_rgb, boxes):
    annotated = frame_rgb.copy()
    box_color = (235, 67, 47)  # COLOR_STATUS_RED as an RGB tuple (frame_rgb is RGB, not BGR)
    for x, y, w, h, region_score in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)
        label_y = y - 8 if y - 8 > 12 else y + h + 20
        cv2.putText(
            annotated, f"{region_score:.2f}", (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA,
        )
    return annotated


# Deterministic per-channel color check - not the anomaly-detection model.
# For structured checks ("this channel must contain this specific color wire")
# a calibrated color comparison is faster, more reliable, and fully explainable
# compared to asking a general anomaly model "does this look wrong."
CHANNEL_CONFIG_PATH = "channel_config.json"
HUE_TOLERANCE = 20  # degrees out of OpenCV's 0-180 hue range

channel_state = {
    "channels": [],  # [{"name": str, "region": [x, y, w, h], "expected_hue": float}, ...]
    "calibration_mode": False,
    "pending_click": None,  # (x, y) of the first clicked corner, or None
}


def load_channels():
    if os.path.isfile(CHANNEL_CONFIG_PATH):
        with open(CHANNEL_CONFIG_PATH) as f:
            return json.load(f)
    return []


def save_channels():
    with open(CHANNEL_CONFIG_PATH, "w") as f:
        json.dump(channel_state["channels"], f, indent=2)


channel_state["channels"] = load_channels()


def sample_hue(frame_rgb, region):
    x, y, w, h = region
    crop = frame_rgb[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return float(hsv.reshape(-1, 3)[:, 0].mean())


def hue_distance(a, b):
    diff = abs(a - b)
    return min(diff, 180 - diff)


def check_channels(frame_rgb):
    results = []
    for ch in channel_state["channels"]:
        actual_hue = sample_hue(frame_rgb, ch["region"])
        ok = hue_distance(actual_hue, ch["expected_hue"]) <= HUE_TOLERANCE
        results.append({"name": ch["name"], "ok": ok, "actual_hue": actual_hue, "expected_hue": ch["expected_hue"]})
    return results


def draw_channel_boxes(frame_rgb, results=None):
    annotated = frame_rgb.copy()
    for i, ch in enumerate(channel_state["channels"]):
        x, y, w, h = ch["region"]
        if results is not None:
            color = (2, 176, 40) if results[i]["ok"] else (235, 67, 47)  # COLOR_STATUS_GREEN / RED
        else:
            color = (62, 107, 226)  # COLOR_PRIMARY_BLUE - neutral, calibration mode
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, ch["name"], (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    if channel_state["pending_click"] is not None:
        cv2.circle(annotated, channel_state["pending_click"], 6, (62, 107, 226), -1)
    return annotated


def toggle_calibration_mode(enabled):
    channel_state["calibration_mode"] = enabled
    channel_state["pending_click"] = None
    if enabled:
        return "Calibration mode ON - click two corners on the image to define each channel"
    return "Calibration mode off - showing live channel status"


def on_channel_image_click(evt: gr.SelectData, next_name):
    if not channel_state["calibration_mode"]:
        return gr.update(), "Enable Calibration Mode to define channels"

    x, y = evt.index
    pending = channel_state["pending_click"]

    if pending is None:
        channel_state["pending_click"] = (x, y)
        return gr.update(), f"First corner set at ({x},{y}) - click the opposite corner"

    x1, y1 = pending
    channel_state["pending_click"] = None
    rx, ry = min(x1, x), min(y1, y)
    rw, rh = abs(x - x1), abs(y - y1)
    if rw < 5 or rh < 5:
        return gr.update(), "Region too small - click the first corner again"

    frame = camera.get_frame()
    expected_hue = sample_hue(frame, (rx, ry, rw, rh)) if frame is not None else 0.0
    name = next_name.strip() or f"channel_{len(channel_state['channels']) + 1}"
    channel_state["channels"].append({"name": name, "region": [rx, ry, rw, rh], "expected_hue": expected_hue})
    save_channels()

    return (
        gr.update(value=f"channel_{len(channel_state['channels']) + 1}"),
        f"Added '{name}' (expected hue {expected_hue:.0f}/180)",
    )


def undo_last_channel():
    if channel_state["channels"]:
        removed = channel_state["channels"].pop()
        save_channels()
        return f"Removed '{removed['name']}'"
    return "No channels to undo"


def clear_all_channels():
    channel_state["channels"] = []
    channel_state["pending_click"] = None
    save_channels()
    return "All channels cleared"


def stream_channel_check():
    """Note this feed is NOT downscaled, unlike the others: calibration clicks
    are recorded in native frame coordinates, so the displayed image has to
    stay in the same pixel space as the sampled one."""
    while True:
        # Idle unless the tab is in front. Beyond the encode, this loop runs
        # real per-frame OpenCV work, so skipping it matters more than it does
        # for the plain camera views.
        if active_tab["name"] != TAB_CHANNEL:
            yield gr.update(), gr.update()
            time.sleep(0.5)
            continue

        frame = camera.get_frame()
        if frame is None:
            yield no_camera_placeholder(), "Waiting for camera..."
            time.sleep(0.3)
            continue

        if channel_state["calibration_mode"]:
            # gr.update() (no value) leaves whatever the click handler last set
            # in the status box alone, instead of overwriting it every 0.3s.
            yield draw_channel_boxes(frame, results=None), gr.update()
        elif not channel_state["channels"]:
            yield frame, "No channels calibrated yet - enable Calibration Mode and click to define channels."
        else:
            results = check_channels(frame)
            lines = [
                f"{'OK  ' if r['ok'] else 'FAIL'} - {r['name']} (hue {r['actual_hue']:.0f}, expected {r['expected_hue']:.0f})"
                for r in results
            ]
            yield draw_channel_boxes(frame, results=results), "\n".join(lines)
        time.sleep(0.3)


DATA_ROOT = "inspection_data/images"

# Hierarchical label taxonomy: category -> list of subtypes.
# "normal" is a leaf category (images go straight in inspection_data/images/normal/).
# "defect" is a parent whose subtypes are the actual capture folders
# (inspection_data/images/defect/missing_component/, etc.) - a category with
# subtypes never gets images saved directly under it.
TAXONOMY = {
    "normal": [],
    "defect": ["missing_component", "wrong_placement", "wrong_color"],
}

# UI palette, per the internal design standards for operator-facing tooling
COLOR_PRIMARY_BLUE = "#3e6be2"
COLOR_WHITE = "#FFFFFF"
COLOR_GREY_BG = "#F4F4F4"
COLOR_STATUS_RED = "#eb432f"
COLOR_STATUS_ORANGE = "#f29137"
COLOR_STATUS_GREEN = "#02b028"
COLOR_STATUS_TEAL = "#3eade1"
COLOR_STATUS_GREY = "#707070"


class CameraController:
    def __init__(self):
        self.device = None
        self.latest_frame = None
        self.running = False
        self.connected = False
        self.lock = threading.Lock()

    def connect(self):
        if not ARENA_AVAILABLE:
            return False
        devices = system.create_device()
        if not devices:
            return False
        self.device = devices[0]
        nodemap = self.device.nodemap
        nodemap.get_node("PixelFormat").value = "BGR8"
        nodemap.get_node("ExposureAuto").value = "Continuous"
        nodemap.get_node("GainAuto").value = "Continuous"
        try:
            nodemap.get_node("BalanceWhiteAuto").value = "Continuous"
        except Exception:
            pass  # not every camera/firmware exposes this node
        self.device.start_stream()
        self.running = True
        self.connected = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        return True

    def _capture_loop(self):
        while self.running:
            try:
                buffer = self.device.get_buffer()
                image = np.ctypeslib.as_array(
                    buffer.pdata, shape=(buffer.height, buffer.width, 3)
                ).copy()
                self.device.requeue_buffer(buffer)
                with self.lock:
                    self.latest_frame = image
            except Exception as e:
                print(f"Capture error: {e}")
                self.connected = False
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)

    def set_exposure_auto(self, auto):
        self.device.nodemap.get_node("ExposureAuto").value = "Continuous" if auto else "Off"

    def set_exposure_value(self, value):
        nodemap = self.device.nodemap
        nodemap.get_node("ExposureAuto").value = "Off"
        nodemap.get_node("ExposureTime").value = float(value)

    def set_gain_auto(self, auto):
        self.device.nodemap.get_node("GainAuto").value = "Continuous" if auto else "Off"

    def set_gain_value(self, value):
        nodemap = self.device.nodemap
        nodemap.get_node("GainAuto").value = "Off"
        nodemap.get_node("Gain").value = float(value)

    def set_wb_auto(self, auto):
        try:
            self.device.nodemap.get_node("BalanceWhiteAuto").value = "Continuous" if auto else "Off"
        except Exception:
            pass

    def get_current_exposure(self):
        if not self.device:
            return None
        return float(self.device.nodemap.get_node("ExposureTime").value)

    def get_current_gain(self):
        if not self.device:
            return None
        return float(self.device.nodemap.get_node("Gain").value)

    def stop(self):
        self.running = False
        if self.device:
            self.device.stop_stream()
            system.destroy_device(self.device)


camera = CameraController()
last_saved_path = {"path": None}

inference_state = {
    "engine": None,
    "enabled": False,
    "threshold": 0.5,
    "latest_overlay": None,
    "latest_score": 0.0,
    "latest_box_count": 0,
}


def toggle_inference(enabled):
    if enabled and not LIZAD_AVAILABLE:
        return "LiZAD client not available on this machine", gr.update(value=False)
    inference_state["enabled"] = enabled
    if enabled and inference_state["engine"] is None:
        client = LiZADClient(host="localhost", port=8000)
        if not client.health():
            inference_state["enabled"] = False
            return "Cannot reach LiZAD inference server on localhost:8000 - is lizad_server.py running?", gr.update(value=False)
        inference_state["engine"] = client
    return ("Live inference enabled" if enabled else "Live inference disabled"), gr.update()


def update_threshold(value):
    inference_state["threshold"] = value


def all_images_in_folder(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)]


def score_images(engine, paths):
    scores = []
    for path in paths:
        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        _, score = engine.run(frame)
        scores.append(score)
    return scores


def multi_score_images(engine, paths):
    scores_by_label = {}
    for path in paths:
        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        for label, score in engine.run_multi_scores(frame).items():
            scores_by_label.setdefault(label, []).append(score)
    return scores_by_label


def compare_prompt_sets(category_choice, subtype_choice):
    engine = inference_state["engine"]
    if engine is None:
        return "Enable Live Inference first so the LiZAD connection is available."

    category, subtype = current_selection(category_choice, subtype_choice)
    normal_paths = all_images_in_folder(label_dir("normal", None))
    defect_paths = [] if category == "normal" else all_images_in_folder(label_dir(category, subtype))

    if not normal_paths or not defect_paths:
        return (
            f"Need at least one captured image in 'normal' and in the currently selected "
            f"category/subtype to compare (have {len(normal_paths)} normal, {len(defect_paths)} "
            f"in '{category}/{subtype or ''}')."
        )

    normal_by_label = multi_score_images(engine, normal_paths)
    defect_by_label = multi_score_images(engine, defect_paths)

    rows = []
    for label in normal_by_label:
        if label not in defect_by_label:
            continue
        max_normal = max(normal_by_label[label])
        min_defect = min(defect_by_label[label])
        rows.append((label, min_defect - max_normal, max_normal, min_defect))
    rows.sort(key=lambda r: r[1], reverse=True)

    lines = [
        f"Comparing prompt sets for '{category}/{subtype or category}' "
        f"({len(normal_paths)} normal, {len(defect_paths)} defect images):"
    ]
    for i, (label, gap, max_normal, min_defect) in enumerate(rows):
        marker = " <- best" if i == 0 else ""
        separation = "clean separation" if gap > 0 else "overlap"
        lines.append(
            f"  {label}: gap={gap:+.3f} (normal max={max_normal:.3f}, "
            f"defect min={min_defect:.3f}, {separation}){marker}"
        )
    return "\n".join(lines)


def suggest_threshold():
    engine = inference_state["engine"]
    if engine is None:
        return gr.update(), "Enable Live Inference first so the LiZAD connection is available."

    tree = discover_taxonomy()
    normal_paths = all_images_in_folder(label_dir("normal", None))
    defect_paths = []
    for subtype in tree.get("defect", []):
        defect_paths.extend(all_images_in_folder(label_dir("defect", subtype)))

    if not normal_paths or not defect_paths:
        return gr.update(), (
            f"Need at least one captured image in both 'normal' and a 'defect' subtype "
            f"to suggest a threshold (have {len(normal_paths)} normal, {len(defect_paths)} defect)."
        )

    normal_scores = score_images(engine, normal_paths)
    defect_scores = score_images(engine, defect_paths)
    max_normal = max(normal_scores)
    min_defect = min(defect_scores)
    suggested = round((max_normal + min_defect) / 2, 3)

    if max_normal >= min_defect:
        note = (
            f"Scores overlap (normal max={max_normal:.3f}, defect min={min_defect:.3f}) - "
            "this is a rough midpoint, expect some misses. Capture more examples or revisit the prompts."
        )
    else:
        note = f"Clean separation - normal max={max_normal:.3f}, defect min={min_defect:.3f}."

    return (
        gr.update(value=suggested),
        f"Suggested threshold: {suggested}. {note} ({len(normal_scores)} normal, {len(defect_scores)} defect images evaluated)",
    )


def lock_threshold(locked):
    return gr.update(interactive=not locked), gr.update(interactive=not locked)


# ---------------------------------------------------------------------------
# Cross-detector comparison. This is the output the platform exists to
# produce: the same captured dataset scored by every available approach, so
# the choice of detector rests on measurement rather than reputation.
# ---------------------------------------------------------------------------

COMPARISON_HISTORY_PATH = "comparison_history.json"
SETUP_PROFILE_PATH = "setup_profile.json"
REPORTS_DIR = "reports"

SETUP_FIELDS = ("part", "defect_class", "distance_mm", "lighting", "mounting", "notes")


def load_setup_profile():
    """Setup rarely changes between runs, so the form is prefilled from the
    last session rather than retyped each time."""
    if os.path.isfile(SETUP_PROFILE_PATH):
        try:
            with open(SETUP_PROFILE_PATH) as f:
                saved = json.load(f)
            return {k: saved.get(k, "") for k in SETUP_FIELDS}
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read {SETUP_PROFILE_PATH} ({e})")
    return {k: "" for k in SETUP_FIELDS}


def save_setup_profile(profile):
    tmp = SETUP_PROFILE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=2)
    os.replace(tmp, SETUP_PROFILE_PATH)


def live_camera_settings():
    """Exposure and gain read from the camera itself - the two settings most
    likely to be misremembered, and the ones that most affect scores."""
    try:
        exposure = camera.get_current_exposure()
        gain = camera.get_current_gain()
    except Exception:
        return {}
    if exposure is None or gain is None:
        return {}
    return {"exposure_us": round(exposure, 1), "gain_db": round(gain, 2)}


def collect_setup(part, defect_class, distance_mm, lighting, mounting, notes):
    profile = {
        "part": (part or "").strip(),
        "defect_class": (defect_class or "").strip(),
        "distance_mm": (distance_mm or "").strip(),
        "lighting": (lighting or "").strip(),
        "mounting": mounting or "",
        "notes": (notes or "").strip(),
    }
    save_setup_profile(profile)
    return {**profile, **live_camera_settings()}


def derive_caveats(results, setup, skipped, n_normal, n_defect):
    """Caveats are generated from the run rather than left to the reader.

    The failure mode this guards against is specific: a measurement taken on
    a desk with a handheld camera being quoted months later as though it were
    controlled. If the limitations travel with the numbers, that can't happen.
    """
    caveats = [
        "Scores are NOT comparable between detectors - each normalises differently. "
        "Compare the gap within a detector, never raw values across them."
    ]
    if setup.get("mounting") != "Fixed / fixtured":
        caveats.append(
            "**Camera and/or part were NOT rigidly fixtured.** Position and lighting "
            "variation between shots is a large confound here, and reference-based "
            "detectors (PatchCore, EfficientAD) are the most damaged by it. Treat any "
            "ranking as provisional."
        )
    thin = [r for r in results if min(r["n_normal"], r["n_defect"]) < MIN_MEANINGFUL_EVAL]
    if thin:
        caveats.append(
            f"{len(thin)} detector(s) scored fewer than {MIN_MEANINGFUL_EVAL} images per "
            "class. Conclusions from a handful of images are indicative at best."
        )
    if results and results[0]["gap"] <= 0:
        caveats.append(
            "No detector separated this defect class. That may mean insufficient or "
            "noisy data rather than an unsuitable approach - it is not evidence that "
            "machine vision cannot do this job."
        )
    elif results and results[0]["gap"] < MARGINAL_GAP:
        caveats.append(
            f"The best gap ({results[0]['gap']:+.3f}) is below {MARGINAL_GAP}, which is "
            "within the frame-to-frame noise we have measured from lighting and "
            "position drift. Not a dependable margin."
        )
    if skipped:
        caveats.append(
            "Not every approach was evaluated - " + "; ".join(skipped) +
            ". This is a partial comparison."
        )
    if not setup.get("exposure_us"):
        caveats.append("Camera exposure/gain were not recorded (no camera connected at run time).")
    return caveats


def build_report(run):
    setup = run.get("setup", {})
    ds = run.get("dataset", {})
    results = run.get("results", [])

    lines = [
        f"# Vision approach comparison - {setup.get('part') or 'unnamed part'}",
        "",
        f"**Run:** {run.get('when', '')}  ",
        f"**Defect class:** {setup.get('defect_class') or 'not recorded'}",
        "",
        "## Setup",
        "",
        "| | |",
        "|---|---|",
        f"| Mounting | {setup.get('mounting') or 'not recorded'} |",
        f"| Camera distance | {setup.get('distance_mm') or 'not recorded'} |",
        f"| Lighting | {setup.get('lighting') or 'not recorded'} |",
        f"| Exposure | {setup.get('exposure_us', 'not recorded')} microseconds |",
        f"| Gain | {setup.get('gain_db', 'not recorded')} dB |",
        f"| Dataset | {ds.get('n_normal', '?')} normal, {ds.get('n_defect', '?')} defect |",
    ]
    if setup.get("notes"):
        lines += ["", f"**Notes:** {setup['notes']}"]

    lines += [
        "",
        "## Results",
        "",
        "The **gap** is the distance between the worst normal score and the best defect "
        "score. Positive means a threshold exists that classifies every evaluated image "
        "correctly; larger means more tolerance for drift.",
        "",
        "| Detector | Normal max | Defect min | Gap | Verdict | Images scored |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['detector']} | {r['max_normal']:.3f} | {r['min_defect']:.3f} | "
            f"{r['gap']:+.3f} | {verdict_for_gap(r['gap'])} | "
            f"{r['n_normal']} normal / {r['n_defect']} defect |"
        )

    if results:
        best = results[0]
        lines += [
            "",
            f"**Best: {best['detector']}**, gap {best['gap']:+.3f} "
            f"({verdict_for_gap(best['gap'])}).",
        ]
        midpoint = (best["max_normal"] + best["min_defect"]) / 2
        if best["gap"] >= MARGINAL_GAP:
            lines.append(
                f"A threshold near **{midpoint:.3f}** would classify this set correctly, "
                f"with roughly {best['gap'] / 2:.3f} of margin either side."
            )
        elif best["gap"] > 0:
            # Deliberately not phrased as a recommendation. A midpoint exists,
            # but quoting it confidently would contradict the caveats below and
            # this report is what someone acts on.
            lines.append(
                f"The separation is too thin to set a dependable threshold. A midpoint "
                f"would fall near {midpoint:.3f}, but with only {best['gap']:+.3f} of gap "
                "it would not survive normal lighting or position variation."
            )
        else:
            lines.append(
                "No threshold separates these classes on this data - the score "
                "distributions overlap."
            )

    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in run.get("caveats", [])]
    lines += [
        "",
        "---",
        "",
        "*Generated by ArgusVision, a bench platform for pre-validating manufacturing "
        "vision approaches. These results indicate which approach suits this defect "
        "class under the conditions above; they are not a production qualification.*",
    ]
    return "\n".join(lines)


def export_last_report():
    history = load_comparison_history()
    if not history:
        return "No comparison runs yet - run one first.", None
    run = history[-1]
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = run.get("when", "").replace(":", "-") or datetime.now().strftime("%Y%m%d_%H%M%S")
    part = (run.get("setup", {}).get("part") or "run").replace(" ", "_")
    # Keep the filename filesystem-safe without dragging in a slug library.
    part = "".join(c for c in part if c.isalnum() or c in "._-") or "run"
    path = os.path.join(REPORTS_DIR, f"{part}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report(run))
    return f"Report written to {os.path.abspath(path)}", path

# A gap can be positive and still worthless. 0.02 between the worst normal and
# the best defect is well inside the frame-to-frame noise we've measured from
# lighting and part-position drift, so calling it "separates" would be
# misleading. Rule of thumb, not a derived constant.
MARGINAL_GAP = 0.10

# anomalib holds back only a fraction of captured images for evaluation, so a
# large capture set can still produce a handful of scored images. Conclusions
# from three images are not conclusions.
MIN_MEANINGFUL_EVAL = 8


def verdict_for_gap(gap):
    if gap <= 0:
        return "OVERLAP"
    if gap < MARGINAL_GAP:
        return "marginal"
    return "separates"


# Only the most recent runs are rendered. The history file keeps everything -
# this is purely to stop the tab getting heavier every time you press Run.
HISTORY_DISPLAY_LIMIT = 30


def rows_to_markdown(headers, rows, empty_message="_No runs yet._"):
    """Render a table as Markdown rather than gr.Dataframe.

    Dataframe is a comparatively heavy component and Gradio builds it lazily
    on first display, which is what made switching to this tab lag - it was
    the only tab with two of them. These tables are read-only, so the widget
    bought nothing.
    """
    if not rows:
        return empty_message
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def load_comparison_history():
    if not os.path.isfile(COMPARISON_HISTORY_PATH):
        return []
    try:
        with open(COMPARISON_HISTORY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read {COMPARISON_HISTORY_PATH} ({e})")
        return []


def save_comparison_run(run):
    history = load_comparison_history()
    history.append(run)
    tmp = COMPARISON_HISTORY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, COMPARISON_HISTORY_PATH)  # atomic
    return history


def captured_counts():
    tree = discover_taxonomy()
    n_normal = len(all_images_in_folder(label_dir("normal", None)))
    n_defect = sum(
        len(all_images_in_folder(label_dir("defect", s))) for s in tree.get("defect", [])
    )
    return n_normal, n_defect


def evaluate_lizad():
    """LiZAD scores every captured image rather than a held-out split - being
    zero-shot, it never trained on any of them, so there is no train/test
    distinction to respect. Noted in the UI because it means its image counts
    won't match the other detectors'."""
    if not LIZAD_AVAILABLE:
        return None
    client = inference_state["engine"] or LiZADClient(host="localhost", port=8000)
    if not client.health():
        return None

    tree = discover_taxonomy()
    normal_paths = all_images_in_folder(label_dir("normal", None))
    defect_paths = []
    for subtype in tree.get("defect", []):
        defect_paths.extend(all_images_in_folder(label_dir("defect", subtype)))
    if not normal_paths or not defect_paths:
        return None

    normal_scores = score_images(client, normal_paths)
    defect_scores = score_images(client, defect_paths)
    max_normal, min_defect = max(normal_scores), min(defect_scores)
    return {
        "detector": "LiZAD (zero-shot)",
        "max_normal": max_normal,
        "min_defect": min_defect,
        "gap": min_defect - max_normal,
        "n_normal": len(normal_scores),
        "n_defect": len(defect_scores),
    }


def run_comparison(part, defect_class, distance_mm, lighting, mounting, notes):
    """Runs every reachable detector over the captured dataset and returns
    (results table markdown, summary, history markdown)."""
    setup = collect_setup(part, defect_class, distance_mm, lighting, mounting, notes)
    n_normal, n_defect = captured_counts()
    if not n_normal or not n_defect:
        return (
            "",
            f"Need images in both categories to compare (have {n_normal} normal, "
            f"{n_defect} defect). Capture some in Live Capture first.",
            history_markdown(),
        )

    results, skipped = [], []

    lizad_result = evaluate_lizad()
    if lizad_result:
        results.append(lizad_result)
    else:
        skipped.append("LiZAD (server unreachable or no images)")

    for runtime, name in ((patchcore, "PatchCore"), (efficientad, "EfficientAD")):
        detector_result = _evaluate_runtime(runtime, name)
        if detector_result:
            results.append(detector_result)
        else:
            skipped.append(f"{name} (server unreachable or untrained)")

    if not results:
        return "", "No detectors available. Start at least one server.", history_markdown()

    results.sort(key=lambda r: r["gap"], reverse=True)
    rows = [
        [
            r["detector"],
            f"{r['max_normal']:.3f}",
            f"{r['min_defect']:.3f}",
            f"**{r['gap']:+.3f}**",
            verdict_for_gap(r["gap"]),
            f"{r['n_normal']}N / {r['n_defect']}D"
            + (" _(too few)_" if min(r["n_normal"], r["n_defect"]) < MIN_MEANINGFUL_EVAL else ""),
        ]
        for r in results
    ]
    table_md = rows_to_markdown(
        ["Detector", "Normal max", "Defect min", "Gap", "Verdict", "Images"], rows
    )

    best = results[0]
    thin = [r for r in results if min(r["n_normal"], r["n_defect"]) < MIN_MEANINGFUL_EVAL]
    if best["gap"] > 0:
        summary = (
            f"Best: {best['detector']} with a gap of {best['gap']:+.3f} "
            f"({verdict_for_gap(best['gap'])}). "
            f"Threshold would sit near {(best['max_normal'] + best['min_defect']) / 2:.3f}."
        )
        if thin:
            summary += (
                f"  CAUTION: {len(thin)} detector(s) scored fewer than "
                f"{MIN_MEANINGFUL_EVAL} images per class - treat as indicative only."
            )
    else:
        summary = (
            "No detector separates this defect class cleanly on the current dataset. "
            "That may mean more images are needed, or that this defect suits a "
            "deterministic check rather than anomaly detection."
        )
    if skipped:
        summary += "  Not evaluated: " + "; ".join(skipped) + "."

    if setup.get("mounting") != "Fixed / fixtured":
        summary += "  NOT FIXTURED - ranking is provisional."

    save_comparison_run(
        {
            "when": datetime.now().isoformat(timespec="seconds"),
            "setup": setup,
            "dataset": {"n_normal": n_normal, "n_defect": n_defect},
            "results": results,
            "skipped": skipped,
            "caveats": derive_caveats(results, setup, skipped, n_normal, n_defect),
        }
    )
    return table_md, summary, history_markdown()


def _evaluate_runtime(runtime, name):
    try:
        r = runtime.evaluate_raw()
    except Exception as e:
        print(f"{name} evaluation failed: {e}")
        return None
    if not r or not r.get("ok"):
        return None
    return {
        "detector": f"{name} (reference-image)",
        "max_normal": r["max_normal"],
        "min_defect": r["min_defect"],
        "gap": r["gap"],
        "n_normal": r["n_normal"],
        "n_defect": r["n_defect"],
    }


HISTORY_HEADERS = ["When", "Part", "Detector", "Gap", "Verdict", "Mounting", "Dataset"]


def history_rows():
    """Mounting is a column, not a footnote - it is the condition most likely
    to invalidate a comparison, and the one most likely to be forgotten when
    someone reads these numbers back later."""
    rows = []
    for run in reversed(load_comparison_history()):
        ds = run.get("dataset", {})
        setup = run.get("setup", {})
        for r in run.get("results", []):
            gap = r.get("gap", 0)
            rows.append(
                [
                    run.get("when", ""),
                    setup.get("part") or "-",
                    r.get("detector", ""),
                    f"{gap:+.3f}",
                    verdict_for_gap(gap),
                    setup.get("mounting") or "not recorded",
                    f"{ds.get('n_normal', '?')}N / {ds.get('n_defect', '?')}D",
                ]
            )
    return rows


def history_markdown():
    rows = history_rows()
    total = len(rows)
    shown = rows[:HISTORY_DISPLAY_LIMIT]
    table = rows_to_markdown(HISTORY_HEADERS, shown, "_No comparison runs recorded yet._")
    if total > len(shown):
        table += f"\n\n_Showing the {len(shown)} most recent of {total} rows._"
    return table


def delete_last_run():
    """For the common case: that run was botched, drop it."""
    history = load_comparison_history()
    if not history:
        return "Nothing to delete.", history_markdown()
    removed = history.pop()
    tmp = COMPARISON_HISTORY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, COMPARISON_HISTORY_PATH)
    when = removed.get("when", "unknown time")
    part = removed.get("setup", {}).get("part") or "unnamed part"
    return f"Removed the run from {when} ({part}).", history_markdown()


def clear_history():
    """Archives rather than deletes. This table is meant to be an auditable
    record, so 'clear' should mean 'clear the view', not 'destroy the
    evidence' - especially since it's one button press away from a demo."""
    if not os.path.isfile(COMPARISON_HISTORY_PATH):
        return "No history to clear.", history_markdown()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    archive = os.path.join(REPORTS_DIR, f"comparison_history_archived_{stamp}.json")
    os.replace(COMPARISON_HISTORY_PATH, archive)
    return f"History cleared. Previous runs archived to {archive}", history_markdown()


def _inference_loop():
    while True:
        if inference_state["enabled"] and inference_state["engine"] is not None:
            frame = camera.get_frame()
            if frame is not None:
                try:
                    anomaly_map, score = inference_state["engine"].run(frame)
                    boxes = boxes_from_anomaly_map(anomaly_map, frame.shape, inference_state["threshold"])
                    overlay = overlay_heatmap(frame, anomaly_map)
                    inference_state["latest_overlay"] = draw_boxes(overlay, boxes)
                    inference_state["latest_score"] = score
                    inference_state["latest_box_count"] = len(boxes)
                except Exception as e:
                    print(f"Inference error: {e}")
        time.sleep(1.0)


threading.Thread(target=_inference_loop, daemon=True).start()


def verdict_html(score, threshold):
    if score >= threshold:
        return f'<div style="font-size:2em;font-weight:700;color:{COLOR_STATUS_RED};">⚠ ANOMALY DETECTED</div>'
    return f'<div style="font-size:2em;font-weight:700;color:{COLOR_STATUS_GREEN};">✓ NORMAL</div>'


def stream_inference():
    while True:
        overlay = inference_state["latest_overlay"]
        score = inference_state["latest_score"]
        threshold = inference_state["threshold"]
        box_count = inference_state["latest_box_count"]
        # Skip the heatmap encode entirely when the LiZAD tab isn't in front;
        # the score and verdict are cheap enough to keep current.
        on_tab = active_tab["name"] == TAB_LIZAD
        if overlay is not None:
            image = for_display(overlay) if on_tab else gr.update()
            yield image, round(score, 3), box_count, verdict_html(score, threshold)
        else:
            image = no_camera_placeholder() if on_tab else gr.update()
            yield image, 0.0, 0, '<div style="font-size:1.2em;color:#707070;">Waiting for inference...</div>'
        time.sleep(0.3)


# Display widgets are 480-560px tall, but camera frames are several
# megapixels. Encoding the full frame so the browser can scale it down is
# pure waste, and it was happening for four widgets at 5 fps.
DISPLAY_MAX_HEIGHT = 620

# Which tab is in front. Streaming to hidden widgets costs a full image
# encode each and buys nothing, so the generators skip them.
active_tab = {"name": "Live Capture"}

TAB_LIVE = "Live Capture"
TAB_LIZAD = "LiZAD (Zero-Shot)"
TAB_PATCHCORE = "PatchCore (Reference-Based)"
TAB_EFFICIENTAD = "EfficientAD (Logical Anomalies)"
TAB_CHANNEL = "Channel Check (Classical CV)"
TAB_CAMERA = "Camera Settings"


def on_tab_select(evt: gr.SelectData):
    active_tab["name"] = evt.value


def for_display(frame):
    """Downscale a frame for the browser.

    NOT used for the Channel Check feed: its calibration stores click
    coordinates in native frame pixels, so scaling what's displayed would
    silently invalidate every calibrated region.
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if h <= DISPLAY_MAX_HEIGHT:
        return frame
    scale = DISPLAY_MAX_HEIGHT / h
    return cv2.resize(
        frame, (int(w * scale), DISPLAY_MAX_HEIGHT), interpolation=cv2.INTER_AREA
    )


def format_training_status(status, label, port):
    """The one line that tells the operator whether the model they're testing
    actually reflects the images they've captured. Staleness is measured by
    image count rather than anything cleverer - simple, and understandable
    without explanation."""
    if status is None:
        return f"{label} server not reachable on port {port} - start it to train or run inference."

    state = status.get("state")
    if state == "training":
        pct = int(round(status.get("progress", 0.0) * 100))
        return f"Training... {pct}%   |   {status.get('message', '')}"
    if state == "error":
        return f"Training failed - {status.get('message', 'unknown error')}"

    current = status.get("current_n_images") or 0
    if not status.get("has_model"):
        # Surface the server's own message rather than assuming "never trained".
        # It distinguishes that from "a checkpoint exists but failed to load",
        # which needs a completely different response from the operator.
        return f"{status.get('message') or 'No model trained yet'}   ({current} normal images captured)"

    n_trained = status.get("n_images")
    if n_trained is None:
        return f"Model loaded, but training details are unknown. {current} normal images on disk."

    line = f"Trained on {n_trained} normal images at {status.get('trained_at', 'unknown time')}."
    delta = current - n_trained
    if delta > 0:
        return f"{line}   {delta} new image{'s' if delta != 1 else ''} captured since - retrain recommended."
    return f"{line}   Up to date."


def make_detector_runtime(label, client_cls, available, port, tab_name, interval=2.0):
    """Both trainable detectors (PatchCore, EfficientAD) need identical runtime
    behaviour: a client, a background scoring loop, a live score/verdict
    stream, and training controls. Built once and instantiated twice, so the
    two tabs cannot drift apart - which matters for a tool whose whole purpose
    is comparing them fairly."""
    state = {"engine": None, "enabled": False, "threshold": 0.5, "latest_score": 0.0}

    def client():
        # Just a URL holder - constructing it opens no connection.
        if state["engine"] is None and available:
            state["engine"] = client_cls(host="localhost", port=port)
        return state["engine"]

    def toggle(enabled):
        if enabled and not available:
            return f"{label} client not available on this machine", gr.update(value=False)
        state["enabled"] = enabled
        if enabled and not client().health():
            state["enabled"] = False
            return (
                f"Cannot reach the {label} server on localhost:{port} - is it running?",
                gr.update(value=False),
            )
        return (
            f"{label} inference enabled" if enabled else f"{label} inference disabled",
            gr.update(),
        )

    def update_threshold(value):
        state["threshold"] = value

    def loop():
        while True:
            if state["enabled"] and state["engine"] is not None:
                frame = camera.get_frame()
                if frame is not None:
                    try:
                        state["latest_score"] = state["engine"].run(frame)
                    except Exception as e:
                        print(f"{label} inference error: {e}")
            # Slower cadence than LiZAD's loop - each call spins up a Lightning
            # predict loop server-side, so polling faster just queues work.
            time.sleep(interval)

    def stream():
        while True:
            # Idle when the tab is hidden. Cheap per tick, but every live
            # generator holds a queue worker, and it was those workers that
            # made tab switching wait.
            if active_tab["name"] != tab_name:
                yield gr.update(), gr.update()
                time.sleep(1.0)
                continue
            if state["enabled"]:
                score = state["latest_score"]
                yield round(score, 3), verdict_html(score, state["threshold"])
            else:
                yield 0.0, (
                    f'<div style="font-size:1.2em;color:{COLOR_STATUS_GREY};">'
                    f"{label} inference disabled</div>"
                )
            time.sleep(0.5)

    def start_training():
        if not available:
            return f"{label} client not available on this machine"
        try:
            return client().train().get("message", "Training requested")
        except Exception as e:
            # Covers a stopped server, a rejected concurrent run, and anything
            # the server raised before it could start the thread.
            return f"Could not start training: {e}"

    def stream_training_status():
        while True:
            # This one makes a blocking HTTP call, so polling it while the tab
            # is hidden was the worst offender - two of these were hitting
            # their servers every 1.5s regardless of what you were looking at.
            if active_tab["name"] != tab_name:
                yield gr.update()
                time.sleep(2.0)
                continue
            if not available:
                yield f"{label} client not available on this machine"
            else:
                yield format_training_status(client().train_status(), label, port)
            time.sleep(1.5)

    def evaluate_raw():
        """Raw /evaluate result, or None if unavailable. Used by the
        cross-detector comparison, which needs the numbers rather than prose."""
        if not available:
            return None
        try:
            return client().evaluate()
        except Exception as e:
            print(f"{label} evaluation failed: {e}")
            return None

    def evaluate():
        """Scores every captured image and reports how well normal separates
        from defect. This is the number the platform exists to produce - a
        live score tells you it runs, this tells you whether it works."""
        if not available:
            return f"{label} client not available on this machine", gr.update()
        try:
            r = client().evaluate()
        except Exception as e:
            return f"Evaluation failed: {e}", gr.update()

        if not r.get("ok"):
            return r.get("message", "Evaluation failed"), gr.update()

        gap, max_normal, min_defect = r["gap"], r["max_normal"], r["min_defect"]
        suggested = r["suggested_threshold"]
        lines = [
            f"{label}: normal max={max_normal:.3f}  defect min={min_defect:.3f}  "
            f"gap={gap:+.3f}",
            f"({r['n_normal']} normal, {r['n_defect']} defect images evaluated)",
        ]
        if gap > 0:
            lines.append(f"Clean separation. Threshold set to {suggested}.")
        else:
            lines.append(
                "Scores OVERLAP - no threshold separates these cleanly. "
                "More images may help; a different approach may be needed."
            )
        # Only move the slider when there's a real gap to sit in - snapping it
        # to the midpoint of overlapping distributions would look authoritative
        # while being meaningless.
        slider = gr.update(value=suggested) if gap > 0 else gr.update()
        if gap > 0:
            state["threshold"] = suggested
        return "\n".join(lines), slider

    threading.Thread(target=loop, daemon=True).start()

    return types.SimpleNamespace(
        state=state,
        toggle=toggle,
        update_threshold=update_threshold,
        stream=stream,
        start_training=start_training,
        stream_training_status=stream_training_status,
        evaluate=evaluate,
        evaluate_raw=evaluate_raw,
    )


patchcore = make_detector_runtime(
    "PatchCore", PatchCoreClient, PATCHCORE_AVAILABLE, 8001, TAB_PATCHCORE
)
efficientad = make_detector_runtime(
    "EfficientAD", EfficientAdClient, EFFICIENTAD_AVAILABLE, 8002, TAB_EFFICIENTAD
)


def status_html():
    if camera.connected:
        return f'<div style="display:flex;align-items:center;gap:8px;font-weight:600;"><span style="width:14px;height:14px;border-radius:50%;background:{COLOR_STATUS_GREEN};display:inline-block;"></span>Camera Connected</div>'
    return f'<div style="display:flex;align-items:center;gap:8px;font-weight:600;"><span style="width:14px;height:14px;border-radius:50%;background:{COLOR_STATUS_RED};display:inline-block;"></span>No Camera</div>'


def compute_sharpness(frame):
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def compute_brightness(frame):
    if frame is None:
        return 0.0
    return round(float(np.mean(frame)), 1)


def no_camera_placeholder():
    img = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.putText(img, "No Camera Connected", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (112, 112, 112), 2, cv2.LINE_AA)
    cv2.putText(img, "Waiting for device...", (70, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (112, 112, 112), 1, cv2.LINE_AA)
    return img


def stream_frames():
    """Feeds the four plain camera views. Only the visible one is actually
    sent - the rest get gr.update(), which is a no-op rather than an encode.
    The numeric readouts are cheap, so they always refresh."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            shown = no_camera_placeholder()
            sharpness = brightness = 0.0
        else:
            shown = for_display(frame)
            # Measured on the full frame, not the downscaled copy - resizing
            # would change the Laplacian variance and make the focus readout
            # depend on display size.
            sharpness = compute_sharpness(frame)
            brightness = compute_brightness(frame)

        tab = active_tab["name"]
        yield (
            shown if tab == TAB_LIVE else gr.update(),
            shown if tab == TAB_CAMERA else gr.update(),
            shown if tab == TAB_PATCHCORE else gr.update(),
            shown if tab == TAB_EFFICIENTAD else gr.update(),
            sharpness,
            sharpness,
            brightness,
            status_html(),
        )
        time.sleep(0.2)


def toggle_exposure_auto(auto):
    camera.set_exposure_auto(auto)
    return gr.update(interactive=not auto)


def update_exposure(value):
    camera.set_exposure_value(value)


def toggle_gain_auto(auto):
    camera.set_gain_auto(auto)
    return gr.update(interactive=not auto)


def update_gain(value):
    camera.set_gain_value(value)


def toggle_wb_auto(auto):
    camera.set_wb_auto(auto)


def lock_camera_settings(locked, exposure_auto_value, gain_auto_value):
    if locked:
        exposure_val = camera.get_current_exposure()
        gain_val = camera.get_current_gain()
        if exposure_val is None or gain_val is None:
            return (
                gr.update(value=False),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                "Cannot lock - no camera connected",
            )
        camera.set_exposure_value(exposure_val)
        camera.set_gain_value(gain_val)
        return (
            gr.update(),
            gr.update(interactive=False),
            gr.update(value=exposure_val, interactive=False),
            gr.update(interactive=False),
            gr.update(value=gain_val, interactive=False),
            gr.update(interactive=False),
            f"Locked - exposure {exposure_val:.0f}µs, gain {gain_val:.1f}dB frozen",
        )
    return (
        gr.update(),
        gr.update(interactive=True),
        gr.update(interactive=not exposure_auto_value),
        gr.update(interactive=True),
        gr.update(interactive=not gain_auto_value),
        gr.update(interactive=True),
        "Unlocked - settings can be adjusted",
    )


def label_from_choice(choice):
    return choice.split(" (")[0] if choice else None


def label_dir(category, subtype):
    return os.path.join(DATA_ROOT, category, subtype) if subtype else os.path.join(DATA_ROOT, category)


def count_images(category, subtype):
    folder = label_dir(category, subtype)
    return len(os.listdir(folder)) if os.path.isdir(folder) else 0


def discover_taxonomy():
    """Merge the built-in TAXONOMY with whatever category/subtype folders
    already exist on disk, so subtypes added via the custom-subtype box in
    past sessions still show up."""
    tree = {category: list(subs) for category, subs in TAXONOMY.items()}
    if not os.path.isdir(DATA_ROOT):
        return tree
    for category in os.listdir(DATA_ROOT):
        if not os.path.isdir(os.path.join(DATA_ROOT, category)):
            continue
        tree.setdefault(category, [])
        cat_path = os.path.join(DATA_ROOT, category)
        for sub in os.listdir(cat_path):
            if os.path.isdir(os.path.join(cat_path, sub)) and sub not in tree[category]:
                tree[category].append(sub)
    return tree


def category_choices(tree):
    choices = []
    for category in sorted(tree):
        subs = tree[category]
        total = sum(count_images(category, s) for s in subs) if subs else count_images(category, None)
        choices.append(f"{category} ({total})")
    return choices


def subtype_choices(tree, category):
    return [f"{s} ({count_images(category, s)})" for s in sorted(tree.get(category, []))]


def choice_for_value(value, choices):
    if not value:
        return None
    return next((c for c in choices if c.startswith(value + " (")), (choices[0] if choices else None))


def current_selection(category_choice, subtype_choice):
    category = label_from_choice(category_choice)
    subtype = label_from_choice(subtype_choice) if subtype_choice else None
    return category, subtype


def images_for(category, subtype):
    folder = label_dir(category, subtype)
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder))[-8:]


def on_category_change(category_choice):
    tree = discover_taxonomy()
    category = label_from_choice(category_choice)
    subs = subtype_choices(tree, category)
    sub_value = subs[0] if subs else None
    subtype = label_from_choice(sub_value) if sub_value else None
    # Switching label clears the selection - those paths belong to the folder
    # you just navigated away from.
    return gr.update(choices=subs, value=sub_value), images_for(category, subtype), [], ""


def on_subtype_change(category_choice, subtype_choice):
    category, subtype = current_selection(category_choice, subtype_choice)
    return images_for(category, subtype), [], ""


def format_selection(selected):
    """Gradio's gallery only highlights the last-clicked thumbnail, so with
    multi-select this list is the operator's only view of what's actually
    staged for deletion. Worth being explicit about the count."""
    if not selected:
        return ""
    header = f"{len(selected)} selected:\n"
    return header + "\n".join(os.path.basename(p) for p in selected)


def on_gallery_select(evt: gr.SelectData, category_choice, subtype_choice, selected):
    """Clicking a thumbnail toggles it in or out of the selection, so several
    can be culled in one pass."""
    category, subtype = current_selection(category_choice, subtype_choice)
    images = images_for(category, subtype)
    selected = list(selected or [])
    if evt.index < len(images):
        path = images[evt.index]
        if path in selected:
            selected.remove(path)
        else:
            selected.append(path)
    return selected, format_selection(selected)


def clear_selection():
    return [], ""


def open_capture_folder(category_choice, subtype_choice):
    """Opens the current label's folder in the desktop file manager, for bulk
    operations the gallery isn't suited to (mass delete, move, inspect).

    Note this opens on whichever machine is *running* app.py - the Jetson -
    not on the machine viewing the browser, if those differ."""
    category, subtype = current_selection(category_choice, subtype_choice)
    folder = os.path.abspath(label_dir(category, subtype))
    os.makedirs(folder, exist_ok=True)
    try:
        subprocess.Popen(["xdg-open", folder])
        return f"Opened {folder} in the file manager on the machine running this app"
    except (FileNotFoundError, OSError) as e:
        # Headless box, or no desktop session - still tell them the path so
        # they can get there over SSH.
        return f"Could not open a file manager ({e}). The folder is: {folder}"


def delete_selected_images(selected_paths, category_choice, subtype_choice):
    category, subtype = current_selection(category_choice, subtype_choice)

    removed, missing, failed = 0, 0, []
    for path in selected_paths or []:
        if not path:
            continue
        if not os.path.exists(path):
            # Already gone - deleted in the file manager, or a stale selection
            # left over from an earlier view. Not an error, but don't silently
            # report it as "nothing selected".
            missing += 1
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            failed.append(f"{os.path.basename(path)} ({e})")

    if not selected_paths:
        msg = "Nothing selected - click thumbnails to select them first"
    else:
        parts = []
        if removed:
            parts.append(f"deleted {removed} image{'s' if removed != 1 else ''}")
        if missing:
            parts.append(f"{missing} already gone")
        if failed:
            parts.append(f"failed on {', '.join(failed)}")
        msg = "; ".join(parts).capitalize() if parts else "Nothing to delete"

    tree = discover_taxonomy()
    new_categories = category_choices(tree)
    new_category_value = choice_for_value(category, new_categories)
    new_subtypes = subtype_choices(tree, category)
    new_subtype_value = choice_for_value(subtype, new_subtypes)

    return (
        msg,
        gr.update(choices=new_categories, value=new_category_value),
        gr.update(choices=new_subtypes, value=new_subtype_value),
        images_for(category, subtype),
        [],
        "",
    )


def capture_and_save(category_choice, subtype_choice, custom_subtype):
    frame = camera.get_frame()
    if frame is None:
        tree = discover_taxonomy()
        category = label_from_choice(category_choice)
        return (
            "No frame available yet - is the camera connected?",
            gr.update(choices=category_choices(tree)),
            gr.update(choices=subtype_choices(tree, category)),
            [],
            "",
        )

    category, subtype = current_selection(category_choice, subtype_choice)
    if custom_subtype.strip():
        subtype = custom_subtype.strip()

    folder = label_dir(category, subtype)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(folder, f"{timestamp}.jpg")
    cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    last_saved_path["path"] = filepath

    tree = discover_taxonomy()
    new_categories = category_choices(tree)
    new_category_value = choice_for_value(category, new_categories)
    new_subtypes = subtype_choices(tree, category)
    new_subtype_value = choice_for_value(subtype, new_subtypes)

    return (
        f"Saved to {filepath}",
        gr.update(choices=new_categories, value=new_category_value),
        gr.update(choices=new_subtypes, value=new_subtype_value),
        images_for(category, subtype),
        "",
    )


def undo_last_capture(category_choice, subtype_choice):
    path = last_saved_path["path"]
    if path and os.path.exists(path):
        os.remove(path)
        last_saved_path["path"] = None
        msg = f"Removed {path}"
    else:
        msg = "Nothing to undo"

    category, subtype = current_selection(category_choice, subtype_choice)
    tree = discover_taxonomy()
    new_categories = category_choices(tree)
    new_category_value = choice_for_value(category, new_categories)
    new_subtypes = subtype_choices(tree, category)
    new_subtype_value = choice_for_value(subtype, new_subtypes)

    return (
        msg,
        gr.update(choices=new_categories, value=new_category_value),
        gr.update(choices=new_subtypes, value=new_subtype_value),
        images_for(category, subtype),
    )


# Light, high-contrast theme built from the internal UI design standards palette,
# not a generic "dark mode" look - this is a glance-readable HMI, not a dev tool demo.
THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
).set(
    body_background_fill=COLOR_GREY_BG,
    body_background_fill_dark=COLOR_GREY_BG,
    block_background_fill=COLOR_WHITE,
    block_background_fill_dark=COLOR_WHITE,
    block_border_color="#dddddd",
    button_primary_background_fill=COLOR_PRIMARY_BLUE,
    button_primary_background_fill_hover="#3457c2",
    button_primary_text_color=COLOR_WHITE,
    button_secondary_background_fill=COLOR_WHITE,
    button_secondary_border_color=COLOR_STATUS_GREY,
    button_secondary_text_color=COLOR_PRIMARY_BLUE,
)

CSS = """
.gradio-container { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

#header {
    text-align: center;
    padding: 1.25em 0 1em 0;
    margin-bottom: 0.75em;
    border-bottom: 3px solid #3e6be2;
}
#header h1 {
    font-weight: 800;
    letter-spacing: -0.03em;
    font-size: 1.8em;
    color: #1a1a1a;
    margin: 0;
}
#header p {
    color: #707070;
    font-size: 0.95em;
    margin-top: 0.25em;
}

.card {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.08) !important;
    border: 1px solid #e5e5e5 !important;
}

#live-feed-panel {
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
}

button.lg { border-radius: 10px !important; font-weight: 600 !important; }

/* ---------------------------------------------------------------------
   Touch targets.

   Deliberately styled against plain HTML elements (button, input, label)
   rather than Gradio's own class names - those changed between 5.x and 6.x
   and silently broke an earlier gallery rule. Element selectors are stable
   across versions.

   48px is the usual minimum for a reliable finger target; the defaults here
   were roughly 32px, which is fine with a mouse and frustrating without one.
   --------------------------------------------------------------------- */
button {
    min-height: 48px !important;
    font-size: 1.05em !important;
    padding-left: 1.1em !important;
    padding-right: 1.1em !important;
}
input[type="text"], input[type="number"], input[type="password"], textarea {
    min-height: 46px !important;
    font-size: 1.05em !important;
}
/* The control itself, and the whole label row, so the text is tappable too
   rather than just the 16px box next to it. */
input[type="checkbox"], input[type="radio"] {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
}
label > span, .gr-form label {
    font-size: 1.03em !important;
}
[data-testid="checkbox"] label, [data-testid="radio"] label,
fieldset label {
    padding: 0.5em 0.4em !important;
    line-height: 1.6 !important;
}

/* Sliders: the default track and thumb are both hard to grab accurately. */
input[type="range"] {
    height: 34px !important;
}
input[type="range"]::-webkit-slider-thumb {
    width: 30px !important;
    height: 30px !important;
}
input[type="range"]::-moz-range-thumb {
    width: 30px !important;
    height: 30px !important;
}

/* Tabs are the primary navigation and were sized for a cursor. */
.tab-nav button, button.tab-nav-button {
    min-height: 54px !important;
    font-size: 1.08em !important;
}

#delete-btn { background: #eb432f !important; color: white !important; border: none !important; }
#delete-btn:hover { background: #c93520 !important; }

/* "What is this approach / when to use it" panel at the top of each detector
   tab - visually distinct from the controls below so it reads as reference
   material rather than another setting. */
.approach-info {
    background: #f8f9fc !important;
    border-left: 4px solid #3e6be2 !important;
    border-radius: 8px !important;
    padding: 0.5em 1.25em !important;
    margin-bottom: 1em !important;
}
.approach-info h3 { margin-top: 0.4em !important; color: #1a1a1a !important; }
.approach-info p { font-size: 0.92em !important; color: #444 !important; }

/* Recent captures: a plain horizontal filmstrip, not a boxed gallery widget.
   Deliberately NOT flex-filling the page - an earlier version stretched to
   fill remaining height, which is what made it read as a separate "window".

   Sizing targets the <img> tags directly rather than Gradio's internal
   wrapper classes, which changed between 5.x and 6.x and silently broke an
   earlier version of this rule. */
#capture-gallery {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
#capture-gallery img {
    height: 140px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 6px;
}
"""

with gr.Blocks(title="ArgusVision", theme=THEME, css=CSS) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("# ArgusVision")
        gr.Markdown(
            "Vision inspection comparison platform — capture one dataset, then evaluate "
            "it against multiple detection approaches to find which fits your process"
        )

    with gr.Tabs() as main_tabs:
        with gr.Tab(TAB_LIVE):
            status_display = gr.HTML(status_html())

            with gr.Row():
                with gr.Column(scale=3):
                    live_feed = gr.Image(label="Live Feed", streaming=True, height=560)
                    sharpness_display = gr.Number(label="Focus Sharpness (higher = sharper)", interactive=False)

                with gr.Column(scale=1):
                    initial_tree = discover_taxonomy()
                    initial_categories = category_choices(initial_tree)
                    category_radio = gr.Radio(choices=initial_categories, label="Category", value=initial_categories[0])
                    initial_category = label_from_choice(initial_categories[0])
                    initial_subtypes = subtype_choices(initial_tree, initial_category)
                    subtype_radio = gr.Radio(
                        choices=initial_subtypes,
                        label="Subtype (defect type, etc.)",
                        value=initial_subtypes[0] if initial_subtypes else None,
                    )
                    custom_subtype = gr.Textbox(label="Or type a new subtype", placeholder="e.g. bent_pin")
                    capture_btn = gr.Button("Capture", variant="primary", size="lg")
                    undo_btn = gr.Button("Undo Last", variant="secondary", size="lg")
                    status = gr.Textbox(label="Status", interactive=False)
                    gr.Markdown(
                        "Click thumbnails to select them (click again to deselect), "
                        "then delete. For bulk work, open the folder instead."
                    )
                    selected_image_display = gr.Textbox(
                        label="Selected", interactive=False, lines=4
                    )
                    with gr.Row():
                        delete_btn = gr.Button("Delete Selected", elem_id="delete-btn")
                        clear_selection_btn = gr.Button("Clear", variant="secondary")
                    open_folder_btn = gr.Button("Open Image Folder", variant="secondary")

            # A list, not a single path - the gallery toggles entries in and out.
            selected_image_path = gr.State([])

            gr.Markdown("**Recent captures** — most recent last")
            gallery = gr.Gallery(
                show_label=False,
                columns=8,
                rows=1,
                height=170,
                object_fit="contain",
                preview=False,
                container=False,
                elem_id="capture-gallery",
            )

        with gr.Tab("Comparison"):
            gr.Markdown(
                """
### Which approach actually fits this process?
Runs **every reachable detector** over the same captured dataset and reports
how cleanly each separates normal from defect. The **gap** is the number that
matters — the distance between the worst normal score and the best defect
score. Positive means a threshold exists that classifies everything correctly;
the larger it is, the more tolerance you have for lighting and position drift.

Scores are **not comparable between detectors** (each normalises differently),
so rank by gap, not by raw values. Every run is saved below with its date and
dataset size, so results survive a restart and can be shown to someone later.

Start the servers for whichever detectors you want included — anything
unreachable or untrained is skipped and listed as such.
""",
                elem_classes="approach-info",
            )
            with gr.Accordion("Setup conditions (recorded with the result)", open=True):
                gr.Markdown(
                    "A result without its conditions can't be acted on — and worse, a "
                    "desk-and-handheld measurement can end up quoted later as if it were "
                    "controlled. These fields are saved with every run and appear in the "
                    "exported report. They persist between sessions, so fill them once."
                )
                _profile = load_setup_profile()
                with gr.Row():
                    setup_part = gr.Textbox(
                        label="Part / process", placeholder="e.g. Inverter driver PCB, rev C",
                        value=_profile["part"],
                    )
                    setup_defect_class = gr.Textbox(
                        label="Defect class under test", placeholder="e.g. missing MOSFETs",
                        value=_profile["defect_class"],
                    )
                with gr.Row():
                    setup_mounting = gr.Radio(
                        choices=["Fixed / fixtured", "Partly fixtured", "Handheld / loose"],
                        label="Camera & part mounting",
                        value=_profile["mounting"] or "Handheld / loose",
                    )
                    setup_distance = gr.Textbox(
                        label="Working distance", placeholder="e.g. 200 mm",
                        value=_profile["distance_mm"],
                    )
                with gr.Row():
                    setup_lighting = gr.Textbox(
                        label="Lighting", placeholder="e.g. ring light, ambient excluded",
                        value=_profile["lighting"],
                    )
                    setup_notes = gr.Textbox(
                        label="Notes", placeholder="anything that would change how this reads",
                        value=_profile["notes"],
                    )
                gr.Markdown(
                    "_Exposure and gain are read from the camera automatically at run time._"
                )

            run_comparison_btn = gr.Button(
                "Run Comparison on Captured Images", variant="primary", size="lg"
            )
            comparison_summary = gr.Textbox(label="Result", interactive=False, lines=3)
            gr.Markdown("#### This run (best first)")
            comparison_table = gr.Markdown()
            with gr.Row():
                export_report_btn = gr.Button("Export Report (Markdown)", variant="secondary")
                export_status = gr.Textbox(label="Export", interactive=False, scale=3)
            report_preview = gr.Markdown()

            gr.Markdown("#### Previous runs")
            with gr.Row():
                delete_last_run_btn = gr.Button("Delete Last Run", variant="secondary")
                clear_history_btn = gr.Button("Clear History", elem_id="delete-btn")
                history_status = gr.Textbox(label="History", interactive=False, scale=3)
            comparison_history_table = gr.Markdown(value=history_markdown())

        with gr.Tab(TAB_LIZAD):
            gr.Markdown(
                """
### LiZAD — zero-shot anomaly detection
**How it works:** compares the live image against generic text prompts
(*"a flawless PCB"* vs *"a damaged PCB"*) using a pretrained vision-language
model. Needs **no example images at all** — nothing to capture, nothing to train.

**Recommended for:** open-ended *surface* defects too varied to enumerate in
advance — scratches, cracks, discoloration, contamination. This is the one
approach that can flag a defect type nobody thought to define up front.

**Weak at:** *absence*-type defects. Measured on this hardware, a board with
7 missing MOSFETs scored only ~0.25 vs ~0.0 for a good board — a real but
fragile margin. An empty pad doesn't read as "damaged" to a generic model,
it just looks like a cleanly-manufactured unpopulated board. Use PatchCore,
EfficientAD, or Channel Check for presence/absence instead.

**No training needed** — that's the whole point of zero-shot. There is no
Train button on this tab because there is nothing to train.
""",
                elem_classes="approach-info",
            )
            inference_toggle = gr.Checkbox(label="Enable Live Inference (LiZAD, trained_on_visa)", value=False)
            inference_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    inference_overlay = gr.Image(label="Anomaly Heatmap + Bounding Boxes", height=560)

                with gr.Column(scale=1):
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Anomaly Threshold"
                    )
                    suggest_threshold_btn = gr.Button(
                        "Suggest Threshold from Captured Images", variant="secondary"
                    )
                    threshold_suggestion_status = gr.Textbox(label="Suggestion Result", interactive=False)
                    compare_prompts_btn = gr.Button(
                        "Compare Prompt Sets for Current Category/Subtype", variant="secondary"
                    )
                    prompt_comparison_result = gr.Textbox(
                        label="Prompt Set Comparison", interactive=False, lines=6
                    )
                    lock_threshold_checkbox = gr.Checkbox(label="Lock Threshold", value=False)
                    score_display = gr.Number(label="Anomaly Score (max, 0-1)", interactive=False)
                    box_count_display = gr.Number(label="Anomaly Regions Detected", interactive=False)
                    verdict_display = gr.HTML()

        with gr.Tab(TAB_PATCHCORE):
            gr.Markdown(
                """
### PatchCore — reference-image anomaly detection
**How it works:** builds a "memory bank" from **your own captured normal
images** (Live Capture → category `normal`), then flags anything that looks
statistically different from that bank. Learns what *your* good parts look
like rather than relying on a generic pretrained idea of "damage".

**Recommended for:** presence/absence and structural defects where you can
supply real good-part examples but don't want to hand-code a rule — the gap
between LiZAD's open-ended detection and Channel Check's fixed rules.

**Measured on this hardware:** cleanly separated missing MOSFETs (normal
`0.000` vs defect `0.521`) where LiZAD only managed a marginal ~0.25 margin.

**Trade-offs:** needs captured normal images before it works at all, and is
slower per frame than LiZAD (hence the ~2s refresh). No heatmap or bounding
boxes yet — score only.

**To train:** capture normal parts in Live Capture, then click **Train** below.
Takes seconds — the "memory bank" is a feature index, not a trained network.
""",
                elem_classes="approach-info",
            )
            with gr.Row():
                patchcore_train_btn = gr.Button("Train PatchCore", variant="primary")
                patchcore_train_status = gr.Textbox(
                    label="Model Status", interactive=False, scale=4
                )
            patchcore_toggle = gr.Checkbox(label="Enable PatchCore Inference", value=False)
            patchcore_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    patchcore_feed = gr.Image(label="Live Feed", height=480)

                with gr.Column(scale=1):
                    patchcore_threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Anomaly Threshold"
                    )
                    patchcore_evaluate_btn = gr.Button(
                        "Evaluate on Captured Images", variant="secondary"
                    )
                    patchcore_eval_result = gr.Textbox(
                        label="Separation", interactive=False, lines=3
                    )
                    patchcore_score_display = gr.Number(label="PatchCore Score", interactive=False)
                    patchcore_verdict_display = gr.HTML()

        with gr.Tab(TAB_EFFICIENTAD):
            gr.Markdown(
                """
### EfficientAD — reference-image detection aimed at *logical* anomalies
**How it works:** a student-teacher network plus an autoencoder branch,
trained on **your own captured normal images**. The autoencoder half exists
specifically to catch **logical** anomalies — a part *missing*, in the wrong
position, or present in the wrong quantity — rather than only surface damage.

**Recommended for:** exactly the class where LiZAD proved weak here. A missing
component isn't "damaged-looking", it's *logically* wrong, and that's what
this model was built to notice.

**Trade-offs:** unlike PatchCore's memory bank, this trains a real network —
so it takes **minutes, not seconds**, and wants substantially more normal
images before its result means anything. Score only, no heatmap yet.

**To train:** capture normal parts in Live Capture, then click **Train** below
and watch the progress. You can leave the tab; training continues server-side.
""",
                elem_classes="approach-info",
            )
            with gr.Row():
                efficientad_train_btn = gr.Button("Train EfficientAD", variant="primary")
                efficientad_train_status = gr.Textbox(
                    label="Model Status", interactive=False, scale=4
                )
            efficientad_toggle = gr.Checkbox(label="Enable EfficientAD Inference", value=False)
            efficientad_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    efficientad_feed = gr.Image(label="Live Feed", height=480)

                with gr.Column(scale=1):
                    efficientad_threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Anomaly Threshold"
                    )
                    efficientad_evaluate_btn = gr.Button(
                        "Evaluate on Captured Images", variant="secondary"
                    )
                    efficientad_eval_result = gr.Textbox(
                        label="Separation", interactive=False, lines=3
                    )
                    efficientad_score_display = gr.Number(
                        label="EfficientAD Score", interactive=False
                    )
                    efficientad_verdict_display = gr.HTML()

        with gr.Tab(TAB_CHANNEL):
            gr.Markdown(
                """
### Channel Check — deterministic color match
**How it works:** you define a region per channel and its expected color once;
every frame is then checked with a direct HSV color comparison. No model, no
GPU, no training data — just a calibrated rule.

**Recommended for:** fully-specifiable checks at *fixed* positions — "channel 3
must contain the blue wire". When you can state the rule exactly, this beats
both anomaly models: faster, fully explainable, and it can name *which*
channel failed rather than just "something looks wrong".

**Requires:** a rigid camera mount and fixed part position — regions are stored
as pixel coordinates, so anything that shifts the framing invalidates the
calibration. Lock camera settings first so exposure drift doesn't move the
sampled hues.

**Setup:** place known-good wires in every channel → enable **Calibration
Mode** → click two opposite corners of each channel to define it and sample
its color → turn Calibration Mode off for live pass/fail.

**No training needed** — this is calibration, not learning. There is no Train
button because there is no model; the calibration above *is* the setup.
""",
                elem_classes="approach-info",
            )
            calibration_mode_checkbox = gr.Checkbox(label="Calibration Mode", value=False)
            with gr.Row():
                with gr.Column(scale=3):
                    channel_image = gr.Image(label="Channel Check", height=480)
                with gr.Column(scale=1):
                    next_channel_name = gr.Textbox(label="Next Channel Name", value="channel_1")
                    channel_status = gr.Textbox(label="Status", interactive=False, lines=10)
                    undo_channel_btn = gr.Button("Undo Last Channel", variant="secondary")
                    clear_channels_btn = gr.Button("Clear All Channels", elem_id="delete-btn")

        with gr.Tab(TAB_CAMERA):
            gr.Markdown(
                "Let **Auto** settle on a good image using the preview below "
                "(aim for sharpness above ~50 and brightness in the 100-180 range), "
                "then check **Lock Camera Settings** to freeze exactly what's currently "
                "active so it can't drift or get bumped during production."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    camera_settings_preview = gr.Image(label="Live Preview", height=360)
                    with gr.Row():
                        cs_sharpness_display = gr.Number(label="Sharpness (higher = sharper)", interactive=False)
                        cs_brightness_display = gr.Number(label="Brightness (target 100-180)", interactive=False)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Exposure — Manual / Auto")
                        exposure_auto = gr.Checkbox(label="Auto Exposure", value=True)
                        exposure_slider = gr.Slider(
                            minimum=10, maximum=100000, value=10000,
                            label="Exposure Time (microseconds)", interactive=False
                        )

                    with gr.Group():
                        gr.Markdown("### Gain — Manual / Auto")
                        gain_auto = gr.Checkbox(label="Auto Gain", value=True)
                        gain_slider = gr.Slider(
                            minimum=0, maximum=48, value=0,
                            label="Gain (dB)", interactive=False
                        )

                    with gr.Group():
                        gr.Markdown("### White Balance — Manual / Auto")
                        wb_auto = gr.Checkbox(label="Auto White Balance", value=True)

                    lock_camera_checkbox = gr.Checkbox(label="Lock Camera Settings", value=False)
                    lock_camera_status = gr.Textbox(label="Lock Status", interactive=False)

    exposure_auto.change(toggle_exposure_auto, inputs=exposure_auto, outputs=exposure_slider)
    exposure_slider.release(update_exposure, inputs=exposure_slider)

    gain_auto.change(toggle_gain_auto, inputs=gain_auto, outputs=gain_slider)
    gain_slider.release(update_gain, inputs=gain_slider)

    wb_auto.change(toggle_wb_auto, inputs=wb_auto)

    lock_camera_checkbox.change(
        lock_camera_settings,
        inputs=[lock_camera_checkbox, exposure_auto, gain_auto],
        outputs=[lock_camera_checkbox, exposure_auto, exposure_slider, gain_auto, gain_slider, wb_auto, lock_camera_status],
    )

    suggest_threshold_btn.click(
        suggest_threshold,
        outputs=[threshold_slider, threshold_suggestion_status],
    )
    compare_prompts_btn.click(
        compare_prompt_sets,
        inputs=[category_radio, subtype_radio],
        outputs=prompt_comparison_result,
    )
    lock_threshold_checkbox.change(
        lock_threshold,
        inputs=lock_threshold_checkbox,
        outputs=[threshold_slider, suggest_threshold_btn],
    )

    category_radio.change(
        on_category_change,
        inputs=category_radio,
        outputs=[subtype_radio, gallery, selected_image_path, selected_image_display],
    )
    subtype_radio.change(
        on_subtype_change,
        inputs=[category_radio, subtype_radio],
        outputs=[gallery, selected_image_path, selected_image_display],
    )

    capture_btn.click(
        capture_and_save,
        inputs=[category_radio, subtype_radio, custom_subtype],
        outputs=[status, category_radio, subtype_radio, gallery, custom_subtype],
    )
    undo_btn.click(
        undo_last_capture,
        inputs=[category_radio, subtype_radio],
        outputs=[status, category_radio, subtype_radio, gallery],
    )

    gallery.select(
        on_gallery_select,
        inputs=[category_radio, subtype_radio, selected_image_path],
        outputs=[selected_image_path, selected_image_display],
    )
    delete_btn.click(
        delete_selected_images,
        inputs=[selected_image_path, category_radio, subtype_radio],
        outputs=[status, category_radio, subtype_radio, gallery, selected_image_path, selected_image_display],
    )
    clear_selection_btn.click(
        clear_selection, outputs=[selected_image_path, selected_image_display]
    )
    open_folder_btn.click(
        open_capture_folder, inputs=[category_radio, subtype_radio], outputs=status
    )

    inference_toggle.change(toggle_inference, inputs=inference_toggle, outputs=[inference_status, inference_toggle])
    threshold_slider.change(update_threshold, inputs=threshold_slider)

    patchcore_toggle.change(
        patchcore.toggle, inputs=patchcore_toggle, outputs=[patchcore_status, patchcore_toggle]
    )
    patchcore_threshold_slider.change(
        patchcore.update_threshold, inputs=patchcore_threshold_slider
    )
    patchcore_train_btn.click(patchcore.start_training, outputs=patchcore_train_status)
    patchcore_evaluate_btn.click(
        patchcore.evaluate, outputs=[patchcore_eval_result, patchcore_threshold_slider]
    )

    efficientad_toggle.change(
        efficientad.toggle, inputs=efficientad_toggle, outputs=[efficientad_status, efficientad_toggle]
    )
    efficientad_threshold_slider.change(
        efficientad.update_threshold, inputs=efficientad_threshold_slider
    )
    efficientad_train_btn.click(efficientad.start_training, outputs=efficientad_train_status)
    efficientad_evaluate_btn.click(
        efficientad.evaluate, outputs=[efficientad_eval_result, efficientad_threshold_slider]
    )

    run_comparison_btn.click(
        run_comparison,
        inputs=[
            setup_part, setup_defect_class, setup_distance,
            setup_lighting, setup_mounting, setup_notes,
        ],
        outputs=[comparison_table, comparison_summary, comparison_history_table],
    )

    def _export_and_preview():
        message, path = export_last_report()
        if not path:
            return message, ""
        with open(path, encoding="utf-8") as f:
            return message, f.read()

    export_report_btn.click(_export_and_preview, outputs=[export_status, report_preview])
    delete_last_run_btn.click(
        delete_last_run, outputs=[history_status, comparison_history_table]
    )
    clear_history_btn.click(clear_history, outputs=[history_status, comparison_history_table])

    # Tracks which tab is in front so the streaming generators can skip
    # encoding images into hidden widgets.
    main_tabs.select(on_tab_select)

    demo.load(
        stream_frames,
        outputs=[
            live_feed, camera_settings_preview, patchcore_feed, efficientad_feed,
            sharpness_display, cs_sharpness_display, cs_brightness_display, status_display,
        ],
    )
    demo.load(stream_inference, outputs=[inference_overlay, score_display, box_count_display, verdict_display])
    demo.load(patchcore.stream, outputs=[patchcore_score_display, patchcore_verdict_display])
    demo.load(efficientad.stream, outputs=[efficientad_score_display, efficientad_verdict_display])

    # These poll their servers, so they also surface "server not running"
    # without the operator having to enable inference to find out.
    demo.load(patchcore.stream_training_status, outputs=patchcore_train_status)
    demo.load(efficientad.stream_training_status, outputs=efficientad_train_status)

    calibration_mode_checkbox.change(
        toggle_calibration_mode, inputs=calibration_mode_checkbox, outputs=channel_status
    )
    channel_image.select(
        on_channel_image_click, inputs=next_channel_name, outputs=[next_channel_name, channel_status]
    )
    undo_channel_btn.click(undo_last_channel, outputs=channel_status)
    clear_channels_btn.click(clear_all_channels, outputs=channel_status)
    demo.load(stream_channel_check, outputs=[channel_image, channel_status])


if __name__ == "__main__":
    if not camera.connect():
        print("WARNING: No camera found at startup - live feed will stay blank until one connects")
    demo.launch(server_name="0.0.0.0", server_port=7860)
